"""
scenes.py — Scene cut detection and smoother reset triggers.

Two mechanisms:
  1. PySceneDetect: hard cut detection via content-aware analysis
  2. Kalman innovation fallback: catches soft transitions PySceneDetect misses

On any detected cut → call smoother.reset() so the camera path doesn't
try to smooth across a scene boundary.
"""

import numpy as np
import logging
from pathlib import Path
from typing import List, Set

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PySceneDetect threshold
# Set conservatively — better to miss a soft cut than false-trigger on action
# ---------------------------------------------------------------------------

# ContentDetector threshold — higher = only true hard cuts detected
# 30 was too sensitive (false cuts on motion-heavy frames)
# 50-60 is a better range for well-produced content
SCENEDETECT_THRESHOLD = 55.0

# Minimum frames between two detected cuts (avoid rapid re-triggering)
MIN_CUT_INTERVAL_FRAMES = 20


# ---------------------------------------------------------------------------
# Pre-analysis: get all cut frame indices before pipeline runs
# ---------------------------------------------------------------------------

def detect_cuts(video_path: str) -> Set[int]:
    """
    Run PySceneDetect on the input file and return a set of frame indices
    where hard cuts occur. Run this BEFORE the frame loop starts.

    Returns empty set if scenedetect is not installed.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except ImportError:
        log.warning("scenedetect not installed — scene cut detection disabled")
        return set()

    log.info("Running PySceneDetect on %s ...", video_path)

    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=SCENEDETECT_THRESHOLD))
    manager.detect_scenes(video, show_progress=False)

    scene_list = manager.get_scene_list()
    cut_frames: Set[int] = set()

    for i, (start, end) in enumerate(scene_list):
        if i > 0:  # First scene has no incoming cut
            cut_frames.add(start.get_frames())

    log.info("PySceneDetect found %d cuts", len(cut_frames))
    return cut_frames


# ---------------------------------------------------------------------------
# Per-frame cut checker (used inside pipeline loop)
# ---------------------------------------------------------------------------

class CutDetector:
    """
    Stateful cut detector used inside the frame loop.
    Combines PySceneDetect pre-computed cuts with Kalman innovation fallback.
    """

    def __init__(
        self,
        precomputed_cuts: Set[int],
        innovation_thresh: float = 0.0,  # 0 = disabled; pass real value from CropSmoother
    ):
        self._cuts = precomputed_cuts
        self._innovation_thresh = innovation_thresh
        self._last_cut_frame = -1

    def is_cut(self, frame_idx: int, innovation: float) -> bool:
        """
        Returns True if a cut should be triggered at this frame.
        Triggers on:
          1. PySceneDetect pre-computed hard cut at this frame index
          2. Kalman innovation spike above threshold (soft transition / missed cut)

        Respects MIN_CUT_INTERVAL_FRAMES to avoid rapid re-triggering.
        """
        if frame_idx - self._last_cut_frame < MIN_CUT_INTERVAL_FRAMES:
            return False

        # Hard cut from PySceneDetect
        if frame_idx in self._cuts:
            log.debug("Hard cut at frame %d (PySceneDetect)", frame_idx)
            self._last_cut_frame = frame_idx
            return True

        # Kalman innovation fallback
        if innovation > self._innovation_thresh:
            log.debug(
                "Soft cut at frame %d (Kalman innovation %.1f > %.1f)",
                frame_idx, innovation, self._innovation_thresh
            )
            self._last_cut_frame = frame_idx
            return True

        return False
