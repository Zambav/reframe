"""
crop.py — Aspect ratio math, crop window computation, and edge case policy.

This is the AutoFlip-inspired layer. All edge cases are explicit conditionals,
not emergent behavior from the smoother.

Edge case policy:
  1. Subject near frame edge, crop would clip them
     → Bias crop toward center; never follow subject into edge.
  2. Subject fully out of frame (occlusion)
     → Hold last known position with exponential decay toward center.
  3. Multiple subjects in opposite quadrants
     → Fall back to scene centroid — never split-the-difference.
  4. No detections this frame
     → Hold last crop position; do not snap to center.
"""

import numpy as np
from typing import Optional, Tuple
import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edge case policy constants
# ---------------------------------------------------------------------------

# How close (px) to frame edge before we start biasing back toward center
EDGE_SAFE_MARGIN_FACTOR = 0.05     # 5% of frame dimension

# Decay rate when subject is occluded (per-frame blend toward center)
OCCLUSION_DECAY_ALPHA = 0.02       # slow drift back to center

# If two clusters of salient pixels are this far apart (as fraction of frame width),
# treat as "opposite quadrants" → fall back to scene centroid
SPLIT_SUBJECT_THRESH = 0.45


# ---------------------------------------------------------------------------
# Crop window
# ---------------------------------------------------------------------------

def compute_crop_window(
    smooth_cx: float,
    smooth_cy: float,
    src_w: int,
    src_h: int,
    out_w: int,
    out_h: int,
) -> Tuple[int, int, int, int]:
    """
    Given a smoothed crop center (cx, cy), compute the (x, y, w, h) crop window
    that fits within the source frame.

    Returns (x, y, crop_w, crop_h) in source pixel coordinates.

    Edge case: if the window would go out of bounds, clamp and bias toward center.
    """
    half_w = out_w / 2
    half_h = out_h / 2

    margin_x = EDGE_SAFE_MARGIN_FACTOR * src_w
    margin_y = EDGE_SAFE_MARGIN_FACTOR * src_h

    # Clamp center so crop window stays within safe margins
    cx = np.clip(smooth_cx, half_w + margin_x, src_w - half_w - margin_x)
    cy = np.clip(smooth_cy, half_h + margin_y, src_h - half_h - margin_y)

    # Compute top-left corner
    x = int(cx - half_w)
    y = int(cy - half_h)

    # Hard clamp to frame bounds (should be redundant with above but safety net)
    x = max(0, min(x, src_w - out_w))
    y = max(0, min(y, src_h - out_h))

    return x, y, out_w, out_h


def apply_crop(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Slice a crop window from a frame."""
    return frame[y:y+h, x:x+w].copy()


# ---------------------------------------------------------------------------
# Opposite-quadrant subject detection
# ---------------------------------------------------------------------------

def detect_split_subjects(heatmap: np.ndarray) -> bool:
    """
    Returns True if the heatmap has significant mass in two distant clusters,
    indicating subjects in opposite quadrants.

    In this case, crop.py should fall back to scene centroid rather than
    trying to split the difference.
    """
    h, w = heatmap.shape
    if heatmap.max() < 1e-4:
        return False

    # Simple left/right split: compare centroid of left vs right half
    left  = heatmap[:, :w//2]
    right = heatmap[:, w//2:]

    left_mass  = left.sum()
    right_mass = right.sum()
    total_mass = left_mass + right_mass

    if total_mass < 1e-4:
        return False

    # Weighted centroid x of each half
    xs = np.arange(w//2, dtype=np.float32)
    if left_mass > 0:
        left_cx  = (left.sum(axis=0) * xs).sum() / left_mass
    else:
        left_cx = w / 4

    if right_mass > 0:
        right_cx = w//2 + (right.sum(axis=0) * xs).sum() / right_mass
    else:
        right_cx = 3 * w / 4

    separation = abs(right_cx - left_cx) / w
    both_significant = (left_mass / total_mass > 0.25) and (right_mass / total_mass > 0.25)

    return both_significant and (separation > SPLIT_SUBJECT_THRESH)


# ---------------------------------------------------------------------------
# Occlusion decay
# ---------------------------------------------------------------------------

class OcclusionHandler:
    """
    Tracks whether subjects are currently visible and handles position decay
    when they disappear (occlusion / out of frame).
    """

    def __init__(self, src_w: int, src_h: int):
        self._src_w = src_w
        self._src_h = src_h
        self._held_x: Optional[float] = None
        self._held_y: Optional[float] = None
        self._occluded_frames: int = 0

    def update(
        self,
        centroid: Optional[Tuple[float, float]],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Given the current frame's centroid (or None if no detections),
        return the effective (cx, cy) accounting for occlusion decay.
        """
        if centroid is not None:
            # Subject visible — reset occlusion state
            self._held_x, self._held_y = centroid
            self._occluded_frames = 0
            return centroid

        # No detection — subject occluded or out of frame
        self._occluded_frames += 1

        if self._held_x is None:
            return None, None

        # Decay held position toward frame center
        center_x = self._src_w / 2
        center_y = self._src_h / 2
        a = OCCLUSION_DECAY_ALPHA

        self._held_x = a * center_x + (1 - a) * self._held_x
        self._held_y = a * center_y + (1 - a) * self._held_y

        if self._occluded_frames % 30 == 0:
            log.debug(
                "Occluded %d frames — decayed to (%.1f, %.1f)",
                self._occluded_frames, self._held_x, self._held_y
            )

        return self._held_x, self._held_y

    def reset(self):
        self._held_x = None
        self._held_y = None
        self._occluded_frames = 0


# ---------------------------------------------------------------------------
# Scene centroid fallback
# ---------------------------------------------------------------------------

def scene_centroid(src_w: int, src_h: int) -> Tuple[float, float]:
    """Dead center of the source frame. Used as fallback for split subjects."""
    return src_w / 2.0, src_h / 2.0
