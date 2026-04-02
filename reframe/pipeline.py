"""
pipeline.py — FFmpeg pipe orchestration
The main loop. Handles all video I/O via FFmpeg subprocess pipes.
Never uses cv2.VideoWriter.
"""

import subprocess
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Generator
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FFmpeg probe
# ---------------------------------------------------------------------------

def probe_video(input_path: str) -> dict:
    """Return basic video metadata via ffprobe."""
    import json
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", input_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")

    # Handle fractional frame rates like "30000/1001"
    fps_str = video.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "duration": float(video.get("duration", 0)),
        "codec": video.get("codec_name", "unknown"),
    }


# ---------------------------------------------------------------------------
# Frame decode generator
# ---------------------------------------------------------------------------

def decode_frames(
    input_path: str,
    width: int,
    height: int,
) -> Generator[np.ndarray, None, None]:
    """
    Yield raw RGB frames from input_path via FFmpeg stdout pipe.
    Never uses cv2.VideoCapture.
    """
    frame_bytes = width * height * 3  # RGB24

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", input_path,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            yield frame.copy()
    finally:
        proc.stdout.close()
        proc.wait()


# ---------------------------------------------------------------------------
# Encoder context manager
# ---------------------------------------------------------------------------

class FrameEncoder:
    """
    Write raw RGB frames to an output file via FFmpeg stdin pipe.
    Use as a context manager.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        input_path: str,           # source — used for audio passthrough
        crf: int = 18,
        preset: str = "fast",
    ):
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_bytes = width * height * 3

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-y",                               # overwrite output
            # video stdin
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-s", f"{width}x{height}",
            "-i", "pipe:0",
            # audio from source
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "1:a:0?",                   # audio optional
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-preset", preset,
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        """Write one RGB frame."""
        assert frame.shape == (self.height, self.width, 3), \
            f"Frame shape mismatch: expected ({self.height}, {self.width}, 3), got {frame.shape}"
        self._proc.stdin.write(frame.tobytes())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._proc.stdin:
            self._proc.stdin.close()
        self._proc.wait()
        stderr = self._proc.stderr.read().decode("utf-8", errors="replace")
        if stderr.strip():
            log.debug("FFmpeg encoder stderr: %s", stderr)


# ---------------------------------------------------------------------------
# Pre-transcode helper
# ---------------------------------------------------------------------------

def transcode_to_h264(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Pre-transcode AV1/VP9/HEVC source to H.264 for pipeline compatibility.
    Returns the output path.
    """
    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_h264.mp4")

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        "-c:a", "copy",
        output_path,
    ]
    log.info("Pre-transcoding %s → %s", input_path, output_path)
    subprocess.run(cmd, check=True)
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str,
    output_path: str,
    target_ar: Tuple[int, int] = (9, 16),
    process_fn=None,    # callable(frame, meta) → cropped_frame; None = center crop
):
    """
    Full pipeline entry point.

    Args:
        input_path:   Source video (H.264 recommended; see transcode_to_h264)
        output_path:  Destination file
        target_ar:    Output aspect ratio as (width_parts, height_parts), e.g. (9,16)
        process_fn:   Frame processor. Receives (frame: ndarray, meta: dict) → ndarray.
                      If None, falls back to dumb center crop (useful for I/O validation).
    """
    meta = probe_video(input_path)
    src_w, src_h = meta["width"], meta["height"]
    fps = meta["fps"]

    # Compute output dimensions
    ar_w, ar_h = target_ar
    out_h = src_h
    out_w = int(src_h * ar_w / ar_h)
    if out_w > src_w:
        out_w = src_w
        out_h = int(src_w * ar_h / ar_w)
    # Ensure even dimensions (x264 requirement)
    out_w = out_w if out_w % 2 == 0 else out_w - 1
    out_h = out_h if out_h % 2 == 0 else out_h - 1

    frame_meta = {
        "src_w": src_w,
        "src_h": src_h,
        "out_w": out_w,
        "out_h": out_h,
        "fps": fps,
        "frame_idx": 0,
    }

    log.info(
        "Pipeline: %dx%d → %dx%d @ %.2f fps",
        src_w, src_h, out_w, out_h, fps
    )

    if process_fn is None:
        process_fn = _center_crop

    with FrameEncoder(output_path, out_w, out_h, fps, input_path) as enc:
        for frame in decode_frames(input_path, src_w, src_h):
            frame_meta["frame_idx"] += 1
            out_frame = process_fn(frame, frame_meta)
            enc.write(out_frame)

    log.info("Done → %s", output_path)


# ---------------------------------------------------------------------------
# Fallback: dumb center crop (I/O validation only)
# ---------------------------------------------------------------------------

def _center_crop(frame: np.ndarray, meta: dict) -> np.ndarray:
    """
    Dumb center crop. Used only to validate FFmpeg pipe I/O before detection is wired in.
    Do not use this in production.
    """
    src_h, src_w = frame.shape[:2]
    out_w, out_h = meta["out_w"], meta["out_h"]
    x = (src_w - out_w) // 2
    y = (src_h - out_h) // 2
    return frame[y:y+out_h, x:x+out_w]
