"""
cli.py — Command-line entry point. Wires all modules together.

Usage:
    python -m reframe.cli input.mp4 output.mp4 --ar 9:16
    python -m reframe.cli input.mp4 output.mp4 --ar 1:1 --device cuda
    python -m reframe.cli input.mp4 output.mp4 --no-insightface  # YOLO-only
    python -m reframe.cli input.mp4 output.mp4 --transcode       # auto pre-transcode
"""

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import run_pipeline, transcode_to_h264, probe_video
from .detect import SaliencyDetector, CLASS_WEIGHTS
from .smooth import CropSmoother
from .crop import compute_crop_window, apply_crop, OcclusionHandler, detect_split_subjects, scene_centroid
from .scenes import detect_cuts, CutDetector


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process function factory
# ---------------------------------------------------------------------------

def make_process_fn(
    detector: SaliencyDetector,
    smoother: CropSmoother,
    occlusion: OcclusionHandler,
    cut_detector: CutDetector,
):
    """
    Returns a process_fn(frame, meta) → cropped_frame suitable for pipeline.run_pipeline().
    Closes over the stateful objects above.
    """

    def process_fn(frame, meta):
        frame_idx  = meta["frame_idx"]
        src_w      = meta["src_w"]
        src_h      = meta["src_h"]
        out_w      = meta["out_w"]
        out_h      = meta["out_h"]

        # --- Detection ---
        detections, heatmap, raw_centroid = detector.process_frame(frame)

        # --- Occlusion handling ---
        cx_raw, cy_raw = occlusion.update(raw_centroid)

        # --- Split-subject fallback ---
        if detect_split_subjects(heatmap):
            log.debug("Frame %d: split subjects → scene centroid fallback", frame_idx)
            cx_raw, cy_raw = scene_centroid(src_w, src_h)

        # --- Smoothing ---
        cx_smooth, cy_smooth = smoother.update(cx_raw, cy_raw)

        # --- Scene cut check → reset smoother ---
        innovation = smoother.last_innovation
        if cut_detector.is_cut(frame_idx, innovation):
            smoother.reset(cx_smooth, cy_smooth)
            occlusion.reset()
            log.info("Smoother reset at frame %d", frame_idx)

        # --- Fallback if still no position ---
        if cx_smooth == 0.0 and cy_smooth == 0.0:
            cx_smooth, cy_smooth = scene_centroid(src_w, src_h)

        # --- Crop ---
        x, y, cw, ch = compute_crop_window(cx_smooth, cy_smooth, src_w, src_h, out_w, out_h)
        return apply_crop(frame, x, y, cw, ch)

    return process_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reframe — saliency-aware video reframing pipeline"
    )
    parser.add_argument("input",  type=str, help="Input video path")
    parser.add_argument("output", type=str, help="Output video path")
    parser.add_argument(
        "--ar", type=str, default="9:16",
        help="Target aspect ratio, e.g. 9:16 or 1:1 (default: 9:16)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda"],
        help="Inference device (default: cpu)"
    )
    parser.add_argument(
        "--yolo-model", type=str, default="yolo11n.pt",
        help="YOLO model weights (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--no-insightface", action="store_true",
        help="Disable InsightFace (YOLO-only detection)"
    )
    parser.add_argument(
        "--transcode", action="store_true",
        help="Pre-transcode input to H.264 before processing (recommended for AV1/VP9 sources)"
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="Output CRF quality (lower = better, default: 18)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse aspect ratio
    try:
        ar_w, ar_h = (int(x) for x in args.ar.split(":"))
    except ValueError:
        print(f"Invalid aspect ratio: {args.ar}. Use format W:H e.g. 9:16")
        sys.exit(1)

    input_path = args.input

    # Optional pre-transcode
    if args.transcode:
        log.info("Pre-transcoding to H.264...")
        input_path = transcode_to_h264(input_path)

    # Pre-compute scene cuts before frame loop
    log.info("Detecting scene cuts...")
    cuts = detect_cuts(input_path)

    # Initialize stateful objects
    meta = probe_video(input_path)
    src_w, src_h = meta["width"], meta["height"]

    detector  = SaliencyDetector(
        yolo_model=args.yolo_model,
        use_insightface=not args.no_insightface,
        device=args.device,
    )
    smoother  = CropSmoother(src_w=src_w, src_h=src_h)
    occlusion = OcclusionHandler(src_w, src_h)
    cut_det   = CutDetector(cuts, innovation_thresh=smoother.innovation_thresh)

    process_fn = make_process_fn(detector, smoother, occlusion, cut_det)

    # Run
    log.info("Starting reframe: %s → %s  [AR %d:%d]", input_path, args.output, ar_w, ar_h)
    run_pipeline(
        input_path=input_path,
        output_path=args.output,
        target_ar=(ar_w, ar_h),
        process_fn=process_fn,
    )


if __name__ == "__main__":
    main()
