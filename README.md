# Reframe — Saliency-Aware Video Reframing Pipeline

> Take any wide video and intelligently crop it to 9:16 (or any aspect ratio) by tracking what matters in the frame.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What It Does

Reframe is a Python-native video reframing pipeline. Feed it a landscape video, and it outputs a vertically framed crop (or any target aspect ratio) that follows the action — faces, objects, cars — like a human camera operator would.

It is a clean-room, modern implementation inspired by Google AutoFlip, built without AutoFlip's broken C++ toolchain or codec-dependent OpenCV VideoWriter.

## Features

- **YOLOv11 object detection** — tracks people, cars, bicycles, sports balls, animals, and more
- **InsightFace face precision** — gated on YOLO person detection for efficiency
- **Saliency heatmap** — Gaussian blobs per detection, class-weighted, normalized [0,1]
- **Shot mode classification** — STATIONARY / PAN / TRACK modes based on centroid variance
- **Kalman + EMA smoothing** — temporal filtering appropriate to each shot type
- **PySceneDetect hard cuts** — pre-computed before the frame loop; resets smoother state
- **Kalman innovation fallback** — catches soft transitions PySceneDetect misses
- **Full edge case policy** — edge bias, occlusion decay, split-subject fallback, no-detection hold
- **FFmpeg I/O only** — codec-proof; no `cv2.VideoWriter`; handles AV1/VP9 via pre-transcode

## Architecture

```
FFmpeg decode pipe → raw numpy frames
       │
       ├──► YOLOv11 ─────────► object boxes (person, car, etc.)
       ├──► InsightFace ─────► face boxes + landmarks (gated on YOLO person)
       └──► (future) text detector ──► text region boxes
                   │
                   ▼
         Saliency Heatmap Builder
         (Gaussian blob per detection, class-weighted, normalized [0,1])
                   │
                   ▼
         Weighted Centroid → target (cx, cy)
                   │
                   ▼
         PySceneDetect boundary check
         ├── Hard cut → reset smoother state
         └── No cut → continue
                   │
                   ▼
         Shot Mode Classifier (20-frame rolling window)
         ├── STATIONARY → lock crop, move only on hard breach
         ├── PAN → EMA, low alpha (~0.05), slow follow
         └── TRACK → Kalman filter, responsive
                   │
                   ▼
         Crop / AR Math + Edge Case Policy
                   │
                   ▼
FFmpeg encode pipe → output file (H.264, CRF 18, audio passthrough)
```

## Installation

```bash
pip install reframe-pipeline
```

Or install from source:

```bash
git clone https://github.com/Zambav/reframe
cd reframe
pip install -e .
```

### Dependencies

All dependencies install automatically. For GPU acceleration:

```bash
# CUDA-enabled YOLO and InsightFace
pip install ultralytics insightface onnxruntime-gpu
```

FFmpeg must be installed on your system:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html or via winget:
winget install ffmpeg
```

## Quick Start

```bash
# Validate I/O first (center crop, no detection — confirms FFmpeg pipes work)
reframe input_h264.mp4 output.mp4

# Full pipeline with GPU
reframe input_h264.mp4 output.mp4 --ar 9:16 --device cuda

# AV1/VP9 source (auto pre-transcode to H.264 first)
reframe input_av1.mkv output.mp4 --ar 9:16 --transcode

# YOLO-only (skip InsightFace for speed)
reframe input.mp4 output.mp4 --ar 9:16 --no-insightface

# Square crop
reframe input.mp4 output.mp4 --ar 1:1
```

## Usage

```
reframe [-h] [--ar AR] [--device {cpu,cuda}] [--yolo-model YOLO_MODEL]
        [--no-insightface] [--transcode] [--crf CRF] [--debug]
        input output
```

| Argument | Default | Description |
|---|---|---|
| `input` | — | Input video path |
| `output` | — | Output video path |
| `--ar` | `9:16` | Target aspect ratio as `W:H` |
| `--device` | `cpu` | Inference device (`cpu` or `cuda`) |
| `--yolo-model` | `yolo11n.pt` | YOLO model (any Ultralytics model) |
| `--no-insightface` | False | Skip InsightFace (YOLO-only detection) |
| `--transcode` | False | Pre-transcode source to H.264 first |
| `--crf` | 18 | Output quality (lower = better, 18 = visually lossless) |
| `--debug` | False | Enable debug logging |

## How It Works

### Detection

YOLOv11 runs on every frame. When a `person` detection fires above confidence threshold (0.4), InsightFace runs too — giving tighter face boxes and head pose. All detections are merged into a **saliency heatmap**: one Gaussian blob per detection, sigma proportional to box size, amplitude proportional to class weight × confidence.

### Smoothing

A 20-frame rolling window tracks centroid variance:

| Variance | Mode | Behavior |
|---|---|---|
| `< 50` | STATIONARY | Lock crop; only moves if subject drifts >80px |
| `50–400` | PAN | Exponential moving average, α=0.05 |
| `> 400` | TRACK | Kalman filter — responsive but temporally filtered |

### Cut Detection

PySceneDetect pre-computes all hard cuts before the frame loop. Inside the loop, a **Kalman innovation** fallback catches soft transitions: if the prediction error exceeds 120px, the smoother resets.

### Edge Cases

- **Subject near edge** — crop biases toward center; never clips a salient region
- **Occlusion** — holds last known position with slow exponential decay toward center
- **Split subjects** (opposite quadrants) — falls back to scene centroid
- **No detections** — holds last crop position; never snaps to center

## Why Not pyautoflip or Google AutoFlip?

| Tool | Problem |
|---|---|
| Google AutoFlip | Bazel + old OpenCV + C++ toolchain = build hell. Unmaintained. |
| pyautoflip | Face-only detection; weak smoother; `cv2.VideoWriter` crashes on AV1/VP9 |

Reframe uses FFmpeg pipes for all I/O (codec-proof), runs YOLOv11 + InsightFace for richer saliency, and implements an explicit edge case policy rather than hoping the smoother handles it.

## Benchmarking

When your pipeline is working, run a three-way compare on the same clip:

1. **pyautoflip** — reference
2. **YOLO-only, no smoothing** — baseline
3. **Full Reframe stack** — this project

This will show you exactly where each approach wins and where it needs tuning.

## Class Weights

Default detection weights in `detect.py`:

```python
CLASS_WEIGHTS = {
    "person":       1.0,
    "face":         1.4,   # InsightFace gets highest weight
    "car":          0.7,
    "motorcycle":   0.7,
    "bicycle":      0.6,
    "sports ball":  0.8,
    "dog":          0.5,
    "cat":          0.5,
    "_default":     0.3,
}
```

Tune these per content type. Sports content benefits from higher `sports ball`. Wildlife docs benefit from higher `dog`, `cat`.

## Key Constants

| File | Constant | Default | Effect |
|---|---|---|---|
| `smooth.py` | `STATIONARY_THRESH` | 50.0 | Variance below → lock crop |
| `smooth.py` | `PAN_THRESH` | 400.0 | Variance below → EMA follow |
| `smooth.py` | `EMA_ALPHA_PAN` | 0.05 | Pan follow speed |
| `smooth.py` | `KALMAN_PROCESS_NOISE` | 0.05 | Track responsiveness |
| `smooth.py` | `INNOVATION_RESET_THRESH` | 120.0 | Cut detection sensitivity |
| `crop.py` | `EDGE_SAFE_MARGIN_FACTOR` | 0.05 | Edge bias margin |
| `crop.py` | `OCCLUSION_DECAY_ALPHA` | 0.02 | Occlusion decay speed |
| `detect.py` | `YOLO_CONF_THRESHOLD` | 0.35 | Minimum detection confidence |
| `detect.py` | `INSIGHTFACE_GATE_THRESHOLD` | 0.4 | Person conf to trigger face detection |

## Module Map

| File | Responsibility |
|---|---|
| `pipeline.py` | FFmpeg pipe I/O, main loop, frame dispatch |
| `detect.py` | YOLOv11 + InsightFace, Gaussian saliency heatmap |
| `smooth.py` | Kalman + EMA + shot mode classifier (STATIONARY/PAN/TRACK) |
| `crop.py` | AR math, crop window, edge case policy |
| `scenes.py` | PySceneDetect + Kalman innovation fallback |
| `cli.py` | argparse entry point, wires everything together |

## License

MIT — see [LICENSE](LICENSE).
