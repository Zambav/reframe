# Reframe Pipeline — Agent Brief
> Everything you need to know before touching a single line of code.

---

## What We're Building

A **local, Python-native, codec-proof video reframing pipeline** — the spiritual successor to Google AutoFlip, but built clean from scratch with modern tooling.

The goal: take a wide/landscape video, intelligently crop it to 9:16 (or any target AR) by tracking what matters in the frame — faces, objects, cars, tools, text regions — using a saliency-aware crop window that moves smoothly like a real camera operator would.

---

## Why Not Use Existing Tools

| Tool | Problem |
|---|---|
| Google AutoFlip | Bazel + old OpenCV + C++ toolchain = build hell. Legacy. Not worth it. |
| pyautoflip | Face-only detection, weak smoother, OpenCV VideoWriter crashes on AV1/modern codecs |
| OpenCV VideoWriter | Codec negotiation is platform-dependent and breaks silently. Never use it for output. |

---

## Architecture Overview

```
FFmpeg decode pipe → raw numpy frames
       │
       ├──► YOLOv11 ─────────► object boxes (person, car, motorcycle, sports ball, etc.)
       ├──► InsightFace ──────► face boxes + landmarks + head pose (only when YOLO person fires)
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
         Shot Mode Classifier (15–30 frame window)
         ├── STATIONARY  → lock crop, move only on hard threshold breach
         ├── PAN         → EMA, low alpha (~0.05), slow smooth follow
         └── TRACK       → Kalman filter, higher process noise, responsive
                   │
                   ▼
         Crop / AR Math + Edge Case Policy
                   │
                   ▼
FFmpeg encode pipe → output file
```

---

## File Responsibilities

```
reframe/
├── pipeline.py     # FFmpeg pipe orchestration — the main loop
├── detect.py       # YOLOv11 + InsightFace, outputs merged saliency heatmap
├── smooth.py       # Kalman + EMA + shot mode classifier
├── crop.py         # AR math, crop window, edge case / padding policy
├── scenes.py       # PySceneDetect integration + Kalman innovation fallback
└── cli.py          # argparse entry point
```

---

## Critical Design Decisions (Do Not Deviate)

### 1. FFmpeg pipes for ALL video I/O
**Never use `cv2.VideoWriter`**. OpenCV's writer has platform-dependent codec negotiation and breaks on AV1, VP9, and VFR clips. All decode and encode goes through FFmpeg subprocess pipes:

```python
# Decode: ffmpeg → stdout → numpy
# Encode: numpy → stdin → ffmpeg
```

Pre-transcode any AV1/VP9 source to H.264 before feeding the pipeline:
```bash
ffmpeg -i input.mkv -c:v libx264 -pix_fmt yuv420p -crf 18 -preset fast -an input_h264.mp4
```

### 2. Gate InsightFace on YOLO person detection
Running both models every frame halves throughput. Only invoke InsightFace when YOLO's `person` class fires above confidence threshold (default: 0.4). No person box → skip InsightFace that frame entirely.

### 3. Saliency heatmap, not raw centroid
Don't average detection centers. Build a Gaussian heatmap — one blob per detection, sigma scaled to box size, amplitude scaled by class weight. The crop target is the heatmap's weighted centroid. This handles edge cases (two subjects at opposite ends of frame) far more gracefully.

### 4. Shot mode classification uses centroid variance
Over a 20-frame rolling window, compute variance of target (cx, cy):
- `variance < STATIONARY_THRESH` → STATIONARY mode, lock crop
- `variance < PAN_THRESH` with consistent direction → PAN mode, EMA follow
- Otherwise → TRACK mode, Kalman

These thresholds are **explicit named constants** in `smooth.py`. Do not magic-number them.

### 5. Crop edge case policy (AutoFlip-inspired)
Google AutoFlip explicitly handles these — so must we:

| Case | Policy |
|---|---|
| Subject near frame edge, crop would clip them | Bias toward center — don't follow subject into edge |
| Subject fully out of frame (occlusion) | Hold last known position with exponential decay toward center |
| Multiple subjects in opposite quadrants | Fall back to scene centroid — never split-the-difference |
| No detections this frame | Hold last crop position, do not snap to center |

These are implemented in `crop.py::compute_crop_window()` as explicit conditionals, not emergent behavior.

### 6. PySceneDetect cut detection + Kalman innovation fallback
PySceneDetect catches hard cuts. Soft transitions / dissolves it misses. Add a secondary check: if the Kalman filter's **innovation** (prediction error) exceeds `INNOVATION_RESET_THRESH`, treat it as a scene boundary and reset smoother state. This catches what PySceneDetect misses.

---

## Dependency Surface

```
ultralytics       # YOLOv11 — pip install ultralytics
insightface       # face precision — pip install insightface
onnxruntime       # InsightFace backend — pip install onnxruntime
scenedetect       # shot cuts — pip install scenedetect
numpy             # frame math
ffmpeg-python     # FFmpeg subprocess wrapper — pip install ffmpeg-python
filterpy          # Kalman filter — pip install filterpy
```

No OpenCV VideoWriter. No pyautoflip. No Bazel. No C++ toolchain.
OpenCV is still used for frame manipulation (resize, Gaussian) — just never for encode.

Install all:
```bash
pip install ultralytics insightface onnxruntime scenedetect numpy ffmpeg-python filterpy opencv-python
```

---

## Class Weights (Starting Defaults)

These go in `detect.py::CLASS_WEIGHTS`. Tune per content type:

```python
CLASS_WEIGHTS = {
    # YOLO COCO class names
    "person":        1.0,
    "face":          1.4,   # InsightFace detections get this weight
    "car":           0.7,
    "motorcycle":    0.7,
    "bicycle":       0.6,
    "sports ball":   0.8,
    "dog":           0.5,
    "cat":           0.5,
    # default for any unspecified class
    "_default":      0.3,
}
```

---

## Build Order — Do Not Skip Ahead

1. **`pipeline.py`** — FFmpeg pipes, dumb center crop, end-to-end I/O validated
2. **`detect.py`** — YOLO only first, confirm detection boxes render correctly
3. **`detect.py`** — add InsightFace, build saliency heatmap merge
4. **`smooth.py`** — Kalman + EMA + shot mode classifier
5. **`scenes.py`** — PySceneDetect + innovation fallback
6. **`crop.py`** — full edge case policy
7. **`cli.py`** — wire everything together
8. **Benchmark** — pyautoflip output vs. raw YOLO crop vs. full pipeline on same clip

---

## Output Spec

Default output: `{input_stem}_reframed_{ar}.mp4`
- Codec: `libx264`, `yuv420p`, CRF 18
- AR: 9:16 default, configurable
- Audio: passthrough from source (FFmpeg handles remux)
- No audio processing in Python — let FFmpeg handle it

---

## What pyautoflip Is Still Good For

Run it on the H.264 pre-transcoded clip. Use its output as a **reference diff** — not a foundation. When our pipeline is working, do a three-way compare:

1. pyautoflip output
2. Our pipeline, YOLO-only, no smoothing
3. Our pipeline, full stack

That diff will tell you exactly where each approach wins and loses.

---

## Notes on Google AutoFlip's Edge Case Handling

AutoFlip handles the crop edge cases through its "retargeting" step, which:
- Scores candidate crop windows across a range of positions
- Penalizes windows that clip salient regions
- Uses a path optimization pass (similar to what we're calling shot mode classification) to select the globally smooth path, not the locally optimal one per frame

Our implementation approximates this with the explicit conditionals in `crop.py` + the Kalman smoother. It's not identical but it covers the same failure modes.
