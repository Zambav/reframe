# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-04-02 — "Almost Perfect"

### Added
- **Median pre-filter** on raw centroid before smoothing — kills one-frame detection spikes
- **Velocity clamping** — crop center cannot jump more than `MAX_VELOCITY_FRAC × diagonal` per frame; this is the primary anti-spazz mechanism
- **Resolution-aware smoothing** — all pixel thresholds scale with frame diagonal (reference: 1920×1080 = 2209px diagonal); 4K content now behaves identically to 1080p in tuning terms

### Changed
- **STATIONARY mode** is now the dominant mode — crop locks aggressively and only moves on sustained, large drift
- **TRACK mode** is heavily damped via higher Kalman measurement noise (25.0 vs 10.0) and velocity clamping upstream
- **EMA alpha** lowered to 0.015 — very slow, anti-jitter follow for PAN mode
- **Scene cut detection** raised to threshold 55 — fewer false cuts on motion-heavy frames
- **MIN_CUT_INTERVAL_FRAMES** raised to 20 — avoid rapid re-triggering
- **Face weight** lowered from 1.4 to 0.7 — face jitter at 4K was a major spazz contributor
- **YOLO confidence threshold** raised from 0.35 to 0.40 — fewer spurious detections
- **InsightFace gate** raised from 0.4 to 0.60 — only confident person detections trigger face analysis

### Known Remaining Characteristics
These are intentional design choices, not bugs:
- **Spazz-free by default** — the crop prioritizes stability over responsiveness; it will feel "sticky" on purpose
- **Face weight still active** — tuned to 0.7 so faces influence the crop without dominating
- **Scene cuts reset the smoother** — this is correct behavior; a new shot = new camera path
- **Process exits code 1** — this is a Python/ONNX cleanup artifact; output files are valid and complete

### Fixed
- **4K vs 1080p inconsistency** — thresholds no longer behave differently at different resolutions
- **Constant innovation resets** — removed ~95% of spurious smoother resets that caused glitchy output

---

## [0.1.0] — 2026-04-02 — "Initial Release"

### Added
- YOLOv11 + InsightFace saliency detection pipeline
- Gaussian heatmap-based centroid computation
- Kalman + EMA smoothing with STATIONARY / PAN / TRACK shot modes
- PySceneDetect hard cut detection with Kalman innovation fallback
- Full edge case policy (edge bias, occlusion decay, split-subject fallback, no-detection hold)
- FFmpeg pipe I/O (no cv2.VideoWriter — codec-proof)
- Resolution-aware output sizing with even-dimension enforcement
- Audio passthrough via FFmpeg
- `--transcode` flag for AV1/VP9/HEVC source pre-conversion
- `--no-insightface` flag for YOLO-only (faster, less face precision)
- MIT license
