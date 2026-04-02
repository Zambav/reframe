"""
smooth.py — Camera path smoothing: Kalman filter, EMA, and shot mode classification.

Shot modes (AutoFlip-inspired):
  STATIONARY  — subject not moving; lock crop, only move on hard threshold breach
  PAN         — slow lateral movement; EMA with low alpha, smooth follow
  TRACK       — active subject movement; Kalman filter, responsive but filtered

All pixel-based thresholds are SCALED to the source frame diagonal.
Reference: 1920x1080 diagonal = 2209 px. All thresholds are expressed as
fractions of this diagonal, then multiplied by the actual frame diagonal.
"""

import numpy as np
from collections import deque
from enum import Enum
from typing import Optional, Tuple
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reference frame diagonal (1920x1080)
# ---------------------------------------------------------------------------
_REF_DIAGONAL = float(np.sqrt(1920**2 + 1080**2))   # 2209.07 px


def _scale(val: float, diagonal: float) -> float:
    """Scale a threshold from reference diagonal to actual frame diagonal."""
    return val * (diagonal / _REF_DIAGONAL)


# ---------------------------------------------------------------------------
# Thresholds expressed as FRACTIONS of reference diagonal
# Tune these. The actual pixel threshold = fraction × (frame_diagonal / 2209)
# ---------------------------------------------------------------------------

# Shot mode classification (based on centroid variance — scale-aware)
WINDOW_FRAMES = 20

# Variance below this (normalized to ref diagonal²) → STATIONARY
# 200/2209 ≈ 9% of diagonal — lock crop aggressively
_STATIONARY_VARIANCE_FRAC  = 0.009          # × ref_diagonal² → threshold in px²
_PAN_VARIANCE_FRAC          = 0.040          # × ref_diagonal² → threshold in px²

# STATIONARY mode: only move crop if centroid drifts this far from lock point
# Expressed as fraction of ref diagonal
_STATIONARY_BREACH_FRAC     = 0.045          # ~100px at 1080p, ~200px at 4K

# EMA alpha for PAN mode (0.0 = frozen, 1.0 = instant snap)
_EMA_ALPHA_PAN_RAW          = 0.025          # very slow — don't chase small movements

# Kalman measurement noise — increase to dampen responsiveness
_KALMAN_MEASUREMENT_NOISE   = 20.0          # was 10 — more smoothing

# Innovation threshold fraction (reset trigger)
_INNOVATION_FRAC            = 0.18           # ~397px at 1080p, ~795px at 4K


# ---------------------------------------------------------------------------
# Shot mode enum
# ---------------------------------------------------------------------------

class ShotMode(Enum):
    STATIONARY = "stationary"
    PAN        = "pan"
    TRACK      = "track"


# ---------------------------------------------------------------------------
# Simple 1D Kalman filter
# ---------------------------------------------------------------------------

class Kalman1D:
    """
    1D constant-velocity Kalman filter.
    State: [position, velocity]
    """

    def __init__(self, process_noise: float, measurement_noise: float):
        self.Q = process_noise
        self.R = measurement_noise
        self.x = None
        self.P = np.eye(2) * 1000

    def reset(self, position: float):
        self.x = np.array([position, 0.0])
        self.P = np.eye(2) * 1000

    def update(self, measurement: float) -> Tuple[float, float]:
        dt = 1.0
        F = np.array([[1, dt], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * self.Q
        R = np.array([[self.R]])

        if self.x is None:
            self.reset(measurement)
            return measurement, 0.0

        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        y = measurement - (H @ x_pred)[0]
        innovation = abs(y)

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self.x = x_pred + (K @ [[y]]).flatten()
        self.P = (np.eye(2) - K @ H) @ P_pred

        return float(self.x[0]), float(innovation)


# ---------------------------------------------------------------------------
# 2D smoother: wraps two Kalman1D (x and y)
# ---------------------------------------------------------------------------

class CropSmoother:
    """
    Full 2D crop center smoother with shot mode classification.

    All thresholds are resolution-aware: scaled by frame diagonal.
    Usage:
        smoother = CropSmoother(src_w=3840, src_h=2160)
        smoother.reset(cx, cy)
        smooth_cx, smooth_cy = smoother.update(raw_cx, raw_cy)
    """

    def __init__(
        self,
        src_w: int,
        src_h: int,
        process_noise: float = 0.03,
        measurement_noise: float = _KALMAN_MEASUREMENT_NOISE,
    ):
        self._src_w = src_w
        self._src_h = src_h
        self._diagonal = float(np.sqrt(src_w**2 + src_h**2))
        self._scale = self._diagonal / _REF_DIAGONAL

        # Pre-compute scaled thresholds
        self._stationary_thresh = _STATIONARY_VARIANCE_FRAC * _REF_DIAGONAL**2
        self._pan_thresh       = _PAN_VARIANCE_FRAC * _REF_DIAGONAL**2
        self._breach_thresh    = _STATIONARY_BREACH_FRAC * _REF_DIAGONAL * self._scale
        self._innovation_thresh = _INNOVATION_FRAC * _REF_DIAGONAL * self._scale
        self._ema_alpha = _EMA_ALPHA_PAN_RAW

        log.debug(
            "CropSmoother (%.0fx%.0f, diagonal=%.0f, scale=%.2f): "
            "stationary_var=%.0f, pan_var=%.0f, breach=%.1f, innovation=%.1f",
            src_w, src_h, self._diagonal, self._scale,
            self._stationary_thresh, self._pan_thresh,
            self._breach_thresh, self._innovation_thresh
        )

        self.kx = Kalman1D(process_noise, measurement_noise)
        self.ky = Kalman1D(process_noise, measurement_noise)

        self._history_x: deque = deque(maxlen=WINDOW_FRAMES)
        self._history_y: deque = deque(maxlen=WINDOW_FRAMES)

        self._ema_x: Optional[float] = None
        self._ema_y: Optional[float] = None
        self._lock_x: Optional[float] = None
        self._lock_y: Optional[float] = None
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None

        self._mode: ShotMode = ShotMode.TRACK
        self._last_innovation: float = 0.0

    # -----------------------------------------------------------------------
    # Accessor for cut detector
    # -----------------------------------------------------------------------

    @property
    def innovation_thresh(self) -> float:
        return self._innovation_thresh

    # -----------------------------------------------------------------------
    # Reset (call on scene cut)
    # -----------------------------------------------------------------------

    def reset(self, cx: Optional[float] = None, cy: Optional[float] = None):
        self._history_x.clear()
        self._history_y.clear()
        self._ema_x = cx
        self._ema_y = cy
        self._lock_x = cx
        self._lock_y = cy
        if cx is not None:
            self.kx.reset(cx)
        if cy is not None:
            self.ky.reset(cy)
        self._mode = ShotMode.TRACK
        log.debug("Smoother reset at (%.1f, %.1f)", cx or 0, cy or 0)

    # -----------------------------------------------------------------------
    # Shot mode classification
    # -----------------------------------------------------------------------

    def _classify_mode(self) -> ShotMode:
        if len(self._history_x) < 5:
            return ShotMode.TRACK
        var = np.var(list(self._history_x)) + np.var(list(self._history_y))
        if var < self._stationary_thresh:
            return ShotMode.STATIONARY
        elif var < self._pan_thresh:
            return ShotMode.PAN
        else:
            return ShotMode.TRACK

    # -----------------------------------------------------------------------
    # Main update
    # -----------------------------------------------------------------------

    def update(
        self,
        raw_cx: Optional[float],
        raw_cy: Optional[float],
    ) -> Tuple[float, float]:
        if raw_cx is None or raw_cy is None:
            if self._last_x is None:
                return 0.0, 0.0
            log.debug("No detections — holding last position")
            return self._last_x, self._last_y

        self._history_x.append(raw_cx)
        self._history_y.append(raw_cy)
        self._mode = self._classify_mode()

        if self._mode == ShotMode.STATIONARY:
            smooth_cx, smooth_cy = self._apply_stationary(raw_cx, raw_cy)
        elif self._mode == ShotMode.PAN:
            smooth_cx, smooth_cy = self._apply_ema(raw_cx, raw_cy)
        else:
            smooth_cx, smooth_cy, innovation = self._apply_kalman(raw_cx, raw_cy)
            self._last_innovation = max(innovation, self._last_innovation * 0.9)

        self._last_x = smooth_cx
        self._last_y = smooth_cy
        return smooth_cx, smooth_cy

    # -----------------------------------------------------------------------
    # Per-mode implementations
    # -----------------------------------------------------------------------

    def _apply_stationary(self, raw_cx: float, raw_cy: float) -> Tuple[float, float]:
        if self._lock_x is None:
            self._lock_x = raw_cx
            self._lock_y = raw_cy
            return raw_cx, raw_cy

        dist = np.sqrt((raw_cx - self._lock_x)**2 + (raw_cy - self._lock_y)**2)
        if dist > self._breach_thresh:
            log.debug("STATIONARY breach (%.1fpx > %.1fpx) — updating lock", dist, self._breach_thresh)
            self._lock_x = raw_cx
            self._lock_y = raw_cy
        return self._lock_x, self._lock_y

    def _apply_ema(self, raw_cx: float, raw_cy: float) -> Tuple[float, float]:
        if self._ema_x is None:
            self._ema_x = raw_cx
            self._ema_y = raw_cy
        else:
            a = self._ema_alpha
            self._ema_x = a * raw_cx + (1 - a) * self._ema_x
            self._ema_y = a * raw_cy + (1 - a) * self._ema_y
        return self._ema_x, self._ema_y

    def _apply_kalman(self, raw_cx: float, raw_cy: float) -> Tuple[float, float, float]:
        sx, ix = self.kx.update(raw_cx)
        sy, iy = self.ky.update(raw_cy)
        innovation = max(ix, iy)
        return sx, sy, innovation

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def mode(self) -> ShotMode:
        return self._mode

    @property
    def last_innovation(self) -> float:
        return self._last_innovation
