"""
smooth.py — Camera path smoothing: Kalman filter, EMA, and shot mode classification.

Core philosophy: be LAZY. The crop should only move when the evidence is overwhelming.
Detection noise (especially faces at 4K) causes jitter — we kill that at the source
with a median pre-filter, then apply very conservative smoothing.

Shot modes (AutoFlip-inspired):
  STATIONARY  — default. Lock crop. Only move on sustained, large drift.
  PAN         — only when centroid is moving consistently in one direction.
  TRACK       — only for genuinely fast subject motion. Capped velocity.
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


# ---------------------------------------------------------------------------
# Median pre-filter window — kills detection spike noise before smoothing
# ---------------------------------------------------------------------------
_MEDIAN_WINDOW = 5          # odd number; must be >= 3


# ---------------------------------------------------------------------------
# Thresholds as FRACTIONS of reference diagonal
# ---------------------------------------------------------------------------

# Shot mode variance thresholds (expressed as fraction of ref diagonal)
# Variance below STATIONARY → lock crop
# Variance below PAN → slow EMA follow
# Above PAN → TRACK (capped velocity) — almost never reached in practice
_VARIANCE_STATIONARY_FRAC = 0.020   # ~44px at 1080p, ~88px at 4K — very sticky
_VARIANCE_PAN_FRAC        = 0.120   # ~265px at 1080p, ~530px at 4K — almost never hit

# STATIONARY breach: only unlock if centroid drifts this far (fraction of ref diagonal)
_BREACH_FRAC = 0.07              # ~155px at 1080p, ~309px at 4K

# EMA alpha for PAN mode — very slow to avoid chasing detection noise
_EMA_ALPHA = 0.010               # very conservative — near-frozen

# Maximum crop center jump per frame (in reference diagonal units)
# This is the KEY anti-spazzing mechanism
_MAX_VELOCITY_FRAC = 0.012        # ~27px at 1080p, ~53px at 4K per frame — slower deliberate pans

# Kalman measurement noise — high = very damped, low = responsive
_KALMAN_MEASUREMENT_NOISE = 25.0  # was 10 — much more smoothing

# Innovation threshold fraction for soft cut detection
_INNOVATION_FRAC = 0.28          # ~618px at 1080p, ~1237px at 4K — very hard to trigger


# ---------------------------------------------------------------------------
# Shot mode enum
# ---------------------------------------------------------------------------

class ShotMode(Enum):
    STATIONARY = "stationary"
    PAN        = "pan"
    TRACK      = "track"


# ---------------------------------------------------------------------------
# 1D Kalman filter
# ---------------------------------------------------------------------------

class Kalman1D:
    """1D constant-velocity Kalman filter — heavily damped for stability."""

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
# CropSmoother — resolution-aware, with median pre-filter + velocity clamping
# ---------------------------------------------------------------------------

class CropSmoother:
    """
    Full 2D crop center smoother.

    Key anti-spazzing design:
    1. Median pre-filter on raw centroid — kills detection spike noise
    2. Very conservative shot mode thresholds — stay in STATIONARY almost always
    3. Velocity clamping — crop center can't jump more than MAX_VELOCITY per frame
    4. High Kalman measurement noise — heavily damped when TRACK is unavoidable

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
        self._var_stationary = (_VARIANCE_STATIONARY_FRAC * _REF_DIAGONAL) ** 2
        self._var_pan        = (_VARIANCE_PAN_FRAC * _REF_DIAGONAL) ** 2
        self._breach_thresh  = _BREACH_FRAC * _REF_DIAGONAL * self._scale
        self._max_velocity   = _MAX_VELOCITY_FRAC * _REF_DIAGONAL * self._scale
        self._innovation_thresh = _INNOVATION_FRAC * _REF_DIAGONAL * self._scale
        self._ema_alpha = _EMA_ALPHA

        log.debug(
            "CropSmoother (%.0fx%.0f, diagonal=%.0f, scale=%.2f): "
            "stationary_var=%.0f, pan_var=%.0f, breach=%.1f, "
            "max_vel=%.1f, innovation=%.1f",
            src_w, src_h, self._diagonal, self._scale,
            self._var_stationary, self._var_pan,
            self._breach_thresh, self._max_velocity, self._innovation_thresh
        )

        # Median pre-filter state
        self._median_x: deque = deque(maxlen=_MEDIAN_WINDOW)
        self._median_y: deque = deque(maxlen=_MEDIAN_WINDOW)

        self.kx = Kalman1D(process_noise, measurement_noise)
        self.ky = Kalman1D(process_noise, measurement_noise)

        self._history_x: deque = deque(maxlen=20)
        self._history_y: deque = deque(maxlen=20)

        self._ema_x: Optional[float] = None
        self._ema_y: Optional[float] = None
        self._lock_x: Optional[float] = None
        self._lock_y: Optional[float] = None
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None

        self._mode: ShotMode = ShotMode.STATIONARY
        self._last_innovation: float = 0.0

    # -----------------------------------------------------------------------
    # Accessor for CutDetector
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
        self._median_x.clear()
        self._median_y.clear()
        self._ema_x = cx
        self._ema_y = cy
        self._lock_x = cx
        self._lock_y = cy
        if cx is not None:
            self.kx.reset(cx)
        if cy is not None:
            self.ky.reset(cy)
        self._mode = ShotMode.STATIONARY
        log.debug("Smoother reset at (%.1f, %.1f)", cx or 0, cy or 0)

    # -----------------------------------------------------------------------
    # Median pre-filter
    # -----------------------------------------------------------------------

    def _median_filter(self, cx: float, cy: float) -> Tuple[float, float]:
        """Rolling median filter — kills detection spike noise."""
        self._median_x.append(cx)
        self._median_y.append(cy)
        if len(self._median_x) < _MEDIAN_WINDOW:
            return cx, cy
        return float(np.median(self._median_x)), float(np.median(self._median_y))

    # -----------------------------------------------------------------------
    # Velocity clamp
    # -----------------------------------------------------------------------

    def _clamp_velocity(
        self,
        raw_x: float,
        raw_y: float,
        prev_x: Optional[float],
        prev_y: Optional[float],
    ) -> Tuple[float, float]:
        """Prevent the crop center from jumping more than MAX_VELOCITY per frame."""
        if prev_x is None or prev_y is None:
            return raw_x, raw_y
        dx = raw_x - prev_x
        dy = raw_y - prev_y
        dist = np.sqrt(dx**2 + dy**2)
        if dist <= self._max_velocity:
            return raw_x, raw_y
        # Scale down to max velocity
        scale = self._max_velocity / dist
        new_x = prev_x + dx * scale
        new_y = prev_y + dy * scale
        log.debug("Velocity clamp: (%.1f,%.1f) → (%.1f,%.1f) [max=%.1f]",
                  raw_x, raw_y, new_x, new_y, self._max_velocity)
        return new_x, new_y

    # -----------------------------------------------------------------------
    # Shot mode classification
    # -----------------------------------------------------------------------

    def _classify_mode(self) -> ShotMode:
        if len(self._history_x) < 5:
            return ShotMode.STATIONARY
        var = np.var(list(self._history_x)) + np.var(list(self._history_y))
        if var < self._var_stationary:
            return ShotMode.STATIONARY
        elif var < self._var_pan:
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

        # Step 1: Median filter on raw detection
        cx, cy = self._median_filter(raw_cx, raw_cy)

        # Step 2: Velocity clamp
        cx, cy = self._clamp_velocity(cx, cy, self._last_x, self._last_y)

        # Step 3: Update history for mode classification
        self._history_x.append(cx)
        self._history_y.append(cy)

        self._mode = self._classify_mode()

        if self._mode == ShotMode.STATIONARY:
            smooth_cx, smooth_cy = self._apply_stationary(cx, cy)
        elif self._mode == ShotMode.PAN:
            smooth_cx, smooth_cy = self._apply_ema(cx, cy)
        else:
            smooth_cx, smooth_cy, innovation = self._apply_kalman(cx, cy)
            self._last_innovation = max(innovation, self._last_innovation * 0.9)

        self._last_x = smooth_cx
        self._last_y = smooth_cy
        return smooth_cx, smooth_cy

    # -----------------------------------------------------------------------
    # Per-mode implementations
    # -----------------------------------------------------------------------

    def _apply_stationary(self, cx: float, cy: float) -> Tuple[float, float]:
        """Lock crop. Only unlock on sustained large drift."""
        if self._lock_x is None:
            self._lock_x = cx
            self._lock_y = cy
            return cx, cy

        dist = np.sqrt((cx - self._lock_x)**2 + (cy - self._lock_y)**2)
        if dist > self._breach_thresh:
            log.debug("STATIONARY breach (%.1fpx > %.1fpx) — updating lock",
                      dist, self._breach_thresh)
            self._lock_x = cx
            self._lock_y = cy
        return self._lock_x, self._lock_y

    def _apply_ema(self, cx: float, cy: float) -> Tuple[float, float]:
        """Very slow EMA — don't chase small movements."""
        if self._ema_x is None:
            self._ema_x = cx
            self._ema_y = cy
        else:
            a = self._ema_alpha
            self._ema_x = a * cx + (1 - a) * self._ema_x
            self._ema_y = a * cy + (1 - a) * self._ema_y
        return self._ema_x, self._ema_y

    def _apply_kalman(self, cx: float, cy: float) -> Tuple[float, float, float]:
        """Kalman — capped by velocity clamp upstream."""
        sx, ix = self.kx.update(cx)
        sy, iy = self.ky.update(cy)
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
