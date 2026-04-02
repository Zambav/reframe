"""
smooth.py — Camera path smoothing: Kalman filter, EMA, and shot mode classification.

Shot modes (AutoFlip-inspired):
  STATIONARY  — subject not moving; lock crop, only move on hard threshold breach
  PAN         — slow lateral movement; EMA with low alpha, smooth follow
  TRACK       — active subject movement; Kalman filter, responsive but filtered
"""

import numpy as np
from collections import deque
from enum import Enum
from typing import Optional, Tuple
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Explicit thresholds — tune these, do not magic-number them in logic
# ---------------------------------------------------------------------------

# Shot mode classification (based on centroid variance over rolling window)
WINDOW_FRAMES         = 20       # rolling window length for mode classification
STATIONARY_THRESH     = 200.0    # variance below this → STATIONARY (raised from 50 — lock more)
PAN_THRESH            = 2000.0   # variance below this (and > STATIONARY) → PAN (raised from 400)
                                 # above PAN_THRESH → TRACK

# STATIONARY mode: only move crop if centroid drifts this many pixels from lock point
STATIONARY_BREACH_PX  = 100      # raised from 80 — less jitter on lock

# EMA alpha for PAN mode (0.0 = frozen, 1.0 = instant snap)
EMA_ALPHA_PAN         = 0.03     # lowered from 0.05 — slower, smoother follow

# Kalman process noise for TRACK mode (higher = more responsive, more jittery)
KALMAN_PROCESS_NOISE  = 0.03     # lowered from 0.05 — less jitter when in TRACK mode

# Kalman innovation threshold for scene cut fallback (see scenes.py)
# Only reset smoother on LARGE sudden jumps — not normal tracking noise
INNOVATION_RESET_THRESH = 500.0  # raised from 120 — was resetting every 10 frames


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

    def __init__(self, process_noise: float = KALMAN_PROCESS_NOISE, measurement_noise: float = 10.0):
        self.Q = process_noise      # process noise covariance
        self.R = measurement_noise  # measurement noise covariance
        self.x = None               # state [pos, vel]
        self.P = np.eye(2) * 1000   # state covariance (high initial uncertainty)

    def reset(self, position: float):
        self.x = np.array([position, 0.0])
        self.P = np.eye(2) * 1000

    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Returns (smoothed_position, innovation).
        innovation = |predicted - measured|, used for scene cut fallback.
        """
        dt = 1.0  # per-frame
        F = np.array([[1, dt], [0, 1]])   # state transition
        H = np.array([[1, 0]])            # measurement matrix
        Q = np.eye(2) * self.Q
        R = np.array([[self.R]])

        if self.x is None:
            self.reset(measurement)
            return measurement, 0.0

        # Predict
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # Innovation
        y = measurement - (H @ x_pred)[0]
        innovation = abs(y)

        # Kalman gain
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        # Update
        self.x = x_pred + (K @ [[y]]).flatten()
        self.P = (np.eye(2) - K @ H) @ P_pred

        return float(self.x[0]), float(innovation)


# ---------------------------------------------------------------------------
# 2D smoother: wraps two Kalman1D (x and y)
# ---------------------------------------------------------------------------

class CropSmoother:
    """
    Full 2D crop center smoother with shot mode classification.

    Usage:
        smoother = CropSmoother()
        smoother.reset(cx, cy)
        smooth_cx, smooth_cy = smoother.update(raw_cx, raw_cy)
    """

    def __init__(
        self,
        process_noise: float = KALMAN_PROCESS_NOISE,
        measurement_noise: float = 10.0,
    ):
        self.kx = Kalman1D(process_noise, measurement_noise)
        self.ky = Kalman1D(process_noise, measurement_noise)

        # Rolling window of raw centroid positions for mode classification
        self._history_x: deque = deque(maxlen=WINDOW_FRAMES)
        self._history_y: deque = deque(maxlen=WINDOW_FRAMES)

        # EMA state
        self._ema_x: Optional[float] = None
        self._ema_y: Optional[float] = None

        # STATIONARY lock point
        self._lock_x: Optional[float] = None
        self._lock_y: Optional[float] = None

        # Last known good position (used when no detections)
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None

        self._mode: ShotMode = ShotMode.TRACK
        self._last_innovation: float = 0.0

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
        if var < STATIONARY_THRESH:
            return ShotMode.STATIONARY
        elif var < PAN_THRESH:
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
        """
        Given a raw target centroid (or None if no detections this frame),
        return a smoothed (cx, cy).
        """
        # No detections: hold last known position
        if raw_cx is None or raw_cy is None:
            if self._last_x is None:
                return 0.0, 0.0     # no history at all, caller will center
            log.debug("No detections — holding last position")
            return self._last_x, self._last_y

        # Update history
        self._history_x.append(raw_cx)
        self._history_y.append(raw_cy)

        # Classify shot mode
        self._mode = self._classify_mode()

        if self._mode == ShotMode.STATIONARY:
            smooth_cx, smooth_cy = self._apply_stationary(raw_cx, raw_cy)

        elif self._mode == ShotMode.PAN:
            smooth_cx, smooth_cy = self._apply_ema(raw_cx, raw_cy)

        else:  # TRACK
            smooth_cx, smooth_cy, innovation = self._apply_kalman(raw_cx, raw_cy)
            self._last_innovation = max(innovation, self._last_innovation * 0.9)

        self._last_x = smooth_cx
        self._last_y = smooth_cy
        return smooth_cx, smooth_cy

    # -----------------------------------------------------------------------
    # Per-mode smoothing implementations
    # -----------------------------------------------------------------------

    def _apply_stationary(self, raw_cx: float, raw_cy: float) -> Tuple[float, float]:
        """Lock crop, only move on hard threshold breach."""
        if self._lock_x is None:
            self._lock_x = raw_cx
            self._lock_y = raw_cy
            return raw_cx, raw_cy

        dist = np.sqrt((raw_cx - self._lock_x)**2 + (raw_cy - self._lock_y)**2)
        if dist > STATIONARY_BREACH_PX:
            log.debug("STATIONARY breach (%.1fpx) — updating lock", dist)
            self._lock_x = raw_cx
            self._lock_y = raw_cy

        return self._lock_x, self._lock_y

    def _apply_ema(self, raw_cx: float, raw_cy: float) -> Tuple[float, float]:
        """Exponential moving average — slow smooth follow."""
        if self._ema_x is None:
            self._ema_x = raw_cx
            self._ema_y = raw_cy
        else:
            a = EMA_ALPHA_PAN
            self._ema_x = a * raw_cx + (1 - a) * self._ema_x
            self._ema_y = a * raw_cy + (1 - a) * self._ema_y
        return self._ema_x, self._ema_y

    def _apply_kalman(
        self, raw_cx: float, raw_cy: float
    ) -> Tuple[float, float, float]:
        """Kalman filter — responsive but temporally filtered."""
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
        """Latest Kalman innovation magnitude — used by scenes.py for cut detection."""
        return self._last_innovation
