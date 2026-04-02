"""
detect.py — YOLOv11 + InsightFace detection, unified saliency heatmap output.

Design rules:
  - InsightFace is ONLY invoked when YOLO fires a 'person' detection above threshold.
  - Output is always a normalized [0,1] float32 heatmap of the same H×W as the input frame.
  - Callers compute the weighted centroid from the heatmap — detect.py never returns a crop.
"""

import numpy as np
import cv2
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class weights — tune per content type via CLI or config
# ---------------------------------------------------------------------------

CLASS_WEIGHTS: dict = {
    "person":       1.0,
    "face":         0.7,    # lowered from 1.4 — face jitter at 4K causes spazzing
    "car":          0.7,
    "motorcycle":   0.7,
    "bicycle":      0.6,
    "sports ball":  0.8,
    "dog":          0.5,
    "cat":          0.5,
    "_default":     0.3,    # any YOLO class not listed above
}

# Minimum YOLO confidence to register a detection
YOLO_CONF_THRESHOLD = 0.40    # raised from 0.35 — fewer false detections

# Minimum YOLO person confidence required to trigger InsightFace
INSIGHTFACE_GATE_THRESHOLD = 0.60   # raised from 0.4 — only confident person → face

# Gaussian blob sigma as a fraction of detection box diagonal
GAUSSIAN_SIGMA_FACTOR = 0.25


# ---------------------------------------------------------------------------
# Detection result dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    weight: float = 1.0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------

class SaliencyDetector:
    """
    Runs YOLOv11 + optional InsightFace, returns a saliency heatmap per frame.
    """

    def __init__(
        self,
        yolo_model: str = "yolo11n.pt",
        use_insightface: bool = True,
        class_weights: Optional[dict] = None,
        device: str = "cpu",            # "cuda" if GPU available
    ):
        self.class_weights = class_weights or CLASS_WEIGHTS
        self.use_insightface = use_insightface
        self.device = device

        # Lazy-load models on first call to avoid import-time overhead
        self._yolo = None
        self._yolo_model_name = yolo_model
        self._face_app = None

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            log.info("Loading YOLO model: %s", self._yolo_model_name)
            self._yolo = YOLO(self._yolo_model_name)
        return self._yolo

    def _load_insightface(self):
        if self._face_app is None:
            import insightface
            from insightface.app import FaceAnalysis
            log.info("Loading InsightFace...")
            self._face_app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            self._face_app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))
        return self._face_app

    # -----------------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single RGB frame.
        Returns list of Detection objects with weights assigned.
        """
        h, w = frame.shape[:2]
        detections: List[Detection] = []
        has_person = False

        # --- YOLO pass ---
        yolo = self._load_yolo()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = yolo(bgr, verbose=False, conf=YOLO_CONF_THRESHOLD, device=self.device)

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = yolo.names[cls_id]
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                weight = self.class_weights.get(label, self.class_weights["_default"])

                d = Detection(
                    label=label, confidence=conf, weight=weight,
                    x1=max(0, x1), y1=max(0, y1),
                    x2=min(w, x2), y2=min(h, y2)
                )
                detections.append(d)

                if label == "person" and conf >= INSIGHTFACE_GATE_THRESHOLD:
                    has_person = True

        # --- InsightFace pass (gated on YOLO person) ---
        if self.use_insightface and has_person:
            try:
                face_app = self._load_insightface()
                faces = face_app.get(bgr)
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                    conf = float(face.det_score) if hasattr(face, "det_score") else 0.9
                    d = Detection(
                        label="face",
                        confidence=conf,
                        weight=self.class_weights.get("face", 1.4),
                        x1=max(0, x1), y1=max(0, y1),
                        x2=min(w, x2), y2=min(h, y2)
                    )
                    detections.append(d)
            except Exception as e:
                log.warning("InsightFace failed on frame: %s", e)

        return detections

    # -----------------------------------------------------------------------
    # Saliency heatmap
    # -----------------------------------------------------------------------

    def build_heatmap(
        self,
        frame_shape: Tuple[int, int],
        detections: List[Detection],
    ) -> np.ndarray:
        """
        Build a normalized [0,1] float32 saliency heatmap from detections.
        One Gaussian blob per detection, sigma scaled to box diagonal, amplitude = weight * confidence.
        """
        h, w = frame_shape
        heatmap = np.zeros((h, w), dtype=np.float32)

        if not detections:
            return heatmap

        for det in detections:
            cx, cy = det.center
            box_diag = np.sqrt((det.x2 - det.x1) ** 2 + (det.y2 - det.y1) ** 2)
            sigma = max(10.0, box_diag * GAUSSIAN_SIGMA_FACTOR)
            amplitude = det.weight * det.confidence

            _add_gaussian(heatmap, cx, cy, sigma, amplitude)

        # Normalize to [0,1]
        hmap_max = heatmap.max()
        if hmap_max > 0:
            heatmap /= hmap_max

        return heatmap

    # -----------------------------------------------------------------------
    # Weighted centroid from heatmap
    # -----------------------------------------------------------------------

    @staticmethod
    def heatmap_centroid(heatmap: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Compute the weighted centroid of the heatmap.
        Returns (cx, cy) in pixel coordinates, or None if heatmap is empty.
        """
        total = heatmap.sum()
        if total < 1e-6:
            return None
        h, w = heatmap.shape
        xs = np.arange(w, dtype=np.float32)
        ys = np.arange(h, dtype=np.float32)
        cx = (heatmap.sum(axis=0) * xs).sum() / total
        cy = (heatmap.sum(axis=1) * ys).sum() / total
        return float(cx), float(cy)

    # -----------------------------------------------------------------------
    # Combined: frame → (detections, heatmap, centroid)
    # -----------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray):
        """
        Convenience: runs detect → build_heatmap → centroid in one call.
        Returns (detections, heatmap, centroid_or_None).
        """
        h, w = frame.shape[:2]
        detections = self.detect(frame)
        heatmap = self.build_heatmap((h, w), detections)
        centroid = self.heatmap_centroid(heatmap)
        return detections, heatmap, centroid


# ---------------------------------------------------------------------------
# Internal: Gaussian blob painter
# ---------------------------------------------------------------------------

def _add_gaussian(
    heatmap: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
    amplitude: float,
):
    """Paint a 2D Gaussian blob into heatmap in-place."""
    h, w = heatmap.shape
    # Bounding box for the blob (3 sigma)
    r = int(3 * sigma)
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r + 1)
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r + 1)

    if x0 >= x1 or y0 >= y1:
        return

    xs = np.arange(x0, x1, dtype=np.float32) - cx
    ys = np.arange(y0, y1, dtype=np.float32) - cy
    xx, yy = np.meshgrid(xs, ys)
    blob = amplitude * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    heatmap[y0:y1, x0:x1] += blob
