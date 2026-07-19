"""Core object-detection wrapper around Ultralytics YOLO.

This is the single place that talks to the model. Everything else (image CLI,
video CLI, webcam web app) uses :class:`ObjectDetector`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """A single detected object: a bounding box + a class label + confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float
    class_id: int

    def as_dict(self) -> dict:
        return asdict(self)


# Deterministic per-class colour so the same class is always drawn the same way.
def _color_for(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id * 9973 + 1)  # stable, class-seeded
    c = rng.integers(60, 256, size=3)
    return int(c[0]), int(c[1]), int(c[2])  # BGR


class ObjectDetector:
    """Detect objects in an image (numpy BGR array) with a YOLO model.

    Parameters
    ----------
    model_path:
        Path or name of the weights. Ultralytics auto-downloads known names
        such as ``yolo11n.pt`` (nano, fast), ``yolo11s.pt``, ``yolo11m.pt`` ...
    conf:
        Minimum confidence threshold for a detection to be reported.
    device:
        ``None`` lets Ultralytics choose (GPU if available, else CPU). Pass
        ``"cpu"`` to force CPU or ``0`` for the first CUDA GPU.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.25,
        device: str | int | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on one BGR image and return the list of detections."""
        results = self.model.predict(
            image, conf=self.conf, device=self.device, verbose=False
        )
        detections: list[Detection] = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        label=names[class_id],
                        confidence=confidence,
                        class_id=class_id,
                    )
                )
        return detections

    @staticmethod
    def draw(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        """Return a copy of ``image`` with bounding boxes and labels drawn."""
        out = image.copy()
        h = out.shape[0]
        thickness = max(1, round(h / 400))
        font_scale = max(0.4, h / 1000)

        for det in detections:
            color = _color_for(det.class_id)
            p1 = (int(det.x1), int(det.y1))
            p2 = (int(det.x2), int(det.y2))
            cv2.rectangle(out, p1, p2, color, thickness)

            label = f"{det.label} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            # Filled label background above the box.
            top = max(0, p1[1] - th - baseline)
            cv2.rectangle(
                out, (p1[0], top), (p1[0] + tw, top + th + baseline), color, -1
            )
            cv2.putText(
                out,
                label,
                (p1[0], top + th),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        return out
