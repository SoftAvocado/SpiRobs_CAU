"""Core object-detection wrapper around Ultralytics YOLO.

This is the single place that talks to the model. Everything else (image CLI,
video CLI, webcam web app) uses :class:`ObjectDetector`.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from .classes import DETECTION_CLASSES

#: Open-vocabulary model used so the full DETECTION_CLASSES vocabulary (COCO +
#: the ~200 table items in classes.py) can be detected without any training.
#: Larger variants (yolov8m/l/x-worldv2.pt) are more accurate but slower.
DEFAULT_MODEL = "yolov8s-worldv2.pt"


def _weights_dir() -> Path:
    """Single directory where downloaded YOLO weights are cached.

    Configurable via ``YOLO_WEIGHTS_DIR``; defaults to a ``weights`` folder next
    to the Ultralytics config dir (``YOLO_CONFIG_DIR``), falling back to the
    user's home. Keeping one fixed location means weights download only once
    instead of into whatever the current working directory happens to be.
    """
    env = os.environ.get("YOLO_WEIGHTS_DIR")
    if env:
        return Path(env)
    config = os.environ.get("YOLO_CONFIG_DIR")
    base = Path(config) if config else Path.home() / ".config" / "Ultralytics"
    return base / "weights"


def _load_model(model_path: str, loader=YOLO):
    """Load a model, caching auto-downloaded weights in one fixed dir.

    An explicit path (existing file, or a name containing a directory) is used
    as-is. A bare known name such as ``yolo11n.pt`` is resolved against the
    weights cache; if absent, Ultralytics downloads it there (once) rather than
    into the current working directory. ``loader`` is the model class to build
    (``YOLO`` for standard models, ``YOLOWorld`` for open-vocabulary ones).
    """
    p = Path(model_path)
    if p.exists() or p.parent != Path("."):
        return loader(str(model_path))

    weights_dir = _weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / p.name
    if target.exists():
        return loader(str(target))

    # Ultralytics downloads bare names into the current working directory, so
    # run the download from inside the cache dir, then restore the cwd.
    cwd = Path.cwd()
    try:
        os.chdir(weights_dir)
        return loader(p.name)
    finally:
        os.chdir(cwd)


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
    """Detect objects in an image (numpy BGR array).

    Always runs open-vocabulary detection over the fixed vocabulary defined in
    :mod:`src.classes` (the 80 COCO classes + ~200 common table items). To
    change what can be detected, edit ``classes.py`` — that is the single,
    only place to add or remove objects.

    Parameters
    ----------
    model_path:
        YOLO-World weights. Defaults to :data:`DEFAULT_MODEL`. Use a larger
        ``-worldv2`` variant for more accuracy at the cost of speed.
    conf:
        Minimum confidence threshold for a detection to be reported.
    device:
        ``None`` lets Ultralytics choose (GPU if available, else CPU). Pass
        ``"cpu"`` to force CPU or ``0`` for the first CUDA GPU.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf: float = 0.25,
        device: str | int | None = None,
    ) -> None:
        from ultralytics import YOLOWorld

        self.conf = conf
        self.device = device
        self.classes = list(DETECTION_CLASSES)
        self.model = _load_model(model_path, loader=YOLOWorld)
        self.model.set_classes(self.classes)

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
