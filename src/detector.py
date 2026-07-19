"""Core object-detection wrapper around Ultralytics YOLO.

This is the single place that talks to the model. Everything else (image CLI,
video CLI, webcam web app) uses :class:`ObjectDetector`.
"""

from __future__ import annotations

import hashlib
import os
import sys
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


def _prompt_cache_path(model_path: str, classes: Sequence[str]) -> Path:
    """Where the *prompted* model for this exact vocabulary is cached.

    The filename embeds a hash of everything the embeddings depend on: the base
    weights, the Ultralytics version (its text encoder could change between
    releases) and the class list itself. So editing ``classes.py`` silently
    produces a different path and the cache is rebuilt — there is no way to
    accidentally keep using embeddings for an old vocabulary.
    """
    import ultralytics

    key = "\n".join([Path(model_path).name, ultralytics.__version__, *classes])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return _weights_dir() / f"{Path(model_path).stem}-vocab-{digest}.pt"


def _save_prompt_cache(model, path: Path) -> None:
    """Persist a freshly prompted model so the next run can skip CLIP.

    Must be called *before* any inference: running ``predict`` fuses conv+BN
    layers in place, and saving a fused model then reloading it changes the
    numerics enough to flip borderline detections (measured: 14 boxes became
    13). Saving straight after ``set_classes`` reproduces detections exactly.

    ``clip_model`` is dropped first. Ultralytics attaches the whole CLIP
    encoder to the model, but it is only needed to *create* the embeddings —
    keeping it would make the cache 329 MB instead of 26 MB.
    """
    try:
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "clip_model"):
            del inner.clip_model
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a crash or two processes racing here must never
        # leave a truncated .pt that later loads as a silently broken model.
        tmp = path.with_name(path.name + ".tmp.pt")
        model.save(str(tmp))
        tmp.replace(path)
    except Exception as exc:  # caching is an optimisation, never a hard error
        print(
            f"warning: could not cache the prompted model ({exc}); "
            "startup will stay slow",
            file=sys.stderr,
        )


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

    Runs open-vocabulary detection over a vocabulary of plain-English phrases.
    By default that vocabulary is the fixed list in :mod:`src.classes` (the 80
    COCO classes + ~200 common table items) — edit ``classes.py`` to change
    what "detect everything" looks for.

    Passing ``classes`` overrides that list, which is how the "find one
    specific thing" feature works: the model is prompted with a single free-text
    description such as ``["blue cup"]`` (see :mod:`src.find`).

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
    classes:
        Vocabulary to detect. ``None`` (default) uses ``DETECTION_CLASSES``
        and is served from an on-disk prompt cache (see
        :func:`_prompt_cache_path`), turning a ~25 s startup into ~2 s.

    Attributes
    ----------
    prompt_cache_hit:
        Whether the vocabulary came from that cache. Useful for telling the
        user which of the two startup costs they are about to pay.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf: float = 0.25,
        device: str | int | None = None,
        classes: Sequence[str] | None = None,
    ) -> None:
        from ultralytics import YOLOWorld

        self.conf = conf
        self.device = device
        self.classes = list(classes) if classes else list(DETECTION_CLASSES)
        if not self.classes:
            raise ValueError("classes must not be empty")

        # Prompting the model is by far the slowest part of startup: loading
        # the weights takes 0.06 s, but set_classes() has to build the CLIP
        # text encoder and embed every phrase, which measured ~25 s for the
        # ~200-phrase default vocabulary. Those embeddings depend only on the
        # class list, so for the FIXED vocabulary we save the prompted model
        # once and reload it in ~0.05 s afterwards.
        #
        # Only the default vocabulary is cached. A custom `classes` list is
        # typically a one-off free-text query from src.find; caching those
        # would spend 26 MB per distinct phrase on entries that are unlikely
        # ever to be reused.
        cache_path = (
            _prompt_cache_path(model_path, self.classes) if not classes else None
        )
        self.prompt_cache_hit = bool(cache_path and cache_path.exists())

        if self.prompt_cache_hit:
            # The cached file already carries txt_feats and names, so
            # set_classes() must NOT be called again — that would undo the
            # entire point and re-run CLIP.
            self.model = _load_model(str(cache_path), loader=YOLOWorld)
        else:
            self.model = _load_model(model_path, loader=YOLOWorld)
            self.model.set_classes(self.classes)
            if cache_path is not None:
                _save_prompt_cache(self.model, cache_path)

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
