"""Core monocular *metric* depth wrapper around MoGe-2.

This is the single place that talks to the depth model. Everything else (image
CLI, video CLI, webcam web app) uses :class:`DepthEstimator`, mirroring how
:class:`src.detector.ObjectDetector` is the single entry point for detection.

Why MoGe-2 and not a classic depth network: most monocular models (MiDaS,
Depth-Anything v1) are *relative* — they tell you which pixel is nearer, but
not by how many metres, so a robot cannot act on the output. MoGe-2 predicts a
metric-scale point map, so :attr:`DepthMap.depth` is in real metres, and it
also returns a full 3D point per pixel, which is what the planned
"distance and angle to an object" feature will consume.

Depth estimation is deliberately independent of object detection: the two run
on the same frame but neither needs the other yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .camera import CameraIntrinsics

#: MoGe-2 ViT-L, metric scale. The ``-normal`` variant additionally predicts
#: surface normals for a little extra cost; we don't need them yet. Weights are
#: pulled from the HuggingFace hub on first use and cached in ``$HF_HOME``.
DEFAULT_MODEL = "Ruicheng/moge-2-vitl"

#: MoGe's internal working resolution. 9 is the model's own default; lower
#: values (e.g. 6-7) are faster and coarser, which matters for live webcam use.
DEFAULT_RESOLUTION_LEVEL = 9


def _pick_device() -> str:
    """GPU when one is available, else CPU (MoGe-2 on CPU is seconds/frame)."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class DepthMap:
    """Per-pixel metric geometry for one frame.

    Attributes
    ----------
    depth:
        ``(H, W)`` float32 distance along the camera's optical axis, in
        **metres**. Invalid pixels (sky, out-of-range) are ``NaN`` — use
        :attr:`mask` or the NaN-aware helpers rather than assuming finiteness.
    points:
        ``(H, W, 3)`` metric 3D point per pixel in OpenCV camera coordinates
        (x right, y down, z forward). Unused today; kept because the angle
        feature needs it. ``NaN`` where invalid.
    mask:
        ``(H, W)`` bool, ``True`` where the model considers the geometry valid.
    intrinsics:
        ``(3, 3)`` intrinsics the model used, **normalised** by image size
        (i.e. ``fx`` is in units of image widths). Use
        :meth:`pixel_intrinsics` for pixel units.
    """

    depth: np.ndarray
    points: np.ndarray
    mask: np.ndarray
    intrinsics: np.ndarray

    def pixel_intrinsics(self) -> np.ndarray:
        """Intrinsics rescaled from normalised units to pixels."""
        h, w = self.depth.shape[:2]
        k = self.intrinsics.astype(np.float64).copy()
        k[0, :] *= w
        k[1, :] *= h
        return k

    def depth_at(self, x: int, y: int) -> float | None:
        """Metric depth at pixel ``(x, y)``, or ``None`` where invalid."""
        if not (0 <= y < self.depth.shape[0] and 0 <= x < self.depth.shape[1]):
            return None
        value = float(self.depth[y, x])
        return None if not np.isfinite(value) else value

    def point_at(self, x: int, y: int) -> tuple[float, float, float] | None:
        """Metric 3D point at pixel ``(x, y)`` in camera coordinates."""
        if not (0 <= y < self.points.shape[0] and 0 <= x < self.points.shape[1]):
            return None
        p = self.points[y, x]
        if not np.all(np.isfinite(p)):
            return None
        return float(p[0]), float(p[1]), float(p[2])

    def range_metres(self, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
        """Robust near/far bounds over valid pixels, as percentiles.

        Percentiles rather than min/max because a handful of speckle pixels at
        0.1 m or 300 m would otherwise squash the entire colour ramp.
        Returns ``(0.0, 1.0)`` if nothing is valid.
        """
        valid = self.depth[np.isfinite(self.depth)]
        if valid.size == 0:
            return 0.0, 1.0
        near = float(np.percentile(valid, low))
        far = float(np.percentile(valid, high))
        if far <= near:
            far = near + 1e-3
        return near, far

    def stats(self) -> dict:
        """Summary suitable for printing or dumping to JSON."""
        valid = self.depth[np.isfinite(self.depth)]
        h, w = self.depth.shape[:2]
        centre = self.depth_at(w // 2, h // 2)
        if valid.size == 0:
            return {
                "width": w,
                "height": h,
                "valid_fraction": 0.0,
                "min_m": None,
                "max_m": None,
                "median_m": None,
                "centre_m": None,
            }
        return {
            "width": w,
            "height": h,
            "valid_fraction": round(float(valid.size) / (h * w), 4),
            "min_m": round(float(valid.min()), 3),
            "max_m": round(float(valid.max()), 3),
            "median_m": round(float(np.median(valid)), 3),
            "centre_m": None if centre is None else round(centre, 3),
        }


class DepthEstimator:
    """Estimate metric depth for an image (numpy BGR array).

    Parameters
    ----------
    model_path:
        HuggingFace id or local checkpoint. Defaults to :data:`DEFAULT_MODEL`.
    device:
        ``None`` auto-selects CUDA when available, else CPU.
    camera:
        Optional :class:`~src.camera.CameraIntrinsics`. When it yields a
        horizontal FOV, that FOV is handed to the model instead of being
        estimated — see :mod:`src.camera` for why that helps.
    resolution_level:
        MoGe's internal working resolution, 0-9. Lower is faster and coarser.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str | None = None,
        camera: CameraIntrinsics | None = None,
        resolution_level: int = DEFAULT_RESOLUTION_LEVEL,
    ) -> None:
        # Imported lazily (like YOLOWorld in detector.py) so that merely
        # importing this module — e.g. for `--help` — doesn't drag in torch.
        import torch

        try:
            from moge.model.v2 import MoGeModel
        except ImportError as exc:  # pragma: no cover - dependency guidance
            raise ImportError(
                "MoGe is not installed. It is not on PyPI, so install it with:\n"
                "    pip install git+https://github.com/microsoft/MoGe.git\n"
                "(it is already part of requirements.txt / the dev container)"
            ) from exc

        self.device = device or _pick_device()
        self.camera = camera or CameraIntrinsics()
        self.resolution_level = resolution_level
        self.model_path = model_path
        self._torch = torch
        self.model = MoGeModel.from_pretrained(model_path).to(self.device).eval()

    @property
    def fov_x_deg(self) -> float | None:
        """Horizontal FOV passed to the model, or ``None`` to let it estimate."""
        return self.camera.horizontal_fov_deg

    def estimate(self, image: np.ndarray) -> DepthMap:
        """Run depth estimation on one BGR image and return a :class:`DepthMap`."""
        torch = self._torch

        # MoGe wants RGB float in [0, 1] as (3, H, W); OpenCV hands us BGR HWC.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).to(self.device)

        with torch.no_grad():
            output: dict[str, Any] = self.model.infer(
                tensor,
                resolution_level=self.resolution_level,
                fov_x=self.fov_x_deg,
                use_fp16=self.device != "cpu",  # fp16 on CPU is slower, not faster
            )

        mask = output["mask"].cpu().numpy().astype(bool)
        depth = output["depth"].float().cpu().numpy().astype(np.float32)
        points = output["points"].float().cpu().numpy().astype(np.float32)
        intrinsics = output["intrinsics"].float().cpu().numpy()

        # Make invalidity explicit and impossible to average over by accident.
        depth = np.where(mask, depth, np.nan)
        points = np.where(mask[..., None], points, np.nan)
        return DepthMap(
            depth=depth, points=points, mask=mask, intrinsics=intrinsics
        )

    @staticmethod
    def colorize(
        depth_map: DepthMap,
        near: float | None = None,
        far: float | None = None,
        colorbar: bool = True,
    ) -> np.ndarray:
        """Render a depth map as a BGR image: **red = near, blue = far**.

        ``near``/``far`` clamp the colour ramp, in metres. Leaving them ``None``
        auto-scales to this frame's robust range, which is fine for a single
        image but makes video flicker — the video and webcam paths therefore
        lock a range once and reuse it for every frame.

        Invalid pixels are drawn dark grey so they are visibly *absent* rather
        than silently rendered as "very far".
        """
        if near is None or far is None:
            auto_near, auto_far = depth_map.range_metres()
            near = auto_near if near is None else near
            far = auto_far if far is None else far
        if far <= near:
            far = near + 1e-3

        depth = depth_map.depth
        valid = np.isfinite(depth)
        norm = np.clip((np.nan_to_num(depth, nan=far) - near) / (far - near), 0, 1)

        # TURBO runs blue -> red as the value rises, so invert to put NEAR at
        # the red end: closer things should read as "hotter"/more urgent.
        ramp = ((1.0 - norm) * 255).astype(np.uint8)
        out = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
        out[~valid] = (40, 40, 40)

        if colorbar:
            out = _draw_colorbar(out, near, far)
        return out


def _draw_colorbar(image: np.ndarray, near: float, far: float) -> np.ndarray:
    """Overlay a vertical scale bar so the colours can be read as metres."""
    out = image.copy()
    h, w = out.shape[:2]
    bar_h = max(60, int(h * 0.45))
    bar_w = max(10, int(w * 0.02))
    x0 = w - bar_w - max(8, int(w * 0.02))
    y0 = (h - bar_h) // 2

    # Same inversion as colorize(): top of the bar is near, bottom is far.
    ramp = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
    strip = cv2.applyColorMap(np.repeat(ramp, bar_w, axis=1), cv2.COLORMAP_TURBO)
    out[y0 : y0 + bar_h, x0 : x0 + bar_w] = strip
    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1)

    font_scale = max(0.35, h / 1400)
    thickness = max(1, round(h / 700))
    for text, y in ((f"{near:.2f} m", y0 - 6), (f"{far:.2f} m", y0 + bar_h + 16)):
        (tw, _), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.putText(
            out,
            text,
            (max(0, x0 + bar_w - tw), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out
