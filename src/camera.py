"""Optional camera intrinsics, loaded from ``camera.json``.

The depth model (see :mod:`src.depth_estimator`) predicts the camera's field of
view on its own, so intrinsics are never *required*. They exist here for two
reasons:

1. Telling MoGe the true horizontal FOV instead of letting it guess makes the
   metric scale noticeably more trustworthy — a FOV error becomes a distance
   error more or less proportionally.
2. The planned "distance *and angle* to an object" feature needs the principal
   point (``cx``/``cy``) and focal length to turn a pixel into a bearing.

Every field is optional; missing ones stay ``None`` and are simply not used.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

#: Repo-root ``camera.json``, used when nothing else is specified.
DEFAULT_CAMERA_CONFIG = Path(__file__).resolve().parent.parent / "camera.json"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixels, valid at one specific capture resolution."""

    name: str = "unnamed camera"
    fov_x_deg: float | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    width: int | None = None
    height: int | None = None

    @property
    def horizontal_fov_deg(self) -> float | None:
        """Horizontal FOV in degrees, or ``None`` if it cannot be determined.

        An explicit ``fov_x_deg`` wins. Otherwise it is derived from ``fx`` and
        ``width`` via ``fov_x = 2 * atan(width / (2 * fx))`` — the standard
        pinhole relation, which is why the two must come from the *same*
        resolution.
        """
        if self.fov_x_deg is not None:
            return float(self.fov_x_deg)
        if self.fx and self.width:
            return math.degrees(2 * math.atan(self.width / (2 * self.fx)))
        return None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "fov_x_deg": self.horizontal_fov_deg,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }


def load_camera(path: str | Path | None = None) -> CameraIntrinsics:
    """Read intrinsics from JSON, tolerating an absent or all-null file.

    Resolution order: explicit ``path``, then ``$CAMERA_CONFIG``, then the
    repo-root ``camera.json``. A missing file is not an error — it just means
    "no intrinsics known", which is a fully supported mode.

    Keys starting with ``_`` (such as the ``_comment`` block in the shipped
    template) and unknown keys are ignored.
    """
    if path is None:
        path = os.environ.get("CAMERA_CONFIG") or DEFAULT_CAMERA_CONFIG
    path = Path(path)
    if not path.exists():
        return CameraIntrinsics()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")

    fields = {f for f in CameraIntrinsics.__dataclass_fields__}
    known = {k: v for k, v in raw.items() if k in fields and v is not None}
    return CameraIntrinsics(**known)
