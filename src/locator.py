"""Turn a place in the image + a depth map into a distance and a bearing.

This is the join between the two halves of the project. Neither half needs the
other on its own — :class:`~src.detector.ObjectDetector` says *what is where in
the image* and :class:`~src.depth_estimator.DepthEstimator` says *how far every
pixel is* — but a robot needs both at once, aimed at one thing: "the blue cup is
1.24 m away, 12 degrees to the right".

There are two ways to say *which* thing, and they share everything after that:

* :func:`locate` — a detection box, i.e. an object someone described in words.
* :func:`locate_point` — a bare pixel, i.e. somewhere a user clicked. No
  detector involved at all; the depth map alone answers it.

Both reduce a region of metric 3D points to one :class:`Measurement`.

Everything here is pure geometry over arrays the models already produce, so
this module loads no weights and is cheap to call.

Coordinates follow MoGe's convention (OpenCV camera frame): **x right, y down,
z forward**, origin at the camera centre. Reported angles are the friendlier
form of that:

* ``bearing_deg`` — horizontal angle off the optical axis, **positive = right**
* ``elevation_deg`` — vertical angle off the optical axis, **positive = up**

Both are angles a robot base can turn by directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CameraIntrinsics
from .detector import Detection

#: Fraction of the box (per side) kept when sampling depth — the centred half.
#:
#: A bounding box is a rectangle around a non-rectangular object, so its corners
#: are usually *background*. On a cup against a far wall those corners are metres
#: behind the cup, and including them pulls the answer toward the wall. Shrinking
#: to the middle keeps the samples on the object itself. Half is a compromise:
#: smaller would be purer but starts to miss thin objects entirely.
DEFAULT_CORE_FRACTION = 0.5

#: Percentile used for :attr:`Measurement.nearest_m` — the near *surface* of the
#: object rather than its middle. A plain minimum would latch onto a single
#: speckle pixel, so this is deliberately not ``min``.
NEAREST_PERCENTILE = 10.0

#: Radius sampled around a clicked pixel, as a fraction of image width.
#:
#: A single pixel is a bad measurement: it can be ``NaN`` outright, and even
#: when valid it carries the model's full per-pixel noise. Sampling a small
#: patch and taking the median of it costs the user nothing and makes a click
#: repeatable. Small enough (~1% of width, so ±6 px at 640) that it still
#: measures the thing under the cursor rather than its surroundings.
DEFAULT_POINT_RADIUS_FRACTION = 0.01


def _clamp_box(det: Detection, width: int, height: int) -> tuple[int, int, int, int]:
    """Detection box as integer pixel bounds clipped to the image."""
    x0 = max(0, min(width - 1, int(math.floor(det.x1))))
    y0 = max(0, min(height - 1, int(math.floor(det.y1))))
    x1 = max(x0 + 1, min(width, int(math.ceil(det.x2))))
    y1 = max(y0 + 1, min(height, int(math.ceil(det.y2))))
    return x0, y0, x1, y1


def _core_region(
    det: Detection, width: int, height: int, fraction: float
) -> tuple[int, int, int, int]:
    """The centred sub-rectangle of the box used for sampling — see above."""
    x0, y0, x1, y1 = _clamp_box(det, width, height)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half_w = max(1.0, (x1 - x0) * fraction / 2.0)
    half_h = max(1.0, (y1 - y0) * fraction / 2.0)
    return (
        max(x0, int(round(cx - half_w))),
        max(y0, int(round(cy - half_h))),
        min(x1, max(int(round(cx + half_w)), int(round(cx - half_w)) + 1)),
        min(y1, max(int(round(cy + half_h)), int(round(cy - half_h)) + 1)),
    )


def _finite_points(points: np.ndarray) -> np.ndarray:
    """``(N, 3)`` of the rows where all three coordinates are finite."""
    flat = points.reshape(-1, 3)
    return flat[np.isfinite(flat).all(axis=1)]


def _fov_x_deg(depth_map, width: int) -> float | None:
    """Horizontal FOV the depth model actually used, in degrees."""
    fx = float(depth_map.pixel_intrinsics()[0, 0])
    if not math.isfinite(fx) or fx <= 0:
        return None
    return math.degrees(2 * math.atan(width / (2 * fx)))


def _angles_from_camera(
    camera: CameraIntrinsics | None, u: float, v: float, width: int, height: int
) -> tuple[float, float] | None:
    """Bearing/elevation of pixel ``(u, v)`` from calibrated intrinsics.

    Returns ``None`` unless ``fx``, ``fy``, ``cx`` and ``cy`` are all known —
    a half-filled ``camera.json`` falls back to the depth model's own geometry
    rather than mixing two different sources of truth into one answer.

    Intrinsics are only valid at the resolution they were measured at, so they
    are rescaled to this frame. That is also why ``width``/``height`` matter in
    ``camera.json``: without them the numbers are unusable at any other size,
    and we can only assume they were measured here.
    """
    if camera is None or not camera.fx or not camera.fy:
        return None
    if camera.cx is None or camera.cy is None:
        return None

    sx = width / camera.width if camera.width else 1.0
    sy = height / camera.height if camera.height else 1.0
    fx, cx = camera.fx * sx, camera.cx * sx
    fy, cy = camera.fy * sy, camera.cy * sy
    if fx <= 0 or fy <= 0:
        return None

    bearing = math.degrees(math.atan2((u - cx) / fx, 1.0))
    elevation = math.degrees(math.atan2(-(v - cy) / fy, 1.0))
    return bearing, elevation


@dataclass(frozen=True)
class Measurement:
    """Where something is, in metres and degrees.

    The same shape whether it came from a detected object or from a clicked
    pixel — only :attr:`detection` differs, so everything downstream (printing,
    JSON, drawing, the browser) handles one type instead of two.

    Attributes
    ----------
    distance_m:
        Straight-line range from the camera centre to the object, in metres.
        This is what you travel; :attr:`depth_m` is only its forward component.
    depth_m:
        Distance along the optical axis (the ``z`` of the sampled point). Equal
        to ``distance_m`` only for an object dead ahead.
    nearest_m:
        Range to the near surface of the sampled region
        (:data:`NEAREST_PERCENTILE`th percentile) — the number that matters for
        reaching or stopping, since the gripper meets the front of the cup, not
        its middle.
    bearing_deg:
        Horizontal angle off the optical axis, positive to the **right**.
    elevation_deg:
        Vertical angle off the optical axis, positive **up**.
    bearing_source:
        ``"camera.json"`` when the angles came from calibrated intrinsics,
        ``"depth model"`` when they came from MoGe's own estimated geometry.
    point:
        The sampled ``(x, y, z)`` in metres, camera frame.
    valid_fraction:
        Share of the sampled region that had valid geometry. A low value means
        the model saw little of it, so treat the numbers with care.
    fov_x_deg:
        Horizontal field of view used by the depth model, for reference.
    pixel:
        The ``(u, v)`` in the image this measurement is about — the centre of
        the box, or the pixel that was clicked.
    detection:
        The box this was measured from, or ``None`` for a clicked point.
    """

    distance_m: float
    depth_m: float
    nearest_m: float
    bearing_deg: float
    elevation_deg: float
    bearing_source: str
    point: tuple[float, float, float]
    valid_fraction: float
    fov_x_deg: float | None
    pixel: tuple[float, float]
    detection: Detection | None = None

    @property
    def side(self) -> str:
        """``"left"`` / ``"right"``, or ``"ahead"`` within half a degree."""
        if abs(self.bearing_deg) < 0.5:
            return "ahead"
        return "right" if self.bearing_deg > 0 else "left"

    def summary(self) -> str:
        """One human-readable line: ``1.24 m · 12° right · 5° up``."""
        parts = [f"{self.distance_m:.2f} m"]
        if self.side == "ahead":
            parts.append("straight ahead")
        else:
            parts.append(f"{abs(self.bearing_deg):.0f}° {self.side}")
        if abs(self.elevation_deg) >= 0.5:
            updown = "up" if self.elevation_deg > 0 else "down"
            parts.append(f"{abs(self.elevation_deg):.0f}° {updown}")
        return " · ".join(parts)

    def as_dict(self) -> dict:
        return {
            "detection": self.detection.as_dict() if self.detection else None,
            "pixel_xy": [round(c, 1) for c in self.pixel],
            "distance_m": round(self.distance_m, 3),
            "depth_m": round(self.depth_m, 3),
            "nearest_m": round(self.nearest_m, 3),
            "bearing_deg": round(self.bearing_deg, 2),
            "elevation_deg": round(self.elevation_deg, 2),
            "bearing_source": self.bearing_source,
            "point_xyz_m": [round(c, 3) for c in self.point],
            "valid_fraction": round(self.valid_fraction, 4),
            "fov_x_deg": None if self.fov_x_deg is None else round(self.fov_x_deg, 2),
        }


def _reduce(
    finite: np.ndarray,
    valid_fraction: float,
    pixel: tuple[float, float],
    depth_map,
    camera: CameraIntrinsics | None,
    detection: Detection | None = None,
) -> Measurement:
    """Reduce a set of valid 3D points to one :class:`Measurement`.

    Shared by both entry points, so a click and a detected object are measured
    by exactly the same rules and their numbers are directly comparable.
    """
    height, width = depth_map.depth.shape[:2]

    # Component-wise median: robust, and each coordinate is reduced by the same
    # rule, so the result behaves like a point at the region's centre of mass.
    centre = np.median(finite, axis=0)
    x, y, z = (float(c) for c in centre)
    distance = float(np.linalg.norm(centre))
    nearest = float(np.percentile(np.linalg.norm(finite, axis=1), NEAREST_PERCENTILE))

    bearing = math.degrees(math.atan2(x, z))
    elevation = math.degrees(math.atan2(-y, z))
    source = "depth model"

    # Calibrated intrinsics beat the model's estimate when they exist: they
    # carry the true principal point, which MoGe assumes is the image centre,
    # and they do not depend on the depth of the sample at all.
    calibrated = _angles_from_camera(camera, pixel[0], pixel[1], width, height)
    if calibrated is not None:
        bearing, elevation = calibrated
        source = "camera.json"

    return Measurement(
        distance_m=distance,
        depth_m=z,
        nearest_m=nearest,
        bearing_deg=bearing,
        elevation_deg=elevation,
        bearing_source=source,
        point=(x, y, z),
        valid_fraction=valid_fraction,
        fov_x_deg=_fov_x_deg(depth_map, width),
        pixel=pixel,
        detection=detection,
    )


def locate(
    detection: Detection,
    depth_map,
    camera: CameraIntrinsics | None = None,
    core_fraction: float = DEFAULT_CORE_FRACTION,
) -> Measurement | None:
    """Measure distance and bearing to one detected object.

    Samples the metric 3D points inside the box (see
    :data:`DEFAULT_CORE_FRACTION` for why only the middle of it) and reduces
    them with a **median**, not a mean: whatever background still leaks in sits
    in the tail of the distribution, where a median ignores it and a mean does
    not.

    Returns ``None`` when the depth model found no valid geometry anywhere in
    the box — an honest "cannot measure this", which the callers report as such
    rather than inventing a number.

    ``depth_map`` is a :class:`~src.depth_estimator.DepthMap` for the *same*
    frame the detection came from; passing a mismatched pair silently measures
    the wrong thing, so callers should compute both from one image.
    """
    height, width = depth_map.depth.shape[:2]

    x0, y0, x1, y1 = _core_region(detection, width, height, core_fraction)
    sampled = depth_map.points[y0:y1, x0:x1]
    finite = _finite_points(sampled)
    valid_fraction = float(finite.shape[0]) / max(1, sampled.shape[0] * sampled.shape[1])

    if finite.shape[0] == 0:
        # Nothing valid in the middle. Before giving up, try the whole box: a
        # thin or hollow object (a mug handle, a wire frame) can be valid only
        # around its edges.
        bx0, by0, bx1, by1 = _clamp_box(detection, width, height)
        box = depth_map.points[by0:by1, bx0:bx1]
        finite = _finite_points(box)
        if finite.shape[0] == 0:
            return None
        valid_fraction = float(finite.shape[0]) / max(1, box.shape[0] * box.shape[1])

    pixel = ((detection.x1 + detection.x2) / 2.0, (detection.y1 + detection.y2) / 2.0)
    return _reduce(finite, valid_fraction, pixel, depth_map, camera, detection)


def locate_point(
    x: float,
    y: float,
    depth_map,
    camera: CameraIntrinsics | None = None,
    radius: int | None = None,
) -> Measurement | None:
    """Measure distance and bearing to one **pixel** — no detector involved.

    This is the whole of "click anywhere and tell me how far that is": the depth
    map already holds a metric 3D point per pixel, so a click needs no object,
    no vocabulary and no second model.

    A small patch around ``(x, y)`` is sampled rather than the single pixel —
    see :data:`DEFAULT_POINT_RADIUS_FRACTION`.

    Returns ``None`` for a click outside the image, or when the patch holds no
    valid geometry (sky, a mirror, a blown-out highlight). That is a real
    answer — "the model does not know how far that is" — not an error.
    """
    height, width = depth_map.depth.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return None

    if radius is None:
        radius = max(2, round(width * DEFAULT_POINT_RADIUS_FRACTION))
    cx, cy = int(round(x)), int(round(y))
    patch = depth_map.points[
        max(0, cy - radius) : min(height, cy + radius + 1),
        max(0, cx - radius) : min(width, cx + radius + 1),
    ]
    finite = _finite_points(patch)
    if finite.shape[0] == 0:
        return None

    valid_fraction = float(finite.shape[0]) / max(1, patch.shape[0] * patch.shape[1])
    return _reduce(finite, valid_fraction, (float(x), float(y)), depth_map, camera)


#: Green, matching the single-match box drawn by the find feature.
_BOX_COLOR = (128, 222, 74)  # BGR


def draw(
    image: np.ndarray,
    location: Measurement | None,
    query: str = "",
    bearing_bar: bool = True,
) -> np.ndarray:
    """Annotate ``image`` with the measurement and a bearing scale.

    Draws the box when the measurement came from a detection, and just the
    crosshair when it came from a clicked pixel. Passing ``None`` returns an
    unannotated copy, so the caller can render a "not found" frame through the
    same path.
    """
    out = image.copy()
    if location is None:
        return out

    h, w = out.shape[:2]
    thickness = max(2, round(h / 300))
    font_scale = max(0.5, h / 900)
    det = location.detection

    if det is not None:
        cv2.rectangle(
            out,
            (int(det.x1), int(det.y1)),
            (int(det.x2), int(det.y2)),
            _BOX_COLOR,
            thickness,
        )

    # Crosshair at the point the measurement refers to, so it is obvious the
    # numbers describe the middle of the object and not, say, its near corner.
    cx, cy = int(round(location.pixel[0])), int(round(location.pixel[1]))
    arm = max(4, round(h / 90))
    cv2.line(out, (cx - arm, cy), (cx + arm, cy), _BOX_COLOR, max(1, thickness // 2))
    cv2.line(out, (cx, cy - arm), (cx, cy + arm), _BOX_COLOR, max(1, thickness // 2))

    if det is not None:
        lines = [f"{query or det.label} {det.confidence:.2f}", location.summary()]
        anchor = (int(det.x1), int(det.y1))
    else:
        lines = [location.summary()]
        anchor = (cx + arm, cy - arm)  # off the crosshair, not over it
    _draw_label(out, lines, anchor[0], anchor[1], font_scale, thickness)

    if bearing_bar and location.fov_x_deg:
        _draw_bearing_bar(out, location, font_scale)
    return out


def _draw_label(
    image: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    font_scale: float,
    thickness: int,
) -> None:
    """Filled multi-line caption above ``(x, y)``, flipped below if it won't fit."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in lines]
    pad = max(3, round(font_scale * 6))
    line_h = max(s[1] for s in sizes) + pad
    block_h = line_h * len(lines) + pad
    block_w = max(s[0] for s in sizes) + 2 * pad

    top = y - block_h
    if top < 0:  # no room above the box; put the caption inside it instead
        top = y
    left = min(max(0, x), max(0, image.shape[1] - block_w))

    cv2.rectangle(
        image, (left, top), (left + block_w, top + block_h), _BOX_COLOR, -1
    )
    for i, text in enumerate(lines):
        baseline = top + pad + line_h * i + sizes[i][1]
        cv2.putText(
            image,
            text,
            (left + pad, baseline),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )


def _draw_bearing_bar(
    image: np.ndarray, location: Measurement, font_scale: float
) -> None:
    """Strip along the bottom spanning the field of view, ticked at the object.

    The bearing is otherwise just a number: this makes it readable at a glance
    and, because the strip spans exactly the camera's FOV, it also shows how
    much of the scene the measurement could possibly be about.
    """
    h, w = image.shape[:2]
    fov = location.fov_x_deg
    if not fov:
        return

    margin = max(8, round(w * 0.04))
    y = h - max(14, round(h * 0.05))
    x0, x1 = margin, w - margin
    cv2.line(image, (x0, y), (x1, y), (200, 200, 200), 1, cv2.LINE_AA)

    # Centre tick = straight ahead, ends = the edges of the field of view.
    for frac in (0.0, 0.5, 1.0):
        tx = int(x0 + frac * (x1 - x0))
        cv2.line(image, (tx, y - 4), (tx, y + 4), (200, 200, 200), 1, cv2.LINE_AA)

    # Clamped: an object at the very edge of frame can land marginally outside
    # the nominal FOV, and a marker drawn off the strip would just vanish.
    frac = 0.5 + location.bearing_deg / fov
    mx = int(x0 + min(1.0, max(0.0, frac)) * (x1 - x0))
    cv2.drawMarker(
        image, (mx, y), _BOX_COLOR, cv2.MARKER_TRIANGLE_DOWN,
        max(10, round(h / 40)), 2, cv2.LINE_AA,
    )

    label = f"{location.bearing_deg:+.0f}°"
    scale = font_scale * 0.7
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(
        image,
        label,
        (min(max(x0, mx - tw // 2), x1 - tw), y + max(16, round(h / 45))),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
