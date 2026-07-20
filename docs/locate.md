# Distance and Bearing to an Object

The fourth mode, and the one the other three were built for. `src.detect` says
*what is in the image*, `src.find` narrows that to *one described object*, and
`src.depth` says *how far every pixel is*. This joins the last two: describe an
object in plain English and get back **how far away it is and which way to
turn**.

## Layout

```text
src/locator.py   the geometry: box + depth map -> distance and bearing (no models)
src/locate.py    CLI: image / video / webcam
src/webcam_server.py  POST /locate, the browser's "Locate object" mode
camera.json      optional intrinsics; a full one makes the bearing exact
```

`locator.py` loads no weights and touches no model. It is pure arithmetic over
arrays the detector and the depth estimator already produce, which keeps the
part that is easy to get subtly wrong small, model-free and testable on
synthetic scenes.

## 1. One image

```bash
python -m src.locate image "blue cup" data/table3.jpg
# -> writes data/table3_located.jpg next to it, and prints:
#   blue cup        conf=0.86
#     distance   0.80 m   (near surface 0.79 m, depth 0.78 m)
#     bearing    12.4 deg right, 1.7 deg down   [depth model]
#     point      x=+0.17  y=+0.02  z=+0.78  (metres, camera frame)
```

The annotated image carries the box, a crosshair on the point that was
measured, the two numbers, and a strip along the bottom spanning the camera's
field of view with the object marked on it — the bearing as a picture.

`--json out.json` writes the same measurement as a machine-readable record,
including the raw `(x, y, z)` and the intrinsics that produced it.

Exit code is 0 only when the object was **found and measured**, so this can be
scripted like `src.find`.

## 2. What the numbers mean

| Field | Meaning |
| --- | --- |
| `distance_m` | straight-line range, camera centre to the middle of the object |
| `depth_m` | only the forward (`z`) component of that — equal to `distance_m` only dead ahead |
| `nearest_m` | range to the object's near surface (10th percentile) — what a gripper meets |
| `bearing_deg` | horizontal angle off the optical axis, **positive = right** |
| `elevation_deg` | vertical angle off the optical axis, **positive = up** |
| `point_xyz_m` | the sampled point in the camera frame: x right, y down, z forward |
| `valid_fraction` | share of the sampled region that had valid depth |

Bearing and elevation are exactly what a base has to turn by, which is why they
are reported instead of the raw `(x, y, z)` alone — though that is there too.

Three distances are reported rather than one because they answer different
questions: how far to travel (`distance_m`), how far to reach before touching
(`nearest_m`), and how far forward the object sits (`depth_m`).

## 3. Why the middle of the box, and why a median

A bounding box is a rectangle around a shape that is not one, so its corners
are usually background. On a cup against a far wall those corners are metres
behind the cup, and averaging them in drags the answer toward the wall.

Two defences, both in `src/locator.py`:

1. **Sample only the centred half of the box** (`DEFAULT_CORE_FRACTION = 0.5`).
   Smaller would be purer but starts to miss thin objects entirely.
2. **Reduce with a median, not a mean.** Whatever background still leaks in sits
   in the tail of the distribution, where a median ignores it.

A synthetic check: an object at 2 m inside a box twice its size, with 8 m
background filling the rest of the box, still measures 2.00 m.

If the model reported no valid geometry anywhere in the box, `locate()` returns
`None` and every caller says "found, but cannot measure" — an honest outcome,
distinct from "not found", and never a fabricated number.

`valid_fraction` below 50% is flagged in all three interfaces: the measurement
stands on very little evidence and should not be read to two decimals.

## 4. Where the angles come from

Two possible sources, and the reported `bearing_source` always says which was
used:

* **`depth model`** (default) — `atan2(x, z)` on the sampled 3D point. MoGe
  assumes the principal point is the image centre.
* **`camera.json`** — `atan2((u - cx) / fx, 1)` from the box centre pixel. Used
  whenever `fx`, `fy`, `cx` **and** `cy` are all filled in.

Calibrated intrinsics win when they exist: they carry the true principal point
instead of assuming a centred lens, and they do not depend on the depth of the
sample at all. A half-filled `camera.json` is ignored rather than mixed with
the model's own geometry — one answer, one source of truth.

Intrinsics are only valid at the resolution they were measured at, so they are
rescaled to the current frame using `width`/`height` from `camera.json`. Fill
those in; without them the numbers can only be assumed to belong to whatever
resolution happens to arrive.

See [depth.md §6](depth.md#6-camera-intrinsics-camerajson--optional) for how to
obtain the values.

## 5. Video

```bash
python -m src.locate video "blue cup" data/table2.mp4 --stride 8
```

Prints where the object first appears, its closest approach, and the range of
distances over the clip; `--json` writes one record per measured frame.

Both models run on every measured frame, so this is the slowest mode by some
margin. `--stride N` measures every Nth frame and is the quickest way to speed a
long clip up. Skipped frames reuse the previous annotation, so the output video
stays smooth rather than flickering the box on and off — but only measured
frames are reported and written to JSON.

`--resolution-level` (0-9, default 9) is the other dial; see
[depth.md §5](depth.md#5-performance).

## 6. Live, from the webcam

```bash
# LINUX host with /dev/video0 passthrough:
python -m src.locate webcam "blue cup" 0

# Windows / macOS (container cannot open the host camera):
python -m src.webcam_server        # then pick "Locate object"
```

The browser mode draws the box, the crosshair, the distance and the bearing
strip over the live video — the frame is left in its own colours, since the
point is to see the object and read its numbers off it, not to look at a depth
map.

`POST /locate` only runs the depth model **after** the object has been found, so
a frame with nothing matching costs a detection and no more. It returns JSON
only (a few hundred bytes) rather than an image: the browser already has the
frame to draw on.

Depth quality is the same selector the depth tab uses. At "Balanced" the
measurement agrees with "Best" to a centimetre or so — the run behind this doc
measured 0.792 m against 0.798 m — which is well inside the model's own error.

## 7. From Python

```python
import cv2
from src import ObjectDetector, DepthEstimator, load_camera, locate
from src.find import pick_unique

camera = load_camera()
detector = ObjectDetector(conf=0.10, classes=["blue cup"])
estimator = DepthEstimator(camera=camera)

frame = cv2.imread("data/table3.jpg")
match = pick_unique(detector.detect(frame))          # at most one object
if match is not None:
    location = locate(match, estimator.estimate(frame), camera=camera)
    if location is not None:
        print(location.summary())                     # 0.80 m · 12° right · 2° down
        print(location.distance_m, location.bearing_deg)
```

`locate()` takes a `Detection` and a `DepthMap` **from the same frame**; a
mismatched pair measures the wrong thing without complaining, so compute both
from one image.

## 8. Accuracy notes

The distance is only as good as MoGe's metric scale, which is why
`camera.json`'s FOV matters (see [depth.md §6](depth.md#6-camera-intrinsics-camerajson--optional)):
a wrong FOV becomes a proportionally wrong distance. The *bearing* is the more
trustworthy of the two — with a calibrated `cx`/`fx` it is straight pinhole
geometry and does not involve the depth model at all.

Things that degrade a measurement, in rough order of impact:

* A loose or wrong box. The distance is measured where the detector points; a
  box that has crept onto the wall behind measures the wall. Raise `--conf` if
  the match is doubtful.
* Reflective, transparent or very dark surfaces, where monocular depth is
  weakest. Watch `valid_fraction`.
* An uncalibrated `camera.json`, which costs metric accuracy and assumes a
  centred lens.
