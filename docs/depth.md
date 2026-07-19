# Depth Estimation Infrastructure

Monocular **metric** depth from a single RGB frame: one distance in metres per
pixel, plus a full 3D point per pixel. This is a standalone feature — it does
not use object detection, and object detection does not use it. Combining the
two ("how far away and in which direction is the blue cup?") is the next step,
sketched at the bottom.

## Layout

```
camera.json                 optional camera intrinsics (see below)
src/camera.py               loads camera.json -> CameraIntrinsics
src/depth_estimator.py      the only place that talks to the depth model
src/depth.py                CLI: image / video / webcam
src/webcam_server.py        POST /depth for the browser app
```

Mirrors the detection side: `depth_estimator.py` is to `src.depth` what
`detector.py` is to `src.detect`.

## 1. Why MoGe-2

Most monocular depth networks (MiDaS, Depth-Anything v1) are **relative**: they
say which pixel is nearer, but not by how much. That is fine for a visual
effect and useless for a robot, which needs to know whether the cup is at 0.3 m
or 1.2 m before it reaches for it.

[MoGe-2](https://github.com/microsoft/MoGe) (CVPR'25, Microsoft) predicts a
**metric-scale** point map, so the output really is in metres. We use the ViT-L
variant, `Ruicheng/moge-2-vitl` (~1.3 GB, pre-downloaded into the dev container
image). The `-normal` variant additionally predicts surface normals; we do not
need them yet.

MoGe is not on PyPI, so `requirements.txt` installs it straight from git.

## 2. Depth for an image

```bash
python -m src.depth image data/table2.jpg
# -> writes data/table2_depth.jpg next to it, and prints:
#   range   0.41 m .. 3.87 m   median 1.62 m
#   centre  1.24 m   valid 97% of pixels
```

Reading the output image: **red = near, blue = far**, dark grey = the model
found no valid geometry there (sky, reflections, out of range). The bar on the
right edge labels the two ends of the ramp in metres.

Useful flags:

| Flag | What it does |
| --- | --- |
| `--side-by-side` | writes the source frame next to the depth map |
| `--npy out.npy` | saves the raw `float32` metres array (`NaN` = invalid) |
| `--json out.json` | saves range/median/centre + the intrinsics actually used |
| `--near`/`--far` | fix the colour ramp bounds in metres instead of auto-scaling |
| `--resolution-level 0-9` | model working resolution; lower = faster, coarser |

The PNG is for humans. `--npy` is the actual measurement — that is what you
want if you are going to compute anything from it.

## 3. Depth for a video

```bash
python -m src.depth video data/table2.mp4 --side-by-side
```

The colour ramp is **locked to the first frame** and reused for the rest of the
video. Auto-scaling each frame independently would make the whole clip pulse as
the nearest and farthest points in the scene shift, and would make colours
incomparable between frames. Pass `--near`/`--far` to pin the range yourself
(sensible when you know the working volume, e.g. `--near 0.2 --far 2.0` for a
tabletop).

## 4. Depth from the webcam

Same split as detection:

```bash
# LINUX host with /dev/video0 passthrough:
python -m src.depth webcam --source 0        # prints centre distance live

# Windows / macOS (container cannot open the host camera):
python -m src.webcam_server                  # then pick the "Depth map" tab
```

In the browser the server renders the colourised map and sends it back as a
JPEG data URL, which replaces the video in the viewport. A full `float32` depth
map would be ~4 MB per frame — far too much for a live loop — so the numbers
you would actually read right now (scene range, centre distance) come back as
JSON alongside the image instead.

The colour ramp is locked on the first frame of each session and re-locked when
you start the camera or switch tabs.

MoGe-2 ViT-L is ~60 ms/frame on a decent GPU and **seconds** per frame on CPU.
On CPU, drop the detection-rate slider to 1/s and lower
`--depth-resolution-level`.

## 5. Camera intrinsics (`camera.json`) — optional

MoGe-2 estimates the camera's field of view itself, so **depth works with
`camera.json` left untouched**. Filling it in only improves accuracy: the FOV
sets how a pixel offset converts to a metric offset, so a wrong FOV guess
becomes a proportionally wrong distance.

Fill in **either** `fov_x_deg`, **or** `fx` together with `width`:

```json
{
  "name": "logitech c920 @ 1280x720",
  "fx": 1000.0, "fy": 1000.0,
  "cx": 640.0,  "cy": 360.0,
  "width": 1280, "height": 720
}
```

`fov_x` is derived as `2 * atan(width / (2 * fx))`, which is why `fx` and
`width` must come from the **same** resolution — intrinsics are only valid at
the resolution they were measured at.

Getting the numbers: print an OpenCV chessboard, take ~20 photos of it at the
resolution you will actually use, and run `cv2.calibrateCamera`. It returns
`fx, fy, cx, cy` in pixels.

`fy`, `cx` and `cy` are not used for the depth map itself. They are stored now
because the angle feature below needs the principal point.

Point at a different file with `--camera PATH` or `$CAMERA_CONFIG`.

## 6. Using depth from Python

```python
from src import DepthEstimator, load_camera
import cv2

estimator = DepthEstimator(camera=load_camera())
depth_map = estimator.estimate(cv2.imread("data/table2.jpg"))

depth_map.depth_at(320, 240)      # metres at one pixel, or None if invalid
depth_map.point_at(320, 240)      # (x, y, z) metres, OpenCV camera coords
depth_map.stats()                 # summary dict
depth_map.pixel_intrinsics()      # 3x3, in pixels
```

`DepthMap.depth` is `NaN` wherever the model reported no valid geometry — the
invalid pixels are made explicit precisely so they cannot be silently averaged
into a distance. Use `np.nanmedian`, not `np.mean`.

`DepthMap.points` is in OpenCV camera coordinates: **x right, y down, z
forward**, origin at the camera centre.

## 7. Next step: distance and angle to an object

The pieces are deliberately shaped for this:

1. `src.find` gives a box for the described object.
2. `estimate()` gives `points` for the same frame.
3. Distance = `np.nanmedian` of `points[..., 2]` inside the box (median, not
   mean, so background pixels leaking into the box do not drag it).
4. Bearing = `atan2(X, Z)` of the median point — or from the pixel and the
   intrinsics: `atan2((u - cx) / fx, 1)`.

Step 4 is the one that genuinely wants a calibrated `cx`/`fx`, which is why
`camera.json` carries them already.
