# Depth Estimation Infrastructure

Monocular **metric** depth from a single RGB frame: one distance in metres per
pixel, plus a full 3D point per pixel. This is a standalone feature — it does
not use object detection, and object detection does not use it. Combining the
two ("how far away and in which direction is the blue cup?") is the next step,
sketched at the bottom.

## Layout

```text
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

If the tab is unusably slow, see the next section — it is almost certainly
running on CPU.

## 5. Performance

**Give the container the GPU. Nothing else comes close.**

Measured in this project's dev container, 640x480 frame, laptop RTX 4070:

| Setup | ms/frame | fps |
| --- | --- | --- |
| ViT-L, level 9, **CPU** | ~45 000 | 0.02 |
| ViT-L, level 9, GPU | 214 | 4.7 |
| ViT-L, level 6, GPU | 160 | 6.2 |
| ViT-L, level 3, GPU | 114 | 8.8 |
| ViT-B, level 9, GPU | 142 | 7.0 |
| ViT-S, level 9, GPU | 115 | 8.7 |

GPU versus CPU is a factor of **~200**. Every other knob is worth 2x at best,
so fix the device first and only then consider tuning.

### Making sure the GPU is actually used

`devcontainer.json` requests it:

```jsonc
"hostRequirements": { "gpu": "optional" }
```

`"optional"` passes the GPU through when the host has one and starts normally
when it doesn't. This needs Docker Desktop with WSL2 (Windows) or the NVIDIA
Container Toolkit (Linux). **Rebuild the dev container after changing it** —
"Dev Containers: Rebuild Container" — since `runArgs`/`hostRequirements` are
applied when the container is created, not on reload.

Verify inside the container:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # want: True
```

The torch wheel in the image is already a CUDA build, so if this prints
`False` the problem is passthrough, not the Python environment. `src.depth`
and the web app both print a loud warning when they fall back to CPU, and the
browser tab shows one in the legend.

### Where the time actually goes

Once the GPU is in use, the request is essentially *all* model. Measured per
640x480 frame on the RTX 4070:

| Stage | ms |
| --- | --- |
| `model.infer()` | 209 |
| GPU→CPU transfer | 0.8 |
| `colorize()` | 1.2 |
| JPEG encode | 0.5 |
| base64 | ~0 |

So optimising the serving path is pointless — 97% of the wall clock is one
call. `resolution_level` is the only dial that moves the number.

### resolution_level: the one real knob

It sets the model's internal token count (1200 at level 0, 3600 at level 9):

| Level | ms | fps | median error vs level 9 |
| --- | --- | --- | --- |
| 9 | 214 | 4.7 | — |
| 6 | 160 | 6.2 | ~1.9% |
| 4 | 125 | 8.0 | ~1.1-1.6% |
| 2 | 98 | 10.2 | ~1.2-1.8% |
| 0 | 71 | 14.0 | ~1.5-5.6% |

The accuracy column is agreement with level 9 over two test scenes — and note
it is **not monotonic**: level 4 beats level 6 and 7 on both images. That means
most of the difference is noise rather than lost signal, and the metric scale
stays stable all the way down. What genuinely degrades at low levels is spatial
detail — thin objects and depth discontinuities get softer — which a
whole-image error metric does not capture. Look at the output before trusting
level 0 for anything fine-grained.

Two defaults, deliberately different:

- **`src.depth` (stills) stays at level 9.** A one-off image has no reason to
  trade quality for 90 ms.
- **The web app defaults to level 4**, and the browser has a *Depth quality*
  selector (Fastest ~70 ms → Best ~215 ms) that overrides it per request.
  Server-side default: `--depth-resolution-level`.

Caveat for measurement: a **single pixel** is noisier than the whole-image
figures above. The centre-pixel reading moved 1.45→1.55 m across levels on the
test image (~6%), against ~1.5% for the image median. If you are going to act
on a distance, take a median over a region (which is what the object-distance
plan below does) and use a high level.

### Things that did *not* help

- **`torch.compile`**: no change (1.00x). MoGe is invoked through `.infer()`
  rather than `.forward()`, so wrapping the model compiles nothing. Compiling
  the inner ViT directly might pay off, but `resolution_level` already offers
  3x for one line of config.
- **`cudnn.benchmark`**: 216 ms vs 214 ms, i.e. nothing. It autotunes
  convolutions and this is a vision *transformer*.
- **Sending smaller frames**: MoGe resamples internally to the resolution
  implied by the token count, so 1920x1080 costs essentially the same as
  640x480. Don't bother downscaling.
- **Cheaper JPEG**: encoding is 0.5 ms at q=85. Nothing to win.

The webcam tab's rate slider only sets how often a frame is *submitted*; an
in-flight request is never overlapped, so the real rate is capped by the model
anyway. Lower it to stop queueing pointless work.

### On switching to a smaller model

`--model Ruicheng/moge-2-vitb-normal` (104M) or `moge-2-vits-normal` (35M) are
about 1.5-2x faster than ViT-L (326M) on GPU. That is a poor trade here:
GPU-vs-CPU already buys 200x, and the variants **disagree on absolute scale**.
On the same test frame ViT-L reported a 1.78 m median depth, ViT-B 0.94 m and
ViT-S 1.05 m. Since the whole point of this feature is *metric* depth that a
robot will act on, don't swap the model for a 2x speedup without first checking
its distances against a tape measure. ViT-L stays the default.

## 6. Camera intrinsics (`camera.json`) — optional

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

## 7. Using depth from Python

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

## 8. Distance and angle to an object — done

This is now `src.locate` / `src/locator.py`; see **[locate.md](locate.md)**.

The pieces were deliberately shaped for it:

1. `src.find` gives a box for the described object.
2. `estimate()` gives `points` for the same frame.
3. Distance = the **median** of the points inside the box (median, not mean, so
   background pixels leaking into the box do not drag it — and only the centred
   half of the box is sampled, since a box's corners are usually background).
4. Bearing = `atan2(X, Z)` of the median point — or, when `camera.json` is fully
   calibrated, from the pixel and the intrinsics: `atan2((u - cx) / fx, 1)`.

Step 4 is the one that genuinely wants a calibrated `cx`/`fx`, which is why
`camera.json` carries them: with them the bearing is exact pinhole geometry
rather than an assumption that the lens is centred.

```bash
python -m src.locate image "blue cup" data/table3.jpg
#     distance   0.80 m   (near surface 0.79 m, depth 0.78 m)
#     bearing    12.4 deg right, 1.7 deg down   [depth model]
```
