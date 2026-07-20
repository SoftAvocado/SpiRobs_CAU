# SpiRobs_CAU

Master-project, based on Spirobs (Logarithmic Spiral-shaped Robots for Versatile Grasping Across Scales)

## How it works

### Object detection

YOLO-based object detection (image / video / live webcam)l

```bash
python -m src.detect image data/table2.jpg             # annotate an image
python -m src.detect video data/table2.mp4             # annotate a video
python -m src.webcam_server                            # live webcam → http://localhost:8000
```

### Finding one specific object

Instead of labelling everything, describe the one thing you want in words:

```bash
python -m src.find image "blue cup" data/table2.jpg    # → table2_found.jpg, or "not found"
python -m src.find video "blue cup" data/table2.mp4    # → which frames contain it
python -m src.webcam_server                            # live → pick "Find one object"
```

The description is assumed to name a **unique** object, so at most one box is
ever reported: if several candidates match, the most confident one wins, and a
tie is broken at random (`--seed` makes that repeatable).

Exit code is 0 when found, 1 when not.

### Depth map (metric distance per pixel)

Monocular **metric** depth with [MoGe-2](https://github.com/microsoft/MoGe)
ViT-L — one RGB frame in, distance in **metres** for every pixel out. Runs
independently of object detection:

```bash
python -m src.depth image data/table2.jpg               # → table2_depth.jpg
python -m src.depth video data/table2.mp4               # → table2_depth.mp4
python -m src.webcam_server                             # live → pick "Depth map"
```

Red is near, blue is far, dark grey means the model found no valid geometry
there. The colour bar on the right gives the two ends in metres. Add
`--side-by-side` to keep the source frame next to the depth map, and `--npy
out.npy` to save the raw `float32` metres array (`NaN` = invalid) instead of
just a picture.

**This needs a GPU.** ViT-L runs at ~0.2 s/frame on a laptop RTX 4070 and
~45 s/frame on CPU — a factor of ~200, so a CPU fallback makes the webcam tab
unusable rather than merely slow. `devcontainer.json` requests GPU passthrough
(`"hostRequirements": {"gpu": "optional"}`); **rebuild the dev container** for
it to take effect, then check `python -c "import torch;
print(torch.cuda.is_available())"` prints `True`.

Camera intrinsics are **optional**: MoGe-2 estimates the field of view itself.
Filling in `camera.json` only makes the metric scale more accurate, since a
wrong FOV guess becomes a proportionally wrong distance.

Full details in [docs/depth.md](docs/depth.md).

### Distance to a point

Click anywhere on the live view and get the distance and bearing to that pixel.
No object detection involved — the depth map already holds a metric 3D point per
pixel, so a click needs no vocabulary and no second model:

```bash
python -m src.webcam_server                              # live → pick "Distance to point"
```

Click once and the reading keeps updating as the scene moves; click elsewhere to
re-aim. A small patch around the pixel is sampled and reduced with a median, so
a click is repeatable rather than at the mercy of one noisy pixel. Points with
no valid geometry (sky, mirrors, blown-out highlights) are reported as
unmeasurable instead of guessed.

This mode is browser-only: it needs somewhere to click, and the container is
headless.

### Distance to an object

Describe one object and get **how far away it is and which way to turn** — the
detection and depth features running on the same frame:

```bash
python -m src.locate image "blue cup" data/table3.jpg    # → table3_located.jpg
python -m src.locate video "blue cup" data/table2.mp4    # → per-frame measurements
python -m src.webcam_server                              # live → pick "Distance to object"
```

```text
Found "blue cup" in table3.jpg:
  blue cup        conf=0.86
    distance   0.80 m   (near surface 0.79 m, depth 0.78 m)
    bearing    12.4 deg right, 1.7 deg down   [depth model]
    point      x=+0.17  y=+0.02  z=+0.78  (metres, camera frame)
```

`distance` is the straight-line range to the middle of the object, `near
surface` is the range to its front (what a gripper actually meets) and `depth`
is only the forward component. Bearing is positive to the **right**, elevation
positive **up** — both relative to the optical axis, so a robot base can turn by
them directly.

Depth is sampled from the middle of the box and reduced with a **median**, so
background leaking into the corners of the box does not drag the distance out.
If the object is found but has no valid depth, that is reported as such rather
than guessed. Exit code is 0 only when the object was both found and measured.

Both models run per frame, so video is the slowest mode — use `--stride N` to
measure every Nth frame. `camera.json` is still optional, but filling in
`fx`/`fy`/`cx`/`cy` makes the bearing exact instead of assuming a centred lens.

Full details in [docs/locate.md](docs/locate.md).

## About the project 

### Dev container 

All the dependencies lie inside a dev container. You can run it via Visual Studio Code.