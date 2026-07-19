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

### Retrieving depth and angle to an object

todo — will combine the two features above: take the box from `src.find`, read
the metric 3D points inside it from `src.depth`, and report distance + bearing.

## About the project 

### Dev container 

All the dependencies lie inside a dev container. You can run it via Visual Studio Code.