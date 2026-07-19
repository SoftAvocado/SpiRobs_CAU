# SpiRobs_CAU

Master-project, based on Spirobs (Logarithmic Spiral-shaped Robots for Versatile Grasping Across Scales)

## How it works

### Object detection

YOLO-based object detection (image / video / live webcam)l

```bash
python -m src.detect image data/table2.jpg    # annotate an image
python -m src.detect video data/table2.mp4     # annotate a video
python -m src.webcam_server                  # live webcam → http://localhost:8000
```

### Finding one specific object

Instead of labelling everything, describe the one thing you want in words:

```bash
python -m src.find image "blue cup" data/table2.jpg    # → table2_found.jpg, or "not found"
python -m src.find video "blue cup" data/table2.mp4    # → which frames contain it
python -m src.find webcam "blue cup"                   # live (Linux host only)
```

The description is assumed to name a **unique** object, so at most one box is
ever reported: if several candidates match, the most confident one wins, and a
tie is broken at random (`--seed` makes that repeatable).

Exit code is 0 when found, 1 when not.

### Retrieving depth and angle to an object
todo

## About the project 

### Dev container 

All the dependencies lie inside a dev container. You can run it via Visual Studio Code.