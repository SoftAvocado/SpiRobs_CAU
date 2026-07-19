# SpiRobs_CAU

Master-project, based on Spirobs (Logarithmic Spiral-shaped Robots for Versatile Grasping Across Scales)

## How it works

### Object detection

YOLO-based object detection (image / video / live webcam)l

```bash
python -m src.detect image path/to/photo.jpg    # annotate an image
python -m src.detect video path/to/clip.mp4     # annotate a video
python -m src.webcam_server                     # live webcam → http://localhost:8000
```

### Retrieving depth and angle to an object
todo

## About the project 

### Dev container 

All the dependencies lie inside a dev container. You can run it via Visual Studio Code.