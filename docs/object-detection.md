# Object Detection Infrastructure

Detect objects (bounding box + label) in an **image**, a **video file**, or a
**live webcam stream**, using [Ultralytics YOLO](https://docs.ultralytics.com/).
Everything runs inside a dev container — no Python setup on the host.

## Layout

```
.devcontainer/
  Dockerfile          # Python 3.11 + OpenCV/ffmpeg libs + Python deps + YOLO weights
  devcontainer.json   # VS Code dev container config, forwards port 8000
requirements.txt      # Python dependencies
src/
  detector.py         # ObjectDetector: YOLO wrapper + box/label drawing (core)
  detect.py           # CLI for image / video / (Linux) webcam
  webcam_server.py    # FastAPI app: browser webcam -> YOLO -> boxes (all platforms)
  static/index.html   # Browser UI for the webcam app
```

## 1. Open the dev container

In VS Code (with the **Dev Containers** extension) open this folder and run
**"Reopen in Container"**. The image builds once and installs all dependencies,
including the default `yolo11n.pt` weights.

To build/run from the CLI instead:

```bash
docker build -t spirobs-detect -f .devcontainer/Dockerfile .
docker run --rm -it -p 8000:8000 -v "${PWD}:/workspaces/SpiRobs_CAU" spirobs-detect bash
```

## 2. Detect objects in an image

```bash
python -m src.detect image path/to/photo.jpg
# -> writes photo_detected.jpg next to it, prints the detected objects
```

Options: `-o out.jpg` (output path), `--json dets.json` (also dump raw
detections), `--conf 0.4` (confidence threshold), `--model yolo11s.pt`.

## 3. Detect objects in a video file

```bash
python -m src.detect video path/to/clip.mp4 -o annotated.mp4
```

Reads every frame, draws boxes, writes an annotated `.mp4`.

## 4. Detect objects from the webcam

**On Windows / macOS** (your case): a container cannot reach the host camera
directly, so use the browser app. The browser captures the webcam and sends
frames to the container for inference:

```bash
python -m src.webcam_server        # then open http://localhost:8000
```

Click **Start camera**, allow webcam access, and you'll see live bounding
boxes. The detection rate slider trades latency for CPU load.

**On a Linux host** you can alternatively give the container the camera device
directly: uncomment `--device=/dev/video0` in `.devcontainer/devcontainer.json`,
rebuild, then:

```bash
python -m src.detect webcam --source 0 -o recording.mp4
```

## Choosing a model

`--model` accepts any Ultralytics weight name; larger = more accurate, slower:

| Model         | Speed      | Accuracy |
|---------------|------------|----------|
| `yolo11n.pt`  | fastest    | good (default) |
| `yolo11s.pt`  | fast       | better   |
| `yolo11m.pt`  | medium     | high     |

Unknown-but-recognized names auto-download on first use. To detect
project-specific objects (e.g. the parts the SpiRobs gripper grasps), train a
custom model with Ultralytics and pass its `.pt` path to `--model`.

## GPU

The container runs on CPU by default. On a Linux host with NVIDIA drivers +
`nvidia-container-toolkit`, add `"--gpus=all"` to `runArgs` in
`devcontainer.json` and pass `--device 0`.
