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
  classes.py          # THE vocabulary: 80 COCO classes + ~200 table items (edit here)
  detector.py         # ObjectDetector: YOLO-World wrapper + box/label drawing (core)
  detect.py           # CLI: detect EVERYTHING in classes.py (image / video / webcam)
  find.py             # CLI: find ONE object you describe in words ("blue cup")
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
detections), `--conf 0.4` (confidence threshold), `--model yolov8m-worldv2.pt`.

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

## 5. Find one specific object (`src.find`)

The commands above answer *"what is in this picture?"*. This one answers
*"is my blue cup in this picture, and where?"* — you describe the object in
plain English and get either an annotated output or `not found` in the console.

```bash
python -m src.find image "blue cup" data/table.jpg
# Found 1 match(es) for "blue cup" in table.jpg:
#   blue cup        conf=0.31  box=(412,180)-(505,297)
# Annotated image written to: data/table_found.jpg

python -m src.find image "unicorn" data/table.jpg
# not found: "unicorn"
```

Videos report which frames contained the object; the webcam mode prints a live
`FOUND` / `not found` status. Both write an annotated output when you pass `-o`.

```bash
python -m src.find video "blue cup" data/clip.mp4 -o found.mp4
python -m src.find webcam "blue cup"        # Linux host only, see section 4
```

The process **exits 0 when the object was found and 1 when it was not**, so it
drops straight into a shell script:

```bash
if python -m src.find image "blue cup" photo.jpg --best; then
    echo "cup is on the table"
fi
```

### How it works

No second model and no training. YOLO-World is *open-vocabulary* — it is
prompted with text — so `src.detect` hands it the ~287 phrases from
`classes.py`, and `src.find` hands it your one phrase instead. Every box it
returns is therefore already a candidate match for your description.

Consequently the description does **not** need to be in `classes.py`, and
`classes.py` is irrelevant to this command.

### Accuracy knobs for `find`

- **`--best`** keeps only the single strongest match. Use it when you know
  there is at most one such object; it removes near-duplicate boxes.
- **`--conf`** defaults to `0.10` here, lower than the `0.25` of `src.detect`,
  because scores against a single free-text prompt are not comparable to scores
  against a large vocabulary. Raise it if you get false matches, lower it if a
  visibly-present object is missed.
- **Attributes are the weak spot.** The model grounds nouns ("cup") far more
  reliably than modifiers ("blue"), so `"blue cup"` may still box a red one.
  Being concrete and visual helps (`"blue ceramic mug"` over `"my cup"`). If
  colour precision turns out to matter, the next step would be re-ranking the
  candidate crops with CLIP — worth doing only if you hit that limit.
- **`--model yolov8m-worldv2.pt`** (or `l`/`x`) improves grounding noticeably.

## What can be detected (and how to change it)

Standard YOLO only knows COCO's 80 classes, so it can't see a pen, a charger,
and most desk clutter. To fix that without any training, this project runs an
open-vocabulary [YOLO-World](https://docs.ultralytics.com/models/yolo-world/)
model over a **fixed, curated vocabulary**: the 80 COCO classes **plus ~200
common table/desk items** (pen, mug, charger, keys, glasses, ...).

That vocabulary lives in one file — [`src/classes.py`](../src/classes.py) — and
editing it is the **only** way to change what gets detected. There are no
class-selection command-line flags, by design.

To add or remove an object, edit the `TABLE_ITEMS` list:

```python
# src/classes.py
TABLE_ITEMS = [
    "pen", "pencil", "marker",
    "screwdriver",          # <- add a new object here (concrete, lower-case)
    # "magazine",           # <- comment out / delete to stop detecting one
    ...
]
```

Save and re-run — the change takes effect immediately. The full list handed to
the model is `COCO_CLASSES + TABLE_ITEMS`, de-duplicated.

First use downloads the YOLO-World weights and a small CLIP text encoder (both
are pre-baked into the dev container image, so a rebuilt container runs offline).

### Accuracy knobs

- **Too many false positives?** Raise the threshold: `--conf 0.4`. A big
  vocabulary (~287 classes) makes the model guessier, so this is the main dial.
- **Missing faint objects?** Lower it: `--conf 0.15`.
- **Need more accuracy overall?** Use a larger model (below). Trim
  `TABLE_ITEMS` down to what you actually care about — fewer, well-chosen
  classes detect more reliably than a huge list.

## Choosing a model

`--model` selects the YOLO-World size; larger = more accurate, slower:

| Model                | Speed   | Accuracy       |
|----------------------|---------|----------------|
| `yolov8s-worldv2.pt` | fast    | good (default) |
| `yolov8m-worldv2.pt` | medium  | better         |
| `yolov8l-worldv2.pt` | slower  | high           |
| `yolov8x-worldv2.pt` | slowest | highest        |

Names auto-download on first use (into the cached weights dir).

## GPU

The container runs on CPU by default. On a Linux host with NVIDIA drivers +
`nvidia-container-toolkit`, add `"--gpus=all"` to `runArgs` in
`devcontainer.json` and pass `--device 0`.
