# Object Detection Infrastructure

Detect objects (bounding box + label) in an **image**, a **video file**, or a
**live webcam stream**, using [Ultralytics YOLO](https://docs.ultralytics.com/).
Everything runs inside a dev container — no Python setup on the host.

## Layout

```text
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
                      #   /detect = everything, /find = one described object
  static/index.html   # Browser UI for the webcam app (both modes)
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
python -m src.find image "blue cup" data/table2.jpg
# Found "blue cup" in table2.jpg:
#   blue cup        conf=0.31  box=(412,180)-(505,297)
#   (3 candidates found, kept the strongest)
# Annotated image written to: data/table2_found.jpg

python -m src.find image "unicorn" data/table2.jpg
# not found: "unicorn"
```

The description is assumed to name a **unique** object, so exactly one box is
reported (or none). See [Picking the one object](#picking-the-one-object) below.

Videos report which frames contained the object; the webcam mode prints a live
`FOUND` / `not found` status. Both write an annotated output when you pass `-o`.

```bash
python -m src.find video "blue cup" data/clip.mp4 -o found.mp4
python -m src.find webcam "blue cup"        # direct camera: Linux host only
```

### Live search from the webcam (all platforms)

Same container-vs-camera problem as section 4, same solution: the browser app
has a **Find one object** mode.

```bash
python -m src.webcam_server        # then open http://localhost:8000
```

Switch to **Find one object**, type a description, and the live view shows a
single green box with `found "blue cup" · 74%`, or a red `not found: "blue cup"`
— the on-screen equivalent of what the CLI prints. You can retype the
description while the camera runs: only the text prompt is recomputed (~ms), so
it responds immediately instead of reloading the model.

The description is **committed** on Enter, on clicking out of the box, or after
a short typing pause — not on every keystroke. Otherwise the request loop would
pick up half-typed words and search for `p` on the way to `person`. While you
are typing a replacement, the previously committed description keeps being
searched rather than the view freezing.

Switching tabs or pressing Stop cancels whatever request is in flight, so a
reply can never be applied to a mode it was not asked for.

Re-prompting always goes through `ObjectDetector.set_classes()`, never
`ultralytics_model.set_classes()` directly. Ultralytics' CLIP wrapper records
the device it was *built* on and uses it to place the token tensor, but the
wrapper is an `nn.Module` hanging off the detection model — so moving that
model to the GPU for the first `predict()` moves CLIP's weights too while
leaving the recorded device at `cpu`. The next re-prompt then feeds CPU tokens
to CUDA weights:

```text
Expected all tensors to be on the same device, but got index is on cpu,
different from other tensors on cuda:0
```

In other words re-prompting worked exactly once per process on a GPU.
`set_classes()` re-syncs the recorded device before prompting, which fixes it
without rebuilding CLIP (that would cost ~10 s per query change).

This is the route to use on Windows/macOS. `python -m src.find webcam` opens the
camera device directly, which only a Linux host can hand to the container.

The process **exits 0 when the object was found and 1 when it was not**, so it
drops straight into a shell script:

```bash
if python -m src.find image "blue cup" photo.jpg; then
    echo "cup is on the table"
fi
```

### Picking the one object

The description is treated as naming a **unique** object, so the command
reports at most one box. Given several candidate boxes above the confidence
threshold, `pick_unique()` in [`src/find.py`](../src/find.py) resolves them:

1. **Highest confidence wins.** The other candidates are discarded (the console
   notes how many there were, so a suspiciously high count tells you the
   description is too vague).
2. **Exact ties are broken at random.** If two boxes share the top confidence
   there is nothing left to prefer one by, so one is picked uniformly. Pass
   `--seed 0` to make that choice repeatable across runs.

In video and webcam mode this happens **per frame** — each frame independently
yields its one best box — so a tie broken differently on consecutive frames can
make the box jump between two similar objects. If you see that, it is a signal
the description doesn't discriminate between them; make it more specific.

### How it works

No second model and no training. YOLO-World is *open-vocabulary* — it is
prompted with text — so `src.detect` hands it the ~214 phrases from
`classes.py`, and `src.find` hands it your one phrase instead. Every box it
returns is therefore already a candidate match for your description.

Consequently the description does **not** need to be in `classes.py`, and
`classes.py` is irrelevant to this command.

### Accuracy knobs for `find`

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
  vocabulary (~214 classes) makes the model guessier, so this is the main dial.
- **Missing faint objects?** Lower it: `--conf 0.15`.
- **Need more accuracy overall?** Use a larger model (below). Trim
  `TABLE_ITEMS` down to what you actually care about — fewer, well-chosen
  classes detect more reliably than a huge list.

## Startup time and the prompt cache

Loading the weights is not what makes startup slow. Measured in the dev
container:

| Step | Time |
| --- | --- |
| `import ultralytics` | 1.8 s |
| loading `yolov8s-worldv2.pt` | 0.06 s |
| **`set_classes()` over the vocabulary** | **~22 s** |

`set_classes()` builds the CLIP text encoder and embeds every phrase in
`classes.py`. That dominates everything else, and it used to run on *every*
invocation.

Those embeddings depend only on the class list, so `ObjectDetector` now saves
the prompted model once and reloads it afterwards:

```text
$YOLO_WEIGHTS_DIR/yolov8s-worldv2-vocab-<hash>.pt    26 MB
```

Cold start ~22 s, warm start **0.05 s**, and detections are identical — the
cached run reproduces the same 14 boxes with the same coordinates, confidences
and labels.

Notes:

- The `<hash>` covers the base weights, the Ultralytics version and the class
  list, so **editing `classes.py` rebuilds the cache automatically**. There is
  no manual invalidation step and no way to keep using stale embeddings.
- The cache is written *before* any inference runs. This matters: `predict()`
  fuses conv+BN layers in place, and caching a fused model then reloading it
  shifts the numerics enough to drop borderline detections (14 boxes became 13
  in testing).
- CLIP is deleted before saving. Ultralytics attaches the full text encoder to
  the model, which would make the file 329 MB instead of 26 MB.
- **`src.find` is not cached.** Its vocabulary is a one-off free-text query, so
  a cache entry would cost 26 MB for a phrase that is unlikely to be reused. It
  pays ~11 s once per process; re-prompting the *same* live model with a new
  query afterwards costs 0.06 s, which is why the web app keeps one find
  detector alive instead of rebuilding it per query.
- Rebuilding the dev container clears the cache, so the first run after a
  rebuild pays the ~22 s again.
- If the weights directory is not writable, caching is skipped with a warning
  and everything still works — just slowly.

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
