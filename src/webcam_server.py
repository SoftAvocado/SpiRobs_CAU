"""Browser-based webcam object detection.

Why this exists: a Docker container on Windows/macOS cannot access the host
webcam directly, and it has no display for an OpenCV window. So instead the
*browser* (which can access the camera and draw to screen) captures frames and
POSTs them to this FastAPI server running inside the container. The server runs
YOLO and returns the bounding boxes as JSON, which the browser overlays on the
live video.

Run it:

    python -m src.webcam_server            # then open http://localhost:8000
    python -m src.webcam_server --model yolo11s.pt --conf 0.4

The container port 8000 is forwarded by devcontainer.json.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .detector import DEFAULT_MODEL, ObjectDetector
from .find import DEFAULT_CONF as FIND_CONF
from .find import pick_unique

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="SpiRobs Object Detection")

# The detector is created once at startup (loading weights is expensive).
# Config comes from environment variables so `python -m src.webcam_server`
# can pass CLI args through to the uvicorn worker.
_detector: ObjectDetector | None = None


def get_detector() -> ObjectDetector:
    global _detector
    if _detector is None:
        _detector = ObjectDetector(
            model_path=os.environ.get("DETECT_MODEL", DEFAULT_MODEL),
            conf=float(os.environ.get("DETECT_CONF", "0.25")),
            device=os.environ.get("DETECT_DEVICE") or None,
        )
    return _detector


# A SECOND detector for the "find one object" mode. It is kept separate from
# the one above so that re-prompting it with a new description cannot disturb
# an in-flight "detect everything" request. Loaded lazily: users who never open
# the find tab never pay for it.
_find_detector: ObjectDetector | None = None
_find_query: str | None = None


def get_find_detector(query: str) -> ObjectDetector:
    """Detector prompted with ``query`` as its entire vocabulary.

    Switching queries re-prompts the existing model rather than loading a new
    one — ``set_classes`` only re-runs the small CLIP text encoder, so typing a
    new description costs milliseconds instead of a full weight load.
    """
    global _find_detector, _find_query
    if _find_detector is None:
        _find_detector = ObjectDetector(
            model_path=os.environ.get("DETECT_MODEL", DEFAULT_MODEL),
            conf=float(os.environ.get("FIND_CONF", str(FIND_CONF))),
            device=os.environ.get("DETECT_DEVICE") or None,
            classes=[query],
        )
        _find_query = query
    elif query != _find_query:
        _find_detector.classes = [query]
        _find_detector.model.set_classes([query])
        _find_query = query
    return _find_detector


@app.get("/")
def index() -> FileResponse:
    # ``no-cache`` = the browser may keep a copy but MUST revalidate before
    # using it. Without an explicit Cache-Control, browsers fall back to
    # heuristic caching and happily serve a stale UI for minutes after
    # index.html changes — which looks exactly like "the new feature isn't
    # there". The etag makes revalidation a cheap 304.
    return FileResponse(
        STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"}
    )


@app.get("/health")
def health() -> JSONResponse:
    d = get_detector()
    return JSONResponse({"status": "ok", "model": str(d.model.model_name)})


@app.post("/detect")
async def detect(frame: UploadFile = File(...)) -> JSONResponse:
    """Accept one JPEG/PNG frame, return detections as JSON."""
    raw = await frame.read()
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "could not decode frame"}, status_code=400)

    detections = get_detector().detect(image)
    height, width = image.shape[:2]
    return JSONResponse(
        {
            "width": width,
            "height": height,
            "detections": [d.as_dict() for d in detections],
        }
    )


@app.post("/find")
async def find(
    frame: UploadFile = File(...), query: str = Form(...)
) -> JSONResponse:
    """Look for ONE object described by ``query`` in a single frame.

    Browser-side counterpart of ``python -m src.find webcam`` — this is how the
    find feature works on Windows/macOS, where the container cannot open the
    host camera itself.
    """
    query = query.strip()
    if not query:
        return JSONResponse({"error": "query must not be empty"}, status_code=400)

    raw = await frame.read()
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "could not decode frame"}, status_code=400)

    # No await between prompting the model and predicting: the two steps stay
    # atomic with respect to other requests hitting this same detector.
    detector = get_find_detector(query)
    candidates = detector.detect(image)
    match = pick_unique(candidates)

    height, width = image.shape[:2]
    return JSONResponse(
        {
            "width": width,
            "height": height,
            "query": query,
            "found": match is not None,
            "candidates": len(candidates),
            "match": match.as_dict() if match else None,
        }
    )


# Serve any additional static assets (kept last so it doesn't shadow routes).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="Webcam detection web server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--find-conf",
        type=float,
        default=FIND_CONF,
        help=f"confidence threshold for 'find one object' mode "
        f"(default {FIND_CONF}; separate from --conf, see docs)",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    os.environ["DETECT_MODEL"] = args.model
    os.environ["DETECT_CONF"] = str(args.conf)
    os.environ["FIND_CONF"] = str(args.find_conf)
    if args.device:
        os.environ["DETECT_DEVICE"] = args.device

    print(f"Loading model '{args.model}' ...")
    get_detector()  # warm up before serving
    print(f"Open http://localhost:{args.port} in your browser.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
