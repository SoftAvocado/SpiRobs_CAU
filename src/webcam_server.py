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
import base64
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .camera import load_camera
from .depth_estimator import DEFAULT_MODEL as DEPTH_MODEL
from .depth_estimator import DEFAULT_RESOLUTION_LEVEL, DepthEstimator
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


# A THIRD model, for depth. Completely independent of the two detectors above:
# depth estimation does not use detection and vice versa. Loaded lazily because
# MoGe-2 ViT-L is a large download that users who never open the depth tab
# should not pay for.
_depth_estimator: DepthEstimator | None = None
#: Colour-ramp bounds, locked on the first frame so the live view does not
#: pulse as the nearest/farthest points in the scene shift (same reasoning as
#: the video path in src/depth.py). Reset when the client starts a new session.
_depth_range: tuple[float, float] | None = None


def get_depth_estimator() -> DepthEstimator:
    global _depth_estimator
    if _depth_estimator is None:
        _depth_estimator = DepthEstimator(
            model_path=os.environ.get("DEPTH_MODEL", DEPTH_MODEL),
            device=os.environ.get("DEPTH_DEVICE") or None,
            camera=load_camera(os.environ.get("CAMERA_CONFIG") or None),
            resolution_level=int(
                os.environ.get("DEPTH_RESOLUTION_LEVEL", DEFAULT_RESOLUTION_LEVEL)
            ),
        )
    return _depth_estimator


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


@app.post("/depth")
async def depth(
    frame: UploadFile = File(...), reset: str = Form("0")
) -> JSONResponse:
    """Metric depth map for one frame, returned as a colourised JPEG.

    Browser-side counterpart of ``python -m src.depth webcam``. The colourised
    image is sent as a data URL rather than a per-pixel depth array: a full
    float32 depth map is ~4 MB per frame, far too much for a live loop, while
    the numbers a caller actually wants right now (scene range, centre
    distance) are small enough to send alongside as JSON.
    """
    global _depth_range

    raw = await frame.read()
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return JSONResponse({"error": "could not decode frame"}, status_code=400)

    try:
        estimator = get_depth_estimator()
    except ImportError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    if reset == "1":
        _depth_range = None

    depth_map = estimator.estimate(image)
    if _depth_range is None:
        _depth_range = depth_map.range_metres()
    near, far = _depth_range

    colored = estimator.colorize(depth_map, near=near, far=far)
    ok, encoded = cv2.imencode(".jpg", colored, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return JSONResponse({"error": "could not encode depth map"}, status_code=500)

    height, width = image.shape[:2]
    return JSONResponse(
        {
            "width": width,
            "height": height,
            "near_m": round(near, 3),
            "far_m": round(far, 3),
            "stats": depth_map.stats(),
            "fov_x_deg": estimator.fov_x_deg,
            # Surfaced so the UI can explain a slow frame rate rather than just
            # looking broken: on CPU this endpoint takes ~45 s per frame.
            "device": estimator.device,
            "image": "data:image/jpeg;base64,"
            + base64.b64encode(encoded.tobytes()).decode("ascii"),
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
    parser.add_argument(
        "--depth-model",
        default=DEPTH_MODEL,
        help=f"MoGe-2 weights for the depth tab (default {DEPTH_MODEL})",
    )
    parser.add_argument(
        "--depth-resolution-level",
        type=int,
        default=DEFAULT_RESOLUTION_LEVEL,
        choices=range(10),
        metavar="0-9",
        help="depth model working resolution; lower is faster and coarser",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="camera intrinsics JSON for the depth tab (default: camera.json)",
    )
    args = parser.parse_args(argv)

    os.environ["DETECT_MODEL"] = args.model
    os.environ["DETECT_CONF"] = str(args.conf)
    os.environ["FIND_CONF"] = str(args.find_conf)
    os.environ["DEPTH_MODEL"] = args.depth_model
    os.environ["DEPTH_RESOLUTION_LEVEL"] = str(args.depth_resolution_level)
    if args.device:
        os.environ["DETECT_DEVICE"] = args.device
        os.environ["DEPTH_DEVICE"] = args.device
    if args.camera:
        os.environ["CAMERA_CONFIG"] = args.camera

    # Say which of the two very different startups is about to happen: a cache
    # hit is ~2 s, a miss has to run the CLIP text encoder over the whole
    # vocabulary and takes ~25 s. Without this the first run just looks hung.
    print(f"Loading model '{args.model}' ...")
    detector = get_detector()  # warm up before serving
    if detector.prompt_cache_hit:
        print("Vocabulary loaded from the prompt cache.")
    else:
        print("Vocabulary embedded and cached; the next start will be quick.")
    print(f"Open http://localhost:{args.port} in your browser.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
