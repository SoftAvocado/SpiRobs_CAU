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
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .detector import ObjectDetector

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
            model_path=os.environ.get("DETECT_MODEL", "yolo11n.pt"),
            conf=float(os.environ.get("DETECT_CONF", "0.25")),
            device=os.environ.get("DETECT_DEVICE") or None,
        )
    return _detector


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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


# Serve any additional static assets (kept last so it doesn't shadow routes).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="Webcam detection web server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    os.environ["DETECT_MODEL"] = args.model
    os.environ["DETECT_CONF"] = str(args.conf)
    if args.device:
        os.environ["DETECT_DEVICE"] = args.device

    print(f"Loading model '{args.model}' ...")
    get_detector()  # warm up before serving
    print(f"Open http://localhost:{args.port} in your browser.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
