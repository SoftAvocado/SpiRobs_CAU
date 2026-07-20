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
import signal
import socket
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .camera import load_camera
from .depth_estimator import DEFAULT_MODEL as DEPTH_MODEL
from .depth_estimator import DepthEstimator
from .detector import DEFAULT_MODEL, ObjectDetector
from .find import DEFAULT_CONF as FIND_CONF
from .find import pick_unique
from .locator import locate

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="SpiRobs Object Detection")


class FrameDecodeError(Exception):
    """The uploaded bytes were not a decodable image."""


def _decode_frame(raw: bytes) -> np.ndarray:
    """Decode uploaded bytes to a BGR image, or raise :class:`FrameDecodeError`.

    Two distinct failure modes, which is why this is shared by every endpoint:
    OpenCV *returns None* for bytes that are not a valid image, but *raises* an
    assertion error for an empty buffer (``!buf.empty()``). A browser that is
    shutting its camera down can easily post zero bytes, so the raising case is
    a normal thing to hit, not a programming error.
    """
    buffer = np.frombuffer(raw, dtype=np.uint8)
    if buffer.size == 0:
        raise FrameDecodeError("empty frame")
    try:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error as exc:  # pragma: no cover - defensive
        raise FrameDecodeError(f"could not decode frame: {exc}") from exc
    if image is None:
        raise FrameDecodeError("could not decode frame")
    return image


@app.exception_handler(FrameDecodeError)
async def _frame_decode_handler(request, exc: FrameDecodeError) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=400)


@app.exception_handler(Exception)
async def _unhandled_handler(request, exc: Exception) -> JSONResponse:
    """Return JSON for *any* unhandled error.

    Starlette's default 500 is the plain text "Internal Server Error", which
    makes the browser's ``response.json()`` blow up with a confusing parse
    error ("Unexpected token 'I'") that hides the real problem. The UI parses
    every response as JSON, so every response must be JSON.
    """
    traceback.print_exc()
    return JSONResponse(
        {"error": f"{type(exc).__name__}: {exc}"}, status_code=500
    )

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
        # Goes through ObjectDetector.set_classes rather than poking the
        # Ultralytics model directly: it also keeps CLIP's recorded device in
        # sync, without which the second query of a GPU session crashes.
        _find_detector.set_classes([query])
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


#: Live view default, deliberately lower than the CLI's. Depth cost is almost
#: entirely the model — measured on a laptop RTX 4070, a 640x480 frame spends
#: 209 ms of its 215 ms in inference, with colourising, JPEG and base64 adding
#: ~1.7 ms between them — so resolution_level is the only meaningful dial.
#: Level 4 runs at 125 ms instead of 214 ms while agreeing with level 9 to
#: ~1-2% (a couple of cm at arm's length), which is the right trade for a live
#: view. Stills keep the slower, sharper default in src/depth.py.
DEFAULT_LIVE_RESOLUTION_LEVEL = 4


def get_depth_estimator() -> DepthEstimator:
    global _depth_estimator
    if _depth_estimator is None:
        _depth_estimator = DepthEstimator(
            model_path=os.environ.get("DEPTH_MODEL", DEPTH_MODEL),
            device=os.environ.get("DEPTH_DEVICE") or None,
            camera=load_camera(os.environ.get("CAMERA_CONFIG") or None),
            resolution_level=int(
                os.environ.get(
                    "DEPTH_RESOLUTION_LEVEL", DEFAULT_LIVE_RESOLUTION_LEVEL
                )
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
    image = _decode_frame(await frame.read())
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

    image = _decode_frame(await frame.read())

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
    frame: UploadFile = File(...),
    reset: str = Form("0"),
    level: str = Form(""),
) -> JSONResponse:
    """Metric depth map for one frame, returned as a colourised JPEG.

    Browser-side counterpart of ``python -m src.depth webcam``. The colourised
    image is sent as a data URL rather than a per-pixel depth array: a full
    float32 depth map is ~4 MB per frame, far too much for a live loop, while
    the numbers a caller actually wants right now (scene range, centre
    distance) are small enough to send alongside as JSON.
    """
    global _depth_range

    image = _decode_frame(await frame.read())

    try:
        estimator = get_depth_estimator()
    except ImportError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    if reset == "1":
        _depth_range = None

    # Speed/detail dial, chosen per request by the UI. Requests are handled one
    # at a time here, so mutating the shared estimator is safe.
    if level:
        try:
            estimator.resolution_level = max(0, min(9, int(level)))
        except ValueError:
            pass  # nonsense value: keep whatever is configured

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
            "resolution_level": estimator.resolution_level,
            # Surfaced so the UI can explain a slow frame rate rather than just
            # looking broken: on CPU this endpoint takes ~45 s per frame.
            "device": estimator.device,
            "image": "data:image/jpeg;base64,"
            + base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    )


@app.post("/locate")
async def locate_object(
    frame: UploadFile = File(...),
    query: str = Form(...),
    level: str = Form(""),
) -> JSONResponse:
    """Distance and bearing to ONE object described by ``query``.

    Browser-side counterpart of ``python -m src.locate webcam``, and the only
    endpoint that runs *both* models on the same frame: the detector locates the
    object in the image, the depth model says how far each pixel is, and
    :func:`src.locator.locate` reduces the two to metres and degrees.

    Depth is the expensive half, so it only runs once the object has actually
    been found — a frame with nothing matching costs a detection and no more.

    No image comes back: the numbers are the point here, and the browser already
    has the frame to draw the box on. That keeps the response a few hundred
    bytes instead of the ~50 kB JPEG the depth tab has to send.
    """
    query = query.strip()
    if not query:
        return JSONResponse({"error": "query must not be empty"}, status_code=400)

    image = _decode_frame(await frame.read())
    height, width = image.shape[:2]

    detector = get_find_detector(query)
    candidates = detector.detect(image)
    match = pick_unique(candidates)

    base = {
        "width": width,
        "height": height,
        "query": query,
        "candidates": len(candidates),
        "found": match is not None,
    }
    if match is None:
        return JSONResponse({**base, "measured": False, "location": None})

    try:
        estimator = get_depth_estimator()
    except ImportError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    if level:
        try:
            estimator.resolution_level = max(0, min(9, int(level)))
        except ValueError:
            pass  # nonsense value: keep whatever is configured

    depth_map = estimator.estimate(image)
    location = locate(match, depth_map, camera=estimator.camera)

    return JSONResponse(
        {
            **base,
            # Found but unmeasurable is a real outcome (the model saw no valid
            # geometry in the box), and distinct from not found — the UI says so
            # rather than showing a made-up distance.
            "measured": location is not None,
            "match": match.as_dict(),
            "location": location.as_dict() if location else None,
            "resolution_level": estimator.resolution_level,
            "device": estimator.device,
        }
    )


# Serve any additional static assets (kept last so it doesn't shadow routes).
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _port_is_free(host: str, port: int) -> bool:
    """Whether ``host:port`` can be bound right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        # Same option uvicorn uses, so this probe answers the question uvicorn
        # will actually ask rather than a stricter one (a socket lingering in
        # TIME_WAIT would otherwise look occupied when it is not).
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
            return True
        except OSError:
            return False


def _ancestor_pids() -> set[int]:
    """Our own PID plus every process above us.

    Anything that *launched* us — the shell, a ``timeout`` wrapper, VS Code's
    task runner — has this module's name somewhere in its command line, so a
    naive scan happily matches it. Killing one of those kills us too (a
    ``timeout python -m src.webcam_server --force`` run terminated itself this
    way), and none of them is ever the process holding the port.
    """
    proc = Path("/proc")
    seen = set()
    pid = os.getpid()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        try:
            status = (proc / str(pid) / "status").read_text()
        except OSError:
            break
        parent = 0
        for line in status.splitlines():
            if line.startswith("PPid:"):
                parent = int(line.split()[1])
                break
        pid = parent
    return seen


def _other_server_pids() -> list[int]:
    """PIDs of *other* processes in this container running this server.

    Read straight from ``/proc`` rather than shelling out to lsof/pgrep, so it
    works regardless of which diagnostic tools the image happens to ship.
    """
    proc = Path("/proc")
    if not proc.is_dir():  # not Linux; nothing we can inspect
        return []
    skip = _ancestor_pids()
    pids = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in skip:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:  # process exited, or not ours to read
            continue
        argv = [part for part in raw.split(b"\0") if part]
        if not argv or b"src.webcam_server" not in b" ".join(argv):
            continue
        # Require the executable itself to be a Python, so wrapper commands
        # that merely mention the module are not mistaken for the server.
        if b"python" not in Path(argv[0].decode(errors="replace")).name.encode():
            continue
        pids.append(int(entry.name))
    return sorted(pids)


def _reclaim_port(host: str, port: int, pids: list[int]) -> bool:
    """Stop older servers and wait for the port to come free."""
    for pid in pids:
        print(f"Stopping previous server (PID {pid}) ...")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"error: not allowed to stop PID {pid}", file=sys.stderr)
            return False

    for _ in range(20):  # up to ~5 s for a graceful shutdown
        if _port_is_free(host, port):
            return True
        time.sleep(0.25)

    for pid in pids:  # graceful shutdown did not work; insist
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    time.sleep(0.5)
    return _port_is_free(host, port)


def _check_port(args: argparse.Namespace) -> bool:
    """Report a busy port clearly, and optionally take it over.

    Checked *before* the model is loaded: warming the detector takes seconds,
    and failing to bind afterwards wastes all of it. uvicorn's own message for
    this is a bare ``[Errno 98] address already in use``, which does not hint
    at the actual cause — a previous server still running in this container,
    which closing the browser does not stop.
    """
    if _port_is_free(args.host, args.port):
        return True

    pids = _other_server_pids()
    if args.force:
        if not pids:
            print(
                f"error: port {args.port} is in use by something that is not "
                "this server; --force cannot help. Try --port 8001.",
                file=sys.stderr,
            )
            return False
        if _reclaim_port(args.host, args.port, pids):
            return True
        print(f"error: could not free port {args.port}", file=sys.stderr)
        return False

    who = (
        f"An earlier server is still running here (PID {', '.join(map(str, pids))})."
        if pids
        else "Another process is holding it."
    )
    print(
        f"error: port {args.port} is already in use.\n"
        f"       {who}\n"
        "       The server is a separate process, so closing the browser does\n"
        "       not stop it. Either take the port over:\n"
        "           python -m src.webcam_server --force\n"
        "       or run somewhere else:\n"
        f"           python -m src.webcam_server --port {args.port + 1}",
        file=sys.stderr,
    )
    return False


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
        default=DEFAULT_LIVE_RESOLUTION_LEVEL,
        choices=range(10),
        metavar="0-9",
        help=f"depth model working resolution (default {DEFAULT_LIVE_RESOLUTION_LEVEL} "
        "for the live view; lower is faster and coarser). The browser's "
        "Depth quality selector overrides this per request.",
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="camera intrinsics JSON for the depth tab (default: camera.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="if the port is busy, stop the previous server and take it over",
    )
    args = parser.parse_args(argv)

    # Before anything expensive: refuse early and helpfully if we cannot bind.
    if not _check_port(args):
        return 1

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
