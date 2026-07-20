"""Find one described object and report how far away and in which direction.

The "distance to object" mode, and the one the earlier three were building
towards: it runs :mod:`src.find`'s open-vocabulary search and :mod:`src.depth`'s
metric depth on the *same* frame, then measures the object's box against the
depth map (:mod:`src.locator`) to get a distance in metres and a bearing in
degrees.

To measure a place rather than an object, the browser app's "distance to point"
mode takes a click and skips the detector entirely (:func:`src.locator.locate_point`).

    python -m src.locate image "blue cup" data/table3.jpg
    #   found "blue cup" (conf 0.31)
    #     distance   1.24 m   (near surface 1.11 m, depth 1.21 m)
    #     bearing    12.4 deg right, 4.8 deg up

    python -m src.locate video "blue cup" data/table2.mp4 --output out.mp4
    python -m src.locate webcam "blue cup" 0        # LINUX host only

Exit code is 0 when the object was found *and* could be measured, 1 otherwise,
so this can be used in shell scripts like ``src.find``.

Both models run on every frame, so video is roughly detection + depth in cost —
see ``--resolution-level`` and ``--stride`` if that is too slow.

Options: --conf, --camera, --resolution-level, --stride, --output, --json,
--model, --depth-model, --device, --seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2

from .camera import load_camera
from .depth_estimator import DEFAULT_MODEL as DEPTH_MODEL
from .depth_estimator import DEFAULT_RESOLUTION_LEVEL, DepthEstimator
from .detector import DEFAULT_MODEL, ObjectDetector
from .find import DEFAULT_CONF, pick_unique
from .locator import Measurement, draw, locate


def _default_output(input_path: Path, suffix: str | None = None) -> Path:
    """``photo.jpg`` -> ``photo_located.jpg``."""
    suffix = suffix or input_path.suffix
    return input_path.with_name(f"{input_path.stem}_located{suffix}")


def _print_location(location: Measurement) -> None:
    det = location.detection
    print(f"  {det.label:<15} conf={det.confidence:.2f}")
    print(
        f"    distance   {location.distance_m:.2f} m   "
        f"(near surface {location.nearest_m:.2f} m, "
        f"depth {location.depth_m:.2f} m)"
    )
    print(
        f"    bearing    {abs(location.bearing_deg):.1f} deg {location.side}, "
        f"{abs(location.elevation_deg):.1f} deg "
        f"{'up' if location.elevation_deg >= 0 else 'down'}"
        f"   [{location.bearing_source}]"
    )
    x, y, z = location.point
    print(f"    point      x={x:+.2f}  y={y:+.2f}  z={z:+.2f}  (metres, camera frame)")
    if location.valid_fraction < 0.5:
        # Not fatal, but the number rests on very little evidence and silently
        # reporting two decimals of it would overstate what is known.
        print(
            f"    warning: only {100 * location.valid_fraction:.0f}% of the sampled "
            "region had valid depth"
        )


def _measure(
    detector: ObjectDetector,
    estimator: DepthEstimator,
    frame,
) -> tuple[Measurement | None, int]:
    """Detect, then measure. Returns ``(location, candidate_count)``.

    ``location`` is ``None`` when the object was not found *or* was found but
    has no valid depth; the candidate count distinguishes the two.
    """
    candidates = detector.detect(frame)
    match = pick_unique(candidates)
    if match is None:
        return None, 0
    depth_map = estimator.estimate(frame)
    return locate(match, depth_map, camera=estimator.camera), len(candidates)


def run_image(
    detector: ObjectDetector, estimator: DepthEstimator, args: argparse.Namespace
) -> int:
    input_path = Path(args.source)
    if not input_path.exists():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        return 1

    image = cv2.imread(str(input_path))
    if image is None:
        print(f"error: could not read image: {input_path}", file=sys.stderr)
        return 1

    location, candidates = _measure(detector, estimator, image)
    if candidates == 0:
        print(f'not found: "{args.query}"')
        return 1
    if location is None:
        print(
            f'found "{args.query}", but the depth model reported no valid '
            "geometry inside its box — cannot measure it.",
            file=sys.stderr,
        )
        return 1

    print(f'Found "{args.query}" in {input_path.name}:')
    _print_location(location)
    if candidates > 1:
        print(f"  ({candidates} candidates found, kept the strongest)")

    output_path = Path(args.output) if args.output else _default_output(input_path)
    cv2.imwrite(str(output_path), draw(image, location, query=args.query))
    print(f"Annotated image written to: {output_path}")

    if args.json:
        payload = location.as_dict()
        payload["query"] = args.query
        payload["camera"] = estimator.camera.as_dict()
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"Measurement (JSON) written to: {args.json}")
    return 0


def run_video(
    detector: ObjectDetector, estimator: DepthEstimator, args: argparse.Namespace
) -> int:
    input_path = Path(args.source)
    if not input_path.exists():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"error: could not open video: {input_path}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    output_path = Path(args.output) if args.output else _default_output(
        input_path, ".mp4"
    )
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_idx = 0
    measured: list[tuple[int, Measurement]] = []
    last: Measurement | None = None
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # With --stride the skipped frames reuse the previous measurement, so
        # the output video stays smooth instead of flickering an annotation on
        # and off. The reported numbers are still only from measured frames.
        if frame_idx % args.stride == 0:
            location, _ = _measure(detector, estimator, frame)
            last = location
            if location is not None:
                measured.append((frame_idx, location))
        writer.write(draw(frame, last, query=args.query))
        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == total:
            pct = f"{100 * frame_idx / total:.0f}%" if total else f"{frame_idx}"
            print(f"\r  processing frame {frame_idx}/{total or '?'} ({pct})", end="")
    print()

    cap.release()
    writer.release()
    elapsed = time.time() - start
    print(
        f"Processed {frame_idx} frame(s) in {elapsed:.1f}s "
        f"({frame_idx / elapsed:.1f} fps). Output: {output_path}"
    )

    if not measured:
        print(f'not found (or not measurable): "{args.query}"')
        return 1

    first_idx, first_loc = measured[0]
    closest_idx, closest_loc = min(measured, key=lambda m: m[1].distance_m)
    distances = [loc.distance_m for _, loc in measured]
    print(
        f'Measured "{args.query}" in {len(measured)} frame(s); '
        f"first at frame {first_idx} (t={first_idx / fps:.1f}s) — {first_loc.summary()}"
    )
    print(
        f"  closest: {closest_loc.summary()} at frame {closest_idx} "
        f"(t={closest_idx / fps:.1f}s)"
    )
    print(
        f"  range over measured frames: {min(distances):.2f} m .. "
        f"{max(distances):.2f} m"
    )

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                [
                    {"frame": idx, "time_s": round(idx / fps, 3), **loc.as_dict()}
                    for idx, loc in measured
                ],
                indent=2,
            )
        )
        print(f"Measurements (JSON) written to: {args.json}")
    return 0


def run_webcam(
    detector: ObjectDetector, estimator: DepthEstimator, args: argparse.Namespace
) -> int:
    source: int | str = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(
            "error: could not open webcam. Note: direct webcam access only works\n"
            "on a LINUX host with --device passthrough in devcontainer.json.\n"
            "On Windows/macOS run the browser app instead:\n"
            "    python -m src.webcam_server",
            file=sys.stderr,
        )
        return 1

    output_path = Path(args.output) if args.output else None
    writer = None
    hits = 0
    print(f'Locating "{args.query}". Press Ctrl+C to stop.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            location, candidates = _measure(detector, estimator, frame)
            annotated = draw(frame, location, query=args.query)
            if output_path is not None:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0,
                        (w, h),
                    )
                writer.write(annotated)
            if location is not None:
                hits += 1
                status = location.summary()
            elif candidates:
                status = "found, but no valid depth"
            else:
                status = "not found"
            print(f"\r  {args.query}: {status:<45}", end="")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Recording written to: {output_path}")

    if not hits:
        print(f'not found (or not measurable): "{args.query}"')
        return 1
    print(f'Measured "{args.query}" in {hits} frame(s).')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distance and bearing to one object described in plain English"
    )
    parser.add_argument(
        "mode", choices=["image", "video", "webcam"], help="input source type"
    )
    parser.add_argument(
        "query",
        help='what to look for, in words, e.g. "blue cup" or "red screwdriver"',
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="path to image/video file, or webcam index (default 0 for webcam)",
    )
    parser.add_argument("--output", "-o", help="output file path")
    parser.add_argument("--json", help="also write the measurement as JSON")
    parser.add_argument(
        "--camera",
        default=None,
        help="camera intrinsics JSON (default: camera.json, or $CAMERA_CONFIG). "
        "Optional, but a fully calibrated file (fx, fy, cx, cy) makes the "
        "bearing exact instead of assuming the lens is centred.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="video mode: measure every Nth frame (default 1). Both models run "
        "per measured frame, so raising this is the quickest way to speed a "
        "long clip up.",
    )
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=DEFAULT_RESOLUTION_LEVEL,
        choices=range(10),
        metavar="0-9",
        help=f"depth model working resolution (default {DEFAULT_RESOLUTION_LEVEL}); "
        "lower is faster and coarser",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO-World weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--depth-model",
        default=DEPTH_MODEL,
        help=f"MoGe-2 weights (default: {DEPTH_MODEL})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"detection confidence threshold (default {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--device", default=None, help="cpu, 0 (cuda:0), ... (default: auto)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed the random tie-break between equally confident candidates",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seed is not None:
        random.seed(args.seed)
    query = args.query.strip()
    if not query:
        print("error: query must not be empty", file=sys.stderr)
        return 1
    args.query = query
    if args.stride < 1:
        print("error: --stride must be at least 1", file=sys.stderr)
        return 1

    try:
        camera = load_camera(args.camera)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    fov = camera.horizontal_fov_deg
    if fov is None:
        print(
            "No camera intrinsics found — the depth model will estimate the field\n"
            "of view, and the bearing will assume the lens is centred. Fill in\n"
            "camera.json for exact distances and angles."
        )
    else:
        print(f"Using camera '{camera.name}': horizontal FOV {fov:.1f} deg")

    detector = ObjectDetector(
        model_path=args.model, conf=args.conf, device=args.device, classes=[query]
    )
    try:
        estimator = DepthEstimator(
            model_path=args.depth_model,
            device=args.device,
            camera=camera,
            resolution_level=args.resolution_level,
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Depth model '{args.depth_model}' loaded on {estimator.device}.")

    if args.mode == "image":
        return run_image(detector, estimator, args)
    if args.mode == "video":
        return run_video(detector, estimator, args)
    return run_webcam(detector, estimator, args)


if __name__ == "__main__":
    raise SystemExit(main())
