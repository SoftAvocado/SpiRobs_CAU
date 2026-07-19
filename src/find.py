"""Find ONE specific object described in plain English.

Where :mod:`src.detect` labels every object it knows about (the fixed
vocabulary in ``src/classes.py``), this command looks for a single thing you
describe in words — ``"blue cup"``, ``"red screwdriver"``, ``"laptop"`` — and
reports whether it is there.

It works because the underlying model (YOLO-World) is *open-vocabulary*: it is
prompted with text, so the description is handed to it directly as its entire
vocabulary instead of ``DETECTION_CLASSES``. Nothing is trained, nothing in
``classes.py`` needs editing, and the description does not have to be a known
class.

Usage (inside the dev container):

    # Image -> annotated image, or "not found" in the console
    python -m src.find image "blue cup" path/to/photo.jpg

    # Video -> annotated video + which frames contained the object
    python -m src.find video "blue cup" path/to/clip.mp4 --output out.mp4

    # Webcam (LINUX host with --device=/dev/video0 passthrough only).
    # The trailing argument is the camera index and defaults to 0.
    python -m src.find webcam "blue cup" 0

Exit code is 0 when the object was found and 1 when it was not (or on error),
so the command can be used in shell scripts.

Options: --best (keep only the single strongest match), --conf, --model,
--device, --output, --json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .detect import _print_detections
from .detector import DEFAULT_MODEL, Detection, ObjectDetector

#: Descriptions are usually harder to match than a plain class name, so the
#: default threshold is lower than the "detect everything" default (0.25).
DEFAULT_CONF = 0.10


def _default_output(input_path: Path, suffix: str | None = None) -> Path:
    """``photo.jpg`` -> ``photo_found.jpg`` (mirrors detect.py's ``_detected``)."""
    suffix = suffix or input_path.suffix
    return input_path.with_name(f"{input_path.stem}_found{suffix}")


def _keep(detections: list[Detection], best_only: bool) -> list[Detection]:
    """Sort matches strongest-first, optionally keeping only the best one."""
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    return ordered[:1] if best_only and ordered else ordered


def _not_found(query: str) -> None:
    print(f'not found: "{query}"')


def run_image(detector: ObjectDetector, args: argparse.Namespace) -> int:
    input_path = Path(args.source)
    if not input_path.exists():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        return 1

    image = cv2.imread(str(input_path))
    if image is None:
        print(f"error: could not read image: {input_path}", file=sys.stderr)
        return 1

    matches = _keep(detector.detect(image), args.best)
    if not matches:
        _not_found(args.query)
        return 1

    print(f'Found {len(matches)} match(es) for "{args.query}" in {input_path.name}:')
    _print_detections(matches)

    output_path = Path(args.output) if args.output else _default_output(input_path)
    cv2.imwrite(str(output_path), detector.draw(image, matches))
    print(f"Annotated image written to: {output_path}")

    if args.json:
        Path(args.json).write_text(json.dumps([d.as_dict() for d in matches], indent=2))
        print(f"Matches (JSON) written to: {args.json}")
    return 0


def run_video(detector: ObjectDetector, args: argparse.Namespace) -> int:
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
    hit_frames: list[int] = []
    best: tuple[float, int] | None = None  # (confidence, frame index)
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        matches = _keep(detector.detect(frame), args.best)
        writer.write(detector.draw(frame, matches))
        if matches:
            hit_frames.append(frame_idx)
            if best is None or matches[0].confidence > best[0]:
                best = (matches[0].confidence, frame_idx)
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

    if not hit_frames:
        _not_found(args.query)
        return 1

    first = hit_frames[0]
    print(
        f'Found "{args.query}" in {len(hit_frames)} of {frame_idx} frame(s); '
        f"first at frame {first} (t={first / fps:.1f}s)"
    )
    if best is not None:
        print(f"  strongest match: conf={best[0]:.2f} at frame {best[1]}")
    return 0


def run_webcam(detector: ObjectDetector, args: argparse.Namespace) -> int:
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
    print(f'Looking for "{args.query}". Press Ctrl+C to stop.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            matches = _keep(detector.detect(frame), args.best)
            annotated = detector.draw(frame, matches)
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
            if matches:
                hits += 1
                status = f"FOUND ({matches[0].confidence:.2f})"
            else:
                status = "not found"
            print(f"\r  {args.query}: {status:<30}", end="")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Recording written to: {output_path}")

    if not hits:
        _not_found(args.query)
        return 1
    print(f'Found "{args.query}" in {hits} frame(s).')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find one specific object described in plain English"
    )
    parser.add_argument(
        "mode", choices=["image", "video", "webcam"], help="detection source type"
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
    parser.add_argument("--best", action="store_true", help="keep only the top match")
    parser.add_argument("--output", "-o", help="output file path")
    parser.add_argument("--json", help="also write matches as JSON to this path")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"YOLO-World weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"confidence threshold (default {DEFAULT_CONF}); raise it if you get "
        "false matches, lower it if a present object is missed",
    )
    parser.add_argument(
        "--device", default=None, help="cpu, 0 (cuda:0), ... (default: auto)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    query = args.query.strip()
    if not query:
        print("error: query must not be empty", file=sys.stderr)
        return 1
    args.query = query

    # The description IS the vocabulary: the model is prompted with this one
    # phrase, so every box it returns is a candidate match for it.
    detector = ObjectDetector(
        model_path=args.model,
        conf=args.conf,
        device=args.device,
        classes=[query],
    )
    if args.mode == "image":
        return run_image(detector, args)
    if args.mode == "video":
        return run_video(detector, args)
    return run_webcam(detector, args)


if __name__ == "__main__":
    raise SystemExit(main())
