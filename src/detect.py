"""Command-line entry point for object detection.

Usage (inside the dev container):

    # Single image -> writes an annotated image next to it (or --output PATH)
    python -m src.detect image path/to/photo.jpg

    # Video file -> writes an annotated video
    python -m src.detect video path/to/clip.mp4 --output out.mp4

    # Webcam (LINUX host with --device=/dev/video0 passthrough only).
    # On Windows/macOS use the web app instead: python -m src.webcam_server
    python -m src.detect webcam --source 0

Common options: --model, --conf, --device.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from .detector import Detection, ObjectDetector


def _print_detections(detections: list[Detection]) -> None:
    for d in detections:
        print(
            f"  {d.label:<15} conf={d.confidence:.2f}  "
            f"box=({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f})"
        )


def _default_output(input_path: Path, suffix: str | None = None) -> Path:
    suffix = suffix or input_path.suffix
    return input_path.with_name(f"{input_path.stem}_detected{suffix}")


def run_image(detector: ObjectDetector, args: argparse.Namespace) -> int:
    input_path = Path(args.source)
    if not input_path.exists():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        return 1

    image = cv2.imread(str(input_path))
    if image is None:
        print(f"error: could not read image: {input_path}", file=sys.stderr)
        return 1

    detections = detector.detect(image)
    print(f"Detected {len(detections)} object(s) in {input_path.name}:")
    _print_detections(detections)

    output_path = Path(args.output) if args.output else _default_output(input_path)
    annotated = detector.draw(image, detections)
    cv2.imwrite(str(output_path), annotated)
    print(f"Annotated image written to: {output_path}")

    if args.json:
        Path(args.json).write_text(
            json.dumps([d.as_dict() for d in detections], indent=2)
        )
        print(f"Detections (JSON) written to: {args.json}")
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector.detect(frame)
        writer.write(detector.draw(frame, detections))
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
    print("Reading from webcam. Press Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector.detect(frame)
            annotated = detector.draw(frame, detections)
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
            labels = ", ".join(sorted({d.label for d in detections})) or "-"
            print(f"\r  {len(detections)} object(s): {labels:<50}", end="")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Recording written to: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO object detection CLI")
    parser.add_argument(
        "mode", choices=["image", "video", "webcam"], help="detection source type"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="path to image/video file, or webcam index (default 0 for webcam)",
    )
    parser.add_argument("--output", "-o", help="output file path")
    parser.add_argument("--json", help="also write detections as JSON to this path")
    parser.add_argument(
        "--model", default="yolo11n.pt", help="YOLO weights (default: yolo11n.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="confidence threshold (default 0.25)"
    )
    parser.add_argument(
        "--device", default=None, help="cpu, 0 (cuda:0), ... (default: auto)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    detector = ObjectDetector(
        model_path=args.model, conf=args.conf, device=args.device
    )
    if args.mode == "image":
        return run_image(detector, args)
    if args.mode == "video":
        return run_video(detector, args)
    return run_webcam(detector, args)


if __name__ == "__main__":
    raise SystemExit(main())
