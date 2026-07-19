"""Command-line entry point for monocular metric depth estimation.

Independent of object detection: this produces a depth map (metres per pixel)
for a frame, nothing else. Combining it with detection to report the distance
and angle to a specific object is a separate, later step.

Usage (inside the dev container):

    # Single image -> writes a colourised depth map next to it (or --output PATH)
    python -m src.depth image path/to/photo.jpg

    # Video file -> writes a depth video
    python -m src.depth video path/to/clip.mp4 --output out.mp4

    # Webcam (LINUX host with --device=/dev/video0 passthrough only).
    # On Windows/macOS use the web app instead: python -m src.webcam_server
    python -m src.depth webcam --source 0

Common options: --camera, --near/--far, --side-by-side, --resolution-level,
--npy, --model, --device. Camera intrinsics are optional — see camera.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import load_camera
from .depth_estimator import (
    DEFAULT_MODEL,
    DEFAULT_RESOLUTION_LEVEL,
    DepthEstimator,
    DepthMap,
)


def _default_output(input_path: Path, suffix: str | None = None) -> Path:
    """``photo.jpg`` -> ``photo_depth.jpg`` (mirrors detect.py's ``_detected``)."""
    suffix = suffix or input_path.suffix
    return input_path.with_name(f"{input_path.stem}_depth{suffix}")


def _print_stats(depth_map: DepthMap) -> None:
    s = depth_map.stats()
    if s["min_m"] is None:
        print("  no valid depth in this frame")
        return
    print(
        f"  range   {s['min_m']:.2f} m .. {s['max_m']:.2f} m   "
        f"median {s['median_m']:.2f} m"
    )
    centre = "n/a" if s["centre_m"] is None else f"{s['centre_m']:.2f} m"
    print(f"  centre  {centre}   valid {100 * s['valid_fraction']:.0f}% of pixels")


def _render(
    estimator: DepthEstimator,
    frame: np.ndarray,
    depth_map: DepthMap,
    args: argparse.Namespace,
) -> np.ndarray:
    """Colourised depth, optionally with the source frame beside it."""
    colored = estimator.colorize(depth_map, near=args.near, far=args.far)
    if not args.side_by_side:
        return colored
    return np.hstack([frame, colored])


def run_image(estimator: DepthEstimator, args: argparse.Namespace) -> int:
    input_path = Path(args.source)
    if not input_path.exists():
        print(f"error: file not found: {input_path}", file=sys.stderr)
        return 1

    image = cv2.imread(str(input_path))
    if image is None:
        print(f"error: could not read image: {input_path}", file=sys.stderr)
        return 1

    depth_map = estimator.estimate(image)
    print(f"Depth for {input_path.name}:")
    _print_stats(depth_map)

    output_path = Path(args.output) if args.output else _default_output(input_path)
    cv2.imwrite(str(output_path), _render(estimator, image, depth_map, args))
    print(f"Depth map written to: {output_path}")

    if args.npy:
        # The colourised PNG is for humans; this is the actual measurement —
        # float32 metres, NaN where the model found no valid geometry.
        np.save(args.npy, depth_map.depth)
        print(f"Raw depth (float32 metres, NaN = invalid) written to: {args.npy}")

    if args.json:
        payload = depth_map.stats()
        payload["intrinsics_px"] = depth_map.pixel_intrinsics().tolist()
        payload["camera"] = estimator.camera.as_dict()
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"Depth summary (JSON) written to: {args.json}")
    return 0


def run_video(estimator: DepthEstimator, args: argparse.Namespace) -> int:
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
    out_width = width * 2 if args.side_by_side else width
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_width, height))

    frame_idx = 0
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        depth_map = estimator.estimate(frame)
        # Lock the colour range to the first frame. Auto-scaling every frame
        # independently would make the whole video pulse as the scene's nearest
        # and farthest points shift, and would make colours incomparable
        # between frames.
        if frame_idx == 0:
            _lock_range(args, depth_map)
        writer.write(_render(estimator, frame, depth_map, args))
        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == total:
            pct = f"{100 * frame_idx / total:.0f}%" if total else f"{frame_idx}"
            print(f"\r  processing frame {frame_idx}/{total or '?'} ({pct})", end="")
    print()

    cap.release()
    writer.release()
    if frame_idx == 0:
        print(f"error: no frames could be read from {input_path}", file=sys.stderr)
        return 1

    elapsed = time.time() - start
    print(
        f"Processed {frame_idx} frame(s) in {elapsed:.1f}s "
        f"({frame_idx / elapsed:.1f} fps). Output: {output_path}"
    )
    print(f"Colour range fixed at {args.near:.2f} m .. {args.far:.2f} m")
    return 0


def run_webcam(estimator: DepthEstimator, args: argparse.Namespace) -> int:
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
    frame_idx = 0
    print("Reading from webcam. Press Ctrl+C to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            depth_map = estimator.estimate(frame)
            if frame_idx == 0:
                _lock_range(args, depth_map)
            rendered = _render(estimator, frame, depth_map, args)
            if output_path is not None:
                if writer is None:
                    h, w = rendered.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0,
                        (w, h),
                    )
                writer.write(rendered)
            frame_idx += 1

            h, w = depth_map.depth.shape[:2]
            centre = depth_map.depth_at(w // 2, h // 2)
            centre_txt = "n/a" if centre is None else f"{centre:.2f} m"
            near, far = depth_map.range_metres()
            print(
                f"\r  centre {centre_txt:<10} scene {near:.2f}-{far:.2f} m"
                f"{' ' * 12}",
                end="",
            )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Recording written to: {output_path}")
    return 0


def _lock_range(args: argparse.Namespace, depth_map: DepthMap) -> None:
    """Fill in whichever of ``--near``/``--far`` the user left unset."""
    near, far = depth_map.range_metres()
    if args.near is None:
        args.near = near
    if args.far is None:
        args.far = far


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MoGe-2 monocular metric depth CLI"
    )
    parser.add_argument(
        "mode", choices=["image", "video", "webcam"], help="input source type"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="path to image/video file, or webcam index (default 0 for webcam)",
    )
    parser.add_argument("--output", "-o", help="output file path")
    parser.add_argument("--npy", help="also write the raw depth array (image mode)")
    parser.add_argument("--json", help="also write a depth summary as JSON")
    parser.add_argument(
        "--camera",
        default=None,
        help="camera intrinsics JSON (default: camera.json, or $CAMERA_CONFIG). "
        "Optional — the model estimates the field of view when absent.",
    )
    parser.add_argument(
        "--near",
        type=float,
        default=None,
        help="metres mapped to the near (red) end of the colour ramp "
        "(default: auto from the frame)",
    )
    parser.add_argument(
        "--far",
        type=float,
        default=None,
        help="metres mapped to the far (blue) end of the colour ramp",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="write the source frame next to the depth map",
    )
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=DEFAULT_RESOLUTION_LEVEL,
        choices=range(10),
        metavar="0-9",
        help=f"model working resolution (default {DEFAULT_RESOLUTION_LEVEL}); "
        "lower is faster and coarser",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"MoGe-2 weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device", default=None, help="cpu, cuda, ... (default: auto)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        camera = load_camera(args.camera)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    fov = camera.horizontal_fov_deg
    if fov is None:
        print(
            "No camera intrinsics found — the model will estimate the field of "
            "view.\nFill in camera.json for more accurate metric distances."
        )
    else:
        print(f"Using camera '{camera.name}': horizontal FOV {fov:.1f} deg")

    try:
        estimator = DepthEstimator(
            model_path=args.model,
            device=args.device,
            camera=camera,
            resolution_level=args.resolution_level,
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Depth model '{args.model}' loaded on {estimator.device}.")

    if args.mode == "image":
        return run_image(estimator, args)
    if args.mode == "video":
        return run_video(estimator, args)
    return run_webcam(estimator, args)


if __name__ == "__main__":
    raise SystemExit(main())
