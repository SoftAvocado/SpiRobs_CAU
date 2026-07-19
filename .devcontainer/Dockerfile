# Dev container image for the object-detection infrastructure.
# Based on the official VS Code Python dev container (non-root "vscode" user).
FROM mcr.microsoft.com/devcontainers/python:3.11

# System libraries required by OpenCV (even the headless build needs libGL/glib
# at import time) and by video decoding (ffmpeg).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so the layer is cached across code changes.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Give Ultralytics a config/cache directory that the non-root "vscode" user can
# write to. Without this it falls back to /tmp/Ultralytics (created by root at
# build time) and logs "Permission denied" when saving its settings cache.
ENV YOLO_CONFIG_DIR=/home/vscode/.config/Ultralytics
# Single cache location for downloaded weights (read by src/detector.py). The
# prewarm below downloads into it so the model is never re-downloaded at runtime.
ENV YOLO_WEIGHTS_DIR=/home/vscode/.config/Ultralytics/weights

# Pre-download the default YOLO weights into the cache dir, then hand the whole
# config dir to the vscode user (the prewarm runs as root and would otherwise
# leave root-owned files there).
RUN mkdir -p "$YOLO_WEIGHTS_DIR" \
    && cd "$YOLO_WEIGHTS_DIR" \
    && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" \
    && chown -R vscode:vscode /home/vscode/.config
