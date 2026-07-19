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

# Pre-download the default YOLO weights so first run is offline/fast.
# Stored under /home/vscode so the non-root user owns it.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')" \
    && mkdir -p /home/vscode/.cache \
    && (cp yolo11n.pt /home/vscode/yolo11n.pt 2>/dev/null || true) \
    && chown -R vscode:vscode /home/vscode

ENV YOLO_CONFIG_DIR=/home/vscode/.config/Ultralytics
