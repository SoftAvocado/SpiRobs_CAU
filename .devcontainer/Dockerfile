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

# Pre-download the open-vocabulary model AND its CLIP text encoder into the
# vscode user's home, so the first real run needs no network. Runs as root (so
# Ultralytics can pip-install the CLIP dependency system-wide) but with
# HOME=/home/vscode so cached files land in the user's home; then chown them.
# set_classes(...) is what triggers the CLIP download, so we call it here.
RUN mkdir -p "$YOLO_WEIGHTS_DIR" \
    && cd "$YOLO_WEIGHTS_DIR" \
    && HOME=/home/vscode python -c "from ultralytics import YOLOWorld; m = YOLOWorld('yolov8s-worldv2.pt'); m.set_classes(['pen', 'cup'])" \
    && chown -R vscode:vscode /home/vscode

# Same idea for the depth model (src/depth_estimator.py): MoGe-2 ViT-L is a
# ~1.3 GB HuggingFace download, so pull it at build time instead of making the
# first `python -m src.depth` wait for it. HF_HOME points at the vscode user's
# cache so the weights are found again at runtime.
ENV HF_HOME=/home/vscode/.cache/huggingface
RUN HOME=/home/vscode python -c "from moge.model.v2 import MoGeModel; MoGeModel.from_pretrained('Ruicheng/moge-2-vitl')" \
    && chown -R vscode:vscode /home/vscode/.cache
