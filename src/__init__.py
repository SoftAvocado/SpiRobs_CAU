"""Object detection and depth infrastructure for the SpiRobs project."""

from .camera import CameraIntrinsics, load_camera
from .depth_estimator import DepthEstimator, DepthMap
from .detector import Detection, ObjectDetector
from .locator import ObjectLocation, locate

__all__ = [
    "CameraIntrinsics",
    "Detection",
    "DepthEstimator",
    "DepthMap",
    "ObjectDetector",
    "ObjectLocation",
    "load_camera",
    "locate",
]
