from .fast_path import FastPathCVDetector
from .onnx_fallback import YOLOv8ONNXFallback
from .detector import BeaconDetector

__all__ = [
    "FastPathCVDetector",
    "YOLOv8ONNXFallback",
    "BeaconDetector",
]