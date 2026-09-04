"""
Unit tests for Module 3 Perception & Detection Engine.
Validates Fast Path CV sub-pixel accuracy, Salt & Pepper noise rejection,
background illumination/fog gradient suppression, dark frame safety,
and protocol type conformance.
"""

import math
import numpy as np
import cv2
import pytest

from src.benchmark.pipeline_adapter import BeaconDetectorProtocol
from src.perception import BeaconDetector, FastPathCVDetector, YOLOv8ONNXFallback


def test_protocol_conformance():
    detector = BeaconDetector()
    assert isinstance(detector, BeaconDetectorProtocol)


def test_fast_path_clean_synthetic_spot():
    detector = FastPathCVDetector()

    # Create dark background frame (640x480) with clean spot at (320, 240)
    frame = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(frame, (320, 240), radius=5, color=255, thickness=-1)

    result = detector.process(frame)
    assert result.detected is True
    assert result.method_used == "FAST_PATH"
    assert result.centroid_x is not None
    assert result.centroid_y is not None
    # Sub-pixel accuracy against known circle center (320.0, 240.0)
    assert math.isclose(result.centroid_x, 320.0, abs_tol=0.5)
    assert math.isclose(result.centroid_y, 240.0, abs_tol=0.5)
    assert result.confidence > 0.5


def test_fast_path_salt_and_pepper_noise_rejection():
    detector = FastPathCVDetector(median_kernel_size=3)

    frame = np.zeros((480, 640), dtype=np.uint8)

    # Inject 5% random Salt & Pepper noise pixels
    rng = np.random.default_rng(42)
    noise_mask = rng.random((480, 640)) < 0.05
    frame[noise_mask] = 255

    # Draw real beacon spot at (150, 100)
    cv2.circle(frame, (150, 100), radius=6, color=255, thickness=-1)

    result = detector.process(frame)
    assert result.detected is True
    assert math.isclose(result.centroid_x, 150.0, abs_tol=0.8)
    assert math.isclose(result.centroid_y, 100.0, abs_tol=0.8)


def test_fast_path_background_fog_gradient_suppression():
    detector = FastPathCVDetector(tophat_kernel_size=15)

    # Create frame with wide background fog brightness gradient (increasing left to right)
    y_indices, x_indices = np.indices((480, 640))
    fog_gradient = (x_indices / 640.0 * 120.0).astype(np.uint8)

    # Add beacon spot at (400, 300)
    frame = fog_gradient.copy()
    cv2.circle(frame, (400, 300), radius=5, color=255, thickness=-1)

    result = detector.process(frame)
    assert result.detected is True
    assert math.isclose(result.centroid_x, 400.0, abs_tol=0.8)
    assert math.isclose(result.centroid_y, 300.0, abs_tol=0.8)


def test_fast_path_dark_frame_no_detection():
    detector = FastPathCVDetector()
    frame = np.zeros((480, 640), dtype=np.uint8)

    result = detector.process(frame)
    assert result.detected is False
    assert result.centroid_x is None
    assert result.centroid_y is None
    assert result.confidence == 0.0


def test_onnx_fallback_missing_model_handling():
    fallback = YOLOv8ONNXFallback(model_path="non_existent_model.onnx")
    assert fallback.is_available is False

    frame = np.zeros((480, 640), dtype=np.uint8)
    result = fallback.detect(frame)
    assert result.detected is False
    assert result.method_used == "AI_YOLO_UNAVAILABLE"


def test_beacon_detector_manager_end_to_end():
    detector = BeaconDetector()

    # Frame 1: Bright spot at (200, 150) -> Fast Path detects cleanly
    frame1 = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(frame1, (200, 150), radius=5, color=255, thickness=-1)
    res1 = detector.detect(frame1)

    assert res1.detected is True
    assert res1.method_used == "FAST_PATH"
    assert math.isclose(res1.centroid_x, 200.0, abs_tol=0.5)
    assert math.isclose(res1.centroid_y, 150.0, abs_tol=0.5)

    # Frame 2: Dark frame -> No detection
    frame2 = np.zeros((480, 640), dtype=np.uint8)
    res2 = detector.detect(frame2)
    assert res2.detected is False
