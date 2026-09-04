"""
Main Perception Manager (BeaconDetector) for SIH 2026 Virtual Camera Tracking System.
Implements a Hybrid Dual-Path Perception Engine:
  1. Fast Path Classical CV (Median + Top-Hat + Sub-pixel moments)
  2. AI Fallback (YOLOv8n ONNX Engine) when Fast Path confidence < threshold.
Conforms to BeaconDetectorProtocol for seamless pipeline integration.
"""

from typing import Optional
import numpy as np

from src.metrics.schemas import DetectionResult
from src.benchmark.pipeline_adapter import BeaconDetectorProtocol
from src.perception.fast_path import FastPathCVDetector
from src.perception.onnx_fallback import YOLOv8ONNXFallback


class BeaconDetector(BeaconDetectorProtocol):
    """
    Hybrid Beacon Detector combining fast classical CV with deep-learning AI fallback.
    """

    def __init__(
        self,
        fallback_threshold: float = 0.35,
        model_path: str = "models/beacon_yolo.onnx",
        median_kernel_size: int = 3,
        tophat_kernel_size: int = 15,
        threshold_sigma: float = 3.0,
    ) -> None:
        self.fallback_threshold = fallback_threshold
        self.fast_path = FastPathCVDetector(
            median_kernel_size=median_kernel_size,
            tophat_kernel_size=tophat_kernel_size,
            threshold_sigma=threshold_sigma,
        )
        self.onnx_fallback = YOLOv8ONNXFallback(model_path=model_path)

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> DetectionResult:
        """
        Executes Fast Path CV first. If confidence < fallback_threshold, delegates to AI Fallback.
        """
        # 1. Run Fast Path Classical CV
        fast_result = self.fast_path.process(frame, frame_id=frame_id, timestamp=timestamp)

        if fast_result.detected and fast_result.confidence >= self.fallback_threshold:
            return fast_result

        # 2. Delegate to AI ONNX Fallback if Fast Path confidence is low
        if self.onnx_fallback.is_available:
            ai_result = self.onnx_fallback.detect(frame)
            if ai_result.detected:
                return ai_result

        # 3. Return Fast Path result (or undetected) if AI fallback did not yield a detection
        return fast_result
