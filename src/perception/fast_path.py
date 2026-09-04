"""
Fast Path Classical Computer Vision Detector for SIH 2026 Virtual Camera Tracking System.
Implements high-speed sub-pixel beacon centroiding using Median Noise Filtering,
Morphological White Top-Hat background suppression, Dynamic Adaptive Thresholding,
and Spatial Intensity Moments.
"""

import math
from typing import Optional, Tuple
import cv2
import numpy as np

from src.metrics.schemas import DetectionResult


class FastPathCVDetector:
    """
    Ultra-fast classical CV detector targeting 1 to 3 ms execution time.
    """

    def __init__(
        self,
        median_kernel_size: int = 3,
        tophat_kernel_size: int = 15,
        threshold_sigma: float = 3.0,
        min_spot_area: float = 4.0,
        max_spot_area: float = 900.0,
        min_circularity: float = 0.25,
    ) -> None:
        self.median_kernel_size = median_kernel_size
        self.tophat_kernel_size = tophat_kernel_size
        self.threshold_sigma = threshold_sigma
        self.min_spot_area = min_spot_area
        self.max_spot_area = max_spot_area
        self.min_circularity = min_circularity

        # Pre-create morphological structuring element (ellipse/circular kernel)
        self._tophat_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.tophat_kernel_size, self.tophat_kernel_size),
        )

    def process(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> DetectionResult:
        """
        Processes a raw frame (BGR or Grayscale) and returns candidate centroid and confidence.

        Args:
            frame: 2D or 3D NumPy array representing camera image (e.g. 640x480).
            frame_id: Frame index.
            timestamp: Video timestamp in seconds.

        Returns:
            DetectionResult with detected=True/False, sub-pixel centroid, and confidence score.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                method_used="FAST_PATH",
                metadata={"reason": "EMPTY_FRAME"},
            )

        # 1. Grayscale Conversion
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 2. Median Filtering (Rejects Salt & Pepper noise up to 10%)
        if self.median_kernel_size > 1:
            denoised = cv2.medianBlur(gray, self.median_kernel_size)
        else:
            denoised = gray

        # 3. Morphological White Top-Hat Filter (Suppresses background illumination & fog gradients)
        tophat = cv2.morphologyEx(
            denoised,
            cv2.MORPH_TOPHAT,
            self._tophat_kernel,
        )

        # 4. Dynamic Adaptive Thresholding (T = mean + k * std)
        mean_val = float(np.mean(tophat))
        std_val = float(np.std(tophat))

        if std_val < 1e-6:
            # Uniform dark background with no distinct spots
            return DetectionResult(
                detected=False,
                confidence=0.0,
                method_used="FAST_PATH",
                metadata={"reason": "UNIFORM_BACKGROUND"},
            )

        threshold_val = mean_val + self.threshold_sigma * std_val
        threshold_val = float(np.clip(threshold_val, 15.0, 250.0))

        _, binary = cv2.threshold(
            tophat,
            threshold_val,
            255,
            cv2.THRESH_BINARY,
        )

        # 5. Contour Extraction & Spot Candidate Selection
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                method_used="FAST_PATH",
                metadata={"threshold_used": threshold_val},
            )

        best_centroid: Optional[Tuple[float, float]] = None
        best_confidence: float = 0.0
        best_area: float = 0.0

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_spot_area or area > self.max_spot_area:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue

            # Circularity metric: 4 * pi * Area / Perimeter^2
            circularity = (4.0 * math.pi * area) / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            # Calculate Sub-Pixel Center of Gravity using Spatial Intensity Moments
            mask = np.zeros_like(tophat)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

            moments_masked = cv2.moments(tophat * (mask > 0))
            m00 = moments_masked["m00"]

            if m00 <= 1e-6:
                continue

            cx = float(moments_masked["m10"] / m00)
            cy = float(moments_masked["m01"] / m00)

            # Confidence calculation based on circularity, intensity peak, and size
            max_intensity = float(np.max(tophat * (mask > 0)))
            intensity_factor = min(max_intensity / 255.0, 1.0)
            circularity_factor = min(circularity, 1.0)

            confidence = float(0.5 * intensity_factor + 0.5 * circularity_factor)
            confidence = float(np.clip(confidence, 0.0, 1.0))

            if confidence > best_confidence:
                best_confidence = confidence
                best_centroid = (cx, cy)
                best_area = area

        if best_centroid is None:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                method_used="FAST_PATH",
                metadata={"threshold_used": threshold_val},
            )

        return DetectionResult(
            detected=True,
            centroid_x=best_centroid[0],
            centroid_y=best_centroid[1],
            confidence=best_confidence,
            method_used="FAST_PATH",
            metadata={
                "area_px": best_area,
                "threshold_used": threshold_val,
                "frame_id": frame_id,
            },
        )
