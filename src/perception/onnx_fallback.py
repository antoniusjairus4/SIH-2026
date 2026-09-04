"""
AI Fallback Engine using ONNX Runtime for YOLOv8n inference on CPU.
Provides deep-learning detection when Fast Path Classical CV confidence is low (heavy fog/rain/occlusion).
Includes safe fallback handling if ONNX Runtime or model files are unavailable.
"""

import os
import logging
from typing import Optional, Tuple
import cv2
import numpy as np

from src.metrics.schemas import DetectionResult

logger = logging.getLogger("YOLOv8ONNXFallback")

try:
    # pyrefly: ignore [missing-import]
    import onnxruntime as ort
    HAS_ONNX_RUNTIME = True
except ImportError:
    HAS_ONNX_RUNTIME = False


class YOLOv8ONNXFallback:
    """
    CPU-bound ONNX inference engine for YOLOv8n beacon detector.
    """

    def __init__(
        self,
        model_path: str = "models/beacon_yolo.onnx",
        conf_threshold: float = 0.25,
        img_size: Tuple[int, int] = (640, 640),
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.session = None

        self._initialize_session()

    def _initialize_session(self) -> None:
        if not HAS_ONNX_RUNTIME:
            logger.info("onnxruntime package not installed; AI ONNX fallback disabled.")
            return

        if not os.path.exists(self.model_path):
            logger.info(f"YOLO ONNX model file not found at '{self.model_path}'; AI ONNX fallback disabled.")
            return

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.info(f"Loaded YOLO ONNX fallback model from '{self.model_path}'.")
        except Exception as e:
            logger.warning(f"Failed to initialize ONNX session: {e}")
            self.session = None

    @property
    def is_available(self) -> bool:
        return self.session is not None

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Executes YOLOv8n ONNX inference on CPU and extracts bounding box center.
        """
        if not self.is_available:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                method_used="AI_YOLO_UNAVAILABLE",
                metadata={"reason": "MODEL_OR_RUNTIME_MISSING"},
            )

        if frame is None or frame.size == 0:
            return DetectionResult(
                detected=False,
                confidence=0.0,
                method_used="AI_YOLO",
                metadata={"reason": "EMPTY_FRAME"},
            )

        try:
            orig_h, orig_w = frame.shape[:2]

            # Preprocess: RGB conversion, resize to 640x640, CHW transpose, normalization [0, 1]
            if len(frame.shape) == 2:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            resized = cv2.resize(rgb, self.img_size)
            chw = resized.transpose(2, 0, 1)
            blob = np.expand_dims(chw, axis=0).astype(np.float32) / 255.0

            # Execute ONNX session
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: blob})

            # Postprocess predictions matrix: shape [1, 5, 8400] -> [5, 8400]
            preds = np.squeeze(outputs[0])
            if len(preds.shape) != 2 or preds.shape[0] < 5:
                return DetectionResult(
                    detected=False,
                    confidence=0.0,
                    method_used="AI_YOLO",
                )

            scores = preds[4, :]
            max_idx = int(np.argmax(scores))
            max_score = float(scores[max_idx])

            if max_score < self.conf_threshold:
                return DetectionResult(
                    detected=False,
                    confidence=max_score,
                    method_used="AI_YOLO",
                )

            # Extract normalized coordinates and map back to original frame dimensions
            raw_cx = float(preds[0, max_idx])
            raw_cy = float(preds[1, max_idx])

            scale_x = orig_w / float(self.img_size[0])
            scale_y = orig_h / float(self.img_size[1])

            cx = raw_cx * scale_x
            cy = raw_cy * scale_y

            return DetectionResult(
                detected=True,
                centroid_x=cx,
                centroid_y=cy,
                confidence=max_score,
                method_used="AI_YOLO",
                metadata={"scaled_from": self.img_size},
            )

        except Exception as e:
            logger.warning(f"Error during YOLO ONNX inference: {e}")
            return DetectionResult(
                detected=False,
                confidence=0.0,
                method_used="AI_YOLO",
                metadata={"error": str(e)},
            )
