"""
Pipeline adapters and protocols for SIH 2026 Virtual Camera Tracking System.
Defines abstract protocols for Beacon Detectors and State Estimators to allow
seamless integration with future team perception and tracking modules without modifying runner code.
"""

from typing import Protocol, Optional, Dict, Any, runtime_checkable
import numpy as np

from src.metrics.schemas import DetectionResult, TrackingResult, LockState


@runtime_checkable
class BeaconDetectorProtocol(Protocol):
    """
    Protocol interface that any Beacon Detector module must satisfy.
    Compatible with Fast Path (Morphological/Top-Hat) and AI Fallback (YOLOv8n ONNX).
    """

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> DetectionResult:
        """
        Processes a single camera frame and returns candidate beacon centroid and confidence.

        Args:
            frame: 2D or 3D NumPy array representing the raw video frame (e.g. 640x480).
            frame_id: Sequential integer index of the frame.
            timestamp: Video timestamp in seconds.

        Returns:
            DetectionResult with detected coordinates and metadata.
        """
        ...


@runtime_checkable
class StateEstimatorProtocol(Protocol):
    """
    Protocol interface that any State Estimator / Kalman Filter module must satisfy.
    Compatible with Constant Acceleration EKF / Discrete Kalman filters.
    """

    def update(self, detection: DetectionResult, dt: float) -> TrackingResult:
        """
        Updates the state filter with the latest detection observation and elapsed time.

        Args:
            detection: Latest DetectionResult from the detector.
            dt: Time delta in seconds since the last update.

        Returns:
            TrackingResult containing filtered coordinates, velocity, and canonical LockState.
        """
        ...

    def reset(self) -> None:
        """
        Resets the internal filter state (e.g., when beginning a new video benchmark).
        """
        ...


class CombinedTrackingPipeline:
    """
    Adapter combining a BeaconDetectorProtocol and a StateEstimatorProtocol
    into a unified step function for the Benchmark-2 execution runner.
    """

    def __init__(
        self,
        detector: BeaconDetectorProtocol,
        state_estimator: StateEstimatorProtocol,
    ) -> None:
        self.detector = detector
        self.state_estimator = state_estimator
        self._last_timestamp: Optional[float] = None

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
    ) -> tuple[DetectionResult, TrackingResult]:
        """
        Executes detection and state estimation sequentially for a single frame.

        Args:
            frame: Raw image array.
            frame_id: Frame sequence number.
            timestamp: Current frame timestamp in seconds.

        Returns:
            Tuple of (DetectionResult, TrackingResult).
        """
        # Calculate time delta for state estimator
        if self._last_timestamp is None:
            dt = 1.0 / 30.0  # Default nominal 30 Hz delta on first frame
        else:
            dt = max(0.0, timestamp - self._last_timestamp)
        self._last_timestamp = timestamp

        # Step 1: Execute Detector
        detection = self.detector.detect(frame, frame_id, timestamp)

        # Step 2: Update State Estimator
        tracking = self.state_estimator.update(detection, dt)

        return detection, tracking

    def reset(self) -> None:
        """Resets the pipeline state."""
        self._last_timestamp = None
        self.state_estimator.reset()


# =====================================================================
# Minimal Null/Mock Adapters for testing and offline harness validation
# (Strictly for unit testing - not production perception/tracking)
# =====================================================================

class NullDetector(BeaconDetectorProtocol):
    """
    Minimal pass-through detector returning no detections.
    Used for baseline, unit testing, or zero-lock scenarios.
    """

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> DetectionResult:
        return DetectionResult(
            detected=False,
            centroid_x=None,
            centroid_y=None,
            confidence=0.0,
            method_used="NULL_DETECTOR",
        )


class NullStateEstimator(StateEstimatorProtocol):
    """
    Minimal pass-through state estimator remaining in SEARCH state.
    Used for baseline, unit testing, or zero-lock scenarios.
    """

    def update(self, detection: DetectionResult, dt: float) -> TrackingResult:
        return TrackingResult(
            lock_state=LockState.SEARCH,
            filtered_x=None,
            filtered_y=None,
            is_valid_track=False,
        )

    def reset(self) -> None:
        pass


class MockThresholdDetector(BeaconDetectorProtocol):
    """
    Simple brightness-threshold centroid detector strictly for harness tests.
    Finds the brightest pixel/region in a monochrome frame.
    """

    def __init__(self, threshold: int = 200) -> None:
        self.threshold = threshold

    def detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> DetectionResult:
        if frame is None or frame.size == 0:
            return DetectionResult(detected=False, method_used="MOCK_THRESHOLD")

        # Convert to grayscale if 3-channel
        if len(frame.shape) == 3:
            gray = np.mean(frame, axis=2).astype(np.uint8)
        else:
            gray = frame

        # Find pixels above threshold
        y_indices, x_indices = np.where(gray >= self.threshold)
        if len(x_indices) > 0:
            cx = float(np.mean(x_indices))
            cy = float(np.mean(y_indices))
            conf = min(1.0, float(np.max(gray)) / 255.0)
            return DetectionResult(
                detected=True,
                centroid_x=cx,
                centroid_y=cy,
                confidence=conf,
                method_used="MOCK_THRESHOLD",
            )

        return DetectionResult(
            detected=False,
            centroid_x=None,
            centroid_y=None,
            confidence=0.0,
            method_used="MOCK_THRESHOLD",
        )


class MockSimpleTracker(StateEstimatorProtocol):
    """
    Simple tracker with canonical state transitions strictly for testing the metric harness.
    """

    def __init__(self, coast_limit_frames: int = 5) -> None:
        self.coast_limit = coast_limit_frames
        self._consecutive_lost = 0
        self._has_ever_locked = False
        self._last_x: Optional[float] = None
        self._last_y: Optional[float] = None

    def update(self, detection: DetectionResult, dt: float) -> TrackingResult:
        if detection.detected and detection.centroid_x is not None and detection.centroid_y is not None:
            self._consecutive_lost = 0
            self._last_x = detection.centroid_x
            self._last_y = detection.centroid_y
            self._has_ever_locked = True
            return TrackingResult(
                lock_state=LockState.TRACK,
                filtered_x=self._last_x,
                filtered_y=self._last_y,
                velocity_x=0.0,
                velocity_y=0.0,
                is_valid_track=True,
            )
        else:
            self._consecutive_lost += 1
            if not self._has_ever_locked:
                return TrackingResult(
                    lock_state=LockState.SEARCH,
                    filtered_x=None,
                    filtered_y=None,
                    is_valid_track=False,
                )
            elif self._consecutive_lost <= self.coast_limit:
                return TrackingResult(
                    lock_state=LockState.COAST,
                    filtered_x=self._last_x,
                    filtered_y=self._last_y,
                    is_valid_track=False,
                )
            else:
                return TrackingResult(
                    lock_state=LockState.REACQUIRE,
                    filtered_x=None,
                    filtered_y=None,
                    is_valid_track=False,
                )

    def reset(self) -> None:
        self._consecutive_lost = 0
        self._has_ever_locked = False
        self._last_x = None
        self._last_y = None
