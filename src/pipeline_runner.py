"""
Master Tracking System Pipeline Supervisor for SIH 2026 Virtual Camera Tracking System.
Interconnects all core modules into a unified operational system:
  - Module 2B: Live TCP Socket Networking Client (localhost:5005)
  - Module 3:  Perception & CV Engine (Fast Path Top-Hat + YOLO ONNX Fallback)
  - Module 4:  Kalman State Estimator (6D Constant Acceleration Filter)
  - Module 5:  PID Gimbal Control Engine (Dual-Axis Pan/Tilt Controller)
  - Module 6:  Metric Logger & Exporters (RMSE, Lock Retention, Acquisition Time, JSON/CSV)
  - Benchmark-2: Pre-recorded .mp4 Video Execution Harness

Designed for instant plug-and-play compatibility when Noorul's Unity 3D C# simulator connects.
"""

import time
import logging
from typing import Optional, Tuple, Callable
import numpy as np

from src.network.network_client import TCPNetworkClient
from src.perception.detector import BeaconDetector
from src.estimation.kalman_filter import BeaconStateEstimator
from src.control.pid_controller import PIDController
from src.control.config import ControllerConfig
from src.control.state import ControlResult
from src.metrics.metric_logger import MetricLogger
from src.metrics.schemas import TelemetryRecord, LockState
from src.benchmark.benchmark2_runner import Benchmark2Runner
from src.benchmark.pipeline_adapter import CombinedTrackingPipeline

logger = logging.getLogger("TrackingSystemPipeline")


class TrackingSystemPipeline:
    """
    Unified Master Supervisor interconnecting Network, Perception, Estimation, Control, and Metrics.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5005,
        detector: Optional[BeaconDetector] = None,
        estimator: Optional[BeaconStateEstimator] = None,
        controller: Optional[PIDController] = None,
        output_dir: str = "results",
    ) -> None:
        self.host = host
        self.port = port
        self.output_dir = output_dir

        self.detector = detector or BeaconDetector()
        self.estimator = estimator or BeaconStateEstimator()
        self.controller = controller or PIDController()
        self.network_client = TCPNetworkClient(host=host, port=port)
        self.metric_logger: Optional[MetricLogger] = None

        self.current_pan_deg: float = 0.0
        self.current_tilt_deg: float = 0.0
        self._last_timestamp: Optional[float] = None
        self._running = False

    def reset(self) -> None:
        """Reset internal estimator, controller, and tracking states."""
        self.estimator.reset()
        self.controller.reset()
        self.current_pan_deg = 0.0
        self.current_tilt_deg = 0.0
        self._last_timestamp = None

    def process_single_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        timestamp: float,
        dt: Optional[float] = None,
    ) -> TelemetryRecord:
        """
        Executes a single frame through Perception -> Estimation -> Control pipeline.

        1. Perception: Runs Fast Path CV (or ONNX Fallback) to detect beacon centroid (cx, cy).
        2. Estimation: Runs 6D CA Kalman Filter to smooth jitter and predict trajectory.
        3. Control: Runs PID Controller to calculate angular pan/tilt commands.
        4. Returns: TelemetryRecord contract for telemetry & metric logging.
        """
        t_start = time.perf_counter()

        # Step 1: Perception (Beacon Detection)
        det_result = self.detector.detect(frame, frame_id=frame_id, timestamp=timestamp)

        # Step 2: Estimation (Kalman Filter Update)
        est_result = self.estimator.step(
            x=det_result.centroid_x if det_result.detected else None,
            y=det_result.centroid_y if det_result.detected else None,
            confidence=det_result.confidence,
            timestamp=timestamp,
        )

        # Calculate time step dt for PID loop
        if dt is None:
            if self._last_timestamp is not None and timestamp > self._last_timestamp:
                dt = timestamp - self._last_timestamp
            else:
                dt = 1.0 / 30.0
        self._last_timestamp = timestamp

        # Step 3: Control (PID Controller Compute)
        ctrl_result = self.controller.compute(
            estimator_result=est_result,
            current_pan_deg=self.current_pan_deg,
            current_tilt_deg=self.current_tilt_deg,
            dt=dt,
        )

        if ctrl_result.should_command:
            self.current_pan_deg += ctrl_result.pan_delta
            self.current_tilt_deg += ctrl_result.tilt_delta

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000.0

        # Convert Estimator TrackerMode to canonical LockState vocabulary
        mode_str = est_result.mode.value.upper()
        if mode_str == "UNINITIALIZED":
            lock_state = LockState.SEARCH
        elif mode_str == "TRACKING":
            lock_state = LockState.TRACK
        elif mode_str == "COASTING":
            lock_state = LockState.COAST
        elif mode_str == "LOST":
            lock_state = LockState.SEARCH
        else:
            lock_state = LockState.SEARCH

        return TelemetryRecord(
            frame_id=frame_id,
            video_timestamp=timestamp,
            processing_latency_ms=latency_ms,
            detector_status=det_result.detected,
            raw_x=det_result.centroid_x,
            raw_y=det_result.centroid_y,
            confidence=det_result.confidence,
            detection_method=det_result.method_used,
            lock_state=lock_state,
            filtered_x=est_result.x,
            filtered_y=est_result.y,
            is_valid_track=(lock_state == LockState.TRACK),
            metadata={
                "vx": est_result.vx,
                "vy": est_result.vy,
                "predicted_x": est_result.predicted_x,
                "predicted_y": est_result.predicted_y,
                "pan_delta": ctrl_result.pan_delta,
                "tilt_delta": ctrl_result.tilt_delta,
                "should_command": ctrl_result.should_command,
                "raw_error_x_deg": ctrl_result.raw_error_x_deg,
                "raw_error_y_deg": ctrl_result.raw_error_y_deg,
                "current_pan_deg": self.current_pan_deg,
                "current_tilt_deg": self.current_tilt_deg,
            },
        )

    def run_live_stream_loop(
        self,
        duration_seconds: float = 10.0,
        control_callback: Optional[Callable[[TelemetryRecord], Tuple[float, float]]] = None,
    ) -> MetricLogger:
        """
        Runs the live socket streaming loop connecting Unity on port 5005 with the Python pipeline.

        Args:
            duration_seconds: Maximum run duration.
            control_callback: Optional function returning (pan_delta, tilt_delta) from control module.

        Returns:
            Populated MetricLogger object.
        """
        logger.info(f"Connecting to Unity Simulator live TCP stream at {self.host}:{self.port}...")
        if not self.network_client.connect():
            raise ConnectionError(f"Could not connect to Unity TCP Server at {self.host}:{self.port}")

        self.metric_logger = MetricLogger(
            benchmark_name="BENCHMARK_1_LIVE_UNITY",
            source_fps=30.0,
        )

        self._running = True
        self.reset()
        start_time = time.time()

        try:
            while self._running and (time.time() - start_time < duration_seconds):
                item = self.network_client.get_latest_frame(block=True, timeout=0.5)
                if item is None:
                    continue

                frame, frame_id, timestamp = item

                # 1. Process frame through Perception -> Estimation -> Control pipeline
                telemetry = self.process_single_frame(frame, frame_id, timestamp)

                # 2. Log telemetry to real-time Metric Logger
                self.metric_logger.log_frame(telemetry)

                # 3. Compute & send PTZ command back to Unity
                if control_callback:
                    pan_delta, tilt_delta = control_callback(telemetry)
                else:
                    pan_delta = telemetry.metadata.get("pan_delta", 0.0)
                    tilt_delta = telemetry.metadata.get("tilt_delta", 0.0)
                    should_cmd = telemetry.metadata.get("should_command", True)
                    if not should_cmd:
                        pan_delta, tilt_delta = 0.0, 0.0

                if pan_delta != 0.0 or tilt_delta != 0.0:
                    self.network_client.send_control_command(pan_delta, tilt_delta)

        finally:
            self._running = False
            self.network_client.disconnect()

        return self.metric_logger

    def run_offline_benchmark2(
        self,
        video_path: str,
        ground_truth_path: Optional[str] = None,
    ):
        """
        Executes offline .mp4 video benchmark using Shylee's Benchmark2Runner.
        """
        combined_pipeline = CombinedTrackingPipeline(
            detector=self.detector,
            state_estimator=self.estimator,
        )
        runner = Benchmark2Runner(pipeline=combined_pipeline, output_dir=f"{self.output_dir}/benchmark2")
        return runner.run_benchmark(video_path=video_path, ground_truth_path=ground_truth_path)

