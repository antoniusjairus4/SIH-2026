"""
QThread Background Worker for FSOC Optical Simulator GUI.
==========================================================
Runs live TCP socket connection / frame simulation or pre-recorded .mp4 video playback (Benchmark-2)
and TrackingSystemPipeline in an isolated background thread without blocking the PyQt UI thread.
"""

import time
from typing import Dict, Any, Optional
import cv2
import numpy as np

# pyrefly: ignore [missing-import]
from PyQt6.QtCore import QThread, pyqtSignal

from src.network import MockUnityServer
from src.pipeline_runner import TrackingSystemPipeline
from src.metrics.schemas import TelemetryRecord


class TrackingPipelineWorker(QThread):
    """
    Asynchronous background worker executing TrackingSystemPipeline live loop.
    Emits telemetry_signal with processed frame and TelemetryRecord at 30+ FPS.
    """

    # Qt Signal emitting (frame_image, telemetry_record, packet_count)
    telemetry_signal = pyqtSignal(np.ndarray, object, int)

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5005,
        use_mock_server: bool = True,
        video_file_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.use_mock_server = use_mock_server
        self.video_file_path = video_file_path
        self.config = config or {}

        self._running = False
        self._paused = False
        self._manual_override = False
        self._manual_pan = 0.0
        self._manual_tilt = 0.0
        self._current_mode = "AUTO"

        self.mock_server: Optional[MockUnityServer] = None
        self.pipeline: Optional[TrackingSystemPipeline] = None

    def run(self):
        """Worker thread main loop."""
        self._running = True
        self._paused = False

        self.pipeline = TrackingSystemPipeline(host=self.host, port=self.port)
        self.pipeline.reset()

        # Apply initial PID & perception parameters if provided in config
        self.apply_config(self.config)

        if self.video_file_path:
            self._run_video_file_mode()
        else:
            self._run_tcp_socket_mode()

    def _run_tcp_socket_mode(self):
        """Streaming mode over TCP socket connection."""
        if self.use_mock_server:
            self.mock_server = MockUnityServer(host=self.host, port=self.port, fps=30.0)
            self.mock_server.start()
            time.sleep(0.2)

        if not self.pipeline.network_client.connect():
            self._running = False
            return

        packet_count = 0

        try:
            while self._running:
                if self._paused:
                    time.sleep(0.05)
                    continue

                item = self.pipeline.network_client.get_latest_frame(block=True, timeout=0.2)
                if item is None:
                    continue

                frame, frame_id, timestamp = item
                packet_count += 1

                self._process_and_emit(frame, frame_id, timestamp, packet_count)

        finally:
            self._running = False
            if self.pipeline:
                self.pipeline.network_client.disconnect()
            if self.mock_server:
                self.mock_server.stop()

    def _run_video_file_mode(self):
        """Offline Benchmark-2 mode reading pre-recorded video frames."""
        cap = cv2.VideoCapture(self.video_file_path)
        if not cap.isOpened():
            self._running = False
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / fps
        frame_id = 0
        start_time = time.time()

        try:
            while self._running:
                if self._paused:
                    time.sleep(0.05)
                    continue

                ret, bgr_frame = cap.read()
                if not ret or bgr_frame is None:
                    # Loop video back to frame 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_id += 1
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                timestamp = frame_id * frame_delay

                self._process_and_emit(rgb_frame, frame_id, timestamp, frame_id)
                time.sleep(frame_delay)

        finally:
            self._running = False
            cap.release()

    def _process_and_emit(self, frame: np.ndarray, frame_id: int, timestamp: float, packet_count: int):
        """Core frame execution routine."""
        if self._manual_override:
            # Under manual control, override gimbal angles directly
            self.pipeline.current_pan_deg = self._manual_pan
            self.pipeline.current_tilt_deg = self._manual_tilt

        telemetry = self.pipeline.process_single_frame(frame, frame_id, timestamp)

        if not self.video_file_path:
            pan_delta = telemetry.metadata.get("pan_delta", 0.0)
            tilt_delta = telemetry.metadata.get("tilt_delta", 0.0)
            if telemetry.metadata.get("should_command", True) and (pan_delta != 0.0 or tilt_delta != 0.0):
                self.pipeline.network_client.send_control_command(pan_delta, tilt_delta)

        self.telemetry_signal.emit(frame, telemetry, packet_count)

    def apply_config(self, config: Dict[str, Any]):
        """Apply dynamic configuration to pipeline components at runtime."""
        if not self.pipeline:
            return

        if "pan_kp" in config or "tilt_kp" in config:
            self.pipeline.controller.config.kp_pan = config.get("pan_kp", self.pipeline.controller.config.kp_pan)
            self.pipeline.controller.config.ki_pan = config.get("pan_ki", self.pipeline.controller.config.ki_pan)
            self.pipeline.controller.config.kd_pan = config.get("pan_kd", self.pipeline.controller.config.kd_pan)
            self.pipeline.controller.config.kp_tilt = config.get("tilt_kp", self.pipeline.controller.config.kp_tilt)
            self.pipeline.controller.config.max_slew_rate_deg_s = config.get("max_slew_deg_s", self.pipeline.controller.config.max_slew_rate_deg_s)

        if "tophat_kernel" in config:
            self.pipeline.detector.fast_path.tophat_kernel_size = config.get("tophat_kernel", 15)
        if "threshold" in config:
            self.pipeline.detector.fast_path.threshold_value = config.get("threshold", 180)
        if "use_onnx_fallback" in config:
            self.pipeline.detector.use_onnx_fallback = config.get("use_onnx_fallback", True)

    def set_manual_command(self, enabled: bool, pan_deg: float, tilt_deg: float):
        """Set manual control override state and angles."""
        self._manual_override = enabled
        self._manual_pan = pan_deg
        self._manual_tilt = tilt_deg

    def pause(self):
        """Pause worker pipeline execution."""
        self._paused = True

    def resume(self):
        """Resume worker pipeline execution."""
        self._paused = False

    def stop(self):
        """Stop worker thread loop."""
        self._running = False
        self.wait(2000)
