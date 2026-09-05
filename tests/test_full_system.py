"""
End-to-End System Integration Test for SIH 2026 Virtual Camera Tracking System.
Tests full inter-module communication across:
MockUnityServer (Port 5012) -> TCPNetworkClient -> BeaconDetector -> BeaconStateEstimator -> MetricLogger.
"""

import time
import math
import pytest

from src.network import MockUnityServer
from src.pipeline_runner import TrackingSystemPipeline


def test_full_system_live_pipeline_integration():
    test_port = 5012

    # 1. Start Mock Unity Server on port 5012
    server = MockUnityServer(host="localhost", port=test_port, fps=30.0)
    server.start()
    time.sleep(0.2)

    # 2. Instantiate Master Pipeline Supervisor
    pipeline = TrackingSystemPipeline(host="localhost", port=test_port)

    # 3. Dummy PTZ control callback (returns pan_delta=0.1, tilt_delta=-0.05)
    def dummy_control_callback(telemetry):
        return 0.1, -0.05

    # 4. Run live stream loop for 1.5 seconds
    logger_result = pipeline.run_live_stream_loop(
        duration_seconds=1.5,
        control_callback=dummy_control_callback,
    )

    time.sleep(0.2)
    server.stop()

    # 5. Verify system-wide integration metrics
    assert logger_result is not None
    assert logger_result.total_processed_frames > 10

    summary = logger_result.compute_summary()
    assert summary.total_video_frames > 10
    assert summary.average_processing_fps > 0.0

    # Verify PTZ deltas received back by Unity server socket
    assert len(server.received_commands) > 0
    pan, tilt = server.received_commands[0]
    assert pytest.approx(pan, 1e-4) == 0.1
    assert pytest.approx(tilt, 1e-4) == -0.05


def test_full_system_auto_pid_control_loop():
    """Verify live pipeline sends automatic PID commands back to Unity socket when control_callback is None."""
    test_port = 5013

    server = MockUnityServer(host="localhost", port=test_port, fps=30.0)
    server.start()
    time.sleep(0.2)

    pipeline = TrackingSystemPipeline(host="localhost", port=test_port)

    # Run live stream loop with automatic PID control loop (control_callback=None)
    logger_result = pipeline.run_live_stream_loop(
        duration_seconds=1.5,
        control_callback=None,
    )

    time.sleep(0.2)
    server.stop()

    assert logger_result is not None
    assert logger_result.total_processed_frames > 10
    # Check that PID commands were computed and sent to socket
    assert len(server.received_commands) > 0

