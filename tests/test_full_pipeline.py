"""
Unit and Integration Tests for TrackingSystemPipeline with PID Controller.
Verifies Perception -> Estimation -> Control -> Telemetry pipeline execution.
"""

import numpy as np
import pytest

from src.pipeline_runner import TrackingSystemPipeline
from src.metrics.schemas import LockState


def test_pipeline_single_frame_processing():
    """Test process_single_frame generates valid telemetry with PID metadata."""
    pipeline = TrackingSystemPipeline()
    pipeline.reset()

    # Create dummy synthetic frame with bright beacon spot at (400, 300)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[295:305, 395:405] = 255

    # Process first frame
    record = pipeline.process_single_frame(frame=frame, frame_id=1, timestamp=0.0)

    assert record.frame_id == 1
    assert record.video_timestamp == 0.0
    assert record.detector_status is True
    assert record.raw_x is not None
    assert record.raw_y is not None
    assert record.lock_state == LockState.TRACK

    # Check PID metadata fields
    assert "pan_delta" in record.metadata
    assert "tilt_delta" in record.metadata
    assert "should_command" in record.metadata
    assert "raw_error_x_deg" in record.metadata
    assert "raw_error_y_deg" in record.metadata
    assert "current_pan_deg" in record.metadata
    assert "current_tilt_deg" in record.metadata

    # Since beacon is at (400, 300) vs center (320, 240), pan_delta should be positive (pan right)
    assert record.metadata["pan_delta"] > 0.0


def test_pipeline_multi_frame_tracking_and_control():
    """Test multi-frame sequence accumulation of PID commands in TrackingSystemPipeline."""
    pipeline = TrackingSystemPipeline()
    pipeline.reset()

    # Beacon offset to the right at x=400, y=240
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[235:245, 395:405] = 255

    initial_pan = pipeline.current_pan_deg

    for i in range(10):
        t = i * (1.0 / 30.0)
        record = pipeline.process_single_frame(frame=frame, frame_id=i, timestamp=t)
        assert record.is_valid_track is True

    # Pan angle should have increased towards target
    assert pipeline.current_pan_deg > initial_pan
