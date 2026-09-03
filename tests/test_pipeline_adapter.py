"""
Unit tests for pipeline protocols and adapters.
"""

import numpy as np
import pytest
from src.metrics.schemas import LockState
from src.benchmark.pipeline_adapter import (
    BeaconDetectorProtocol,
    StateEstimatorProtocol,
    CombinedTrackingPipeline,
    NullDetector,
    NullStateEstimator,
    MockThresholdDetector,
    MockSimpleTracker,
)


def test_protocol_conformance():
    null_det = NullDetector()
    null_est = NullStateEstimator()
    assert isinstance(null_det, BeaconDetectorProtocol)
    assert isinstance(null_est, StateEstimatorProtocol)

    mock_det = MockThresholdDetector()
    mock_est = MockSimpleTracker()
    assert isinstance(mock_det, BeaconDetectorProtocol)
    assert isinstance(mock_est, StateEstimatorProtocol)


def test_null_pipeline():
    pipeline = CombinedTrackingPipeline(detector=NullDetector(), state_estimator=NullStateEstimator())
    frame = np.zeros((480, 640), dtype=np.uint8)

    det, track = pipeline.process_frame(frame, frame_id=0, timestamp=0.0)
    assert det.detected is False
    assert det.centroid_x is None
    assert track.lock_state == LockState.SEARCH
    assert track.filtered_x is None
    assert track.is_valid_track is False


def test_mock_threshold_detector_and_simple_tracker():
    detector = MockThresholdDetector(threshold=200)
    tracker = MockSimpleTracker(coast_limit_frames=2)
    pipeline = CombinedTrackingPipeline(detector=detector, state_estimator=tracker)

    # Frame 0: Dark frame -> SEARCH
    dark_frame = np.zeros((480, 640), dtype=np.uint8)
    det0, trk0 = pipeline.process_frame(dark_frame, frame_id=0, timestamp=0.0)
    assert det0.detected is False
    assert trk0.lock_state == LockState.SEARCH

    # Frame 1: Bright spot at (320, 240) -> TRACK
    bright_frame = np.zeros((480, 640), dtype=np.uint8)
    bright_frame[240, 320] = 255
    det1, trk1 = pipeline.process_frame(bright_frame, frame_id=1, timestamp=0.033)
    assert det1.detected is True
    assert det1.centroid_x == 320.0
    assert det1.centroid_y == 240.0
    assert trk1.lock_state == LockState.TRACK
    assert trk1.filtered_x == 320.0
    assert trk1.filtered_y == 240.0
    assert trk1.is_valid_track is True

    # Frame 2: Dark frame (dropout 1) -> COAST (preserves last coordinates)
    det2, trk2 = pipeline.process_frame(dark_frame, frame_id=2, timestamp=0.066)
    assert det2.detected is False
    assert trk2.lock_state == LockState.COAST
    assert trk2.filtered_x == 320.0
    assert trk2.is_valid_track is False

    # Frame 3: Dark frame (dropout 2) -> COAST
    det3, trk3 = pipeline.process_frame(dark_frame, frame_id=3, timestamp=0.099)
    assert trk3.lock_state == LockState.COAST

    # Frame 4: Dark frame (dropout 3 > limit of 2) -> REACQUIRE
    det4, trk4 = pipeline.process_frame(dark_frame, frame_id=4, timestamp=0.132)
    assert trk4.lock_state == LockState.REACQUIRE

    # Pipeline reset
    pipeline.reset()
    assert tracker._consecutive_lost == 0
