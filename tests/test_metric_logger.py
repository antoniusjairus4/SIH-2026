"""
Unit tests for MetricLogger mathematical calculations and lock state machine.
"""

import math
import pytest
from src.metrics.schemas import (
    LockState,
    TelemetryRecord,
    GroundTruthRecord,
)
from src.metrics.metric_logger import MetricLogger


def test_spatial_error_and_rmse_calculation():
    logger = MetricLogger(benchmark_name="TEST", source_fps=30.0)

    # Frame 0: error = sqrt(3^2 + 4^2) = 5.0
    t0 = TelemetryRecord(
        frame_id=0,
        video_timestamp=0.0,
        processing_latency_ms=10.0,
        detector_status=True,
        raw_x=100.0,
        raw_y=100.0,
        confidence=0.9,
        detection_method="FAST_PATH",
        lock_state=LockState.TRACK,
        filtered_x=100.0,
        filtered_y=100.0,
        is_valid_track=True,
    )
    gt0 = GroundTruthRecord(frame_id=0, gt_x=103.0, gt_y=104.0)
    rec0 = logger.log_frame(t0, gt0)
    assert rec0.centroid_error_px == 5.0
    assert rec0.squared_error_px2 == 25.0

    # Frame 1: error = 0.0
    t1 = TelemetryRecord(
        frame_id=1,
        video_timestamp=0.033,
        processing_latency_ms=10.0,
        detector_status=True,
        raw_x=200.0,
        raw_y=200.0,
        confidence=0.95,
        detection_method="FAST_PATH",
        lock_state=LockState.TRACK,
        filtered_x=200.0,
        filtered_y=200.0,
        is_valid_track=True,
    )
    gt1 = GroundTruthRecord(frame_id=1, gt_x=200.0, gt_y=200.0)
    rec1 = logger.log_frame(t1, gt1)
    assert rec1.centroid_error_px == 0.0
    assert rec1.squared_error_px2 == 0.0

    # Frame 2: error = sqrt(6^2 + 8^2) = 10.0
    t2 = TelemetryRecord(
        frame_id=2,
        video_timestamp=0.066,
        processing_latency_ms=10.0,
        detector_status=True,
        raw_x=300.0,
        raw_y=300.0,
        confidence=0.88,
        detection_method="FAST_PATH",
        lock_state=LockState.TRACK,
        filtered_x=300.0,
        filtered_y=300.0,
        is_valid_track=True,
    )
    gt2 = GroundTruthRecord(frame_id=2, gt_x=306.0, gt_y=308.0)
    rec2 = logger.log_frame(t2, gt2)
    assert rec2.centroid_error_px == 10.0
    assert rec2.squared_error_px2 == 100.0

    summary = logger.compute_summary()
    assert summary.evaluated_frames_count == 3
    # Mean error = (5 + 0 + 10) / 3 = 5.0
    assert math.isclose(summary.mean_centroid_error_px, 5.0, abs_tol=1e-3)
    # RMSE = sqrt((25 + 0 + 100) / 3) = sqrt(125/3) = 6.45497
    expected_rmse = math.sqrt(125.0 / 3.0)
    assert math.isclose(summary.rmse_px, expected_rmse, abs_tol=1e-3)
    # Max error = 10.0
    assert summary.max_error_px == 10.0


def test_lock_state_transitions_acquisition_and_reacquisition():
    logger = MetricLogger(benchmark_name="TEST_LOCK", source_fps=10.0)

    # Sequence of states and timestamps
    states_and_times = [
        (LockState.SEARCH, 0.0),    # Frame 0: Search
        (LockState.SEARCH, 0.1),    # Frame 1: Search
        (LockState.TRACK, 0.2),     # Frame 2: Initial Acquisition achieved at t=0.2
        (LockState.TRACK, 0.3),     # Frame 3: Track
        (LockState.COAST, 0.4),     # Frame 4: Loss event 1 starts at t=0.4
        (LockState.COAST, 0.5),     # Frame 5: Coasting
        (LockState.TRACK, 0.6),     # Frame 6: Reacquisition achieved! Duration = 0.6 - 0.4 = 0.2s
        (LockState.TRACK, 0.7),     # Frame 7: Track
    ]

    for fid, (st, t) in enumerate(states_and_times):
        telem = TelemetryRecord(
            frame_id=fid,
            video_timestamp=t,
            processing_latency_ms=10.0,
            detector_status=(st == LockState.TRACK),
            raw_x=100.0 if st == LockState.TRACK else None,
            raw_y=100.0 if st == LockState.TRACK else None,
            confidence=0.9 if st == LockState.TRACK else 0.0,
            detection_method="MOCK",
            lock_state=st,
            filtered_x=100.0 if st in (LockState.TRACK, LockState.COAST) else None,
            filtered_y=100.0 if st in (LockState.TRACK, LockState.COAST) else None,
            is_valid_track=(st == LockState.TRACK),
        )
        logger.log_frame(telem)

    summary = logger.compute_summary()
    assert summary.total_video_frames == 8
    # 4 TRACK frames out of 8 -> 50%
    assert summary.lock_retention_rate_pct == 50.0
    assert summary.target_loss_rate_pct == 50.0
    # Acquisition time = 0.2s - 0.0s = 0.2s
    assert math.isclose(summary.acquisition_time_s, 0.2, abs_tol=1e-3)
    # Loss events = 1
    assert summary.target_loss_event_count == 1
    # Reacquisition time = 0.2s
    assert summary.reacquisition_time_stats["count"] == 1
    assert math.isclose(summary.reacquisition_time_stats["mean_s"], 0.2, abs_tol=1e-3)
    assert summary.lock_state_frame_counts[LockState.SEARCH.value] == 2
    assert summary.lock_state_frame_counts[LockState.TRACK.value] == 4
    assert summary.lock_state_frame_counts[LockState.COAST.value] == 2


def test_zero_lock_sequence():
    logger = MetricLogger(benchmark_name="TEST_ZERO_LOCK")

    for fid in range(10):
        telem = TelemetryRecord(
            frame_id=fid,
            video_timestamp=fid * 0.033,
            processing_latency_ms=5.0,
            detector_status=False,
            raw_x=None,
            raw_y=None,
            confidence=0.0,
            detection_method="NONE",
            lock_state=LockState.SEARCH,
            filtered_x=None,
            filtered_y=None,
            is_valid_track=False,
        )
        logger.log_frame(telem)

    summary = logger.compute_summary()
    assert summary.total_video_frames == 10
    assert summary.lock_retention_rate_pct == 0.0
    assert summary.acquisition_time_s is None
    assert summary.target_loss_event_count == 0
    assert summary.mean_centroid_error_px is None
    assert summary.rmse_px is None


def test_missing_ground_truth_behavior():
    logger = MetricLogger(benchmark_name="TEST_NO_GT")

    for fid in range(5):
        telem = TelemetryRecord(
            frame_id=fid,
            video_timestamp=fid * 0.033,
            processing_latency_ms=10.0,
            detector_status=True,
            raw_x=150.0,
            raw_y=150.0,
            confidence=0.9,
            detection_method="FAST_PATH",
            lock_state=LockState.TRACK,
            filtered_x=150.0,
            filtered_y=150.0,
            is_valid_track=True,
        )
        rec = logger.log_frame(telem, ground_truth=None)
        assert rec.gt_x is None
        assert rec.centroid_error_px is None

    summary = logger.compute_summary()
    assert summary.evaluated_frames_count == 0
    assert summary.mean_centroid_error_px is None
    assert summary.rmse_px is None
    assert summary.lock_retention_rate_pct == 100.0


def test_latency_statistics():
    logger = MetricLogger(benchmark_name="TEST_LATENCY")
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]

    for fid, lat in enumerate(latencies):
        telem = TelemetryRecord(
            frame_id=fid,
            video_timestamp=fid * 0.1,
            processing_latency_ms=lat,
            detector_status=True,
            raw_x=100.0,
            raw_y=100.0,
            confidence=1.0,
            detection_method="TEST",
            lock_state=LockState.TRACK,
            filtered_x=100.0,
            filtered_y=100.0,
            is_valid_track=True,
        )
        logger.log_frame(telem)

    summary = logger.compute_summary()
    assert summary.latency_ms_stats["min_ms"] == 10.0
    assert summary.latency_ms_stats["max_ms"] == 50.0
    assert summary.latency_ms_stats["median_ms"] == 30.0
    assert summary.latency_ms_stats["mean_ms"] == 30.0
    # 5 frames in 150 ms total = 5 / 0.150 = 33.33 FPS
    assert math.isclose(summary.average_processing_fps, 33.33, abs_tol=0.1)
