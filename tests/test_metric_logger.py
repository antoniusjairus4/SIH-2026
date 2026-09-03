"""
Unit tests and edge-case validation for MetricLogger mathematical calculations and lock state machine.
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


def test_intermittent_detection_loss_multiple_cycles():
    logger = MetricLogger(benchmark_name="TEST_MULTI_LOSS", source_fps=10.0)

    # Cycle 1: Acquired at 0.1, lost at 0.3, restored at 0.5 (loss duration = 0.2)
    # Cycle 2: Lost at 0.7, restored at 1.0 (loss duration = 0.3)
    # Final unrecovered: Lost at 1.2 to 1.5 (loss duration = 0.3)
    sequence = [
        (LockState.SEARCH, 0.0),
        (LockState.TRACK, 0.1),
        (LockState.TRACK, 0.2),
        (LockState.COAST, 0.3),     # Loss 1 start
        (LockState.COAST, 0.4),
        (LockState.TRACK, 0.5),     # Loss 1 end (dur=0.2)
        (LockState.TRACK, 0.6),
        (LockState.COAST, 0.7),     # Loss 2 start
        (LockState.REACQUIRE, 0.8),
        (LockState.REACQUIRE, 0.9),
        (LockState.TRACK, 1.0),     # Loss 2 end (dur=0.3)
        (LockState.TRACK, 1.1),
        (LockState.COAST, 1.2),     # Loss 3 start (unrecovered at benchmark end)
        (LockState.COAST, 1.3),
        (LockState.SEARCH, 1.4),
        (LockState.SEARCH, 1.5),    # End of video
    ]

    for fid, (st, t) in enumerate(sequence):
        telem = TelemetryRecord(
            frame_id=fid,
            video_timestamp=t,
            processing_latency_ms=10.0,
            detector_status=(st == LockState.TRACK),
            raw_x=200.0 if st == LockState.TRACK else None,
            raw_y=200.0 if st == LockState.TRACK else None,
            confidence=1.0 if st == LockState.TRACK else 0.0,
            detection_method="TEST",
            lock_state=st,
            filtered_x=200.0 if st in (LockState.TRACK, LockState.COAST) else None,
            filtered_y=200.0 if st in (LockState.TRACK, LockState.COAST) else None,
            is_valid_track=(st == LockState.TRACK),
        )
        logger.log_frame(telem)

    summary = logger.compute_summary()
    assert summary.total_video_frames == 16
    assert summary.target_loss_event_count == 3
    assert summary.reacquisition_time_stats["count"] == 2
    # Recovered reacq times: [0.2, 0.3] -> mean = 0.25
    assert math.isclose(summary.reacquisition_time_stats["mean_s"], 0.25, abs_tol=1e-3)
    assert math.isclose(summary.reacquisition_time_stats["min_s"], 0.2, abs_tol=1e-3)
    assert math.isclose(summary.reacquisition_time_stats["max_s"], 0.3, abs_tol=1e-3)
    # Total loss duration: 0.2 + 0.3 + (1.5 - 1.2 = 0.3) = 0.8s
    assert math.isclose(summary.total_target_loss_duration_s, 0.8, abs_tol=1e-3)


def test_full_canonical_state_transitions():
    logger = MetricLogger(benchmark_name="TEST_CANONICAL_STATES")
    states = [
        LockState.SEARCH,
        LockState.ACQUIRE,
        LockState.TRACK,
        LockState.COAST,
        LockState.REACQUIRE,
        LockState.TRACK,
    ]

    for fid, st in enumerate(states):
        telem = TelemetryRecord(
            frame_id=fid,
            video_timestamp=fid * 0.1,
            processing_latency_ms=10.0,
            detector_status=(st in (LockState.ACQUIRE, LockState.TRACK)),
            raw_x=100.0 if st in (LockState.ACQUIRE, LockState.TRACK) else None,
            raw_y=100.0 if st in (LockState.ACQUIRE, LockState.TRACK) else None,
            confidence=0.8,
            detection_method="CANONICAL",
            lock_state=st,
            filtered_x=100.0 if st in (LockState.TRACK, LockState.COAST) else None,
            filtered_y=100.0 if st in (LockState.TRACK, LockState.COAST) else None,
            is_valid_track=(st == LockState.TRACK),
        )
        logger.log_frame(telem)

    summary = logger.compute_summary()
    assert summary.lock_state_frame_counts[LockState.SEARCH.value] == 1
    assert summary.lock_state_frame_counts[LockState.ACQUIRE.value] == 1
    assert summary.lock_state_frame_counts[LockState.TRACK.value] == 2
    assert summary.lock_state_frame_counts[LockState.COAST.value] == 1
    assert summary.lock_state_frame_counts[LockState.REACQUIRE.value] == 1


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


def test_perfect_zero_error_tracking():
    logger = MetricLogger(benchmark_name="TEST_PERFECT")

    for fid in range(20):
        pos_x = 100.0 + fid * 2.0
        pos_y = 200.0 + fid * 1.5
        telem = TelemetryRecord(
            frame_id=fid,
            video_timestamp=fid * 0.033,
            processing_latency_ms=10.0,
            detector_status=True,
            raw_x=pos_x,
            raw_y=pos_y,
            confidence=1.0,
            detection_method="PERFECT",
            lock_state=LockState.TRACK,
            filtered_x=pos_x,
            filtered_y=pos_y,
            is_valid_track=True,
        )
        gt = GroundTruthRecord(frame_id=fid, gt_x=pos_x, gt_y=pos_y)
        rec = logger.log_frame(telem, gt)
        assert rec.centroid_error_px == 0.0

    summary = logger.compute_summary()
    assert summary.evaluated_frames_count == 20
    assert summary.mean_centroid_error_px == 0.0
    assert summary.rmse_px == 0.0
    assert summary.max_error_px == 0.0
    assert summary.std_dev_error_px == 0.0
    assert summary.lock_retention_rate_pct == 100.0


def test_large_tracking_error():
    logger = MetricLogger(benchmark_name="TEST_LARGE_ERROR")
    telem = TelemetryRecord(
        frame_id=0,
        video_timestamp=0.0,
        processing_latency_ms=10.0,
        detector_status=True,
        raw_x=0.0,
        raw_y=0.0,
        confidence=1.0,
        detection_method="LARGE",
        lock_state=LockState.TRACK,
        filtered_x=0.0,
        filtered_y=0.0,
        is_valid_track=True,
    )
    gt = GroundTruthRecord(frame_id=0, gt_x=1000.0, gt_y=1000.0)
    rec = logger.log_frame(telem, gt)
    expected_dist = math.sqrt(1000.0**2 + 1000.0**2)
    assert math.isclose(rec.centroid_error_px, expected_dist, abs_tol=1e-3)


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


def test_nan_inf_predictions_and_gt_safety():
    logger = MetricLogger(benchmark_name="TEST_NAN_INF")
    telem_nan = TelemetryRecord(
        frame_id=0,
        video_timestamp=0.0,
        processing_latency_ms=10.0,
        detector_status=True,
        raw_x=float("nan"),
        raw_y=float("nan"),
        confidence=0.5,
        detection_method="TEST",
        lock_state=LockState.TRACK,
        filtered_x=float("nan"),
        filtered_y=float("nan"),
        is_valid_track=True,
    )
    gt_inf = GroundTruthRecord(frame_id=0, gt_x=float("inf"), gt_y=100.0)
    rec = logger.log_frame(telem_nan, gt_inf)
    assert rec.centroid_error_px is None

    summary = logger.compute_summary()
    assert summary.evaluated_frames_count == 0
    assert summary.mean_centroid_error_px is None


def test_latency_and_fps_zero_and_minimal_data():
    logger = MetricLogger(benchmark_name="TEST_EMPTY")

    # 0 frames
    summary_empty = logger.compute_summary()
    assert summary_empty.total_video_frames == 0
    assert summary_empty.average_processing_fps == 0.0
    assert summary_empty.latency_ms_stats["mean_ms"] is None
    assert summary_empty.lock_retention_rate_pct == 0.0
    assert summary_empty.target_loss_rate_pct == 0.0

    # 1 frame with 0.0 ms latency input (should be clamped safely)
    telem_single = TelemetryRecord(
        frame_id=0,
        video_timestamp=0.0,
        processing_latency_ms=0.0,
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
    logger.log_frame(telem_single)
    summary_single = logger.compute_summary()
    assert summary_single.total_video_frames == 1
    assert summary_single.average_processing_fps > 0.0
    assert summary_single.latency_ms_stats["min_ms"] == 0.001
    assert summary_single.lock_retention_rate_pct == 100.0


def test_reset_clears_all_internal_state():
    logger = MetricLogger(benchmark_name="TEST_RESET")
    telem = TelemetryRecord(
        frame_id=0,
        video_timestamp=0.0,
        processing_latency_ms=10.0,
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
    logger.log_frame(telem, GroundTruthRecord(frame_id=0, gt_x=100.0, gt_y=100.0))
    assert logger.total_processed_frames == 1

    logger.reset()
    assert logger.total_processed_frames == 0
    assert len(logger.frame_records) == 0
    summary = logger.compute_summary()
    assert summary.total_video_frames == 0
    assert summary.evaluated_frames_count == 0
