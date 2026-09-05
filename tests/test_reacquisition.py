"""
Unit Tests for Module 5 Archimedean Spiral Reacquisition Engine.
"""

import math
import pytest

from src.control.reacquisition import ReacquisitionEngine, ReacquisitionState
from src.estimation.state import EstimatorResult, TrackerMode


def _make_result(
    mode: TrackerMode = TrackerMode.TRACKING,
    confidence: float = 0.9,
    timestamp: float = 0.0,
) -> EstimatorResult:
    return EstimatorResult(
        x=320.0,
        y=240.0,
        vx=0.0,
        vy=0.0,
        ax=0.0,
        ay=0.0,
        predicted_x=320.0,
        predicted_y=240.0,
        mode=mode,
        prediction_only=(mode == TrackerMode.COASTING),
        measurement_available=(mode == TrackerMode.TRACKING),
        measurement_rejected=False,
        confidence=confidence,
        missing_frames=0,
        coast_time=0.0,
        inside_fov=True,
        timestamp=timestamp,
    )



def test_initial_state_is_idle():
    """Verify initial state is IDLE."""
    engine = ReacquisitionEngine()
    assert engine.state == ReacquisitionState.IDLE


def test_coasting_during_brief_loss():
    """Verify state remains COASTING during loss <= 0.5 s."""
    engine = ReacquisitionEngine(dropout_threshold_s=0.5)
    dt = 1.0 / 30.0

    # 10 frames of LOST state (0.33s < 0.5s)
    for i in range(10):
        t = i * dt
        res = engine.update(_make_result(mode=TrackerMode.LOST, confidence=0.0, timestamp=t), 0.0, 0.0, dt)
        assert res.state == ReacquisitionState.COASTING
        assert res.should_command is False


def test_spiral_search_triggered_after_dropout_threshold():
    """Verify SPIRAL_SEARCHING is triggered when loss > 0.5 s."""
    engine = ReacquisitionEngine(dropout_threshold_s=0.5, max_slew_deg_s=5.0)
    dt = 1.0 / 30.0

    # 20 frames of LOST state (0.66s > 0.5s)
    last_res = None
    for i in range(20):
        t = i * dt
        last_res = engine.update(_make_result(mode=TrackerMode.LOST, confidence=0.0, timestamp=t), 0.0, 0.0, dt)

    assert last_res is not None
    assert last_res.state == ReacquisitionState.SPIRAL_SEARCHING
    assert last_res.should_command is True
    assert last_res.spiral_radius_deg > 0.0

    # Verify max slew speed compliance (<= max_slew_deg_s * dt)
    max_delta = 5.0 * dt + 1e-6
    assert abs(last_res.pan_delta) <= max_delta
    assert abs(last_res.tilt_delta) <= max_delta


def test_instant_recovery_on_redetection():
    """Verify state immediately transitions back to REACQUIRED -> IDLE upon beacon re-detection."""
    engine = ReacquisitionEngine(dropout_threshold_s=0.5)
    dt = 1.0 / 30.0

    # Trigger spiral search (20 frames of LOST)
    for i in range(20):
        t = i * dt
        engine.update(_make_result(mode=TrackerMode.LOST, confidence=0.0, timestamp=t), 0.0, 0.0, dt)

    assert engine.state == ReacquisitionState.SPIRAL_SEARCHING

    # Next frame beacon is re-detected with high confidence
    res = engine.update(_make_result(mode=TrackerMode.TRACKING, confidence=0.9, timestamp=1.0), 0.0, 0.0, dt)

    assert res.state == ReacquisitionState.REACQUIRED
    assert res.should_command is False
    assert res.elapsed_loss_time_s == 0.0
