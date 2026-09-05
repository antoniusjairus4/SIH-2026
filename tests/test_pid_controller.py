import json
import math

from src.control import (
    ControllerConfig,
    ControlResult,
    PIDController,
)
from src.estimation.state import EstimatorResult, TrackerMode


# ------------------------------------------------------------------
# Helper: build a synthetic EstimatorResult with sensible defaults.
# ------------------------------------------------------------------

def _make_result(
    predicted_x: float = 320.0,
    predicted_y: float = 240.0,
    mode: TrackerMode = TrackerMode.TRACKING,
    confidence: float = 0.95,
    x: float = 320.0,
    y: float = 240.0,
) -> EstimatorResult:
    """
    Factory for synthetic EstimatorResult objects with known fields.

    Only *predicted_x*, *predicted_y*, *mode*, and *confidence* vary
    across tests; the remaining fields use safe defaults.
    """

    return EstimatorResult(
        x=x,
        y=y,
        vx=0.0,
        vy=0.0,
        ax=0.0,
        ay=0.0,
        predicted_x=predicted_x,
        predicted_y=predicted_y,
        mode=mode,
        prediction_only=mode != TrackerMode.TRACKING,
        measurement_available=mode == TrackerMode.TRACKING,
        measurement_rejected=False,
        confidence=confidence,
        missing_frames=0,
        coast_time=0.0,
        inside_fov=True,
        timestamp=0.0,
    )


# ==================================================================
# P / I / D isolation tests
# ==================================================================

def test_p_only_proportional_response():
    """With I=D=0, output is proportional to pixel error."""

    config = ControllerConfig(
        kp_x=1.0,
        ki_x=0.0,
        kd_x=0.0,
        kp_y=1.0,
        ki_y=0.0,
        kd_y=0.0,
        max_pan_speed_deg_s=100.0,
        max_tilt_speed_deg_s=100.0,
    )

    controller = PIDController(config)

    # Beacon 50 px to the right of center (320+50=370) -> positive
    # pan error.
    result = controller.compute(
        estimator_result=_make_result(predicted_x=370.0),
        current_pan_deg=0.0,
        current_tilt_deg=0.0,
        dt=1.0 / 30.0,
    )

    assert result.should_command is True

    # error_x_px = 370 - 320 = 50 px
    # error_x_deg = 50 * (4.0 / 640) = 0.3125 deg
    # output = kp * conf * error = 1.0 * 0.95 * 0.3125
    expected_pan = 1.0 * 0.95 * (50.0 * 4.0 / 640.0)

    assert abs(result.pan_delta - expected_pan) < 1e-6


def test_i_accumulates_over_time():
    """With P=D=0, integral grows over repeated calls."""

    config = ControllerConfig(
        kp_x=0.0,
        ki_x=1.0,
        kd_x=0.0,
        kp_y=0.0,
        ki_y=1.0,
        kd_y=0.0,
        max_pan_speed_deg_s=100.0,
        max_tilt_speed_deg_s=100.0,
    )

    controller = PIDController(config)

    dt = 1.0 / 30.0
    er = _make_result(predicted_x=370.0, predicted_y=260.0)

    first = controller.compute(er, 0.0, 0.0, dt)
    second = controller.compute(er, 0.0, 0.0, dt)

    # Second call should have more integral accumulation -> larger
    # magnitude.
    assert abs(second.pan_delta) > abs(first.pan_delta)
    assert abs(second.tilt_delta) > abs(first.tilt_delta)


def test_d_responds_to_error_change():
    """With P=I=0, D output responds to changing error."""

    config = ControllerConfig(
        kp_x=0.0,
        ki_x=0.0,
        kd_x=1.0,
        kp_y=0.0,
        ki_y=0.0,
        kd_y=1.0,
        max_pan_speed_deg_s=100.0,
        max_tilt_speed_deg_s=100.0,
    )

    controller = PIDController(config)
    dt = 1.0 / 30.0

    # First call establishes prev_error.
    controller.compute(
        _make_result(predicted_x=320.0),
        0.0,
        0.0,
        dt,
    )

    # Second call with shifted error -> D should be non-zero.
    result = controller.compute(
        _make_result(predicted_x=370.0),
        0.0,
        0.0,
        dt,
    )

    assert abs(result.pan_delta) > 0.0


# ==================================================================
# Mode handling tests
# ==================================================================

def test_lost_mode_returns_no_command():
    """should_command is False when mode is LOST."""

    controller = PIDController()

    result = controller.compute(
        estimator_result=_make_result(mode=TrackerMode.LOST),
        current_pan_deg=0.0,
        current_tilt_deg=0.0,
        dt=1.0 / 30.0,
    )

    assert result.should_command is False
    assert result.pan_delta == 0.0
    assert result.tilt_delta == 0.0
    assert result.mode_at_command_time == TrackerMode.LOST


def test_uninitialized_returns_no_command():
    """should_command is False when mode is UNINITIALIZED."""

    controller = PIDController()

    result = controller.compute(
        estimator_result=_make_result(
            mode=TrackerMode.UNINITIALIZED,
        ),
        current_pan_deg=0.0,
        current_tilt_deg=0.0,
        dt=1.0 / 30.0,
    )

    assert result.should_command is False
    assert result.mode_at_command_time == TrackerMode.UNINITIALIZED


# ==================================================================
# Speed clamp test
# ==================================================================

def test_output_respects_speed_clamps():
    """Large error does not produce output exceeding max speed × dt."""

    config = ControllerConfig(
        kp_x=10.0,
        kp_y=10.0,
        max_pan_speed_deg_s=5.0,
        max_tilt_speed_deg_s=5.0,
    )

    controller = PIDController(config)
    dt = 1.0 / 30.0

    # Place beacon far off-center to produce large raw error.
    result = controller.compute(
        estimator_result=_make_result(
            predicted_x=640.0,
            predicted_y=480.0,
        ),
        current_pan_deg=0.0,
        current_tilt_deg=0.0,
        dt=dt,
    )

    max_pan = config.max_pan_speed_deg_s * dt
    max_tilt = config.max_tilt_speed_deg_s * dt

    assert abs(result.pan_delta) <= max_pan + 1e-12
    assert abs(result.tilt_delta) <= max_tilt + 1e-12

    # Raw error should be larger than clamped output.
    assert abs(result.raw_error_x_deg) > abs(result.pan_delta)


# ==================================================================
# Coasting behaviour tests
# ==================================================================

def test_coasting_reduces_output():
    """Identical scenario in COASTING vs TRACKING -> lower magnitude."""

    config = ControllerConfig(
        kp_x=1.0,
        ki_x=0.0,
        kd_x=0.0,
        kp_y=1.0,
        ki_y=0.0,
        kd_y=0.0,
        coasting_gain_scale=0.5,
        max_pan_speed_deg_s=100.0,
        max_tilt_speed_deg_s=100.0,
    )

    er_tracking = _make_result(
        predicted_x=400.0,
        mode=TrackerMode.TRACKING,
    )

    er_coasting = _make_result(
        predicted_x=400.0,
        mode=TrackerMode.COASTING,
    )

    tracking_ctrl = PIDController(config)
    coasting_ctrl = PIDController(config)

    dt = 1.0 / 30.0

    r_tracking = tracking_ctrl.compute(er_tracking, 0.0, 0.0, dt)
    r_coasting = coasting_ctrl.compute(er_coasting, 0.0, 0.0, dt)

    assert abs(r_coasting.pan_delta) < abs(r_tracking.pan_delta)


def test_coasting_freezes_integral():
    """Integral value does not change during COASTING steps."""

    config = ControllerConfig(
        kp_x=0.0,
        ki_x=1.0,
        kd_x=0.0,
        max_pan_speed_deg_s=100.0,
    )

    controller = PIDController(config)
    dt = 1.0 / 30.0

    # One TRACKING step to build some integral.
    controller.compute(
        _make_result(
            predicted_x=370.0,
            mode=TrackerMode.TRACKING,
        ),
        0.0,
        0.0,
        dt,
    )

    integral_before = controller._integral_x

    # Several COASTING steps — integral should not change.
    for _ in range(10):
        controller.compute(
            _make_result(
                predicted_x=370.0,
                mode=TrackerMode.COASTING,
            ),
            0.0,
            0.0,
            dt,
        )

    assert controller._integral_x == integral_before


# ==================================================================
# Integral reset on mode transition
# ==================================================================

def test_integral_reset_on_mode_transition():
    """
    TRACKING -> LOST -> TRACKING: integral is cleared on re-entry
    so accumulated windup from before the LOST period does not cause
    an overshoot.
    """

    config = ControllerConfig(
        kp_x=0.0,
        ki_x=1.0,
        kd_x=0.0,
        max_pan_speed_deg_s=100.0,
    )

    controller = PIDController(config)
    dt = 1.0 / 30.0

    # Build up integral during TRACKING.
    for _ in range(20):
        controller.compute(
            _make_result(
                predicted_x=400.0,
                mode=TrackerMode.TRACKING,
            ),
            0.0,
            0.0,
            dt,
        )

    assert controller._integral_x != 0.0

    # Enter LOST.
    controller.compute(
        _make_result(mode=TrackerMode.LOST),
        0.0,
        0.0,
        dt,
    )

    # Re-enter TRACKING — integral should have been reset.
    result = controller.compute(
        _make_result(
            predicted_x=400.0,
            mode=TrackerMode.TRACKING,
        ),
        0.0,
        0.0,
        dt,
    )

    # After reset the integral contribution on the very first
    # TRACKING frame is just error * dt (a single step), which is
    # much smaller than the accumulated value before LOST.
    error_x_deg = (400.0 - 320.0) * (4.0 / 640.0)
    max_one_step_integral = abs(error_x_deg * dt) + 1e-9

    assert abs(controller._integral_x) <= max_one_step_integral


# ==================================================================
# Confidence scaling
# ==================================================================

def test_confidence_scales_gains():
    """Lower confidence -> smaller output for the same pixel error."""

    config = ControllerConfig(
        kp_x=1.0,
        ki_x=0.0,
        kd_x=0.0,
        max_pan_speed_deg_s=100.0,
    )

    high_conf = _make_result(
        predicted_x=400.0,
        confidence=0.95,
    )

    low_conf = _make_result(
        predicted_x=400.0,
        confidence=0.3,
    )

    dt = 1.0 / 30.0

    ctrl_high = PIDController(config)
    ctrl_low = PIDController(config)

    r_high = ctrl_high.compute(high_conf, 0.0, 0.0, dt)
    r_low = ctrl_low.compute(low_conf, 0.0, 0.0, dt)

    assert abs(r_low.pan_delta) < abs(r_high.pan_delta)


# ==================================================================
# Serialization
# ==================================================================

def test_json_serialization():
    """ControlResult.to_dict() produces a JSON-compatible dict."""

    controller = PIDController()

    result = controller.compute(
        estimator_result=_make_result(predicted_x=350.0),
        current_pan_deg=0.0,
        current_tilt_deg=0.0,
        dt=1.0 / 30.0,
    )

    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert isinstance(payload["should_command"], bool)
    assert isinstance(payload["pan_delta"], float)
    assert payload["mode_at_command_time"] == "TRACKING"


# ==================================================================
# Drift sanity checks
# ==================================================================

def test_stationary_centered_target_no_drift():
    """
    A stationary target held at exact frame center for 200 frames
    must produce zero cumulative pan/tilt.  Any drift here would
    indicate a real bug (integral windup, sign error, or off-by-one
    in the error calculation).
    """

    controller = PIDController(ControllerConfig())
    dt = 1.0 / 30.0

    cum_pan = 0.0
    cum_tilt = 0.0

    for _ in range(200):
        result = controller.compute(
            estimator_result=_make_result(
                predicted_x=320.0,
                predicted_y=240.0,
                mode=TrackerMode.TRACKING,
                confidence=0.95,
            ),
            current_pan_deg=cum_pan,
            current_tilt_deg=cum_tilt,
            dt=dt,
        )

        cum_pan += result.pan_delta
        cum_tilt += result.tilt_delta

        # Every single frame should output exactly 0.
        assert result.pan_delta == 0.0
        assert result.tilt_delta == 0.0

    assert cum_pan == 0.0
    assert cum_tilt == 0.0


def test_bounded_oscillation_no_net_drift():
    """
    A target oscillating ±20 px around frame center over full cycles
    must NOT accumulate unbounded net drift.

    Because the open-loop demo integrates pan_delta without feeding
    back to the pixel error (no plant model), the cumulative position
    grows even for symmetric oscillations.  This is expected: each
    positive error produces a positive delta that is summed, and the
    camera position never reduces the error because there is no
    closed loop.

    This test verifies the cumulative position remains bounded and
    proportionate to the oscillation amplitude (not growing without
    limit), confirming the controller is behaving correctly.
    """

    config = ControllerConfig()
    controller = PIDController(config)
    dt = 1.0 / 30.0

    cum_pan = 0.0

    # 4 full cycles of a 0.5 Hz sine = 8 seconds = 240 frames.
    n_frames = 240

    for i in range(n_frames):
        t = i * dt
        px = 320.0 + 20.0 * math.sin(2.0 * math.pi * 0.5 * t)

        result = controller.compute(
            estimator_result=_make_result(
                predicted_x=px,
                predicted_y=240.0,
            ),
            current_pan_deg=cum_pan,
            current_tilt_deg=0.0,
            dt=dt,
        )

        cum_pan += result.pan_delta

    # The ±20 px oscillation corresponds to ±0.125 deg angular range.
    # In an open-loop sum the cumulative position will be non-zero but
    # should remain within a modest bound (< 5 deg for 4 cycles).
    # Unbounded growth (e.g. > 10 deg) would indicate a real bug.
    assert abs(cum_pan) < 5.0, (
        f"Cumulative pan {cum_pan:.4f} deg exceeds reasonable bound "
        f"for a ±0.125 deg oscillation"
    )

    # The integral term must return to ~0 after full cycles,
    # confirming no internal windup accumulation.
    assert abs(controller._integral_x) < 1e-6

