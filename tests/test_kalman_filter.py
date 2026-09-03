import json
import math

from src.estimation import (
    BeaconStateEstimator,
    EstimatorConfig,
    TrackerMode,
)


def test_initialization():
    estimator = BeaconStateEstimator()

    result = estimator.step(
        x=320,
        y=240,
        confidence=0.95,
        timestamp=0.0,
    )

    assert result.mode == TrackerMode.TRACKING
    assert result.x == 320.0
    assert result.y == 240.0

    assert result.vx == 0.0
    assert result.vy == 0.0

    assert result.ax == 0.0
    assert result.ay == 0.0

    assert result.missing_frames == 0
    assert result.prediction_only is False
    assert result.measurement_available is True
    assert result.inside_fov is True


def test_uninitialized_without_detection():
    estimator = BeaconStateEstimator()

    result = estimator.step(
        x=None,
        y=None,
        confidence=0.0,
        timestamp=0.0,
    )

    assert result.mode == TrackerMode.UNINITIALIZED
    assert result.x is None
    assert result.y is None
    assert result.confidence == 0.0


def test_stationary_target_remains_stable():
    estimator = BeaconStateEstimator()

    timestamp = 0.0

    for _ in range(50):
        result = estimator.step(
            x=320.0,
            y=240.0,
            confidence=0.95,
            timestamp=timestamp,
        )

        timestamp += 1.0 / 30.0

    assert abs(result.x - 320.0) < 1.0
    assert abs(result.y - 240.0) < 1.0

    assert abs(result.vx) < 1.0
    assert abs(result.vy) < 1.0


def test_constant_velocity_motion():
    estimator = BeaconStateEstimator()

    dt = 1.0 / 30.0

    x = 100.0
    y = 150.0

    vx_true = 30.0
    vy_true = 15.0

    timestamp = 0.0

    for _ in range(120):
        result = estimator.step(
            x=x,
            y=y,
            confidence=0.95,
            timestamp=timestamp,
        )

        timestamp += dt
        x += vx_true * dt
        y += vy_true * dt

    assert abs(result.vx - vx_true) < 5.0
    assert abs(result.vy - vy_true) < 5.0

    assert result.mode == TrackerMode.TRACKING


def test_accelerating_target():
    estimator = BeaconStateEstimator()

    dt = 1.0 / 30.0

    x = 100.0
    y = 100.0

    vx = 5.0
    vy = 3.0

    ax = 8.0
    ay = 4.0

    timestamp = 0.0

    for _ in range(180):
        result = estimator.step(
            x=x,
            y=y,
            confidence=0.98,
            timestamp=timestamp,
        )

        x = x + vx * dt + 0.5 * ax * dt**2
        y = y + vy * dt + 0.5 * ay * dt**2

        vx += ax * dt
        vy += ay * dt

        timestamp += dt

    assert math.isfinite(result.ax)
    assert math.isfinite(result.ay)

    assert result.mode == TrackerMode.TRACKING


def test_inside_fov_is_python_bool():
    estimator = BeaconStateEstimator()

    result = estimator.step(
        320,
        240,
        0.95,
        0.0,
    )

    assert isinstance(result.inside_fov, bool)
    assert result.inside_fov is True


def test_outside_fov_not_clamped():
    estimator = BeaconStateEstimator()

    result = estimator.step(
        700,
        500,
        0.95,
        0.0,
    )

    assert result.x == 700.0
    assert result.y == 500.0
    assert result.inside_fov is False


def test_json_serialization():
    estimator = BeaconStateEstimator()

    result = estimator.step(
        320,
        240,
        0.95,
        0.0,
    )

    payload = result.to_dict()

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert isinstance(payload["inside_fov"], bool)
    assert isinstance(payload["confidence"], float)
    assert isinstance(payload["missing_frames"], int)
    assert payload["mode"] == "TRACKING"


def test_confidence_is_bounded():
    estimator = BeaconStateEstimator()

    result = estimator.step(
        320,
        240,
        5.0,
        0.0,
    )

    assert 0.0 <= result.confidence <= 1.0


def test_low_and_high_detector_confidence():
    config = EstimatorConfig(
        gating_warmup_measurements=100
    )

    high = BeaconStateEstimator(config)
    low = BeaconStateEstimator(config)

    # Same starting state
    for estimator in (high, low):
        estimator.step(100, 100, 1.0, 0.0)
        estimator.step(100, 100, 1.0, 0.033)

    high_result = high.step(
        120,
        100,
        0.95,
        0.066,
    )

    low_result = low.step(
        120,
        100,
        0.25,
        0.066,
    )

    # Higher-confidence detection should pull estimate farther
    # toward x=120.
    assert high_result.x > low_result.x