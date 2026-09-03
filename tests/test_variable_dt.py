import math

from src.estimation import (
    BeaconStateEstimator,
    TrackerMode,
)


def assert_finite_result(result):
    for value in (
        result.x,
        result.y,
        result.vx,
        result.vy,
        result.ax,
        result.ay,
        result.predicted_x,
        result.predicted_y,
        result.confidence,
    ):
        if value is not None:
            assert math.isfinite(value)


def test_variable_frame_intervals():
    estimator = BeaconStateEstimator()

    timestamps = [
        0.000,
        0.033,
        0.069,
        0.097,
        0.138,
        0.170,
    ]

    for index, timestamp in enumerate(timestamps):
        result = estimator.step(
            300 + index,
            200,
            0.95,
            timestamp,
        )

        assert_finite_result(result)

    assert result.mode == TrackerMode.TRACKING


def test_repeated_timestamp():
    estimator = BeaconStateEstimator()

    estimator.step(
        300,
        200,
        0.95,
        1.0,
    )

    result = estimator.step(
        301,
        200,
        0.95,
        1.0,
    )

    assert_finite_result(result)


def test_backward_timestamp():
    estimator = BeaconStateEstimator()

    estimator.step(
        300,
        200,
        0.95,
        10.0,
    )

    result = estimator.step(
        301,
        200,
        0.95,
        9.0,
    )

    assert_finite_result(result)


def test_large_timestamp_gap():
    estimator = BeaconStateEstimator()

    estimator.step(
        300,
        200,
        0.95,
        0.0,
    )

    result = estimator.step(
        301,
        200,
        0.95,
        100.0,
    )

    assert_finite_result(result)


def test_nan_and_inf_measurements():
    estimator = BeaconStateEstimator()

    estimator.step(
        300,
        200,
        0.95,
        0.0,
    )

    result_nan = estimator.step(
        float("nan"),
        200,
        0.95,
        0.033,
    )

    assert_finite_result(result_nan)

    result_inf = estimator.step(
        float("inf"),
        200,
        0.95,
        0.066,
    )

    assert_finite_result(result_inf)