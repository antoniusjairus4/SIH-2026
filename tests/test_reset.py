import numpy as np

from python.estimation import (
    BeaconStateEstimator,
    TrackerMode,
)


def test_reset_returns_to_uninitialized():
    estimator = BeaconStateEstimator()

    estimator.step(
        320,
        240,
        0.95,
        0.0,
    )

    estimator.step(
        325,
        245,
        0.95,
        0.033,
    )

    estimator.reset()

    result = estimator.step(
        None,
        None,
        0.0,
        1.0,
    )

    assert result.mode == TrackerMode.UNINITIALIZED
    assert result.x is None
    assert result.y is None
    assert result.confidence == 0.0


def test_covariance_is_finite_and_symmetric():
    estimator = BeaconStateEstimator()

    timestamp = 0.0

    for i in range(100):
        estimator.step(
            200 + i * 0.5,
            150 + i * 0.25,
            0.9,
            timestamp,
        )

        timestamp += 1.0 / 30.0

    covariance = estimator._P

    assert np.all(np.isfinite(covariance))

    assert np.allclose(
        covariance,
        covariance.T,
        atol=1e-10,
    )


def test_invalid_confidence_does_not_crash():
    estimator = BeaconStateEstimator()

    result = estimator.step(
        320,
        240,
        float("nan"),
        0.0,
    )

    assert result.mode == TrackerMode.UNINITIALIZED

    result = estimator.step(
        320,
        240,
        -10,
        0.033,
    )

    assert result.mode == TrackerMode.UNINITIALIZED

    result = estimator.step(
        320,
        240,
        5.0,
        0.066,
    )

    assert result.mode == TrackerMode.TRACKING