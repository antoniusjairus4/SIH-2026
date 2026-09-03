from python.estimation import (
    BeaconStateEstimator,
    TrackerMode,
)


def test_large_outlier_is_rejected():
    estimator = BeaconStateEstimator()

    dt = 1.0 / 30.0

    timestamp = 0.0

    # Establish a stable track.
    for i in range(10):
        estimator.step(
            x=300 + i,
            y=200,
            confidence=0.95,
            timestamp=timestamp,
        )

        timestamp += dt

    before = estimator.step(
        310,
        200,
        0.95,
        timestamp,
    )

    timestamp += dt

    outlier = estimator.step(
        x=1500,
        y=-800,
        confidence=0.95,
        timestamp=timestamp,
    )

    assert outlier.measurement_available is True
    assert outlier.measurement_rejected is True

    assert outlier.prediction_only is True
    assert outlier.mode == TrackerMode.COASTING

    # The filter must not jump anywhere near the false observation.
    assert abs(outlier.x - before.x) < 100
    assert abs(outlier.y - before.y) < 100


def test_tracking_resumes_after_outlier():
    estimator = BeaconStateEstimator()

    timestamp = 0.0
    dt = 1.0 / 30.0

    for i in range(10):
        estimator.step(
            300 + i,
            200,
            0.95,
            timestamp,
        )

        timestamp += dt

    rejected = estimator.step(
        1400,
        -700,
        0.99,
        timestamp,
    )

    assert rejected.measurement_rejected is True

    timestamp += dt

    recovered = estimator.step(
        311,
        200,
        0.95,
        timestamp,
    )

    assert recovered.measurement_rejected is False
    assert recovered.mode == TrackerMode.TRACKING