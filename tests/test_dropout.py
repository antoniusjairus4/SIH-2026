from src.estimation import (
    BeaconStateEstimator,
    TrackerMode,
)


def build_tracking_estimator():
    estimator = BeaconStateEstimator()

    estimator.step(300, 200, 0.95, 0.0)
    estimator.step(301, 201, 0.95, 0.1)
    estimator.step(302, 202, 0.95, 0.2)

    return estimator


def test_short_dropout_enters_coasting():
    estimator = build_tracking_estimator()

    result = estimator.step(
        x=None,
        y=None,
        confidence=0.0,
        timestamp=0.3,
    )

    assert result.mode == TrackerMode.COASTING
    assert result.prediction_only is True
    assert result.missing_frames == 1
    assert result.coast_time > 0.0


def test_short_dropout_recovers():
    estimator = build_tracking_estimator()

    result = estimator.step(
        None,
        None,
        0.0,
        0.3,
    )

    assert result.mode == TrackerMode.COASTING

    result = estimator.step(
        304,
        204,
        0.95,
        0.4,
    )

    assert result.mode == TrackerMode.TRACKING
    assert result.missing_frames == 0
    assert result.coast_time == 0.0
    assert result.prediction_only is False


def test_long_dropout_enters_lost():
    estimator = build_tracking_estimator()

    timestamp = 0.3

    result = None

    for _ in range(15):
        result = estimator.step(
            None,
            None,
            0.0,
            timestamp,
        )

        timestamp += 0.1

    assert result.mode == TrackerMode.LOST
    assert result.confidence == 0.0


def test_lost_can_recover_from_valid_detection():
    estimator = build_tracking_estimator()

    timestamp = 0.3

    for _ in range(15):
        result = estimator.step(
            None,
            None,
            0.0,
            timestamp,
        )

        timestamp += 0.1

    assert result.mode == TrackerMode.LOST

    recovery = estimator.step(
        450,
        300,
        0.95,
        timestamp,
    )

    assert recovery.mode == TrackerMode.TRACKING
    assert recovery.missing_frames == 0
    assert recovery.coast_time == 0.0

    assert abs(recovery.x - 450.0) < 1e-6
    assert abs(recovery.y - 300.0) < 1e-6