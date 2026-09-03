from dataclasses import dataclass


@dataclass
class EstimatorConfig:
    """
    Configuration for the FSOC beacon state estimator.

    Units:
        position     -> pixels
        velocity     -> pixels / second
        acceleration -> pixels / second^2
        time         -> seconds
    """

    # Process model
    # White-noise jerk intensity used by the constant-acceleration model.
    process_noise: float = 20.0

    # Base detector measurement variance in pixels^2.
    measurement_noise: float = 150.0

    # Initial covariance values
    initial_position_covariance: float = 100.0
    initial_velocity_covariance: float = 1000.0
    initial_acceleration_covariance: float = 1000.0

    # Maximum time for prediction-only tracking before declaring LOST.
    max_coast_time: float = 1.0

    # Detector confidence handling
    min_confidence: float = 0.05
    confidence_noise_scaling: float = 2.0

    # Mahalanobis squared-distance threshold.
    # 9.21 corresponds approximately to a 99% chi-square threshold
    # for a 2-dimensional measurement.
    gating_threshold: float = 9.21

    # Timestamp protection
    min_dt: float = 1e-3
    max_dt: float = 0.25

    # Default FSOC virtual camera resolution
    frame_width: int = 640
    frame_height: int = 480

    # Number of accepted measurements before strict gating starts.
    gating_warmup_measurements: int = 3

    # Prediction horizon used for predicted_x / predicted_y.
    # ~33.3 ms corresponds to approximately one frame at 30 FPS.
    prediction_horizon: float = 1.0 / 30.0