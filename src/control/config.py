from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """
    Configuration for the FSOC PID pan/tilt controller.

    Units:
        angles       -> degrees
        angular rate -> degrees / second
        pixel coords -> pixels
        time         -> seconds

    # TODO: All PID gains below are initial starting points and need
    # tuning against real motion profiles (straight-line, circular,
    # figure-8, random) once integrated with the full tracking
    # pipeline.  Follow the grid-search methodology documented in
    # the Kalman filter walkthrough.md for systematic optimization.
    """

    # -- Per-axis PID gains ------------------------------------------------
    # Pan (horizontal / x-axis) gains.
    kp_x: float = 0.5
    ki_x: float = 0.05
    kd_x: float = 0.1

    # Tilt (vertical / y-axis) gains.
    kp_y: float = 0.5
    ki_y: float = 0.05
    kd_y: float = 0.1

    # -- Actuator speed limits ---------------------------------------------
    # Maximum commanded angular rate.
    # Spec allows 5-10 deg/s; default to conservative end.
    max_pan_speed_deg_s: float = 5.0
    max_tilt_speed_deg_s: float = 5.0

    # -- Camera optics -----------------------------------------------------
    # Field of view used to convert pixel error to angular error.
    camera_fov_x_deg: float = 4.0
    camera_fov_y_deg: float = 3.0

    # Frame resolution — must match EstimatorConfig.frame_width/height.
    frame_width: int = 640
    frame_height: int = 480

    # -- Anti-windup -------------------------------------------------------
    # Maximum absolute value the integral term may accumulate (degrees).
    integral_clamp: float = 2.0

    # -- Mode-aware gain scaling -------------------------------------------
    # Multiplicative factor applied to all gains during COASTING mode.
    coasting_gain_scale: float = 0.5

    # Minimum confidence multiplier.  Gains are scaled by
    # max(confidence, min_confidence_floor) so they never fully
    # zero out, even under very low estimator confidence.
    min_confidence_floor: float = 0.2
