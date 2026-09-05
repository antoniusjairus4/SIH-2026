from __future__ import annotations

import math
from typing import Optional

from src.estimation.state import EstimatorResult, TrackerMode

from .config import ControllerConfig
from .state import ControlResult


class PIDController:
    """
    PID controller for FSOC beacon pan/tilt tracking.

    Converts pixel-domain tracking error from the BeaconStateEstimator
    into angular pan/tilt delta commands, respecting tracker mode,
    estimator confidence, and coasting state.

    The controller uses ``estimator_result.predicted_x / predicted_y``
    (which already include lead compensation from the estimator's
    velocity + acceleration model) as the target position, and
    computes the error relative to frame center.

    # TODO: PID gains in ControllerConfig are initial starting points.
    # Tune against real motion profiles (straight-line, circular,
    # figure-8, random) using a systematic grid-search, matching the
    # approach documented in the Kalman filter walkthrough.md.
    """

    # Modes under which the controller should NOT issue commands.
    _NO_COMMAND_MODES = frozenset({
        TrackerMode.LOST,
        TrackerMode.UNINITIALIZED,
    })

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
    ) -> None:

        self.config = config or ControllerConfig()

        self.reset()

    def reset(self) -> None:
        """Clear all internal integral and previous-error state."""

        self._integral_x: float = 0.0
        self._integral_y: float = 0.0

        self._prev_error_x: float = 0.0
        self._prev_error_y: float = 0.0

        self._prev_mode: TrackerMode = TrackerMode.UNINITIALIZED

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _finite_number(value: object) -> bool:
        """Return True if *value* is a finite float-convertible number."""
        try:
            return bool(math.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------

    def compute(
        self,
        estimator_result: EstimatorResult,
        current_pan_deg: float,
        current_tilt_deg: float,
        dt: float,
    ) -> ControlResult:
        """
        Compute a single PID control step.

        Parameters
        ----------
        estimator_result:
            Full output from ``BeaconStateEstimator.step()``.
        current_pan_deg:
            Current pan angle of the camera (degrees).
        current_tilt_deg:
            Current tilt angle of the camera (degrees).
        dt:
            Time elapsed since the previous control step (seconds).

        Returns
        -------
        ControlResult
            Angular delta commands and metadata.
        """

        mode = estimator_result.mode

        # -- 1. No-command modes ----------------------------------------
        if mode in self._NO_COMMAND_MODES:
            self._prev_mode = mode
            return self._no_command_result(mode)

        # -- 2. Integral reset on mode transition -----------------------
        #    Prevents windup-driven overshoot on reacquisition.
        if self._prev_mode in self._NO_COMMAND_MODES:
            self._integral_x = 0.0
            self._integral_y = 0.0
            self._prev_error_x = 0.0
            self._prev_error_y = 0.0

        # -- 3. Guard: predicted position must be available -------------
        if (
            estimator_result.predicted_x is None
            or estimator_result.predicted_y is None
            or not self._finite_number(estimator_result.predicted_x)
            or not self._finite_number(estimator_result.predicted_y)
        ):
            self._prev_mode = mode
            return self._no_command_result(mode)

        # -- 4. Compute pixel error from frame center ------------------
        cfg = self.config

        setpoint_x = cfg.frame_width / 2.0
        setpoint_y = cfg.frame_height / 2.0

        error_x_px = float(estimator_result.predicted_x) - setpoint_x
        error_y_px = float(estimator_result.predicted_y) - setpoint_y

        # -- 5. Convert pixel error to angular error -------------------
        deg_per_px_x = cfg.camera_fov_x_deg / cfg.frame_width
        deg_per_px_y = cfg.camera_fov_y_deg / cfg.frame_height

        error_x_deg = error_x_px * deg_per_px_x
        error_y_deg = error_y_px * deg_per_px_y

        # -- 6. Effective gain scaling ---------------------------------
        confidence = float(estimator_result.confidence)
        confidence_scale = max(confidence, cfg.min_confidence_floor)

        is_coasting = mode == TrackerMode.COASTING

        mode_scale = (
            cfg.coasting_gain_scale if is_coasting else 1.0
        )

        effective_scale = confidence_scale * mode_scale

        kp_x = cfg.kp_x * effective_scale
        ki_x = cfg.ki_x * effective_scale
        kd_x = cfg.kd_x * effective_scale

        kp_y = cfg.kp_y * effective_scale
        ki_y = cfg.ki_y * effective_scale
        kd_y = cfg.kd_y * effective_scale

        # -- 7. PID computation — pan (x-axis) -------------------------
        safe_dt = dt if self._finite_number(dt) and dt > 0.0 else 0.0

        p_x = kp_x * error_x_deg

        # Integral: freeze during COASTING to prevent windup.
        if not is_coasting and safe_dt > 0.0:
            self._integral_x += error_x_deg * safe_dt
            self._integral_x = float(
                max(
                    -cfg.integral_clamp,
                    min(cfg.integral_clamp, self._integral_x),
                )
            )

        i_x = ki_x * self._integral_x

        if safe_dt > 0.0:
            d_x = kd_x * (
                (error_x_deg - self._prev_error_x) / safe_dt
            )
        else:
            d_x = 0.0

        pan_raw = p_x + i_x + d_x

        # -- 8. PID computation — tilt (y-axis) ------------------------
        p_y = kp_y * error_y_deg

        if not is_coasting and safe_dt > 0.0:
            self._integral_y += error_y_deg * safe_dt
            self._integral_y = float(
                max(
                    -cfg.integral_clamp,
                    min(cfg.integral_clamp, self._integral_y),
                )
            )

        i_y = ki_y * self._integral_y

        if safe_dt > 0.0:
            d_y = kd_y * (
                (error_y_deg - self._prev_error_y) / safe_dt
            )
        else:
            d_y = 0.0

        tilt_raw = p_y + i_y + d_y

        # -- 9. Clamp to max actuator speed ----------------------------
        max_pan_delta = cfg.max_pan_speed_deg_s * max(safe_dt, 0.0)
        max_tilt_delta = cfg.max_tilt_speed_deg_s * max(safe_dt, 0.0)

        pan_delta = float(
            max(-max_pan_delta, min(max_pan_delta, pan_raw))
        )
        tilt_delta = float(
            max(-max_tilt_delta, min(max_tilt_delta, tilt_raw))
        )

        # -- 10. Update internal state ---------------------------------
        self._prev_error_x = error_x_deg
        self._prev_error_y = error_y_deg
        self._prev_mode = mode

        return ControlResult(
            pan_delta=pan_delta,
            tilt_delta=tilt_delta,
            should_command=True,
            mode_at_command_time=mode,
            raw_error_x_deg=error_x_deg,
            raw_error_y_deg=error_y_deg,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _no_command_result(mode: TrackerMode) -> ControlResult:
        """Return a zero-output result indicating no command."""

        return ControlResult(
            pan_delta=0.0,
            tilt_delta=0.0,
            should_command=False,
            mode_at_command_time=mode,
            raw_error_x_deg=0.0,
            raw_error_y_deg=0.0,
        )
