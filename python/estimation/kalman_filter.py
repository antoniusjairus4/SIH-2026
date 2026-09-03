from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .config import EstimatorConfig
from .state import EstimatorResult, TrackerMode


class BeaconStateEstimator:
    """
    State estimator for the FSOC beacon.

    State vector:
        [x, y, vx, vy, ax, ay]^T

    Measurement:
        [x, y]^T

    The estimator uses a discrete linear constant-acceleration
    Kalman filter driven by white-noise jerk.
    """

    STATE_SIZE = 6
    MEASUREMENT_SIZE = 2

    def __init__(
        self,
        config: Optional[EstimatorConfig] = None,
    ) -> None:

        self.config = config or EstimatorConfig()

        self._H = np.zeros(
            (self.MEASUREMENT_SIZE, self.STATE_SIZE),
            dtype=float,
        )

        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0

        self._I = np.eye(self.STATE_SIZE, dtype=float)

        self.reset()

    def reset(self) -> None:
        """Reset estimator to its initial uninitialized state."""

        self._state = np.zeros((self.STATE_SIZE, 1), dtype=float)

        self._P = self._initial_covariance()

        self._last_timestamp: Optional[float] = None

        self._mode = TrackerMode.UNINITIALIZED

        self._missing_frames = 0
        self._coast_time = 0.0

        self._accepted_measurements = 0
        self._last_detector_confidence = 0.0

    def _initial_covariance(self) -> np.ndarray:
        cfg = self.config

        return np.diag(
            [
                cfg.initial_position_covariance,
                cfg.initial_position_covariance,
                cfg.initial_velocity_covariance,
                cfg.initial_velocity_covariance,
                cfg.initial_acceleration_covariance,
                cfg.initial_acceleration_covariance,
            ]
        ).astype(float)

    @staticmethod
    def _finite_number(value: object) -> bool:
        try:
            return bool(math.isfinite(float(value)))
        except (TypeError, ValueError, OverflowError):
            return False

    def _sanitize_confidence(self, confidence: object) -> float:
        if not self._finite_number(confidence):
            return 0.0

        return float(np.clip(float(confidence), 0.0, 1.0))

    def _valid_measurement(
        self,
        x: object,
        y: object,
        confidence: float,
    ) -> bool:

        if x is None or y is None:
            return False

        if not self._finite_number(x) or not self._finite_number(y):
            return False

        return confidence >= self.config.min_confidence

    def _safe_timestamp(self, timestamp: object) -> float:
        if not self._finite_number(timestamp):
            if self._last_timestamp is not None:
                return float(self._last_timestamp)

            return 0.0

        return float(timestamp)

    def _compute_dt(self, timestamp: float) -> float:
        """
        Compute a safe time interval.

        A backward timestamp never replaces the stored valid timestamp.
        """

        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            return 0.0

        raw_dt = timestamp - self._last_timestamp

        if raw_dt < 0.0:
            # Do not corrupt timestamp history with backward time.
            return self.config.min_dt

        if raw_dt == 0.0:
            return self.config.min_dt

        self._last_timestamp = timestamp

        return float(
            np.clip(
                raw_dt,
                self.config.min_dt,
                self.config.max_dt,
            )
        )

    def _build_F(self, dt: float) -> np.ndarray:
        dt2 = dt * dt

        return np.array(
            [
                [1.0, 0.0, dt, 0.0, 0.5 * dt2, 0.0],
                [0.0, 1.0, 0.0, dt, 0.0, 0.5 * dt2],
                [0.0, 0.0, 1.0, 0.0, dt, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def _build_Q(self, dt: float) -> np.ndarray:
        """
        Process covariance for a constant-acceleration model driven by
        continuous white-noise jerk.

        A single-axis state is [position, velocity, acceleration].
        """

        q = self.config.process_noise

        dt2 = dt**2
        dt3 = dt**3
        dt4 = dt**4
        dt5 = dt**5

        q_axis = q * np.array(
            [
                [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
                [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
                [dt3 / 6.0, dt2 / 2.0, dt],
            ],
            dtype=float,
        )

        Q = np.zeros((6, 6), dtype=float)

        x_indices = [0, 2, 4]
        y_indices = [1, 3, 5]

        Q[np.ix_(x_indices, x_indices)] = q_axis
        Q[np.ix_(y_indices, y_indices)] = q_axis

        return Q

    def _predict(self, dt: float) -> None:
        if dt <= 0.0:
            return

        F = self._build_F(dt)
        Q = self._build_Q(dt)

        self._state = F @ self._state
        self._P = F @ self._P @ F.T + Q

        # Protect against tiny numerical asymmetry.
        self._P = 0.5 * (self._P + self._P.T)

    def _measurement_covariance(
        self,
        confidence: float,
    ) -> np.ndarray:

        cfg = self.config

        safe_confidence = max(
            confidence,
            cfg.min_confidence,
        )

        scale = safe_confidence ** cfg.confidence_noise_scaling

        variance = cfg.measurement_noise / max(scale, 1e-12)

        return np.eye(2, dtype=float) * variance

    def _innovation(
        self,
        measurement: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        residual = measurement - self._H @ self._state

        S = self._H @ self._P @ self._H.T + R

        return residual, S

    def _mahalanobis_distance_squared(
        self,
        residual: np.ndarray,
        S: np.ndarray,
    ) -> float:

        try:
            solved = np.linalg.solve(S, residual)
            value = float((residual.T @ solved).item())
        except np.linalg.LinAlgError:
            return float("inf")

        return value

    def _should_reject_measurement(
        self,
        measurement: np.ndarray,
        R: np.ndarray,
    ) -> bool:

        # Warm-up avoids over-gating immediately after initialization.
        if (
            self._accepted_measurements
            < self.config.gating_warmup_measurements
        ):
            return False

        residual, S = self._innovation(measurement, R)

        distance_squared = self._mahalanobis_distance_squared(
            residual,
            S,
        )

        return distance_squared > self.config.gating_threshold

    def _correct(
        self,
        measurement: np.ndarray,
        R: np.ndarray,
    ) -> None:

        residual, S = self._innovation(measurement, R)

        try:
            PHt = self._P @ self._H.T

            # Equivalent to PHt @ inv(S), but avoids explicit inversion.
            K = np.linalg.solve(S.T, PHt.T).T

        except np.linalg.LinAlgError:
            return

        self._state = self._state + K @ residual

        I_KH = self._I - K @ self._H

        # Joseph covariance update.
        self._P = (
            I_KH @ self._P @ I_KH.T
            + K @ R @ K.T
        )

        self._P = 0.5 * (self._P + self._P.T)

    def _initialize(
        self,
        x: float,
        y: float,
        confidence: float,
    ) -> None:

        self._state = np.array(
            [
                [x],
                [y],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ],
            dtype=float,
        )

        self._P = self._initial_covariance()

        self._mode = TrackerMode.TRACKING

        self._missing_frames = 0
        self._coast_time = 0.0

        self._accepted_measurements = 1
        self._last_detector_confidence = confidence

    def _recover_from_lost(
        self,
        x: float,
        y: float,
        confidence: float,
    ) -> bool:
        """
        Recover from LOST using an already supplied detector measurement.

        This is NOT target search/reacquisition logic.

        A sufficiently credible measurement re-centres the estimator
        around the returned detector observation.
        """

        if confidence < self.config.min_confidence:
            return False

        self._initialize(x, y, confidence)

        return True

    def _estimate_confidence(
        self,
        accepted_measurement: bool,
        detector_confidence: float,
    ) -> float:

        if self._mode == TrackerMode.UNINITIALIZED:
            return 0.0

        if self._mode == TrackerMode.LOST:
            return 0.0

        position_uncertainty = max(
            float(self._P[0, 0]),
            float(self._P[1, 1]),
            0.0,
        )

        covariance_score = 1.0 / (
            1.0 + position_uncertainty / 100.0
        )

        if accepted_measurement:
            score = (
                0.7 * detector_confidence
                + 0.3 * covariance_score
            )

        else:
            if self.config.max_coast_time <= 0:
                coast_factor = 0.0
            else:
                coast_factor = max(
                    0.0,
                    1.0
                    - self._coast_time
                    / self.config.max_coast_time,
                )

            score = (
                0.5 * self._last_detector_confidence
                + 0.5 * covariance_score
            ) * coast_factor

        return float(np.clip(score, 0.0, 1.0))

    def _future_position(self) -> tuple[float, float]:
        horizon = max(
            0.0,
            float(self.config.prediction_horizon),
        )

        x, y, vx, vy, ax, ay = self._state[:, 0]

        predicted_x = (
            x
            + vx * horizon
            + 0.5 * ax * horizon * horizon
        )

        predicted_y = (
            y
            + vy * horizon
            + 0.5 * ay * horizon * horizon
        )

        return float(predicted_x), float(predicted_y)

    def _inside_fov(self, x: float, y: float) -> bool:
        return bool(
            0.0 <= x < float(self.config.frame_width)
            and
            0.0 <= y < float(self.config.frame_height)
        )

    def _empty_result(
        self,
        timestamp: float,
        measurement_available: bool,
    ) -> EstimatorResult:

        return EstimatorResult(
            x=None,
            y=None,
            vx=None,
            vy=None,
            ax=None,
            ay=None,
            predicted_x=None,
            predicted_y=None,
            mode=TrackerMode.UNINITIALIZED,
            prediction_only=False,
            measurement_available=measurement_available,
            measurement_rejected=False,
            confidence=0.0,
            missing_frames=0,
            coast_time=0.0,
            inside_fov=False,
            timestamp=float(timestamp),
        )

    def step(
        self,
        x: Optional[float],
        y: Optional[float],
        confidence: float,
        timestamp: float,
    ) -> EstimatorResult:
        """
        Process one detector observation.

        Only detector-style information enters this method:
            x, y, confidence, timestamp
        """

        safe_timestamp = self._safe_timestamp(timestamp)
        safe_confidence = self._sanitize_confidence(confidence)

        measurement_available = self._valid_measurement(
            x,
            y,
            safe_confidence,
        )

        # Before initialization, ignore invalid/missing detections.
        if self._mode == TrackerMode.UNINITIALIZED:
            if not measurement_available:
                return self._empty_result(
                    safe_timestamp,
                    measurement_available=False,
                )

            self._last_timestamp = safe_timestamp

            self._initialize(
                float(x),
                float(y),
                safe_confidence,
            )

            return self._build_result(
                timestamp=safe_timestamp,
                measurement_available=True,
                measurement_rejected=False,
                prediction_only=False,
                accepted_measurement=True,
                detector_confidence=safe_confidence,
            )

        dt = self._compute_dt(safe_timestamp)

        self._predict(dt)

        measurement_rejected = False
        accepted_measurement = False

        if measurement_available:

            measurement = np.array(
                [[float(x)], [float(y)]],
                dtype=float,
            )

            # If already LOST, allow recovery from an incoming credible
            # detector measurement instead of indefinitely gating against
            # a stale prediction.
            if self._mode == TrackerMode.LOST:
                accepted_measurement = self._recover_from_lost(
                    float(x),
                    float(y),
                    safe_confidence,
                )

                if accepted_measurement:
                    self._last_timestamp = safe_timestamp

            else:
                R = self._measurement_covariance(
                    safe_confidence
                )

                measurement_rejected = (
                    self._should_reject_measurement(
                        measurement,
                        R,
                    )
                )

                if not measurement_rejected:
                    self._correct(measurement, R)

                    accepted_measurement = True

                    self._accepted_measurements += 1

                    self._last_detector_confidence = (
                        safe_confidence
                    )

        if accepted_measurement:
            self._mode = TrackerMode.TRACKING
            self._missing_frames = 0
            self._coast_time = 0.0

        else:
            self._missing_frames += 1
            self._coast_time += max(dt, 0.0)

            if self._coast_time > self.config.max_coast_time:
                self._mode = TrackerMode.LOST
            else:
                self._mode = TrackerMode.COASTING

        return self._build_result(
            timestamp=safe_timestamp,
            measurement_available=measurement_available,
            measurement_rejected=measurement_rejected,
            prediction_only=not accepted_measurement,
            accepted_measurement=accepted_measurement,
            detector_confidence=safe_confidence,
        )

    def _build_result(
        self,
        timestamp: float,
        measurement_available: bool,
        measurement_rejected: bool,
        prediction_only: bool,
        accepted_measurement: bool,
        detector_confidence: float,
    ) -> EstimatorResult:

        x, y, vx, vy, ax, ay = [
            float(value)
            for value in self._state[:, 0]
        ]

        predicted_x, predicted_y = self._future_position()

        estimator_confidence = self._estimate_confidence(
            accepted_measurement,
            detector_confidence,
        )

        return EstimatorResult(
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            ax=ax,
            ay=ay,
            predicted_x=predicted_x,
            predicted_y=predicted_y,
            mode=self._mode,
            prediction_only=bool(prediction_only),
            measurement_available=bool(measurement_available),
            measurement_rejected=bool(measurement_rejected),
            confidence=float(estimator_confidence),
            missing_frames=int(self._missing_frames),
            coast_time=float(self._coast_time),
            inside_fov=self._inside_fov(x, y),
            timestamp=float(timestamp),
        )


class BeaconKalmanFilter:
    """
    Compatibility adapter class wrapping BeaconStateEstimator for benchmark harness scripts.
    """

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 8.0,
        measurement_noise_std: float = 3.0,
        max_dead_reckoning_frames: int = 30,
    ) -> None:
        self.dt = dt
        cfg = EstimatorConfig(
            process_noise=process_noise_std ** 2,
            measurement_noise=measurement_noise_std ** 2,
            max_coast_time=max_dead_reckoning_frames * dt,
        )
        self._estimator = BeaconStateEstimator(cfg)
        self._current_time = 0.0

    def predict(self) -> None:
        """Prediction update step."""
        pass

    def update(
        self,
        x_meas: Optional[float],
        y_meas: Optional[float],
        confidence: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        """Full predict -> update cycle returning (x_est, y_est, vx, vy)."""
        self._current_time += self.dt
        res = self._estimator.step(
            x=x_meas,
            y=y_meas,
            confidence=confidence,
            timestamp=self._current_time,
        )
        return res.x, res.y, res.vx, res.vy