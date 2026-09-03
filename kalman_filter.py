"""
Module 4 — State Estimator: Beacon Kalman Filter
=================================================
Owner : Dhanya
Project : SIH PS-169 — AI-Based Virtual Camera Tracking for FSOC (ISRO)

Implements a Discrete 2-D Constant-Acceleration (CA) Kalman Filter
that smooths raw pixel detections from Module 3 (Jairus's detector)
and feeds filtered state to Module 5 (PID Controller).

State Vector (6-D):
    x_k = [x, y, vx, vy, ax, ay]^T

Measurement Vector (2-D):
    z_k = [x_meas, y_meas]^T

Key capabilities:
    • +/- 20 px / frame jitter rejection via tuned R & Q matrices
    • Dead-reckoning prediction for up to 1.0 s (30 frames @ 30 Hz)
      when measurements are missing (occlusion / fog / dropped frame)
    • Sub-millisecond per-cycle on commodity hardware (pure NumPy)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


class BeaconKalmanFilter:
    """Discrete 2-D Constant-Acceleration Kalman Filter for beacon tracking.

    Parameters
    ----------
    dt : float
        Nominal time-step between frames (default 1/30 s for 30 Hz).
    process_noise_std : float
        Standard deviation of the jerk (rate-of-change of acceleration)
        used to build the continuous white-noise jerk Q matrix.
        Higher → filter trusts measurements more, reacts faster to manoeuvres.
        Lower  → filter trusts its model more, smooths harder.
    measurement_noise_std : float
        Standard deviation of pixel-level measurement noise (detector + jitter).
        Should be set to roughly half the peak jitter amplitude so the filter
        absorbs ±20 px spikes without sluggishness.
    max_dead_reckoning_frames : int
        Maximum consecutive prediction-only cycles before the filter
        considers the track lost (default 30 → 1.0 s @ 30 Hz).
    """

    # ── Construction ──────────────────────────────────────────────────

    def __init__(
        self,
        dt: float = 1.0 / 30.0,
        process_noise_std: float = 8.0,
        measurement_noise_std: float = 3.0,
        max_dead_reckoning_frames: int = 30,
    ) -> None:
        self.dt = dt
        self.max_dead_reckoning_frames = max_dead_reckoning_frames

        # Dimensionality
        self._n = 6  # state dimension
        self._m = 2  # measurement dimension

        # ── State Transition Matrix  F (6×6) ──────────────────────────
        # Constant-acceleration kinematic equations per axis:
        #   x_k  = x_{k-1} + vx·dt + 0.5·ax·dt²
        #   vx_k = vx_{k-1} + ax·dt
        #   ax_k = ax_{k-1}                        (const. accel. assumption)
        # Arranged as [x, y, vx, vy, ax, ay].
        self.F = self._build_transition_matrix(dt)

        # ── Measurement Matrix  H (2×6) ──────────────────────────────
        # We observe only position: z = H·x → [x, y]
        self.H = np.zeros((self._m, self._n), dtype=np.float64)
        self.H[0, 0] = 1.0  # x
        self.H[1, 1] = 1.0  # y

        # ── Process Noise Covariance  Q (6×6) ─────────────────────────
        # Piecewise constant white-noise jerk model (Singer-style).
        # Q = G · G^T · sigma_jerk²  where G is the jerk-input vector.
        self.Q = self._build_process_noise(dt, process_noise_std)

        # ── Measurement Noise Covariance  R (2×2) ─────────────────────
        self.R = np.eye(self._m, dtype=np.float64) * (measurement_noise_std ** 2)

        # ── Initial State & Covariance ────────────────────────────────
        self.x = np.zeros((self._n, 1), dtype=np.float64)  # state estimate
        self.P = np.eye(self._n, dtype=np.float64) * 500.0  # high initial uncertainty

        # ── Book-keeping ──────────────────────────────────────────────
        self._initialised: bool = False
        self._consecutive_misses: int = 0

    # ── Public API ────────────────────────────────────────────────────

    def predict(self) -> None:
        """Time-update (prediction) step.

        Propagates state and covariance forward by one time-step dt:
            x⁻_k  = F · x_{k-1}
            P⁻_k  = F · P_{k-1} · F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(
        self,
        x_meas: Optional[float],
        y_meas: Optional[float],
        confidence: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        """Full predict → conditional-correct cycle.

        Parameters
        ----------
        x_meas : float | None
            Detected beacon x-coordinate (pixels).  ``None`` when detection
            failed (occlusion, fog, dropped frame).
        y_meas : float | None
            Detected beacon y-coordinate (pixels).
        confidence : float
            Detector confidence in [0, 1].  ``0.0`` triggers dead-reckoning.

        Returns
        -------
        (x_est, y_est, vx, vy) : tuple[float, float, float, float]
            Filtered position and velocity estimates.
        """

        has_measurement = (
            x_meas is not None
            and y_meas is not None
            and confidence > 0.0
        )

        # ── First valid measurement → seed state directly ─────────
        if has_measurement and not self._initialised:
            self.x[0, 0] = x_meas
            self.x[1, 0] = y_meas
            # velocities & accelerations stay zero — filter will learn
            self._initialised = True
            self._consecutive_misses = 0
            return self._output()

        # ── Prediction step (always runs) ─────────────────────────
        self.predict()

        # ── Measurement correction (conditional) ──────────────────
        if has_measurement:
            self._consecutive_misses = 0

            z = np.array([[x_meas], [y_meas]], dtype=np.float64)

            # Adaptive R scaling: inflate R when confidence is low
            # to let the filter lean on its model during marginal detections.
            R_scaled = self.R / max(confidence, 0.1)

            # Innovation
            y_innov = z - self.H @ self.x

            # Innovation covariance
            S = self.H @ self.P @ self.H.T + R_scaled

            # Kalman gain
            K = self.P @ self.H.T @ np.linalg.inv(S)

            # State correction
            self.x = self.x + K @ y_innov

            # Covariance correction (Joseph form for numerical stability)
            I_KH = np.eye(self._n, dtype=np.float64) - K @ self.H
            self.P = I_KH @ self.P @ I_KH.T + K @ R_scaled @ K.T

        else:
            # Dead-reckoning: prediction already applied above, skip correction.
            self._consecutive_misses += 1

        return self._output()

    def is_track_lost(self) -> bool:
        """Return True if dead-reckoning has exceeded the allowed duration."""
        return self._consecutive_misses > self.max_dead_reckoning_frames

    def get_miss_count(self) -> int:
        """Return the number of consecutive frames without a valid measurement."""
        return self._consecutive_misses

    def reset(self) -> None:
        """Reset filter state for re-acquisition after a track loss."""
        self.x = np.zeros((self._n, 1), dtype=np.float64)
        self.P = np.eye(self._n, dtype=np.float64) * 500.0
        self._initialised = False
        self._consecutive_misses = 0

    # ── Private helpers ───────────────────────────────────────────────

    def _output(self) -> Tuple[float, float, float, float]:
        """Extract the 4-element output tuple from the state vector."""
        return (
            float(self.x[0, 0]),  # x_est
            float(self.x[1, 0]),  # y_est
            float(self.x[2, 0]),  # vx
            float(self.x[3, 0]),  # vy
        )

    @staticmethod
    def _build_transition_matrix(dt: float) -> np.ndarray:
        """Build the 6×6 constant-acceleration state transition matrix.

        Layout:  [x, y, vx, vy, ax, ay]

        Per-axis block (3×3):
            | 1   dt   0.5·dt² |
            | 0   1    dt      |
            | 0   0    1       |
        """
        dt2 = 0.5 * dt * dt
        F = np.eye(6, dtype=np.float64)

        # x-axis block: indices 0 (x), 2 (vx), 4 (ax)
        F[0, 2] = dt
        F[0, 4] = dt2
        F[2, 4] = dt

        # y-axis block: indices 1 (y), 3 (vy), 5 (ay)
        F[1, 3] = dt
        F[1, 5] = dt2
        F[3, 5] = dt

        return F

    @staticmethod
    def _build_process_noise(dt: float, sigma: float) -> np.ndarray:
        """Build Q using a discrete white-noise acceleration (DWNA) model.

        This places process noise directly into the acceleration states,
        modelling the idea that acceleration can change by up to +/- sigma
        per time step.  The kinematic coupling in F then propagates this
        uncertainty into the velocity and position states naturally.

        Per-axis 3x3 sub-block (position, velocity, acceleration):

            G = [dt^2/2, dt, 1]^T   (how a unit acceleration impulse
                                      propagates into pos, vel, acc)

            Q_axis = sigma^2 * G * G^T

        This produces much larger position-level uncertainty than the
        white-noise jerk model (which uses dt^5 terms that vanish at
        high frame rates like 30 Hz), allowing the filter to stay
        responsive during sharp manoeuvres.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        s2 = sigma * sigma

        # G = [dt^2/2, dt, 1]^T  -->  Q_block = s2 * G @ G.T
        q_block = s2 * np.array([
            [dt4 / 4.0,  dt3 / 2.0,  dt2 / 2.0],
            [dt3 / 2.0,  dt2,        dt        ],
            [dt2 / 2.0,  dt,         1.0       ],
        ], dtype=np.float64)

        Q = np.zeros((6, 6), dtype=np.float64)

        # x-axis block: rows/cols [0, 2, 4]
        idx_x = [0, 2, 4]
        for i, qi in enumerate(idx_x):
            for j, qj in enumerate(idx_x):
                Q[qi, qj] = q_block[i, j]

        # y-axis block: rows/cols [1, 3, 5]
        idx_y = [1, 3, 5]
        for i, qi in enumerate(idx_y):
            for j, qj in enumerate(idx_y):
                Q[qi, qj] = q_block[i, j]

        return Q
