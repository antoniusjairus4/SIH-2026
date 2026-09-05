"""
Module 5 — Archimedean Spiral Reacquisition Engine for FSOC Virtual Camera Tracking.
====================================================================================
Owner   : Jairus
Project : SIH PS-169 - AI-Based Virtual Camera Tracking for FSOC (ISRO)

Handles automatic target reacquisition when target lock is lost for > 0.5 s (15 frames @ 30Hz).
Executes a smooth Archimedean spiral search pattern (r = b * theta) centered at the last known
beacon coordinates, obeying physical gimbal slew speed limits (<= 5.0 deg/s), and yields control
immediately back to the PID controller upon target re-detection.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from src.estimation.state import EstimatorResult, TrackerMode


class ReacquisitionState(enum.Enum):
    """Operational states of the Reacquisition Engine FSM."""

    IDLE = "IDLE"                  # Target is tracked or uninitialized; reacquisition inactive.
    COASTING = "COASTING"          # Brief dropout (<= 0.5 s); dead-reckoning active.
    SPIRAL_SEARCHING = "SPIRAL_SEARCHING"  # Extended loss (> 0.5 s); Archimedean spiral active.
    REACQUIRED = "REACQUIRED"      # Beacon re-detected; transitioning back to standard PID.


@dataclass
class ReacquisitionResult:
    """Output of a single Reacquisition Engine step."""

    pan_delta: float               # Commanded pan delta angle (degrees).
    tilt_delta: float              # Commanded tilt delta angle (degrees).
    should_command: bool           # True if spiral search is actively commanding gimbal movement.
    state: ReacquisitionState      # Current FSM state.
    spiral_radius_deg: float       # Current spiral search radius (degrees).
    elapsed_loss_time_s: float     # Time elapsed since target loss began (seconds).


class ReacquisitionEngine:
    """
    Archimedean Spiral Reacquisition Engine.

    Parameters
    ----------
    dropout_threshold_s:
        Time duration of lock loss (seconds) required before initiating spiral search. Default 0.5s.
    max_slew_deg_s:
        Maximum allowed gimbal angular rate (deg/s). Default 5.0 deg/s (ISRO spec limit).
    spiral_pitch_deg:
        Radial growth per full 360-degree rotation (degrees). Default 1.0 deg.
    search_timeout_s:
        Maximum duration to attempt spiral search before resetting. Default 5.0s.
    """

    def __init__(
        self,
        dropout_threshold_s: float = 0.5,
        max_slew_deg_s: float = 5.0,
        spiral_pitch_deg: float = 1.0,
        search_timeout_s: float = 5.0,
    ) -> None:
        self.dropout_threshold_s = float(dropout_threshold_s)
        self.max_slew_deg_s = float(max_slew_deg_s)
        self.spiral_pitch_deg = float(spiral_pitch_deg)
        self.search_timeout_s = float(search_timeout_s)

        # Archimedean spiral constant b = pitch / (2 * pi)
        self.b = self.spiral_pitch_deg / (2.0 * math.pi)

        self.reset()

    def reset(self) -> None:
        """Reset internal state machine and spiral variables."""
        self.state = ReacquisitionState.IDLE
        self.loss_start_time: Optional[float] = None
        self.total_loss_duration: float = 0.0
        self.theta: float = 0.0
        self.spiral_center_pan: float = 0.0
        self.spiral_center_tilt: float = 0.0

    def update(
        self,
        estimator_result: EstimatorResult,
        current_pan_deg: float,
        current_tilt_deg: float,
        dt: float,
    ) -> ReacquisitionResult:
        """
        Executes a single step of the reacquisition state machine.

        Parameters
        ----------
        estimator_result:
            Output from BeaconStateEstimator.step().
        current_pan_deg:
            Current camera pan angle (degrees).
        current_tilt_deg:
            Current camera tilt angle (degrees).
        dt:
            Time step since last update (seconds).

        Returns
        -------
        ReacquisitionResult
        """
        mode = estimator_result.mode
        confidence = float(estimator_result.confidence)

        # Safe time delta
        safe_dt = max(dt, 0.0) if math.isfinite(dt) else 1.0 / 30.0

        # -- Case 1: Target actively tracked with high confidence ----------------
        if mode == TrackerMode.TRACKING and confidence >= 0.6:
            if self.state in (ReacquisitionState.SPIRAL_SEARCHING, ReacquisitionState.COASTING):
                self.state = ReacquisitionState.REACQUIRED
            else:
                self.state = ReacquisitionState.IDLE

            self.loss_start_time = None
            self.total_loss_duration = 0.0
            self.theta = 0.0

            return ReacquisitionResult(
                pan_delta=0.0,
                tilt_delta=0.0,
                should_command=False,
                state=self.state,
                spiral_radius_deg=0.0,
                elapsed_loss_time_s=0.0,
            )

        # -- Case 2: Target lock lost or uninitialized ---------------------------
        if self.loss_start_time is None:
            self.loss_start_time = estimator_result.timestamp
            self.spiral_center_pan = current_pan_deg
            self.spiral_center_tilt = current_tilt_deg

        self.total_loss_duration += safe_dt

        # Check if loss duration is within brief coasting threshold (<= 0.5s)
        if self.total_loss_duration < self.dropout_threshold_s:
            self.state = ReacquisitionState.COASTING
            return ReacquisitionResult(
                pan_delta=0.0,
                tilt_delta=0.0,
                should_command=False,
                state=ReacquisitionState.COASTING,
                spiral_radius_deg=0.0,
                elapsed_loss_time_s=self.total_loss_duration,
            )

        # -- Case 3: Extended loss (> 0.5s) -> Archimedean Spiral Search -----------
        self.state = ReacquisitionState.SPIRAL_SEARCHING

        # Check search timeout
        if self.total_loss_duration > (self.dropout_threshold_s + self.search_timeout_s):
            # Reset spiral angle to cycle pattern if max timeout exceeded
            self.theta = 0.0

        # Archimedean spiral radius: r = b * theta
        r_deg = self.b * self.theta

        # Angular velocity omega (rad/s) designed to maintain target linear scanning speed
        # v = r * omega <= max_slew_deg_s
        target_v = self.max_slew_deg_s
        min_r = 0.1  # Prevent divide-by-zero at spiral center
        omega = target_v / max(r_deg, min_r)

        # Update spiral angle
        self.theta += omega * safe_dt
        r_next_deg = self.b * self.theta

        # Compute next target polar offsets from spiral center
        target_pan_offset = r_next_deg * math.cos(self.theta)
        target_tilt_offset = r_next_deg * math.sin(self.theta)

        # Compute required pan/tilt delta relative to current camera position
        target_pan = self.spiral_center_pan + target_pan_offset
        target_tilt = self.spiral_center_tilt + target_tilt_offset

        pan_raw_delta = target_pan - current_pan_deg
        tilt_raw_delta = target_tilt - current_tilt_deg

        # Speed clamping to max_slew_deg_s
        max_delta = self.max_slew_deg_s * safe_dt
        pan_delta = max(-max_delta, min(max_delta, pan_raw_delta))
        tilt_delta = max(-max_delta, min(max_delta, tilt_raw_delta))

        return ReacquisitionResult(
            pan_delta=pan_delta,
            tilt_delta=tilt_delta,
            should_command=True,
            state=ReacquisitionState.SPIRAL_SEARCHING,
            spiral_radius_deg=r_next_deg,
            elapsed_loss_time_s=self.total_loss_duration,
        )
