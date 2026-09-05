"""
Module 5 — Reacquisition Engine Demonstration Script
=====================================================
Demonstrates Archimedean Spiral Reacquisition when target is lost for > 0.5 s (1.5 s dropout window).
Plots the camera pan/tilt trajectory, spiral radius expansion, and FSM state timeline.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control import ReacquisitionEngine
from src.estimation.state import TrackerMode, EstimatorResult

OUTPUT_PLOT = "reacquisition_demo_output.png"
FPS = 30.0
DT = 1.0 / FPS
TOTAL_FRAMES = 180  # 6.0 seconds

# Dropout window (frames 45 to 120 -> 2.5 seconds loss)
DROPOUT_START = 45
DROPOUT_END = 120


def main() -> None:
    engine = ReacquisitionEngine(dropout_threshold_s=0.5, max_slew_deg_s=5.0)

    timestamps: list[float] = []
    states: list[str] = []
    pan_deltas: list[float] = []
    tilt_deltas: list[float] = []
    spiral_radii: list[float] = []
    cum_pan: list[float] = []
    cum_tilt: list[float] = []

    current_pan = 0.0
    current_tilt = 0.0

    for frame in range(TOTAL_FRAMES):
        t = frame * DT

        # Simulate target state
        if DROPOUT_START <= frame < DROPOUT_END:
            mode = TrackerMode.LOST
            confidence = 0.0
        else:
            mode = TrackerMode.TRACKING
            confidence = 0.95

        est_result = EstimatorResult(
            x=320.0, y=240.0, vx=0.0, vy=0.0, ax=0.0, ay=0.0,
            predicted_x=320.0, predicted_y=240.0,
            mode=mode,
            prediction_only=(mode == TrackerMode.COASTING),
            measurement_available=(mode == TrackerMode.TRACKING),
            measurement_rejected=False,
            confidence=confidence,
            missing_frames=0,
            coast_time=0.0,
            inside_fov=True,
            timestamp=t,
        )


        res = engine.update(est_result, current_pan, current_tilt, dt=DT)

        if res.should_command:
            current_pan += res.pan_delta
            current_tilt += res.tilt_delta

        timestamps.append(t)
        states.append(res.state.value)
        pan_deltas.append(res.pan_delta)
        tilt_deltas.append(res.tilt_delta)
        spiral_radii.append(res.spiral_radius_deg)
        cum_pan.append(current_pan)
        cum_tilt.append(current_tilt)

    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Archimedean Spiral Reacquisition Engine Demo", fontsize=14, fontweight="bold")

    # Panel 1: Cumulative Pan/Tilt Gimbal Position
    axes[0].plot(timestamps, cum_pan, label="Gimbal Pan (deg)", color="#2563eb", linewidth=1.5)
    axes[0].plot(timestamps, cum_tilt, label="Gimbal Tilt (deg)", color="#d97706", linewidth=1.5)
    axes[0].axvspan(DROPOUT_START * DT, DROPOUT_END * DT, color="#fef08a", alpha=0.4, label="Target Lost Window")
    axes[0].set_ylabel("Gimbal Angle (deg)")
    axes[0].set_title("Gimbal Pan / Tilt Position During Spiral Search")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Archimedean Spiral Radius Growth
    axes[1].plot(timestamps, spiral_radii, label="Spiral Radius r(theta) (deg)", color="#7c3aed", linewidth=1.5)
    axes[1].axvspan(DROPOUT_START * DT, DROPOUT_END * DT, color="#fef08a", alpha=0.4)
    axes[1].set_ylabel("Radius (deg)")
    axes[1].set_title("Archimedean Spiral Radius Expansion")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: State Machine Timeline
    mode_color_map = {
        "IDLE": "#9ca3af",
        "COASTING": "#f59e0b",
        "SPIRAL_SEARCHING": "#ef4444",
        "REACQUIRED": "#10b981",
    }
    for i in range(len(timestamps)):
        color = mode_color_map.get(states[i], "#9ca3af")
        axes[2].axvspan(timestamps[i] - DT / 2, timestamps[i] + DT / 2, color=color, alpha=0.6)

    from matplotlib.patches import Patch
    legend_patches = [Patch(color=c, label=m, alpha=0.6) for m, c in mode_color_map.items()]
    axes[2].legend(handles=legend_patches, loc="upper right", fontsize=8)
    axes[2].set_ylabel("State")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Reacquisition FSM State Timeline")
    axes[2].set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()

    print("\n" + "=" * 60)
    print("  ARCHIMEDEAN SPIRAL REACQUISITION DEMO — SUMMARY")
    print("=" * 60)
    print(f"  Total frames:        {TOTAL_FRAMES}")
    print(f"  Dropout window:      {DROPOUT_START * DT:.2f}s to {DROPOUT_END * DT:.2f}s ({DROPOUT_END - DROPOUT_START} frames)")
    print(f"  Max spiral radius:   {max(spiral_radii):.2f} deg")
    print(f"  Plot exported to:    {OUTPUT_PLOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
