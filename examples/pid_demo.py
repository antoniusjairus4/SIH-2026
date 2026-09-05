"""
PID Controller Demo
===================

Synthetic scenario that feeds a moving beacon through
BeaconStateEstimator, then through PIDController, and
plots the resulting pan/tilt commands alongside the
estimator mode at each step.

Usage:
    python examples/pid_demo.py
"""

import math
import os
import sys

# Ensure the project root is on sys.path so `from src...` works
# when running directly via `python examples/pid_demo.py`.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.estimation import BeaconStateEstimator, EstimatorConfig
from src.control import ControllerConfig, PIDController


# ==============================================================
# SIMULATION CONFIGURATION
# ==============================================================

SEED = 42
rng = np.random.default_rng(SEED)

FPS = 30.0
DT = 1.0 / FPS
TOTAL_FRAMES = 300

# Detector dropout window -> COASTING then LOST
DROPOUT_START = 120
DROPOUT_END = 160

NOISE_STD = 5.0

OUTPUT_PATH = "pid_demo_output.png"


# ==============================================================
# GROUND-TRUTH TRAJECTORY
# ==============================================================

def true_trajectory(t: float) -> tuple[float, float]:
    """
    Known ground-truth beacon trajectory in pixel coordinates.

    The path combines linear drift with smooth sinusoidal components
    so the PID controller is tested on realistic changing targets.
    """

    x = (
        250.0
        + 30.0 * t
        + 15.0 * math.sin(0.6 * t)
    )

    y = (
        200.0
        + 12.0 * t
        + 10.0 * math.sin(0.4 * t)
    )

    return x, y


# ==============================================================
# RUN SIMULATION
# ==============================================================

def main() -> None:
    estimator = BeaconStateEstimator(EstimatorConfig())

    # TODO: PID gains are initial starting points — tune against
    # real motion profiles (straight-line, circular, figure-8,
    # random) using a systematic grid-search, matching the approach
    # documented in the Kalman filter walkthrough.md.
    controller = PIDController(ControllerConfig())

    # Storage
    timestamps: list[float] = []
    modes: list[str] = []
    confidences: list[float] = []
    pan_deltas: list[float] = []
    tilt_deltas: list[float] = []
    raw_err_x: list[float] = []
    raw_err_y: list[float] = []

    current_pan = 0.0
    current_tilt = 0.0

    for frame in range(TOTAL_FRAMES):
        t = frame * DT

        gt_x, gt_y = true_trajectory(t)

        # Simulate detector
        if DROPOUT_START <= frame < DROPOUT_END:
            det_x = None
            det_y = None
            det_conf = 0.0
        else:
            det_x = gt_x + rng.normal(0.0, NOISE_STD)
            det_y = gt_y + rng.normal(0.0, NOISE_STD)
            det_conf = 0.9 + 0.1 * rng.random()

        # Estimator step
        est_result = estimator.step(
            x=det_x,
            y=det_y,
            confidence=det_conf,
            timestamp=t,
        )

        # Controller step
        ctrl_result = controller.compute(
            estimator_result=est_result,
            current_pan_deg=current_pan,
            current_tilt_deg=current_tilt,
            dt=DT,
        )

        # Integrate commands (simple accumulation for demo)
        if ctrl_result.should_command:
            current_pan += ctrl_result.pan_delta
            current_tilt += ctrl_result.tilt_delta

        # Record
        timestamps.append(t)
        modes.append(est_result.mode.value)
        confidences.append(est_result.confidence)
        pan_deltas.append(ctrl_result.pan_delta)
        tilt_deltas.append(ctrl_result.tilt_delta)
        raw_err_x.append(ctrl_result.raw_error_x_deg)
        raw_err_y.append(ctrl_result.raw_error_y_deg)

    # ==============================================================
    # PLOT
    # ==============================================================

    timestamps_arr = np.array(timestamps)

    mode_color_map = {
        "UNINITIALIZED": "#888888",
        "TRACKING": "#22c55e",
        "COASTING": "#eab308",
        "LOST": "#ef4444",
    }

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "PID Controller Demo — Pan/Tilt Commands from Estimator Output",
        fontsize=14,
        fontweight="bold",
    )

    # Panel 1: Pan / Tilt delta commands
    ax1 = axes[0]
    ax1.plot(
        timestamps_arr,
        pan_deltas,
        label="pan_delta (deg)",
        color="#3b82f6",
        linewidth=1.2,
    )
    ax1.plot(
        timestamps_arr,
        tilt_deltas,
        label="tilt_delta (deg)",
        color="#f97316",
        linewidth=1.2,
    )
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax1.set_ylabel("Delta Command (deg)")
    ax1.set_title("Pan / Tilt Delta Commands")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Raw angular error (before clamp)
    ax2 = axes[1]
    ax2.plot(
        timestamps_arr,
        raw_err_x,
        label="raw_error_x (deg)",
        color="#3b82f6",
        linewidth=1.0,
        alpha=0.8,
    )
    ax2.plot(
        timestamps_arr,
        raw_err_y,
        label="raw_error_y (deg)",
        color="#f97316",
        linewidth=1.0,
        alpha=0.8,
    )
    ax2.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Angular Error (deg)")
    ax2.set_title("Raw Angular Error (pre-clamp)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Mode timeline
    ax3 = axes[2]
    for i in range(len(timestamps)):
        color = mode_color_map.get(modes[i], "#888888")
        ax3.axvspan(
            timestamps[i] - DT / 2,
            timestamps[i] + DT / 2,
            color=color,
            alpha=0.6,
        )

    # Legend patches
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=c, label=m, alpha=0.6)
        for m, c in mode_color_map.items()
    ]
    ax3.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=8,
    )
    ax3.set_ylabel("Mode")
    ax3.set_title("Estimator Mode Timeline")
    ax3.set_yticks([])

    # Panel 4: Confidence
    ax4 = axes[3]
    ax4.fill_between(
        timestamps_arr,
        confidences,
        alpha=0.4,
        color="#8b5cf6",
    )
    ax4.plot(
        timestamps_arr,
        confidences,
        color="#8b5cf6",
        linewidth=1.2,
    )
    ax4.set_ylabel("Confidence")
    ax4.set_xlabel("Time (s)")
    ax4.set_title("Estimator Confidence")
    ax4.set_ylim(-0.05, 1.05)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    plt.close()

    # ==============================================================
    # SUMMARY TABLE
    # ==============================================================

    mode_counts: dict[str, int] = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    print("\n" + "=" * 60)
    print("  PID Controller Demo — Summary")
    print("=" * 60)
    print(f"  Total frames:   {TOTAL_FRAMES}")
    print(f"  FPS:            {FPS}")
    print(f"  Dropout window: frames {DROPOUT_START}-{DROPOUT_END}")
    print()
    print("  Mode distribution:")
    for mode_name, count in sorted(mode_counts.items()):
        pct = 100.0 * count / TOTAL_FRAMES
        print(f"    {mode_name:20s}  {count:4d}  ({pct:5.1f}%)")
    print()

    pan_arr = np.array(pan_deltas)
    tilt_arr = np.array(tilt_deltas)

    print(f"  Pan delta  — mean: {pan_arr.mean():+.5f} deg, "
          f"max abs: {np.abs(pan_arr).max():.5f} deg")
    print(f"  Tilt delta — mean: {tilt_arr.mean():+.5f} deg, "
          f"max abs: {np.abs(tilt_arr).max():.5f} deg")
    print()
    print(f"  Final pan position:  {current_pan:+.4f} deg")
    print(f"  Final tilt position: {current_tilt:+.4f} deg")
    print()
    print(f"  Plot saved to: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
