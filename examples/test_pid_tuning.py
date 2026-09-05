"""
Module 5 - Test Script: PID Controller Grid-Search Tuning & Slew Optimization
==============================================================================
Owner   : Jairus
Project : SIH PS-169 - AI-Based Virtual Camera Tracking for FSOC (ISRO)

Official Spec Target (SIH26169): Tracking Error <= 10 pixels (average)
Max Gimbal Slew Speed Limit    : <= 5.0 deg/s

Generates synthetic 2D beacon trajectories (Straight-Line, Circular, Figure-8, Random Walk),
simulates closed-loop camera pan/tilt tracking, and produces:
    1. Grid search optimizing (Kp, Ki, Kd, max_slew_rate) for minimum tracking RMSE <= 10 px.
    2. Slew rate compliance check (ensuring actuator limit of 5 deg/s is respected).
    3. CSV export of all candidate evaluations ('pid_grid_search_results.csv').
    4. Comparative visualization plots ('pid_tuning_results.png').
"""

import csv
import math
import os
import sys
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.control import ControllerConfig, PIDController
from src.estimation import BeaconStateEstimator, EstimatorConfig

SPEC_TARGET_PX = 10.0
MAX_SLEW_SPEC_DEG_S = 5.0
FPS = 30.0
DT = 1.0 / FPS
TOTAL_FRAMES = 300
OUTPUT_CSV = "pid_grid_search_results.csv"
OUTPUT_PLOT = "pid_tuning_results.png"


# =====================================================================
#  TRAJECTORY GENERATORS
# =====================================================================

def generate_straight_line(num_frames: int = TOTAL_FRAMES) -> np.ndarray:
    """Straight line motion passing smoothly through frame center (320, 240)."""
    t = np.linspace(-5.0, 5.0, num_frames)
    x = 320.0 + 20.0 * t  # 220 -> 420 px
    y = 240.0 + 10.0 * t  # 190 -> 290 px
    return np.column_stack([x, y])


def generate_circular(num_frames: int = TOTAL_FRAMES) -> np.ndarray:
    """Circular trajectory starting at frame center offset (320, 240)."""
    t = np.linspace(0.0, 2.0 * np.pi * 2.0, num_frames)
    radius_x = 80.0
    radius_y = 60.0
    x = 320.0 + radius_x * (np.cos(t) - 1.0)
    y = 240.0 + radius_y * np.sin(t)
    return np.column_stack([x, y])


def generate_figure8(num_frames: int = TOTAL_FRAMES) -> np.ndarray:
    """Figure-8 (Lissajous 1:2) trajectory starting at frame center (320, 240)."""
    t = np.linspace(0.0, 2.0 * np.pi * 2.0, num_frames)
    x = 320.0 + 90.0 * np.sin(t)
    y = 240.0 + 50.0 * np.sin(2.0 * t)
    return np.column_stack([x, y])


def generate_random_walk(num_frames: int = TOTAL_FRAMES, seed: int = 42) -> np.ndarray:
    """Random walk trajectory starting at frame center (320, 240)."""
    rng = np.random.default_rng(seed)
    dx = rng.normal(0.0, 2.5, size=num_frames)
    dy = rng.normal(0.0, 1.8, size=num_frames)
    x = np.cumsum(dx) + 320.0
    y = np.cumsum(dy) + 240.0
    return np.column_stack([x, y])



# =====================================================================
#  SIMULATION ENGINE
# =====================================================================

AttributeMetrics = Dict[str, float]



def simulate_closed_loop(
    trajectory: np.ndarray,
    config: ControllerConfig,
) -> Tuple[float, float, float, float]:
    """
    Simulates closed-loop tracking of a target trajectory using Estimator + PID.

    Returns:
        rmse_px: Root Mean Square tracking error in pixels.
        max_error_px: Maximum tracking error in pixels.
        max_slew_deg_s: Maximum commanded angular rate (deg/s).
        slew_compliance_pct: Percentage of frames meeting max_slew_speed limit.
    """
    estimator = BeaconStateEstimator(EstimatorConfig())
    controller = PIDController(config)

    camera_pan_deg = 0.0
    camera_tilt_deg = 0.0

    errors_px: List[float] = []
    slew_rates_deg_s: List[float] = []

    # Camera FOV degrees per pixel mapping
    deg_per_px_x = config.camera_fov_x_deg / config.frame_width
    deg_per_px_y = config.camera_fov_y_deg / config.frame_height

    for i in range(len(trajectory)):
        t = i * DT
        world_x, world_y = trajectory[i]

        # Calculate pixel location on camera FPA after camera pan/tilt offset
        pixel_x = world_x - (camera_pan_deg / deg_per_px_x)
        pixel_y = world_y - (camera_tilt_deg / deg_per_px_y)

        # 1. Estimator step
        est_result = estimator.step(x=pixel_x, y=pixel_y, confidence=0.95, timestamp=t)

        # 2. Controller step
        ctrl_result = controller.compute(
            estimator_result=est_result,
            current_pan_deg=camera_pan_deg,
            current_tilt_deg=camera_tilt_deg,
            dt=DT,
        )

        if ctrl_result.should_command:
            camera_pan_deg += ctrl_result.pan_delta
            camera_tilt_deg += ctrl_result.tilt_delta

        # Calculate error between current pixel position and frame center (320, 240)
        err_x = pixel_x - (config.frame_width / 2.0)
        err_y = pixel_y - (config.frame_height / 2.0)
        dist_px = math.hypot(err_x, err_y)
        errors_px.append(dist_px)

        pan_rate = abs(ctrl_result.pan_delta) / DT
        tilt_rate = abs(ctrl_result.tilt_delta) / DT
        slew_rates_deg_s.append(max(pan_rate, tilt_rate))

    rmse_px = float(np.sqrt(np.mean(np.square(errors_px))))
    max_error_px = float(np.max(errors_px))
    max_slew_deg_s = float(np.max(slew_rates_deg_s))
    compliant_count = sum(1 for s in slew_rates_deg_s if s <= MAX_SLEW_SPEC_DEG_S + 1e-5)
    slew_compliance_pct = 100.0 * compliant_count / len(slew_rates_deg_s)

    return rmse_px, max_error_px, max_slew_deg_s, slew_compliance_pct


# =====================================================================
#  GRID SEARCH & BENCHMARKING
# =====================================================================

def run_grid_search() -> Tuple[ControllerConfig, List[Dict[str, str]]]:
    trajectories = {
        "Straight-Line": generate_straight_line(),
        "Circular": generate_circular(),
        "Figure-8": generate_figure8(),
        "Random Walk": generate_random_walk(),
    }

    # Search space grid
    kp_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ki_vals = [0.0, 0.01, 0.02, 0.05]
    kd_vals = [0.0, 0.01, 0.02, 0.05]
    max_slew_vals = [5.0]

    best_config: Optional[ControllerConfig] = None
    best_score = float("inf")
    results_rows: List[Dict[str, str]] = []

    total_combinations = len(kp_vals) * len(ki_vals) * len(kd_vals) * len(max_slew_vals)
    print(f"Running PID Grid Search across {total_combinations} candidate parameter sets...")

    for kp in kp_vals:
        for ki in ki_vals:
            for kd in kd_vals:
                for slew in max_slew_vals:
                    cfg = ControllerConfig(
                        kp_x=kp, ki_x=ki, kd_x=kd,
                        kp_y=kp, ki_y=ki, kd_y=kd,
                        max_pan_speed_deg_s=slew,
                        max_tilt_speed_deg_s=slew,
                    )

                    traj_rmses: List[float] = []
                    traj_slew_compliances: List[float] = []

                    for name, traj in trajectories.items():
                        rmse, max_err, max_slew, comp_pct = simulate_closed_loop(traj, cfg)
                        traj_rmses.append(rmse)
                        traj_slew_compliances.append(comp_pct)

                    avg_rmse = float(np.mean(traj_rmses))
                    avg_compliance = float(np.mean(traj_slew_compliances))

                    # Composite score penalizing high error and non-compliance
                    score = avg_rmse + (100.0 - avg_compliance) * 2.0

                    if score < best_score:
                        best_score = score
                        best_config = cfg

                    results_rows.append({
                        "kp": f"{kp:.2f}",
                        "ki": f"{ki:.2f}",
                        "kd": f"{kd:.2f}",
                        "max_slew_deg_s": f"{slew:.1f}",
                        "avg_rmse_px": f"{avg_rmse:.4f}",
                        "slew_compliance_pct": f"{avg_compliance:.1f}",
                        "spec_compliant": "PASS" if avg_rmse <= SPEC_TARGET_PX and avg_compliance >= 99.0 else "FAIL",
                    })


    # Save results to CSV
    fieldnames = ["kp", "ki", "kd", "max_slew_deg_s", "avg_rmse_px", "slew_compliance_pct", "spec_compliant"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"Grid search complete. Full results exported to '{OUTPUT_CSV}'.")
    return best_config or ControllerConfig(), results_rows


# =====================================================================
#  PLOT GENERATION
# =====================================================================

def generate_verification_plots(best_cfg: ControllerConfig) -> None:
    trajectories = {
        "Straight-Line": generate_straight_line(),
        "Circular": generate_circular(),
        "Figure-8": generate_figure8(),
        "Random Walk": generate_random_walk(),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"PID Controller Optimization & Verification (Kp={best_cfg.kp_x}, Ki={best_cfg.ki_x}, Kd={best_cfg.kd_x})",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (name, traj) in enumerate(trajectories.items()):
        ax = axes[idx // 2, idx % 2]

        estimator = BeaconStateEstimator(EstimatorConfig())
        controller = PIDController(best_cfg)
        deg_per_px_x = best_cfg.camera_fov_x_deg / best_cfg.frame_width
        deg_per_px_y = best_cfg.camera_fov_y_deg / best_cfg.frame_height

        cam_pan, cam_tilt = 0.0, 0.0
        errors: List[float] = []

        for i in range(len(traj)):
            t = i * DT
            wx, wy = traj[i]
            px = wx - (cam_pan / deg_per_px_x)
            py = wy - (cam_tilt / deg_per_px_y)

            est_result = estimator.step(x=px, y=py, confidence=0.95, timestamp=t)
            ctrl_result = controller.compute(est_result, cam_pan, cam_tilt, dt=DT)

            if ctrl_result.should_command:
                cam_pan += ctrl_result.pan_delta
                cam_tilt += ctrl_result.tilt_delta

            dist = math.hypot(px - 320.0, py - 240.0)
            errors.append(dist)

        time_arr = np.linspace(0.0, TOTAL_FRAMES * DT, TOTAL_FRAMES)
        ax.plot(time_arr, errors, label="Tracking Error (px)", color="#2563eb", linewidth=1.5)
        ax.axhline(SPEC_TARGET_PX, color="#dc2626", linestyle="--", label=f"ISRO Spec ({SPEC_TARGET_PX}px)")
        ax.set_title(f"Trajectory: {name} (RMSE = {np.sqrt(np.mean(np.square(errors))):.2f} px)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Pixel Error (px)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    plt.close()
    print(f"Verification plots saved to '{OUTPUT_PLOT}'.")


# =====================================================================
#  MAIN ENTRYPOINT
# =====================================================================

def main() -> None:
    print("==========================================================")
    print("  MODULE 5 — PID CONTROLLER GRID SEARCH & OPTIMIZATION")
    print("==========================================================")

    best_cfg, results = run_grid_search()

    print("\n" + "=" * 60)
    print("  OPTIMAL PID GAIN CONFIGURATION")
    print("=" * 60)
    print(f"  Kp (Pan/Tilt)         : {best_cfg.kp_x}")
    print(f"  Ki (Pan/Tilt)         : {best_cfg.ki_x}")
    print(f"  Kd (Pan/Tilt)         : {best_cfg.kd_x}")
    print(f"  Max Speed Limit       : {best_cfg.max_pan_speed_deg_s} deg/s")
    print("=" * 60)

    generate_verification_plots(best_cfg)


if __name__ == "__main__":
    main()
