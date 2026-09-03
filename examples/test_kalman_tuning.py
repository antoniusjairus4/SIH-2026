"""
Module 4 - Test Script: Kalman Filter Verification & Parameter Tuning
======================================================================
Owner : Dhanya
Project : SIH PS-169 - AI-Based Virtual Camera Tracking for FSOC (ISRO)

Official Spec Target (SIH26169): Tracking Error <= 10 pixels (average)

Generates synthetic 2-D beacon trajectories (Straight-Line, Circular,
Figure-8) on a 640x480 pixel canvas, corrupts them with +/-20 px Gaussian
noise and a 1.0 s occlusion gap, runs BeaconKalmanFilter, and produces:
    1. Multi-trajectory grid search optimising for filtered RMSE <= 10 px
    2. Robustness check across noise levels (std=10, 20, 30 px)
    3. CSV export of all results
    4. Final verification plots with honest spec compliance labelling
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend -- no plt.show() blocking
import matplotlib.pyplot as plt

from src.estimation import BeaconKalmanFilter

# Official spec target
SPEC_TARGET_PX = 10.0


# =====================================================================
#  TRAJECTORY GENERATORS
# =====================================================================

def generate_figure8(
    num_frames: int = 300,
    cx: float = 320.0,
    cy: float = 240.0,
    amp_x: float = 150.0,
    amp_y: float = 100.0,
    periods: float = 2.0,
) -> np.ndarray:
    """Generate a figure-8 (Lissajous 1:2) trajectory.

    Returns ndarray of shape (num_frames, 2) with [x, y] per frame.
    """
    t = np.linspace(0.0, 2.0 * np.pi * periods, num_frames)
    x = cx + amp_x * np.sin(t)
    y = cy + amp_y * np.sin(2.0 * t)
    return np.column_stack([x, y])


def generate_circular(
    num_frames: int = 300,
    cx: float = 320.0,
    cy: float = 240.0,
    radius: float = 120.0,
    periods: float = 2.0,
) -> np.ndarray:
    """Generate a circular trajectory.

    Returns ndarray of shape (num_frames, 2) with [x, y] per frame.
    """
    t = np.linspace(0.0, 2.0 * np.pi * periods, num_frames)
    x = cx + radius * np.cos(t)
    y = cy + radius * np.sin(t)
    return np.column_stack([x, y])


def generate_straight_line(
    num_frames: int = 300,
    start: Tuple[float, float] = (80.0, 100.0),
    end: Tuple[float, float] = (560.0, 380.0),
) -> np.ndarray:
    """Generate a straight-line trajectory crossing the 640x480 frame.

    Returns ndarray of shape (num_frames, 2) with [x, y] per frame.
    """
    t = np.linspace(0.0, 1.0, num_frames)
    x = start[0] + (end[0] - start[0]) * t
    y = start[1] + (end[1] - start[1]) * t
    return np.column_stack([x, y])


# =====================================================================
#  NOISE & OCCLUSION INJECTION
# =====================================================================

def add_jitter_noise(
    coords: np.ndarray,
    noise_std: float = 20.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian noise to simulate camera jitter."""
    if rng is None:
        rng = np.random.default_rng(42)
    noise = rng.normal(0.0, noise_std, size=coords.shape)
    return coords + noise


def inject_occlusion(
    noisy: np.ndarray,
    start_frame: int,
    duration_frames: int,
) -> np.ndarray:
    """Set coordinates to NaN for a contiguous block (simulated occlusion)."""
    occluded = noisy.copy()
    end_frame = min(start_frame + duration_frames, len(occluded))
    occluded[start_frame:end_frame] = np.nan
    return occluded


# =====================================================================
#  RMSE COMPUTATION
# =====================================================================

def rmse(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """Euclidean RMSE over all frames (ignoring NaN rows)."""
    mask = ~np.isnan(predictions).any(axis=1)
    if mask.sum() == 0:
        return float("inf")
    diff = predictions[mask] - ground_truth[mask]
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


# =====================================================================
#  SINGLE-RUN FILTER EVALUATION
# =====================================================================

@dataclass
class RunResult:
    """Stores results from a single Kalman filter evaluation run."""
    trajectory_name: str
    noise_std: float
    process_noise_std: float
    measurement_noise_std: float
    rmse_noisy: float
    rmse_filtered: float
    noise_reduction_factor: float
    dead_reckoning_rmse: float
    filtered_positions: np.ndarray
    ground_truth: np.ndarray
    noisy_measurements: np.ndarray
    occluded_measurements: np.ndarray
    occlusion_start: int
    occlusion_end: int


def run_filter(
    ground_truth: np.ndarray,
    trajectory_name: str = "Test",
    noise_std: float = 20.0,
    occlusion_start: int = 100,
    occlusion_frames: int = 30,
    fps: float = 30.0,
    process_noise_std: float = 8.0,
    measurement_noise_std: float = 3.0,
    seed: int = 42,
) -> RunResult:
    """Run the Kalman filter on a noisy+occluded trajectory and return metrics.

    This function does NOT plot or print anything -- it just returns data.
    """
    num_frames = len(ground_truth)
    dt = 1.0 / fps
    rng = np.random.default_rng(seed)

    # Corrupt ground truth
    noisy = add_jitter_noise(ground_truth, noise_std=noise_std, rng=rng)
    occluded = inject_occlusion(noisy, occlusion_start, occlusion_frames)

    # Run filter
    kf = BeaconKalmanFilter(
        dt=dt,
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std,
    )

    filtered = np.zeros((num_frames, 2), dtype=np.float64)
    for i in range(num_frames):
        mx, my = occluded[i]
        if np.isnan(mx):
            x_est, y_est, vx, vy = kf.update(None, None, confidence=0.0)
        else:
            x_est, y_est, vx, vy = kf.update(float(mx), float(my), confidence=1.0)
        filtered[i] = [x_est, y_est]

    # Compute metrics
    rmse_noisy = rmse(noisy, ground_truth)
    rmse_filtered = rmse(filtered, ground_truth)
    noise_reduction = rmse_noisy / max(rmse_filtered, 1e-9)

    occ_end = min(occlusion_start + occlusion_frames, num_frames)
    dead_reck_err = rmse(
        filtered[occlusion_start:occ_end],
        ground_truth[occlusion_start:occ_end],
    )

    return RunResult(
        trajectory_name=trajectory_name,
        noise_std=noise_std,
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std,
        rmse_noisy=rmse_noisy,
        rmse_filtered=rmse_filtered,
        noise_reduction_factor=noise_reduction,
        dead_reckoning_rmse=dead_reck_err,
        filtered_positions=filtered,
        ground_truth=ground_truth,
        noisy_measurements=noisy,
        occluded_measurements=occluded,
        occlusion_start=occlusion_start,
        occlusion_end=occ_end,
    )


# =====================================================================
#  PLOTTING (NON-BLOCKING -- saves to file, never calls plt.show())
# =====================================================================

def save_verification_plot(result: RunResult, output_path: str) -> None:
    """Generate and save a dual-panel verification plot to disk."""
    spec_met = result.rmse_filtered <= SPEC_TARGET_PX
    spec_label = "MEETS SPEC" if spec_met else "ABOVE SPEC"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Kalman Filter - {result.trajectory_name}  |  "
        f"Q_std={result.process_noise_std:.0f}  R_std={result.measurement_noise_std:.0f}  |  "
        f"RMSE={result.rmse_filtered:.1f}px  NRF={result.noise_reduction_factor:.2f}x  |  "
        f"{spec_label} (target <={SPEC_TARGET_PX:.0f}px)",
        fontsize=11, fontweight="bold",
    )

    gt = result.ground_truth
    occ = result.occluded_measurements
    flt = result.filtered_positions
    noisy = result.noisy_measurements
    occ_s = result.occlusion_start
    occ_e = result.occlusion_end

    # --- XY Trajectory Plot ---
    ax = axes[0]
    ax.plot(gt[:, 0], gt[:, 1], "g-", lw=1.5, alpha=0.7, label="Ground Truth")
    ax.scatter(occ[:, 0], occ[:, 1], s=4, c="red", alpha=0.3, label="Noisy Measurements")
    ax.plot(flt[:, 0], flt[:, 1], "b-", lw=1.2, label="Kalman Filtered")
    ax.plot(flt[occ_s:occ_e, 0], flt[occ_s:occ_e, 1],
            "m--", lw=2.0, label="Dead-Reckoning (occlusion)")
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title("XY Plane")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    # --- Per-Frame Error ---
    ax2 = axes[1]
    frames = np.arange(len(gt))
    err_noisy = np.sqrt(np.sum((noisy - gt) ** 2, axis=1))
    err_filtered = np.sqrt(np.sum((flt - gt) ** 2, axis=1))
    ax2.plot(frames, err_noisy, "r-", alpha=0.4, lw=0.8, label="Noisy Error")
    ax2.plot(frames, err_filtered, "b-", lw=1.2, label="Filtered Error")
    ax2.axhline(y=SPEC_TARGET_PX, color="green", ls="--", lw=1.5,
                label=f"Spec Target ({SPEC_TARGET_PX:.0f} px)")
    ax2.axvspan(occ_s, occ_e, alpha=0.15, color="magenta", label="Occlusion Window")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Euclidean Error (px)")
    ax2.set_title("Per-Frame Tracking Error")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {output_path}")


# =====================================================================
#  MULTI-TRAJECTORY GRID SEARCH
# =====================================================================

@dataclass
class MultiTrajectoryScore:
    """Aggregated score for one (Q_std, R_std) combination across all trajectories."""
    process_noise_std: float
    measurement_noise_std: float
    avg_rmse_filtered: float          # Primary metric: lower is better
    worst_rmse_filtered: float        # Worst-case across trajectories
    avg_nrf: float                    # Secondary: higher is better
    min_nrf: float                    # All must be > 1.0
    avg_dr_rmse: float                # Dead-reckoning quality
    per_trajectory: Dict[str, RunResult]


def multi_trajectory_grid_search(
    trajectories: Dict[str, np.ndarray],
    process_noise_values: List[float],
    measurement_noise_values: List[float],
    noise_std: float = 20.0,
    occlusion_start: int = 100,
    occlusion_frames: int = 30,
    fps: float = 30.0,
) -> List[MultiTrajectoryScore]:
    """Grid search across all trajectories simultaneously.

    For each (Q_std, R_std), runs all trajectories and computes an aggregate
    score. Returns list sorted by avg_rmse_filtered (lowest first), with
    NRF > 1.0 on all trajectories as a hard constraint.
    """
    scores: List[MultiTrajectoryScore] = []
    total = len(process_noise_values) * len(measurement_noise_values)
    idx = 0

    for q_std in process_noise_values:
        for r_std in measurement_noise_values:
            idx += 1
            per_traj: Dict[str, RunResult] = {}

            for traj_name, gt in trajectories.items():
                result = run_filter(
                    ground_truth=gt,
                    trajectory_name=traj_name,
                    noise_std=noise_std,
                    occlusion_start=occlusion_start,
                    occlusion_frames=occlusion_frames,
                    fps=fps,
                    process_noise_std=q_std,
                    measurement_noise_std=r_std,
                )
                per_traj[traj_name] = result

            results_list = list(per_traj.values())
            avg_rmse = np.mean([r.rmse_filtered for r in results_list])
            worst_rmse = max(r.rmse_filtered for r in results_list)
            avg_nrf = np.mean([r.noise_reduction_factor for r in results_list])
            min_nrf = min(r.noise_reduction_factor for r in results_list)
            avg_dr = np.mean([r.dead_reckoning_rmse for r in results_list])

            score = MultiTrajectoryScore(
                process_noise_std=q_std,
                measurement_noise_std=r_std,
                avg_rmse_filtered=float(avg_rmse),
                worst_rmse_filtered=float(worst_rmse),
                avg_nrf=float(avg_nrf),
                min_nrf=float(min_nrf),
                avg_dr_rmse=float(avg_dr),
                per_trajectory=per_traj,
            )
            scores.append(score)

            nrf_ok = "OK" if min_nrf > 1.0 else "FAIL"
            print(f"  [{idx:2d}/{total}] Q={q_std:4.0f} R={r_std:4.0f}  |  "
                  f"avg_RMSE={avg_rmse:6.2f}  worst_RMSE={worst_rmse:6.2f}  "
                  f"min_NRF={min_nrf:5.2f}x({nrf_ok})  avg_DR={avg_dr:6.2f}")

    # Sort: primary = lowest avg_rmse_filtered
    scores.sort(key=lambda s: s.avg_rmse_filtered)
    return scores


def find_best_multi_params(
    scores: List[MultiTrajectoryScore],
) -> Optional[MultiTrajectoryScore]:
    """Find best (Q_std, R_std) optimising for lowest avg filtered RMSE.

    Hard constraint: NRF > 1.0 on ALL trajectories (sanity check).
    If no combination satisfies NRF > 1.0 everywhere, relax and return
    the lowest avg RMSE regardless.
    """
    # First pass: lowest avg RMSE with NRF > 1.0 everywhere
    for s in scores:
        if s.min_nrf > 1.0:
            return s

    # Relaxed: return lowest avg RMSE regardless
    print(f"\n  [WARNING] No combination achieves NRF > 1.0 on all trajectories.")
    print(f"  Relaxing NRF constraint -- returning lowest avg RMSE.")
    return scores[0] if scores else None


def print_multi_grid_table(scores: List[MultiTrajectoryScore], top_n: int = 20) -> None:
    """Print the top N grid search results as a formatted console table."""
    header = (
        f"{'Rank':>4}  {'Q_std':>5}  {'R_std':>5}  "
        f"{'Avg_RMSE':>8}  {'Worst_RMSE':>10}  "
        f"{'Avg_NRF':>7}  {'Min_NRF':>7}  {'Avg_DR':>6}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    show = min(top_n, len(scores))
    for i in range(show):
        s = scores[i]
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"{i+1:4d}  {s.process_noise_std:5.0f}  {s.measurement_noise_std:5.0f}  "
            f"{s.avg_rmse_filtered:8.2f}  {s.worst_rmse_filtered:10.2f}  "
            f"{s.avg_nrf:7.2f}x  {s.min_nrf:7.2f}x  {s.avg_dr_rmse:6.1f}"
            f"{marker}"
        )
    if len(scores) > top_n:
        print(f"  ... ({len(scores) - top_n} more rows omitted)")


def export_multi_grid_csv(
    scores: List[MultiTrajectoryScore],
    csv_path: str,
) -> None:
    """Export multi-trajectory grid search results to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "process_noise_std", "measurement_noise_std",
            "avg_rmse_filtered_px", "worst_rmse_filtered_px",
            "avg_nrf", "min_nrf", "avg_dead_reckoning_rmse_px",
        ])
        for i, s in enumerate(scores):
            writer.writerow([
                i + 1, s.process_noise_std, s.measurement_noise_std,
                f"{s.avg_rmse_filtered:.4f}", f"{s.worst_rmse_filtered:.4f}",
                f"{s.avg_nrf:.4f}", f"{s.min_nrf:.4f}",
                f"{s.avg_dr_rmse:.4f}",
            ])
    print(f"  [SAVED] {csv_path}")


# =====================================================================
#  PRINT SUMMARY (HONEST SPEC LABELLING)
# =====================================================================

def spec_status(rmse_val: float) -> str:
    """Return honest spec compliance label."""
    if rmse_val <= SPEC_TARGET_PX:
        return f"MEETS SPEC (<={SPEC_TARGET_PX:.0f}px)"
    else:
        gap = rmse_val - SPEC_TARGET_PX
        return f"IN PROGRESS ({gap:.1f}px over target)"


def print_run_summary(result: RunResult) -> None:
    """Print formatted metrics for a single filter run with honest labelling."""
    print(f"\n{'=' * 70}")
    print(f"  Trajectory : {result.trajectory_name}  (noise_std={result.noise_std:.0f}px)")
    print(f"  Q_std      : {result.process_noise_std:.0f}  |  "
          f"R_std: {result.measurement_noise_std:.0f}")
    print(f"{'-' * 70}")
    print(f"  RMSE (noisy raw)      : {result.rmse_noisy:8.2f} px")
    print(f"  RMSE (Kalman filtered): {result.rmse_filtered:8.2f} px")
    print(f"  Noise Reduction Factor: {result.noise_reduction_factor:8.2f}x "
          f"{'(OK)' if result.noise_reduction_factor > 1.0 else '(BELOW 1.0)'}")
    print(f"  Dead-Reckoning RMSE   : {result.dead_reckoning_rmse:8.2f} px")
    print(f"  Official Spec (<=10px): {spec_status(result.rmse_filtered)}")
    print(f"{'=' * 70}")


# =====================================================================
#  ROBUSTNESS CHECK
# =====================================================================

def run_robustness_check(
    trajectories: Dict[str, np.ndarray],
    process_noise_std: float,
    measurement_noise_std: float,
    noise_levels: List[float],
) -> List[RunResult]:
    """Test the chosen parameters across different noise intensities.

    Returns all RunResults for reporting.
    """
    all_results: List[RunResult] = []

    for noise_std in noise_levels:
        print(f"\n  --- Noise std = {noise_std:.0f} px ---")
        for traj_name, gt in trajectories.items():
            result = run_filter(
                ground_truth=gt,
                trajectory_name=traj_name,
                noise_std=noise_std,
                process_noise_std=process_noise_std,
                measurement_noise_std=measurement_noise_std,
            )
            all_results.append(result)
            print(f"    {traj_name:15s}  RMSE_filt={result.rmse_filtered:6.2f}px  "
                  f"NRF={result.noise_reduction_factor:5.2f}x  "
                  f"DR={result.dead_reckoning_rmse:6.2f}px  "
                  f"Spec: {spec_status(result.rmse_filtered)}")

    return all_results


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  Module 4 - BeaconKalmanFilter Verification & Tuning Suite")
    print("  SIH PS-169 - ISRO FSOC Coarse Alignment Tracker")
    print(f"  Official Spec Target: Tracking Error <= {SPEC_TARGET_PX:.0f} px (average)")
    print("=" * 70)

    # -- Generate ground-truth trajectories -------------------------
    NUM_FRAMES = 300
    gt_fig8 = generate_figure8(num_frames=NUM_FRAMES, periods=2.0)
    gt_circ = generate_circular(num_frames=NUM_FRAMES, periods=2.0)
    gt_line = generate_straight_line(num_frames=NUM_FRAMES)

    trajectories = {
        "Straight-Line": gt_line,
        "Circular": gt_circ,
        "Figure-8": gt_fig8,
    }

    # ===============================================================
    #  PHASE 1: Multi-trajectory Grid Search
    # ===============================================================
    print("\n" + "=" * 70)
    print("  PHASE 1: Multi-trajectory Grid Search (optimising for RMSE <= 10px)")
    print("=" * 70)

    Q_VALUES = [3, 5, 8, 10, 12, 15, 18, 20, 25, 30]
    R_VALUES = [1, 2, 3, 4, 5, 6, 8, 10]

    scores = multi_trajectory_grid_search(
        trajectories=trajectories,
        process_noise_values=Q_VALUES,
        measurement_noise_values=R_VALUES,
    )

    print_multi_grid_table(scores, top_n=20)
    export_multi_grid_csv(scores, "grid_search_results.csv")

    # ===============================================================
    #  PHASE 2: Select Best Parameters
    # ===============================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: Best Parameter Selection (lowest avg RMSE, NRF > 1.0)")
    print("=" * 70)

    best = find_best_multi_params(scores)

    if best is None:
        print("  [ERROR] Grid search returned no results. Cannot continue.")
        exit(1)

    best_q = best.process_noise_std
    best_r = best.measurement_noise_std
    print(f"\n  BEST PARAMETERS FOUND:")
    print(f"    process_noise_std     = {best_q:.0f}")
    print(f"    measurement_noise_std = {best_r:.0f}")
    print(f"    Avg Filtered RMSE     : {best.avg_rmse_filtered:.2f} px")
    print(f"    Worst Filtered RMSE   : {best.worst_rmse_filtered:.2f} px")
    print(f"    Min NRF (all trajs)   : {best.min_nrf:.2f}x")
    print(f"    Avg Dead-Reckoning    : {best.avg_dr_rmse:.2f} px")

    # Per-trajectory breakdown
    print(f"\n  PER-TRAJECTORY BREAKDOWN:")
    for tname, r in best.per_trajectory.items():
        print(f"    {tname:15s}: RMSE={r.rmse_filtered:6.2f}px  "
              f"NRF={r.noise_reduction_factor:.2f}x  "
              f"Spec: {spec_status(r.rmse_filtered)}")

    print(f"\n  WHY Q_std={best_q:.0f}, R_std={best_r:.0f} WORKS:")
    print(f"    The Q/R ratio ({best_q/best_r:.1f}:1) creates a measurement-dominated")
    print(f"    filter that trusts fresh detections over the constant-acceleration")
    print(f"    model. This prevents over-extrapolation during sharp direction")
    print(f"    changes (Figure-8 / Circular), while the 6-state CA model still")
    print(f"    provides temporal smoothing that averages out jitter noise.")
    print(f"    R_std={best_r:.0f} (R={best_r**2:.0f} px^2) is lower than the true")
    print(f"    sensor noise (std=20px) because the filter relies on the kinematic")
    print(f"    model's coupling between position, velocity, and acceleration to")
    print(f"    provide additional implicit smoothing beyond what R alone controls.")

    # ===============================================================
    #  PHASE 3: Final Verification with best params
    # ===============================================================
    print("\n" + "=" * 70)
    print("  PHASE 3: Final Verification (all trajectories, best params)")
    print("=" * 70)

    any_spec_met = False
    any_spec_missed = False

    for name, gt in trajectories.items():
        result = run_filter(
            ground_truth=gt,
            trajectory_name=name,
            process_noise_std=best_q,
            measurement_noise_std=best_r,
        )
        print_run_summary(result)

        plot_filename = f"kalman_test_{name.lower().replace('-', '')}.png"
        save_verification_plot(result, plot_filename)

        if result.rmse_filtered <= SPEC_TARGET_PX:
            any_spec_met = True
        else:
            any_spec_missed = True

    # ===============================================================
    #  PHASE 4: Robustness Check (noise mismatch)
    # ===============================================================
    print("\n" + "=" * 70)
    print("  PHASE 4: Robustness Check (noise std = 10, 20, 30 px)")
    print(f"  Testing Q_std={best_q:.0f}, R_std={best_r:.0f} against varied noise levels")
    print("=" * 70)

    robustness_results = run_robustness_check(
        trajectories=trajectories,
        process_noise_std=best_q,
        measurement_noise_std=best_r,
        noise_levels=[10.0, 20.0, 30.0],
    )

    # Export robustness results
    with open("robustness_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "noise_std", "trajectory", "rmse_filtered_px",
            "noise_reduction_factor", "dead_reckoning_rmse_px",
            "meets_spec_10px",
        ])
        for r in robustness_results:
            writer.writerow([
                r.noise_std, r.trajectory_name,
                f"{r.rmse_filtered:.4f}", f"{r.noise_reduction_factor:.4f}",
                f"{r.dead_reckoning_rmse:.4f}",
                "YES" if r.rmse_filtered <= SPEC_TARGET_PX else "NO",
            ])
    print(f"\n  [SAVED] robustness_results.csv")

    # ===============================================================
    #  FINAL VERDICT
    # ===============================================================
    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("-" * 70)
    if not any_spec_missed:
        print("  ALL trajectories MEET the official spec (RMSE <= 10px).")
        print("  Module 4 is ready for integration with Deepthi's PID controller.")
    elif any_spec_met:
        print("  SOME trajectories meet the official spec (RMSE <= 10px).")
        print("  High-curvature trajectories remain above the 10px target.")
        print("  Module 4 filtering is functional but spec gap remains --")
        print("  see Remaining Work in walkthrough.md for next steps.")
    else:
        print("  NO trajectory meets the official spec (RMSE <= 10px).")
        print("  Further filter architecture or tuning improvements needed.")
    print("=" * 70)
