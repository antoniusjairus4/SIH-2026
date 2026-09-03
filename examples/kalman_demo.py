import math
import time

import matplotlib.pyplot as plt
import numpy as np

from src.estimation import BeaconStateEstimator, EstimatorConfig


# ==============================================================
# SIMULATION CONFIGURATION
# ==============================================================

SEED = 42
rng = np.random.default_rng(SEED)

FPS = 30.0
DT = 1.0 / FPS
TOTAL_FRAMES = 300

# Temporary detector loss / occlusion
DROPOUT_START = 140
DROPOUT_END = 165

# One deliberately false detector observation
OUTLIER_FRAME = 220


# ==============================================================
# GROUND-TRUTH TARGET TRAJECTORY
# ==============================================================

def true_trajectory(t: float) -> tuple[float, float]:
    """
    Generate the known ground-truth trajectory of the virtual
    FSOC beacon.

    The trajectory contains forward motion together with smooth
    nonlinear movement so that the Kalman estimator is tested
    on a realistic changing target path.

    Returns:
        x, y position in pixels.
    """

    x = (
        120.0
        + 45.0 * t
        + 8.0 * math.sin(0.8 * t)
    )

    y = (
        180.0
        + 18.0 * t
        + 12.0 * math.sin(0.5 * t)
    )

    return x, y


# ==============================================================
# RMSE CALCULATION
# ==============================================================

def rmse(
    truth: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """
    Calculate 2D positional Root Mean Square Error.
    """

    error = truth - estimate

    distance_squared = np.sum(
        error ** 2,
        axis=1,
    )

    return float(
        np.sqrt(
            np.mean(distance_squared)
        )
    )


# ==============================================================
# MAIN DEMONSTRATION
# ==============================================================

def main():

    # ----------------------------------------------------------
    # Create estimator
    # ----------------------------------------------------------

    config = EstimatorConfig()

    estimator = BeaconStateEstimator(config)

    # ----------------------------------------------------------
    # Storage
    # ----------------------------------------------------------

    truth_points = []
    raw_points = []
    filtered_points = []

    valid_raw_truth = []
    valid_raw_measurements = []

    filtered_truth = []
    filtered_measurements = []

    timestamps = []

    processing_times = []

    rejected_count = 0
    prediction_only_count = 0
    max_coast_time = 0.0

    # ----------------------------------------------------------
    # Run simulation
    # ----------------------------------------------------------

    for frame in range(TOTAL_FRAMES):

        timestamp = frame * DT

        true_x, true_y = true_trajectory(timestamp)

        truth_points.append(
            [true_x, true_y]
        )

        timestamps.append(timestamp)

        # ------------------------------------------------------
        # Simulated detector noise
        # ------------------------------------------------------

        gaussian_noise_x = rng.normal(
            0.0,
            5.0,
        )

        gaussian_noise_y = rng.normal(
            0.0,
            5.0,
        )

        # Simulate platform/camera jitter of up to ±20 pixels.
        jitter_x = rng.uniform(
            -20.0,
            20.0,
        )

        jitter_y = rng.uniform(
            -20.0,
            20.0,
        )

        detected_x = (
            true_x
            + gaussian_noise_x
            + jitter_x
        )

        detected_y = (
            true_y
            + gaussian_noise_y
            + jitter_y
        )

        confidence = float(
            rng.uniform(
                0.75,
                0.98,
            )
        )

        # ------------------------------------------------------
        # Simulated temporary detector dropout
        # ------------------------------------------------------

        if DROPOUT_START <= frame <= DROPOUT_END:

            input_x = None
            input_y = None
            confidence = 0.0

            raw_points.append(
                [np.nan, np.nan]
            )

        else:

            input_x = detected_x
            input_y = detected_y

            raw_points.append(
                [
                    detected_x,
                    detected_y,
                ]
            )

            valid_raw_truth.append(
                [
                    true_x,
                    true_y,
                ]
            )

            valid_raw_measurements.append(
                [
                    detected_x,
                    detected_y,
                ]
            )

        # ------------------------------------------------------
        # Inject one extreme false measurement
        # ------------------------------------------------------

        if frame == OUTLIER_FRAME:

            input_x = 1500.0
            input_y = -800.0
            confidence = 0.97

            raw_points[-1] = [
                input_x,
                input_y,
            ]

            valid_raw_measurements[-1] = [
                input_x,
                input_y,
            ]

        # ------------------------------------------------------
        # Execute Kalman state estimator
        # ------------------------------------------------------

        start = time.perf_counter()

        result = estimator.step(
            x=input_x,
            y=input_y,
            confidence=confidence,
            timestamp=timestamp,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        processing_times.append(
            elapsed
        )

        # ------------------------------------------------------
        # Collect estimator statistics
        # ------------------------------------------------------

        if result.measurement_rejected:
            rejected_count += 1

        if result.prediction_only:
            prediction_only_count += 1

        max_coast_time = max(
            max_coast_time,
            result.coast_time,
        )

        # ------------------------------------------------------
        # Store Kalman estimate
        # ------------------------------------------------------

        if result.x is not None:

            filtered_points.append(
                [
                    result.x,
                    result.y,
                ]
            )

            filtered_truth.append(
                [
                    true_x,
                    true_y,
                ]
            )

            filtered_measurements.append(
                [
                    result.x,
                    result.y,
                ]
            )

        else:

            filtered_points.append(
                [
                    np.nan,
                    np.nan,
                ]
            )

    # ==========================================================
    # CONVERT RESULTS TO NUMPY ARRAYS
    # ==========================================================

    truth_points = np.asarray(
        truth_points,
        dtype=float,
    )

    raw_points = np.asarray(
        raw_points,
        dtype=float,
    )

    filtered_points = np.asarray(
        filtered_points,
        dtype=float,
    )

    valid_raw_truth = np.asarray(
        valid_raw_truth,
        dtype=float,
    )

    valid_raw_measurements = np.asarray(
        valid_raw_measurements,
        dtype=float,
    )

    filtered_truth = np.asarray(
        filtered_truth,
        dtype=float,
    )

    filtered_measurements = np.asarray(
        filtered_measurements,
        dtype=float,
    )

    timestamps_array = np.asarray(
        timestamps,
        dtype=float,
    )

    # ==========================================================
    # PERFORMANCE METRICS
    # ==========================================================

    # ----------------------------------------------------------
    # Raw detector RMSE INCLUDING intentional false target
    # ----------------------------------------------------------

    raw_rmse = rmse(
        valid_raw_truth,
        valid_raw_measurements,
    )

    # ----------------------------------------------------------
    # Kalman RMSE
    # ----------------------------------------------------------

    filtered_rmse = rmse(
        filtered_truth,
        filtered_measurements,
    )

    # ----------------------------------------------------------
    # Fair raw RMSE EXCLUDING deliberate extreme outlier
    # ----------------------------------------------------------

    raw_without_outlier_truth = []
    raw_without_outlier_measurements = []

    for frame in range(TOTAL_FRAMES):

        # Ignore frames where detector has no measurement.
        if DROPOUT_START <= frame <= DROPOUT_END:
            continue

        # Ignore intentionally injected extreme false target.
        if frame == OUTLIER_FRAME:
            continue

        raw_without_outlier_truth.append(
            truth_points[frame]
        )

        raw_without_outlier_measurements.append(
            raw_points[frame]
        )

    raw_without_outlier_truth = np.asarray(
        raw_without_outlier_truth,
        dtype=float,
    )

    raw_without_outlier_measurements = np.asarray(
        raw_without_outlier_measurements,
        dtype=float,
    )

    raw_rmse_without_outlier = rmse(
        raw_without_outlier_truth,
        raw_without_outlier_measurements,
    )

    # ----------------------------------------------------------
    # Filtered positional errors
    # ----------------------------------------------------------

    filtered_error = np.linalg.norm(
        filtered_truth
        - filtered_measurements,
        axis=1,
    )

    mean_filtered_error = float(
        np.mean(
            filtered_error
        )
    )

    max_filtered_error = float(
        np.max(
            filtered_error
        )
    )

    # ----------------------------------------------------------
    # Processing performance
    # ----------------------------------------------------------

    average_processing_time = float(
        np.mean(
            processing_times
        )
    )

    if average_processing_time > 0:

        estimated_processing_fps = (
            1.0
            / average_processing_time
        )

    else:

        estimated_processing_fps = (
            float("inf")
        )

    # ----------------------------------------------------------
    # Fair RMSE improvement
    #
    # IMPORTANT:
    # We compare against raw detector RMSE with the deliberate
    # extreme outlier removed. This prevents the improvement
    # percentage from being artificially inflated.
    # ----------------------------------------------------------

    improvement = (
        (
            raw_rmse_without_outlier
            - filtered_rmse
        )
        / raw_rmse_without_outlier
        * 100.0
    )

    # ==========================================================
    # PRINT RESULTS
    # ==========================================================

    print()

    print(
        "=============================================="
    )

    print(
        "FSOC KALMAN STATE ESTIMATOR DEMONSTRATION"
    )

    print(
        "=============================================="
    )

    print(
        f"Raw RMSE incl. outlier  : "
        f"{raw_rmse:.3f} pixels"
    )

    print(
        f"Raw RMSE excl. outlier  : "
        f"{raw_rmse_without_outlier:.3f} pixels"
    )

    print(
        f"Kalman filtered RMSE    : "
        f"{filtered_rmse:.3f} pixels"
    )

    print(
        f"Fair RMSE improvement   : "
        f"{improvement:.2f}%"
    )

    print(
        f"Mean filtered error     : "
        f"{mean_filtered_error:.3f} pixels"
    )

    print(
        f"Maximum filtered error  : "
        f"{max_filtered_error:.3f} pixels"
    )

    print(
        f"Rejected measurements   : "
        f"{rejected_count}"
    )

    print(
        f"Prediction-only frames  : "
        f"{prediction_only_count}"
    )

    print(
        f"Maximum coast duration  : "
        f"{max_coast_time:.3f} seconds"
    )

    print(
        f"Average estimator time  : "
        f"{average_processing_time * 1000:.4f} ms"
    )

    print(
        f"Estimator throughput    : "
        f"{estimated_processing_fps:.1f} FPS"
    )

    print(
        "=============================================="
    )

    print()

    # ==========================================================
    # VISUALIZATION
    # ==========================================================

    plt.figure(
        figsize=(12, 6)
    )

    # Ground truth
    plt.plot(
        timestamps_array,
        truth_points[:, 0],
        label="Ground Truth X",
        linewidth=2,
    )

    # Raw detector output
    plt.scatter(
        timestamps_array,
        raw_points[:, 0],
        label="Raw Detector X",
        s=10,
        alpha=0.45,
    )

    # Kalman output
    plt.plot(
        timestamps_array,
        filtered_points[:, 0],
        label="Kalman Estimate X",
        linewidth=2,
    )

    # ----------------------------------------------------------
    # Mark detector dropout period
    # ----------------------------------------------------------

    dropout_start_time = (
        DROPOUT_START
        * DT
    )

    dropout_end_time = (
        DROPOUT_END
        * DT
    )

    plt.axvspan(
        dropout_start_time,
        dropout_end_time,
        alpha=0.15,
        label="Detector Dropout",
    )

    # ----------------------------------------------------------
    # Mark deliberate outlier
    # ----------------------------------------------------------

    outlier_time = (
        OUTLIER_FRAME
        * DT
    )

    plt.axvline(
        outlier_time,
        linestyle="--",
        alpha=0.5,
        label="Injected Outlier",
    )

    # ----------------------------------------------------------
    # Graph labels
    # ----------------------------------------------------------

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Horizontal Beacon Position (pixels)"
    )

    plt.title(
        "FSOC Beacon Tracking — "
        "Raw Detection vs Kalman Estimate"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    # ----------------------------------------------------------
    # Save high-resolution result
    # ----------------------------------------------------------

    plt.savefig(
        "examples/kalman_tracking_demo.png",
        dpi=200,
    )

    print(
        "Graph saved to: "
        "examples/kalman_tracking_demo.png"
    )

    # Display graph
    plt.show()


# ==============================================================
# PROGRAM ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    main()