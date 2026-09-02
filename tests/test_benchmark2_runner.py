"""
Integration tests for Benchmark-2 runner with synthetic video generation and ground truth.
"""

import os
import csv
import json
import tempfile
import pytest
import cv2
import numpy as np

from src.benchmark.benchmark2_runner import (
    Benchmark2Runner,
    VideoProcessingError,
)
from src.benchmark.pipeline_adapter import (
    CombinedTrackingPipeline,
    MockThresholdDetector,
    MockSimpleTracker,
    NullDetector,
    NullStateEstimator,
)


def _generate_synthetic_video_and_gt(
    output_dir: str,
    num_frames: int = 30,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> tuple[str, str]:
    """
    Helper to synthesize a test MP4 video containing a moving optical spot and matching ground truth CSV.
    """
    video_path = os.path.join(output_dir, "synthetic_beacon.mp4")
    gt_path = os.path.join(output_dir, "synthetic_beacon_gt.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=True)

    gt_rows = [["frame_id", "gt_x", "gt_y", "timestamp"]]

    # Linear motion trajectory
    start_x, start_y = 300.0, 200.0
    vx, vy = 2.0, 1.0  # px per frame

    for fid in range(num_frames):
        ts = fid / fps
        x = start_x + fid * vx
        y = start_y + fid * vy

        # Create dark background frame (simulating dark-space FPA)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw optical beacon spot (10x10 px Gaussian spot / circle)
        cv2.circle(frame, (int(round(x)), int(round(y))), radius=4, color=(255, 255, 255), thickness=-1)

        writer.write(frame)
        gt_rows.append([str(fid), f"{x:.2f}", f"{y:.2f}", f"{ts:.4f}"])

    writer.release()

    with open(gt_path, mode="w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(gt_rows)

    return video_path, gt_path


def test_synthetic_end_to_end_benchmark2_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path, gt_path = _generate_synthetic_video_and_gt(tmpdir, num_frames=30)
        output_results_dir = os.path.join(tmpdir, "results")

        pipeline = CombinedTrackingPipeline(
            detector=MockThresholdDetector(threshold=180),
            state_estimator=MockSimpleTracker(coast_limit_frames=5),
        )

        runner = Benchmark2Runner(pipeline=pipeline, output_dir=output_results_dir)
        summary, csv_path, json_path = runner.run_benchmark(
            video_path=video_path,
            ground_truth_path=gt_path,
            save_csv=True,
            save_json=True,
        )

        # 1. Check summary metrics
        assert summary.total_video_frames == 30
        assert summary.evaluated_frames_count == 30
        assert summary.lock_retention_rate_pct == 100.0
        assert summary.target_loss_event_count == 0
        assert summary.acquisition_time_s == 0.0
        assert summary.mean_centroid_error_px is not None
        assert summary.mean_centroid_error_px < 1.0  # Sub-pixel accuracy against synthetic GT circle center
        assert summary.rmse_px is not None
        assert summary.rmse_px < 1.0
        assert summary.average_processing_fps > 0.0

        # 2. Check exported files
        assert csv_path is not None and os.path.exists(csv_path)
        assert json_path is not None and os.path.exists(json_path)

        # 3. Validate JSON content structure
        with open(json_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["benchmark_metadata"]["total_video_frames"] == 30
            assert data["lock_performance"]["lock_retention_rate_pct"] == 100.0
            assert data["tracking_accuracy"]["mean_centroid_error_px"] < 1.0


def test_missing_video_raises_error():
    runner = Benchmark2Runner()
    with pytest.raises(FileNotFoundError):
        runner.run_benchmark("non_existent_video_path_9999.mp4")


def test_execution_without_ground_truth():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path, _ = _generate_synthetic_video_and_gt(tmpdir, num_frames=15)
        output_results_dir = os.path.join(tmpdir, "results")

        pipeline = CombinedTrackingPipeline(
            detector=MockThresholdDetector(threshold=180),
            state_estimator=MockSimpleTracker(coast_limit_frames=5),
        )

        runner = Benchmark2Runner(pipeline=pipeline, output_dir=output_results_dir)
        summary, csv_path, json_path = runner.run_benchmark(
            video_path=video_path,
            ground_truth_path=None,  # No ground truth
            auto_discover_gt=False,
        )

        assert summary.total_video_frames == 15
        assert summary.evaluated_frames_count == 0
        assert summary.mean_centroid_error_px is None
        assert summary.rmse_px is None
        assert summary.lock_retention_rate_pct == 100.0


def test_zero_lock_run_with_null_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path, gt_path = _generate_synthetic_video_and_gt(tmpdir, num_frames=10)
        output_results_dir = os.path.join(tmpdir, "results")

        # Null pipeline will never detect or track
        pipeline = CombinedTrackingPipeline(
            detector=NullDetector(),
            state_estimator=NullStateEstimator(),
        )

        runner = Benchmark2Runner(pipeline=pipeline, output_dir=output_results_dir)
        summary, _, _ = runner.run_benchmark(
            video_path=video_path,
            ground_truth_path=gt_path,
        )

        assert summary.total_video_frames == 10
        assert summary.evaluated_frames_count == 0
        assert summary.lock_retention_rate_pct == 0.0
        assert summary.acquisition_time_s is None
        assert summary.mean_centroid_error_px is None
