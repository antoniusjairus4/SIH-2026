"""
Benchmark-2 Execution Runner for SIH 2026.
Executes frame-by-frame evaluation on pre-recorded .mp4 video files,
routing frames through the perception and state-estimation pipeline (bypassing PTZ actuation)
and streaming telemetry to the automated MetricLogger.
"""

import os
import sys
import time
import logging
import argparse
from typing import Optional, Tuple

import cv2
import numpy as np

from src.metrics.schemas import (
    TelemetryRecord,
    BenchmarkSummary,
    LockState,
)
from src.metrics.metric_logger import MetricLogger
from src.metrics.reporters import save_frame_metrics_csv, save_summary_json
from src.benchmark.ground_truth_loader import GroundTruthLoader
from src.benchmark.pipeline_adapter import (
    CombinedTrackingPipeline,
    BeaconDetectorProtocol,
    StateEstimatorProtocol,
    NullDetector,
    NullStateEstimator,
    MockThresholdDetector,
    MockSimpleTracker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Benchmark2Runner")


class VideoProcessingError(Exception):
    """Raised when video file cannot be opened or decoded."""
    pass


class Benchmark2Runner:
    """
    Harness for executing Benchmark-2 (.mp4 evaluation) on FSOC tracking pipelines.
    """

    def __init__(
        self,
        pipeline: Optional[CombinedTrackingPipeline] = None,
        output_dir: str = "results/benchmark2",
    ) -> None:
        self.pipeline = pipeline or CombinedTrackingPipeline(
            detector=NullDetector(),
            state_estimator=NullStateEstimator(),
        )
        self.output_dir = output_dir

    def run_benchmark(
        self,
        video_path: str,
        ground_truth_path: Optional[str] = None,
        auto_discover_gt: bool = True,
        save_csv: bool = True,
        save_json: bool = True,
    ) -> Tuple[BenchmarkSummary, Optional[str], Optional[str]]:
        """
        Executes frame-by-frame evaluation over a video file.

        Args:
            video_path: Path to the .mp4 file.
            ground_truth_path: Optional path to companion ground truth CSV.
            auto_discover_gt: Whether to auto-search for companion GT file if not explicitly provided.
            save_csv: Whether to write frame-level CSV report.
            save_json: Whether to write benchmark summary JSON.

        Returns:
            Tuple of (BenchmarkSummary, csv_file_path, json_file_path).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video file: {video_path}")

        try:
            # Read video stream metadata
            total_frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0 or np.isnan(fps) or np.isinf(fps):
                fps = 30.0  # Fallback to nominal 30 FPS

            logger.info(
                f"Starting Benchmark-2 on: {video_path} "
                f"({width}x{height} @ {fps:.1f} FPS, ~{total_frames_reported} frames)"
            )

            # Resolve companion ground truth if not explicitly provided
            gt_path_resolved = ground_truth_path
            if not gt_path_resolved and auto_discover_gt:
                candidate_gt = self._discover_ground_truth(video_path)
                if candidate_gt:
                    gt_path_resolved = candidate_gt
                    logger.info(f"Auto-discovered companion ground truth: {gt_path_resolved}")

            gt_loader = GroundTruthLoader(gt_path_resolved) if gt_path_resolved else None

            # Initialize Metric Logger
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            logger_engine = MetricLogger(
                benchmark_name="BENCHMARK_2_OFFLINE_MP4",
                source_fps=fps,
                video_resolution={"width": width, "height": height},
                video_file=video_path,
                ground_truth_file=gt_path_resolved,
            )

            # Reset tracking pipeline
            self.pipeline.reset()

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Determine video timestamp
                video_ts = frame_id / fps

                # Process through perception and state estimator
                t_start = time.perf_counter()
                detection, tracking = self.pipeline.process_frame(frame, frame_id, video_ts)
                t_end = time.perf_counter()

                latency_ms = max(0.001, (t_end - t_start) * 1000.0)

                # Assemble TelemetryRecord (Ground truth is NEVER passed here)
                telemetry = TelemetryRecord(
                    frame_id=frame_id,
                    video_timestamp=video_ts,
                    processing_latency_ms=latency_ms,
                    detector_status=detection.detected,
                    raw_x=detection.centroid_x,
                    raw_y=detection.centroid_y,
                    confidence=detection.confidence,
                    detection_method=detection.method_used,
                    lock_state=tracking.lock_state,
                    filtered_x=tracking.filtered_x,
                    filtered_y=tracking.filtered_y,
                    velocity_x=tracking.velocity_x,
                    velocity_y=tracking.velocity_y,
                    is_valid_track=tracking.is_valid_track,
                    metadata=tracking.metadata,
                )

                # Retrieve ground truth strictly for evaluation
                gt_record = gt_loader.get_ground_truth(frame_id) if gt_loader else None

                # Log frame in evaluation engine
                logger_engine.log_frame(telemetry, gt_record)

                frame_id += 1

            logger.info(f"Completed processing {frame_id} frames.")

            # Compute summary
            summary = logger_engine.compute_summary()

            # Save Reports
            csv_path: Optional[str] = None
            json_path: Optional[str] = None

            if save_csv:
                csv_filename = os.path.join(self.output_dir, f"{video_basename}_frame_metrics.csv")
                csv_path = save_frame_metrics_csv(logger_engine.frame_records, csv_filename)
                logger.info(f"Frame metrics CSV written to: {csv_path}")

            if save_json:
                json_filename = os.path.join(self.output_dir, f"{video_basename}_summary.json")
                json_path = save_summary_json(summary, json_filename)
                logger.info(f"Benchmark summary JSON written to: {json_path}")

            return summary, csv_path, json_path

        finally:
            cap.release()

    def _discover_ground_truth(self, video_path: str) -> Optional[str]:
        """Looks for common companion ground-truth naming patterns."""
        base_dir = os.path.dirname(video_path)
        stem = os.path.splitext(os.path.basename(video_path))[0]

        candidates = [
            os.path.join(base_dir, f"{stem}_ground_truth.csv"),
            os.path.join(base_dir, f"{stem}_gt.csv"),
            os.path.join(base_dir, f"{stem}.csv"),
            os.path.join("data", "benchmark2", f"{stem}_ground_truth.csv"),
            os.path.join("data", "benchmark2", f"{stem}_gt.csv"),
        ]

        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return None


def main() -> None:
    """CLI entry point for Benchmark-2 runner."""
    parser = argparse.ArgumentParser(
        description="SIH 2026 Benchmark-2: Offline .mp4 Centroiding & Tracking Evaluator"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input .mp4 video file")
    parser.add_argument("--gt", type=str, default=None, help="Path to companion ground-truth CSV")
    parser.add_argument(
        "--output-dir", type=str, default="results/benchmark2", help="Directory to save evaluation reports"
    )
    parser.add_argument(
        "--use-mock-pipeline",
        action="store_true",
        help="Use built-in mock threshold detector and tracker for testing",
    )

    args = parser.parse_args()

    if args.use_mock_pipeline:
        logger.info("Instantiating harness with test mock perception/tracker adapter...")
        pipeline = CombinedTrackingPipeline(
            detector=MockThresholdDetector(threshold=180),
            state_estimator=MockSimpleTracker(coast_limit_frames=5),
        )
    else:
        logger.info("Instantiating harness with standard Null adapters (pass-through)...")
        pipeline = CombinedTrackingPipeline(
            detector=NullDetector(),
            state_estimator=NullStateEstimator(),
        )

    runner = Benchmark2Runner(pipeline=pipeline, output_dir=args.output_dir)
    try:
        summary, csv_p, json_p = runner.run_benchmark(
            video_path=args.video,
            ground_truth_path=args.gt,
        )
        print("\n=======================================================")
        print("           BENCHMARK-2 EXECUTION SUMMARY               ")
        print("=======================================================")
        print(f"Video:             {summary.video_file}")
        print(f"Frames Processed:  {summary.total_video_frames}")
        print(f"Processing FPS:    {summary.average_processing_fps:.1f}")
        print(f"Lock Retention:    {summary.lock_retention_rate_pct:.1f}%")
        print(f"Acquisition Time:  {summary.acquisition_time_s} s")
        if summary.mean_centroid_error_px is not None:
            print(f"Mean Error:        {summary.mean_centroid_error_px:.3f} px")
            print(f"RMSE:              {summary.rmse_px:.3f} px")
            print(f"Max Error:         {summary.max_error_px:.3f} px")
        else:
            print("Spatial Errors:    N/A (No Ground Truth)")
        print(f"Summary JSON:      {json_p}")
        print(f"Frame CSV:         {csv_p}")
        print("=======================================================\n")
    except Exception as e:
        logger.error(f"Benchmark-2 execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
