"""
Report Serializers for SIH 2026 Evaluation Suite.
Handles structured CSV export for frame-level metrics and JSON export for benchmark summaries.
"""

import os
import csv
import json
from dataclasses import asdict
from typing import List, Optional

from src.metrics.schemas import FrameMetricRecord, BenchmarkSummary


def save_frame_metrics_csv(records: List[FrameMetricRecord], output_path: str) -> str:
    """
    Saves a list of FrameMetricRecord instances to a CSV file.

    Args:
        records: List of evaluated frame records.
        output_path: Target path for the CSV output.

    Returns:
        Absolute path to the created CSV file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fieldnames = [
        "frame_id",
        "video_timestamp_s",
        "gt_x",
        "gt_y",
        "raw_detected_x",
        "raw_detected_y",
        "filtered_x",
        "filtered_y",
        "confidence",
        "detection_method",
        "lock_state",
        "centroid_error_px",
        "squared_error_px2",
        "processing_latency_ms",
        "instantaneous_fps",
    ]

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            row = asdict(rec)
            # Format None as empty string for CSV cleanliness
            formatted_row = {k: ("" if v is None else v) for k, v in row.items()}
            writer.writerow(formatted_row)

    return os.path.abspath(output_path)


def save_summary_json(summary: BenchmarkSummary, output_path: str, indent: int = 2) -> str:
    """
    Saves a BenchmarkSummary instance to a structured JSON report file.

    Args:
        summary: Aggregated benchmark summary.
        output_path: Target path for the JSON report.
        indent: JSON indentation for human readability.

    Returns:
        Absolute path to the created JSON file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Convert dataclass to structured dictionary
    summary_dict = {
        "benchmark_metadata": {
            "benchmark_type": summary.benchmark_type,
            "video_file": summary.video_file,
            "ground_truth_file": summary.ground_truth_file,
            "execution_timestamp": summary.execution_timestamp,
            "video_resolution": summary.video_resolution,
            "total_video_frames": summary.total_video_frames,
            "video_duration_s": summary.video_duration_s,
            "source_fps": summary.source_fps,
        },
        "tracking_accuracy": {
            "evaluated_frames_count": summary.evaluated_frames_count,
            "mean_centroid_error_px": summary.mean_centroid_error_px,
            "rmse_px": summary.rmse_px,
            "max_error_px": summary.max_error_px,
            "std_dev_error_px": summary.std_dev_error_px,
        },
        "lock_performance": {
            "lock_retention_rate_pct": summary.lock_retention_rate_pct,
            "target_loss_rate_pct": summary.target_loss_rate_pct,
            "acquisition_time_s": summary.acquisition_time_s,
            "target_loss_event_count": summary.target_loss_event_count,
            "total_target_loss_duration_s": summary.total_target_loss_duration_s,
            "reacquisition_time_stats": summary.reacquisition_time_stats,
            "lock_state_frame_counts": summary.lock_state_frame_counts,
        },
        "system_performance": {
            "average_processing_fps": summary.average_processing_fps,
            "latency_ms_stats": summary.latency_ms_stats,
        },
    }

    with open(output_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=indent)

    return os.path.abspath(output_path)
