"""
Unit tests and edge-case validation for CSV and JSON report serialisation.
"""

import os
import csv
import json
import tempfile
import pytest
from src.metrics.schemas import FrameMetricRecord, BenchmarkSummary
from src.metrics.reporters import save_frame_metrics_csv, save_summary_json


def test_save_frame_metrics_csv():
    records = [
        FrameMetricRecord(
            frame_id=0,
            video_timestamp_s=0.0,
            gt_x=320.0,
            gt_y=240.0,
            raw_detected_x=321.0,
            raw_detected_y=240.5,
            filtered_x=320.8,
            filtered_y=240.2,
            confidence=0.95,
            detection_method="FAST_PATH",
            lock_state="TRACK",
            centroid_error_px=0.8246,
            squared_error_px2=0.68,
            processing_latency_ms=12.5,
            instantaneous_fps=80.0,
        ),
        FrameMetricRecord(
            frame_id=1,
            video_timestamp_s=0.033,
            gt_x=None,
            gt_y=None,
            raw_detected_x=None,
            raw_detected_y=None,
            filtered_x=None,
            filtered_y=None,
            confidence=0.0,
            detection_method="NONE",
            lock_state="SEARCH",
            centroid_error_px=None,
            squared_error_px2=None,
            processing_latency_ms=5.0,
            instantaneous_fps=200.0,
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_metrics.csv")
        saved_path = save_frame_metrics_csv(records, csv_path)
        assert os.path.exists(saved_path)

        with open(saved_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 2
            assert rows[0]["frame_id"] == "0"
            assert rows[0]["lock_state"] == "TRACK"
            assert rows[0]["centroid_error_px"] == "0.8246"
            assert rows[1]["frame_id"] == "1"
            assert rows[1]["gt_x"] == ""  # None formatted as empty string
            assert rows[1]["lock_state"] == "SEARCH"


def test_save_summary_json():
    summary = BenchmarkSummary(
        benchmark_type="BENCHMARK_2_OFFLINE_MP4",
        video_file="test.mp4",
        ground_truth_file="test_gt.csv",
        execution_timestamp="2026-09-02T20:00:00Z",
        video_resolution={"width": 640, "height": 480},
        total_video_frames=100,
        video_duration_s=3.333,
        source_fps=30.0,
        evaluated_frames_count=95,
        mean_centroid_error_px=2.15,
        rmse_px=2.85,
        max_error_px=6.5,
        std_dev_error_px=0.95,
        lock_retention_rate_pct=95.0,
        target_loss_rate_pct=5.0,
        acquisition_time_s=0.066,
        target_loss_event_count=1,
        total_target_loss_duration_s=0.166,
        reacquisition_time_stats={"count": 1, "mean_s": 0.166, "min_s": 0.166, "max_s": 0.166, "median_s": 0.166},
        lock_state_frame_counts={"SEARCH": 2, "ACQUIRE": 0, "TRACK": 95, "COAST": 3, "REACQUIRE": 0},
        average_processing_fps=65.2,
        latency_ms_stats={"mean_ms": 15.3, "median_ms": 14.8, "p95_ms": 18.0, "min_ms": 10.0, "max_ms": 22.0},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_summary.json")
        saved_path = save_summary_json(summary, json_path)
        assert os.path.exists(saved_path)

        with open(saved_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            assert "benchmark_metadata" in data
            assert data["benchmark_metadata"]["video_file"] == "test.mp4"
            assert "tracking_accuracy" in data
            assert data["tracking_accuracy"]["rmse_px"] == 2.85
            assert "lock_performance" in data
            assert data["lock_performance"]["lock_retention_rate_pct"] == 95.0
            assert "system_performance" in data
            assert data["system_performance"]["average_processing_fps"] == 65.2


def test_empty_results_export():
    records = []
    summary = BenchmarkSummary(
        benchmark_type="BENCHMARK_2_OFFLINE_MP4",
        video_file="empty.mp4",
        ground_truth_file=None,
        execution_timestamp="2026-09-02T20:00:00Z",
        video_resolution={"width": 640, "height": 480},
        total_video_frames=0,
        video_duration_s=0.0,
        source_fps=30.0,
        evaluated_frames_count=0,
        mean_centroid_error_px=None,
        rmse_px=None,
        max_error_px=None,
        std_dev_error_px=None,
        lock_retention_rate_pct=0.0,
        target_loss_rate_pct=0.0,
        acquisition_time_s=None,
        target_loss_event_count=0,
        total_target_loss_duration_s=0.0,
        reacquisition_time_stats={"count": 0, "mean_s": None, "min_s": None, "max_s": None, "median_s": None},
        lock_state_frame_counts={"SEARCH": 0, "ACQUIRE": 0, "TRACK": 0, "COAST": 0, "REACQUIRE": 0},
        average_processing_fps=0.0,
        latency_ms_stats={"mean_ms": None, "median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None},
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "empty_metrics.csv")
        json_path = os.path.join(tmpdir, "empty_summary.json")

        save_frame_metrics_csv(records, csv_path)
        save_summary_json(summary, json_path)

        # CSV should have header line and 0 data rows
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert len(list(reader)) == 0

        # JSON should parse with null fields
        with open(json_path, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["benchmark_metadata"]["total_video_frames"] == 0
            assert data["tracking_accuracy"]["rmse_px"] is None


def test_nested_directory_creation():
    records = []
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_csv = os.path.join(tmpdir, "level1", "level2", "deep_metrics.csv")
        save_frame_metrics_csv(records, nested_csv)
        assert os.path.exists(nested_csv)


def test_overwrite_existing_files():
    rec1 = FrameMetricRecord(
        frame_id=0, video_timestamp_s=0.0, gt_x=10.0, gt_y=10.0,
        raw_detected_x=10.0, raw_detected_y=10.0, filtered_x=10.0, filtered_y=10.0,
        confidence=1.0, detection_method="TEST", lock_state="TRACK",
        centroid_error_px=0.0, squared_error_px2=0.0, processing_latency_ms=10.0, instantaneous_fps=100.0
    )
    rec2 = FrameMetricRecord(
        frame_id=1, video_timestamp_s=0.033, gt_x=20.0, gt_y=20.0,
        raw_detected_x=20.0, raw_detected_y=20.0, filtered_x=20.0, filtered_y=20.0,
        confidence=1.0, detection_method="TEST", lock_state="TRACK",
        centroid_error_px=0.0, squared_error_px2=0.0, processing_latency_ms=10.0, instantaneous_fps=100.0
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "overwrite_test.csv")
        # First save with 1 record
        save_frame_metrics_csv([rec1], csv_path)
        with open(csv_path, mode="r", encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 1

        # Second save with 2 records
        save_frame_metrics_csv([rec1, rec2], csv_path)
        with open(csv_path, mode="r", encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == 2
