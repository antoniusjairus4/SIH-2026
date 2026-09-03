"""
Metrics and evaluation package for SIH 2026 Virtual Camera Tracking System.
"""

from src.metrics.schemas import (
    LockState,
    GroundTruthRecord,
    DetectionResult,
    TrackingResult,
    TelemetryRecord,
    FrameMetricRecord,
    BenchmarkSummary,
)
from src.metrics.metric_logger import MetricLogger
from src.metrics.reporters import save_frame_metrics_csv, save_summary_json

__all__ = [
    "LockState",
    "GroundTruthRecord",
    "DetectionResult",
    "TrackingResult",
    "TelemetryRecord",
    "FrameMetricRecord",
    "BenchmarkSummary",
    "MetricLogger",
    "save_frame_metrics_csv",
    "save_summary_json",
]
