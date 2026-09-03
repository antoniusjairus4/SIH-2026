"""
Data contracts and schemas for SIH 2026 Virtual Camera Tracking System.
Defines data structures for Ground Truth, Detection, Tracking, Telemetry, and Evaluation Metrics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List


class LockState(str, Enum):
    """
    Canonical lock state machine vocabulary for the FSOC tracking system.
    """
    SEARCH = "SEARCH"
    ACQUIRE = "ACQUIRE"
    TRACK = "TRACK"
    COAST = "COAST"
    REACQUIRE = "REACQUIRE"

    @classmethod
    def from_string(cls, state_str: Optional[str]) -> "LockState":
        if state_str is None:
            return cls.SEARCH
        cleaned = state_str.strip().upper()
        for member in cls:
            if member.value == cleaned:
                return member
        return cls.SEARCH


@dataclass
class GroundTruthRecord:
    """
    Ground truth coordinate for a specific video frame.
    Strictly for evaluation/metrics - never passed to perception/control pipeline.
    """
    frame_id: int
    gt_x: Optional[float]
    gt_y: Optional[float]
    timestamp: Optional[float] = None
    is_occluded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """
    Output contract for Beacon Detection Engine.
    """
    detected: bool
    centroid_x: Optional[float] = None
    centroid_y: Optional[float] = None
    confidence: float = 0.0
    method_used: str = "NONE"  # e.g., "FAST_PATH", "AI_YOLO", "NONE"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackingResult:
    """
    Output contract for State Estimator (Kalman / EKF).
    """
    lock_state: LockState = LockState.SEARCH
    filtered_x: Optional[float] = None
    filtered_y: Optional[float] = None
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    covariance_trace: Optional[float] = None
    is_valid_track: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryRecord:
    """
    Per-frame diagnostic telemetry emitted during live or offline execution.
    """
    frame_id: int
    video_timestamp: float
    processing_latency_ms: float
    detector_status: bool
    raw_x: Optional[float]
    raw_y: Optional[float]
    confidence: float
    detection_method: str
    lock_state: LockState
    filtered_x: Optional[float]
    filtered_y: Optional[float]
    velocity_x: Optional[float] = None
    velocity_y: Optional[float] = None
    is_valid_track: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameMetricRecord:
    """
    Per-frame combined evaluation record containing telemetry, ground truth, and errors.
    """
    frame_id: int
    video_timestamp_s: float
    gt_x: Optional[float]
    gt_y: Optional[float]
    raw_detected_x: Optional[float]
    raw_detected_y: Optional[float]
    filtered_x: Optional[float]
    filtered_y: Optional[float]
    confidence: float
    detection_method: str
    lock_state: str
    centroid_error_px: Optional[float]
    squared_error_px2: Optional[float]
    processing_latency_ms: float
    instantaneous_fps: Optional[float]


@dataclass
class BenchmarkSummary:
    """
    Aggregated statistical summary of benchmark execution.
    """
    # Benchmark Metadata
    benchmark_type: str
    video_file: str
    ground_truth_file: Optional[str]
    execution_timestamp: str
    video_resolution: Dict[str, int]
    total_video_frames: int
    video_duration_s: float
    source_fps: float

    # Accuracy Metrics
    evaluated_frames_count: int
    mean_centroid_error_px: Optional[float]
    rmse_px: Optional[float]
    max_error_px: Optional[float]
    std_dev_error_px: Optional[float]

    # Lock Performance
    lock_retention_rate_pct: float
    target_loss_rate_pct: float
    acquisition_time_s: Optional[float]
    target_loss_event_count: int
    total_target_loss_duration_s: float
    reacquisition_time_stats: Dict[str, Any]
    lock_state_frame_counts: Dict[str, int]

    # System Performance
    average_processing_fps: float
    latency_ms_stats: Dict[str, Optional[float]]
