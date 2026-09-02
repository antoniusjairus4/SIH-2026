"""
Automated Performance Metric Logger for SIH 2026 Virtual Camera Tracking System.
Consumes diagnostic telemetry from perception/tracking pipelines (offline Benchmark-2 or live Unity Benchmark-1),
tracks canonical lock-state transitions, and computes spatial and temporal performance metrics.
"""

import math
import time
from typing import Optional, List, Dict, Any

from src.metrics.schemas import (
    LockState,
    TelemetryRecord,
    GroundTruthRecord,
    FrameMetricRecord,
    BenchmarkSummary,
)


class MetricLogger:
    """
    Real-time & offline metric logger.
    Accumulates per-frame telemetry and ground truth to calculate
    Euclidean error, RMSE, lock retention, acquisition time, loss events,
    re-acquisition time, and processing throughput.
    """

    def __init__(
        self,
        benchmark_name: str = "BENCHMARK_2_OFFLINE",
        source_fps: float = 30.0,
        video_resolution: Optional[Dict[str, int]] = None,
        video_file: str = "",
        ground_truth_file: Optional[str] = None,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.source_fps = source_fps
        self.video_resolution = video_resolution or {"width": 640, "height": 480}
        self.video_file = video_file
        self.ground_truth_file = ground_truth_file

        self._frame_records: List[FrameMetricRecord] = []
        self._lock_state_counts: Dict[str, int] = {state.value: 0 for state in LockState}
        self._latencies_ms: List[float] = []
        self._centroid_errors: List[float] = []

        # Lock state machine tracking variables
        self._prev_lock_state: Optional[LockState] = None
        self._first_track_timestamp: Optional[float] = None
        self._has_acquired: bool = False
        self._current_loss_start_time: Optional[float] = None
        self._loss_event_count: int = 0
        self._total_loss_duration_s: float = 0.0
        self._reacquisition_times_s: List[float] = []
        self._first_frame_timestamp: Optional[float] = None
        self._last_frame_timestamp: Optional[float] = None

    @property
    def total_processed_frames(self) -> int:
        return len(self._frame_records)

    @property
    def frame_records(self) -> List[FrameMetricRecord]:
        return self._frame_records

    def reset(self) -> None:
        """Resets all accumulators and state variables for a new run."""
        self._frame_records.clear()
        self._lock_state_counts = {state.value: 0 for state in LockState}
        self._latencies_ms.clear()
        self._centroid_errors.clear()
        self._prev_lock_state = None
        self._first_track_timestamp = None
        self._has_acquired = False
        self._current_loss_start_time = None
        self._loss_event_count = 0
        self._total_loss_duration_s = 0.0
        self._reacquisition_times_s.clear()
        self._first_frame_timestamp = None
        self._last_frame_timestamp = None

    def log_frame(
        self,
        telemetry: TelemetryRecord,
        ground_truth: Optional[GroundTruthRecord] = None,
    ) -> FrameMetricRecord:
        """
        Consumes a single frame telemetry record and optional ground-truth record.
        Updates state transitions, computes instantaneous error, and records frame metrics.

        Args:
            telemetry: Diagnostic telemetry emitted by the pipeline.
            ground_truth: Ground truth record for the frame (if available).

        Returns:
            FrameMetricRecord containing combined data and evaluated metrics.
        """
        current_state = telemetry.lock_state
        video_t = telemetry.video_timestamp

        if self._first_frame_timestamp is None:
            self._first_frame_timestamp = video_t
        self._last_frame_timestamp = video_t

        # 1. State Counts
        state_str = current_state.value if isinstance(current_state, LockState) else str(current_state)
        self._lock_state_counts[state_str] = self._lock_state_counts.get(state_str, 0) + 1

        # 2. Lock State Transitions & Timing
        self._update_lock_state_transitions(current_state, video_t)

        # 3. Spatial Centroid Error Computation
        centroid_err: Optional[float] = None
        sq_err: Optional[float] = None

        if ground_truth is not None and ground_truth.gt_x is not None and ground_truth.gt_y is not None:
            # Determine effective prediction coordinate: prefer filtered state if valid, else raw detected
            pred_x: Optional[float] = None
            pred_y: Optional[float] = None

            if telemetry.filtered_x is not None and telemetry.filtered_y is not None:
                pred_x = telemetry.filtered_x
                pred_y = telemetry.filtered_y
            elif telemetry.raw_x is not None and telemetry.raw_y is not None:
                pred_x = telemetry.raw_x
                pred_y = telemetry.raw_y

            if pred_x is not None and pred_y is not None:
                if not (
                    math.isnan(pred_x) or math.isnan(pred_y) or
                    math.isinf(pred_x) or math.isinf(pred_y) or
                    math.isnan(ground_truth.gt_x) or math.isnan(ground_truth.gt_y) or
                    math.isinf(ground_truth.gt_x) or math.isinf(ground_truth.gt_y)
                ):
                    dx = pred_x - ground_truth.gt_x
                    dy = pred_y - ground_truth.gt_y
                    sq_err = dx * dx + dy * dy
                    centroid_err = math.sqrt(sq_err)
                    self._centroid_errors.append(centroid_err)

        # 4. Latency & Instantaneous FPS
        latency_ms = max(0.001, telemetry.processing_latency_ms)
        self._latencies_ms.append(latency_ms)
        instant_fps = 1000.0 / latency_ms if latency_ms > 0 else None

        # 5. Build Frame Record
        frame_rec = FrameMetricRecord(
            frame_id=telemetry.frame_id,
            video_timestamp_s=round(video_t, 4),
            gt_x=round(ground_truth.gt_x, 3) if (ground_truth and ground_truth.gt_x is not None) else None,
            gt_y=round(ground_truth.gt_y, 3) if (ground_truth and ground_truth.gt_y is not None) else None,
            raw_detected_x=round(telemetry.raw_x, 3) if telemetry.raw_x is not None else None,
            raw_detected_y=round(telemetry.raw_y, 3) if telemetry.raw_y is not None else None,
            filtered_x=round(telemetry.filtered_x, 3) if telemetry.filtered_x is not None else None,
            filtered_y=round(telemetry.filtered_y, 3) if telemetry.filtered_y is not None else None,
            confidence=round(telemetry.confidence, 4),
            detection_method=telemetry.detection_method,
            lock_state=state_str,
            centroid_error_px=round(centroid_err, 4) if centroid_err is not None else None,
            squared_error_px2=round(sq_err, 4) if sq_err is not None else None,
            processing_latency_ms=round(latency_ms, 3),
            instantaneous_fps=round(instant_fps, 2) if instant_fps is not None else None,
        )

        self._frame_records.append(frame_rec)
        self._prev_lock_state = current_state
        return frame_rec

    def _update_lock_state_transitions(self, current_state: LockState, current_time: float) -> None:
        """
        Maintains lock acquisition, loss detection, and reacquisition timestamps.
        """
        # Initial Acquisition
        if current_state == LockState.TRACK and not self._has_acquired:
            self._has_acquired = True
            self._first_track_timestamp = current_time

        # If we have already acquired once, track dropouts and re-acquisitions
        if self._has_acquired:
            is_prev_track = self._prev_lock_state == LockState.TRACK
            is_curr_track = current_state == LockState.TRACK

            # Transition: TRACK -> NON-TRACK (Loss event begins)
            if is_prev_track and not is_curr_track:
                self._loss_event_count += 1
                self._current_loss_start_time = current_time

            # Transition: NON-TRACK -> TRACK (Loss event ends, re-acquisition achieved)
            elif not is_prev_track and is_curr_track and self._current_loss_start_time is not None:
                reacq_duration = max(0.0, current_time - self._current_loss_start_time)
                self._reacquisition_times_s.append(reacq_duration)
                self._total_loss_duration_s += reacq_duration
                self._current_loss_start_time = None

    def compute_summary(self) -> BenchmarkSummary:
        """
        Computes final statistical benchmark summary from all accumulated frames.
        All metrics are strictly calculated from executed measurements.
        """
        total_frames = len(self._frame_records)

        # Video duration
        if self._first_frame_timestamp is not None and self._last_frame_timestamp is not None:
            video_duration_s = max(0.0, self._last_frame_timestamp - self._first_frame_timestamp)
        elif total_frames > 0 and self.source_fps > 0:
            video_duration_s = total_frames / self.source_fps
        else:
            video_duration_s = 0.0

        # Close any pending unrecovered loss event at end of benchmark
        total_loss_duration = self._total_loss_duration_s
        if self._current_loss_start_time is not None and self._last_frame_timestamp is not None:
            final_loss_period = max(0.0, self._last_frame_timestamp - self._current_loss_start_time)
            total_loss_duration += final_loss_period

        # 1. Accuracy Metrics (Centroid Error & RMSE)
        eval_count = len(self._centroid_errors)
        if eval_count > 0:
            mean_error = sum(self._centroid_errors) / eval_count
            sq_sum = sum(e * e for e in self._centroid_errors)
            rmse = math.sqrt(sq_sum / eval_count)
            max_error = max(self._centroid_errors)
            variance = sum((e - mean_error) ** 2 for e in self._centroid_errors) / eval_count
            std_dev_error = math.sqrt(variance)
        else:
            mean_error = None
            rmse = None
            max_error = None
            std_dev_error = None

        # 2. Lock Performance
        track_frames = self._lock_state_counts.get(LockState.TRACK.value, 0)
        lock_retention_rate = (track_frames / total_frames * 100.0) if total_frames > 0 else 0.0
        target_loss_rate = (100.0 - lock_retention_rate) if total_frames > 0 else 0.0

        # Acquisition time
        if self._first_track_timestamp is not None and self._first_frame_timestamp is not None:
            acq_time = max(0.0, self._first_track_timestamp - self._first_frame_timestamp)
        else:
            acq_time = None

        # Reacquisition stats
        if self._reacquisition_times_s:
            sorted_reacq = sorted(self._reacquisition_times_s)
            reacq_stats = {
                "count": len(self._reacquisition_times_s),
                "mean_s": round(sum(self._reacquisition_times_s) / len(self._reacquisition_times_s), 4),
                "min_s": round(sorted_reacq[0], 4),
                "max_s": round(sorted_reacq[-1], 4),
                "median_s": round(sorted_reacq[len(sorted_reacq) // 2], 4),
            }
        else:
            reacq_stats = {
                "count": 0,
                "mean_s": None,
                "min_s": None,
                "max_s": None,
                "median_s": None,
            }

        # 3. Latency & Throughput
        if self._latencies_ms:
            sorted_lat = sorted(self._latencies_ms)
            p95_idx = int(math.ceil(0.95 * len(sorted_lat))) - 1
            p95_idx = max(0, min(p95_idx, len(sorted_lat) - 1))
            mean_lat = sum(self._latencies_ms) / len(self._latencies_ms)
            total_time_s = sum(self._latencies_ms) / 1000.0
            avg_fps = (total_frames / total_time_s) if total_time_s > 0 else 0.0

            latency_stats: Dict[str, Optional[float]] = {
                "mean_ms": round(mean_lat, 3),
                "median_ms": round(sorted_lat[len(sorted_lat) // 2], 3),
                "p95_ms": round(sorted_lat[p95_idx], 3),
                "min_ms": round(sorted_lat[0], 3),
                "max_ms": round(sorted_lat[-1], 3),
            }
        else:
            avg_fps = 0.0
            latency_stats = {
                "mean_ms": None,
                "median_ms": None,
                "p95_ms": None,
                "min_ms": None,
                "max_ms": None,
            }

        return BenchmarkSummary(
            benchmark_type=self.benchmark_name,
            video_file=self.video_file,
            ground_truth_file=self.ground_truth_file,
            execution_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            video_resolution=self.video_resolution,
            total_video_frames=total_frames,
            video_duration_s=round(video_duration_s, 3),
            source_fps=round(self.source_fps, 2),
            evaluated_frames_count=eval_count,
            mean_centroid_error_px=round(mean_error, 4) if mean_error is not None else None,
            rmse_px=round(rmse, 4) if rmse is not None else None,
            max_error_px=round(max_error, 4) if max_error is not None else None,
            std_dev_error_px=round(std_dev_error, 4) if std_dev_error is not None else None,
            lock_retention_rate_pct=round(lock_retention_rate, 2),
            target_loss_rate_pct=round(target_loss_rate, 2),
            acquisition_time_s=round(acq_time, 4) if acq_time is not None else None,
            target_loss_event_count=self._loss_event_count,
            total_target_loss_duration_s=round(total_loss_duration, 4),
            reacquisition_time_stats=reacq_stats,
            lock_state_frame_counts=dict(self._lock_state_counts),
            average_processing_fps=round(avg_fps, 2),
            latency_ms_stats=latency_stats,
        )
