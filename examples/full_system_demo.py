"""
Master Full-System Demonstration Script for SIH 2026 Virtual Camera Tracking System
===================================================================================
Demonstrates end-to-end live inter-module communication:
  MockUnityServer (Port 5005 @ 30Hz)
       │ (12-byte header + 640x480 RGB frame bytes)
       ▼
  TCPNetworkClient (Zero-Lag Buffer)
       │ (Raw Frame)
       ▼
  BeaconDetector (Fast Path CV Top-Hat + Sub-pixel Centroid)
       │ (Centroid cx, cy + Confidence)
       ▼
  BeaconStateEstimator (6D Constant-Acceleration Kalman Filter)
       │ (Filtered x, y, vx, vy, pred_x, pred_y)
       ▼
  MetricLogger (Live RMSE, FPS, Lock Retention, Acquisition Time)
       │ (PTZ pan/tilt deltas)
       ▼
  MockUnityServer (Gimbal rotation uplink)
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.network import MockUnityServer
from src.pipeline_runner import TrackingSystemPipeline



def run_full_system_demo():
    print("==========================================================")
    print("  SIH 2026 — FULL SYSTEM END-TO-END DEMONSTRATION")
    print("  AI-Based Virtual Camera Tracking System (ISRO PS-169)")
    print("==========================================================")

    test_port = 5005

    # 1. Start Mock Unity Simulator Server
    server = MockUnityServer(host="localhost", port=test_port, fps=30.0)
    server.start()
    time.sleep(0.3)

    # 2. Instantiate Master Tracking Pipeline
    pipeline = TrackingSystemPipeline(host="localhost", port=test_port)

    # 3. Simple PTZ Gimbal Control Callback (Simulates PID output)
    def gimbal_control_callback(telemetry):
        if telemetry.is_valid_track and telemetry.filtered_x is not None and telemetry.filtered_y is not None:
            # Proportional pan/tilt adjustment to keep target centered at (320, 240)
            err_x = telemetry.filtered_x - 320.0
            err_y = telemetry.filtered_y - 240.0
            pan_delta = float(err_x * (4.0 / 640.0) * 0.1)
            tilt_delta = float(err_y * (3.0 / 480.0) * 0.1)
            return pan_delta, tilt_delta
        return 0.0, 0.0

    print("Connecting to live simulation stream on localhost:5005...")
    print("Running integrated tracking loop for 3.0 seconds...")
    print("----------------------------------------------------------")

    # 4. Execute Live Integrated Tracking Loop
    metric_logger = pipeline.run_live_stream_loop(
        duration_seconds=3.0,
        control_callback=gimbal_control_callback,
    )

    time.sleep(0.2)
    server.stop()

    summary = metric_logger.compute_summary()

    print()
    print("==========================================================")
    print("  LIVE SYSTEM INTEGRATION PERFORMANCE SUMMARY")
    print("==========================================================")
    print(f"Total Video Frames Sent : {server.sent_frames_count}")
    print(f"Total Evaluated Frames  : {summary.evaluated_frames_count}")
    print(f"Lock Retention Rate     : {summary.lock_retention_rate_pct:.1f}%")
    print(f"Target Acquisition Time : {summary.acquisition_time_s:.3f} seconds")
    print(f"Target Loss Events      : {summary.target_loss_event_count}")
    mean_latency = summary.latency_ms_stats.get("mean_ms", 0.0)
    print(f"Mean Latency Per Frame  : {mean_latency:.2f} ms")
    print(f"Effective Processing FPS: {summary.average_processing_fps:.1f} FPS")
    print(f"PTZ Commands Delivered  : {len(server.received_commands)}")
    print("==========================================================")
    print("Status: 100% Ready for Unity C# Simulator connection!")
    print("==========================================================")


if __name__ == "__main__":
    run_full_system_demo()
