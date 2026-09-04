"""
Module 3 — Perception & CV Engine Demonstration Script
======================================================
Demonstrates the Fast Path Classical CV Detector (Median Filter + Top-Hat + Sub-pixel moments)
running over synthetic beacon frames with Salt & Pepper noise and background gradients.
"""

import time
import math
import cv2
import numpy as np

from src.perception import BeaconDetector


def run_perception_demo():
    print("==============================================")
    print("FSOC PERCEPTION & CV DETECTOR DEMONSTRATION")
    print("==============================================")

    detector = BeaconDetector()
    width, height = 640, 480
    total_frames = 200

    start_x, start_y = 100.0, 100.0
    vx, vy = 2.0, 1.5

    latencies_ms = []
    errors_px = []
    detected_count = 0

    rng = np.random.default_rng(42)

    for i in range(total_frames):
        ts = i / 30.0
        gt_x = start_x + i * vx
        gt_y = start_y + i * vy

        # 1. Create dark background frame with linear fog gradient
        y_idx, x_idx = np.indices((height, width))
        fog_background = (x_idx / width * 60.0).astype(np.uint8)
        frame = fog_background.copy()

        # 2. Add 5% Salt & Pepper noise
        noise_mask = rng.random((height, width)) < 0.05
        frame[noise_mask] = 255

        # 3. Draw true beacon spot
        cv2.circle(
            frame,
            (int(round(gt_x)), int(round(gt_y))),
            radius=5,
            color=255,
            thickness=-1,
        )

        # 4. Measure detection time
        t_start = time.perf_counter()
        res = detector.detect(frame, frame_id=i, timestamp=ts)
        t_end = time.perf_counter()

        latency_ms = (t_end - t_start) * 1000.0
        latencies_ms.append(latency_ms)

        if res.detected and res.centroid_x is not None and res.centroid_y is not None:
            detected_count += 1
            err = math.hypot(res.centroid_x - gt_x, res.centroid_y - gt_y)
            errors_px.append(err)

    mean_latency = float(np.mean(latencies_ms))
    fps = 1000.0 / max(mean_latency, 1e-6)
    mean_error = float(np.mean(errors_px)) if errors_px else 0.0
    max_error = float(np.max(errors_px)) if errors_px else 0.0
    detection_rate = (detected_count / total_frames) * 100.0

    print(f"Total Test Frames       : {total_frames}")
    print(f"Detection Success Rate  : {detection_rate:.1f}%")
    print(f"Mean Centroid Error     : {mean_error:.4f} pixels")
    print(f"Max Centroid Error      : {max_error:.4f} pixels")
    print(f"Average Detection Time  : {mean_latency:.4f} ms")
    print(f"Detector Throughput     : {fps:.1f} FPS")
    print("==============================================")


if __name__ == "__main__":
    run_perception_demo()
