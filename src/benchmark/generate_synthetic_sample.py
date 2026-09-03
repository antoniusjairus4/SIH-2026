"""
Utility script to generate a synthetic Benchmark-2 test video (640x480 monochrome FPA)
with dynamic beacon motion, noise, camera jitter, and temporary occlusion,
along with its ground-truth CSV file in data/benchmark2/.
"""

import os
import csv
import cv2
import numpy as np


def generate_sample_dataset(
    output_dir: str = "data/benchmark2",
    num_frames: int = 60,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
) -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "sample_beacon_tracking.mp4")
    gt_path = os.path.join(output_dir, "sample_beacon_tracking_gt.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=True)

    gt_rows = [["frame_id", "gt_x", "gt_y", "timestamp", "is_occluded"]]

    # Trajectory: Circular motion centered at (320, 240) with radius 80
    center_x, center_y = 320.0, 240.0
    radius = 80.0
    omega = 2.0 * np.pi / 45.0  # 1 rotation per 45 frames

    # Frame 35 to 40 will simulate an optical occlusion (e.g. cloud/fog drop)
    occlusion_range = range(35, 41)

    for fid in range(num_frames):
        ts = fid / fps
        angle = fid * omega
        true_x = center_x + radius * np.cos(angle)
        true_y = center_y + radius * np.sin(angle)

        is_occluded = fid in occlusion_range

        # Create dark-space frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add background dark current / sensor noise (Gaussian sigma=5)
        noise = np.random.normal(0, 5, (height, width, 3)).astype(np.int16)
        frame_noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if not is_occluded:
            # Render beacon spot (10x10 Gaussian laser spot)
            cx, cy = int(round(true_x)), int(round(true_y))
            cv2.circle(frame_noisy, (cx, cy), radius=5, color=(255, 255, 255), thickness=-1)
            cv2.circle(frame_noisy, (cx, cy), radius=2, color=(255, 255, 255), thickness=-1)

        writer.write(frame_noisy)
        gt_rows.append([
            str(fid),
            f"{true_x:.3f}",
            f"{true_y:.3f}",
            f"{ts:.4f}",
            "true" if is_occluded else "false",
        ])

    writer.release()

    with open(gt_path, mode="w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(gt_rows)

    print(f"Generated synthetic video: {video_path}")
    print(f"Generated ground truth:   {gt_path}")
    return video_path, gt_path


if __name__ == "__main__":
    generate_sample_dataset()
