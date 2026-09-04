"""
Module 2B — TCP Socket Network Client Demonstration Script
===========================================================
Demonstrates live TCP socket round-trip streaming between MockUnityServer and TCPNetworkClient @ 30 Hz.
"""

import time
from src.network import TCPNetworkClient, MockUnityServer


def run_network_demo():
    print("==============================================")
    print("FSOC TCP SOCKET NETWORK DEMONSTRATION")
    print("==============================================")

    test_port = 5005

    # 1. Start Mock Unity Server on port 5005
    server = MockUnityServer(host="localhost", port=test_port, fps=30.0)
    server.start()
    time.sleep(0.3)

    # 2. Start TCP Network Client
    client = TCPNetworkClient(host="localhost", port=test_port)
    if not client.connect():
        print("[ERROR] Could not connect to Mock Unity Server.")
        server.stop()
        return

    print("Connected to Mock Unity Server on localhost:5005")
    print("Streaming live 640x480 frames @ 30 FPS for 3 seconds...")
    print("----------------------------------------------")

    start_time = time.time()
    frames_processed = 0
    roundtrips = 0

    while time.time() - start_time < 3.0:
        item = client.get_latest_frame(block=True, timeout=0.5)
        if item is None:
            continue

        frame_np, frame_id, timestamp = item
        frames_processed += 1

        # Send test PTZ command back to Unity
        pan_delta = 0.1 * (frame_id % 5)
        tilt_delta = -0.05 * (frame_id % 5)
        if client.send_control_command(pan_delta, tilt_delta):
            roundtrips += 1

        time.sleep(0.01)  # Simulate 10ms CV processing work

    duration = time.time() - start_time
    effective_fps = frames_processed / duration

    print(f"Total Video Frames Sent : {server.sent_frames_count}")
    print(f"Total Frames Received   : {client.total_received}")
    print(f"Frames Ingested by CV   : {frames_processed}")
    print(f"Zero-Lag Dropped Frames : {client.total_dropped}")
    print(f"Successful PTZ Uplinks  : {roundtrips}")
    print(f"Effective Streaming Rate: {effective_fps:.1f} FPS")
    print("==============================================")

    client.disconnect()
    server.stop()


if __name__ == "__main__":
    run_network_demo()
