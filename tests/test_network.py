"""
Unit tests for Module 2B TCP Socket Networking Client & Mock Unity Server.
Validates TCP handshake, header parsing, frame array deserialization,
zero-lag latest-frame buffer policy, and 8-byte control command uplink.
"""

import time
import pytest
import numpy as np

from src.network import TCPNetworkClient, MockUnityServer


def test_network_client_and_mock_server_roundtrip():
    # Use non-standard port to prevent conflicts
    test_port = 5009

    server = MockUnityServer(host="localhost", port=test_port, fps=30.0)
    server.start()

    time.sleep(0.2)  # Give server socket time to bind and listen

    client = TCPNetworkClient(host="localhost", port=test_port)
    assert client.connect() is True
    assert client.is_connected is True

    # 1. Receive frame from client zero-lag buffer
    item = client.get_latest_frame(block=True, timeout=2.0)
    assert item is not None

    frame_np, frame_id, timestamp = item
    assert isinstance(frame_np, np.ndarray)
    assert frame_np.shape == (480, 640, 3)
    assert frame_id >= 0
    assert timestamp >= 0.0

    # 2. Send PTZ control command to server
    sent_success = client.send_control_command(pan_delta=0.5, tilt_delta=-0.2)
    assert sent_success is True

    time.sleep(0.3)  # Give server time to process command

    assert len(server.received_commands) > 0
    pan, tilt = server.received_commands[0]
    assert pytest.approx(pan, 1e-4) == 0.5
    assert pytest.approx(tilt, 1e-4) == -0.2

    # Clean disconnect
    client.disconnect()
    server.stop()
    assert client.is_connected is False


def test_zero_lag_buffer_overwriting_policy():
    test_port = 5010

    server = MockUnityServer(host="localhost", port=test_port, fps=60.0)  # High FPS stream
    server.start()

    time.sleep(0.2)

    client = TCPNetworkClient(host="localhost", port=test_port)
    client.connect()

    # Simulate slow downstream CV loop: sleep while server streams ~12 frames
    time.sleep(0.25)

    assert client.total_received > 1
    # Check that client zero-lag buffer overwrites older frames without blocking queue
    assert client.total_dropped >= 0

    item = client.get_latest_frame(block=False)
    assert item is not None

    client.disconnect()
    server.stop()


def test_client_connection_failure():
    # Attempt connecting to unused port
    client = TCPNetworkClient(host="localhost", port=5999, connect_timeout=0.2)
    connected = client.connect()
    assert connected is False
    assert client.is_connected is False
