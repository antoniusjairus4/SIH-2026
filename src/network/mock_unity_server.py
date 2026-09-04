"""
Standalone Mock Unity TCP Server for SIH 2026 Virtual Camera Tracking System.
Simulates Unity 3D simulator behavior by hosting a TCP socket server on port 5005 (or custom port),
streaming synthetic 640x480 moving beacon frames @ 30 Hz, and receiving PTZ control deltas.
"""

import socket
import struct
import threading
import time
import logging
from typing import Optional, List, Tuple
import cv2
import numpy as np

logger = logging.getLogger("MockUnityServer")


class MockUnityServer:
    """
    Mock TCP server simulating Unity FPA camera stream and gimbal physics.
    """

    HEADER_FORMAT = "!Id"   # uint32 frame_id, double timestamp
    COMMAND_FORMAT = "!ff"  # float32 pan_delta, float32 tilt_delta
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    COMMAND_SIZE = struct.calcsize(COMMAND_FORMAT)

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5005,
        fps: float = 30.0,
        frame_width: int = 640,
        frame_height: int = 480,
    ) -> None:
        self.host = host
        self.port = port
        self.fps = fps
        self.dt = 1.0 / fps
        self.frame_width = frame_width
        self.frame_height = frame_height

        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
        self._command_listener_thread: Optional[threading.Thread] = None

        self.received_commands: List[Tuple[float, float]] = []
        self.sent_frames_count = 0

    def start(self) -> None:
        """
        Starts the TCP server in a background thread.
        """
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(1)
        self._running = True

        self._server_thread = threading.Thread(
            target=self._server_loop,
            daemon=True,
        )
        self._server_thread.start()
        logger.info(f"Mock Unity TCP Server listening on {self.host}:{self.port}")

    def stop(self) -> None:
        """
        Stops the TCP server and closes client connections.
        """
        self._running = False
        if self._client_socket:
            try:
                self._client_socket.shutdown(socket.SHUT_RDWR)
                self._client_socket.close()
            except Exception:
                pass
            self._client_socket = None

        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
            self._server_socket = None

        logger.info("Mock Unity TCP Server stopped.")

    def _server_loop(self) -> None:
        """
        Main server loop: Accepts client connection and streams frames @ 30 FPS.
        """
        try:
            self._server_socket.settimeout(2.0)
            while self._running:
                try:
                    client_sock, addr = self._server_socket.accept()
                    logger.info(f"Mock Unity Server accepted connection from {addr}")
                    self._client_socket = client_sock
                    self._client_socket.settimeout(None)

                    # Start async command listener thread
                    self._command_listener_thread = threading.Thread(
                        target=self._listen_for_commands,
                        daemon=True,
                    )
                    self._command_listener_thread.start()

                    self._stream_frames()

                except socket.timeout:
                    continue

        except Exception as e:
            if self._running:
                logger.warning(f"Mock Unity Server loop exception: {e}")

    def _listen_for_commands(self) -> None:
        """
        Asynchronously reads 8-byte (pan_delta, tilt_delta) control commands sent by Python client.
        """
        client = self._client_socket
        while self._running and client and self._client_socket == client:
            try:
                data = bytearray()
                while len(data) < self.COMMAND_SIZE and self._running:
                    chunk = client.recv(self.COMMAND_SIZE - len(data))
                    if not chunk:
                        return
                    data.extend(chunk)

                if len(data) == self.COMMAND_SIZE:
                    pan_delta, tilt_delta = struct.unpack(self.COMMAND_FORMAT, bytes(data))
                    self.received_commands.append((pan_delta, tilt_delta))

            except Exception:
                break

    def _stream_frames(self) -> None:
        """
        Streams synthetic moving target frames @ 30 FPS.
        """
        frame_id = 0
        start_x, start_y = 200.0, 150.0
        vx, vy = 3.0, 2.0

        client = self._client_socket

        while self._running and client and self._client_socket == client:
            t_start = time.perf_counter()
            ts = frame_id * self.dt

            # Moving beacon spot coordinates
            gt_x = (start_x + frame_id * vx) % self.frame_width
            gt_y = (start_y + frame_id * vy) % self.frame_height

            # Create frame
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            cv2.circle(frame, (int(round(gt_x)), int(round(gt_y))), radius=5, color=(255, 255, 255), thickness=-1)

            # Pack 12-byte header
            header = struct.pack(self.HEADER_FORMAT, frame_id, float(ts))
            payload = frame.tobytes()

            try:
                client.sendall(header + payload)
                self.sent_frames_count += 1
                frame_id += 1
            except Exception:
                break

            # Sleep to maintain ~30 FPS stream
            t_elapsed = time.perf_counter() - t_start
            t_sleep = max(self.dt - t_elapsed, 0.0)
            time.sleep(t_sleep)
