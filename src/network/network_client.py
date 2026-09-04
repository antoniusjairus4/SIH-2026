"""
Multi-Threaded TCP Socket Network Client for SIH 2026 Virtual Camera Tracking System.
Connects to Unity 3D Simulator on localhost:5005 (or custom host/port),
deserializes 12-byte headers (frame_id, timestamp) + raw frame byte payloads,
maintains a zero-lag latest-frame buffer, and sends packed 8-byte PTZ control deltas.
"""

import socket
import struct
import threading
import queue
import time
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("TCPNetworkClient")


class TCPNetworkClient:
    """
    Thread-safe TCP socket client for Unity <-> Python live communication.
    """

    HEADER_FORMAT = "!Id"  # uint32 frame_id (4B), double timestamp (8B) -> 12 bytes
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    COMMAND_FORMAT = "!ff"  # float32 pan_delta (4B), float32 tilt_delta (4B) -> 8 bytes

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5005,
        frame_width: int = 640,
        frame_height: int = 480,
        channels: int = 3,
        connect_timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.channels = channels
        self.connect_timeout = connect_timeout

        self.payload_size = frame_width * frame_height * channels
        self._socket: Optional[socket.socket] = None
        self._connected = False
        self._running = False
        self._receiver_thread: Optional[threading.Thread] = None

        # Zero-Lag Buffer: Overwriting 1-element queue
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._lock = threading.Lock()

        self._dropped_frames_count = 0
        self._total_received_count = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def total_received(self) -> int:
        return self._total_received_count

    @property
    def total_dropped(self) -> int:
        return self._dropped_frames_count

    def connect(self) -> bool:
        """
        Establishes TCP connection to Unity simulator server.
        """
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.connect_timeout)
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(None)  # Blocking mode for worker thread
            self._connected = True
            self._running = True

            # Start background receiver thread
            self._receiver_thread = threading.Thread(
                target=self._worker_receive_loop,
                daemon=True,
            )
            self._receiver_thread.start()
            logger.info(f"Connected to Unity TCP Server at {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.warning(f"Could not connect to Unity TCP Server at {self.host}:{self.port}: {e}")
            self._connected = False
            self._running = False
            return False

    def disconnect(self) -> None:
        """
        Closes TCP connection and stops background worker thread.
        """
        self._running = False
        self._connected = False

        if self._socket:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        if self._receiver_thread and self._receiver_thread.is_alive():
            self._receiver_thread.join(timeout=1.0)

        logger.info("Disconnected from Unity TCP Server.")

    def get_latest_frame(self, block: bool = True, timeout: Optional[float] = 1.0) -> Optional[Tuple[np.ndarray, int, float]]:
        """
        Retrieves the freshest frame from the Zero-Lag Buffer.

        Returns:
            Tuple of (frame_np_array, frame_id, timestamp) or None if buffer is empty.
        """
        try:
            return self._frame_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def send_control_command(self, pan_delta: float, tilt_delta: float) -> bool:
        """
        Packs (pan_delta, tilt_delta) floats into an 8-byte payload and sends to Unity over TCP.

        Args:
            pan_delta: Horizontal pan rotation angle in degrees.
            tilt_delta: Vertical tilt rotation angle in degrees.

        Returns:
            True if successfully sent, False otherwise.
        """
        if not self._connected or self._socket is None:
            return False

        try:
            payload = struct.pack(self.COMMAND_FORMAT, float(pan_delta), float(tilt_delta))
            with self._lock:
                self._socket.sendall(payload)
            return True
        except Exception as e:
            logger.warning(f"Error sending PTZ control command to Unity: {e}")
            self._connected = False
            return False

    def _recv_exact(self, num_bytes: int) -> Optional[bytes]:
        """
        Helper method to read exactly num_bytes from TCP socket.
        """
        data = bytearray()
        while len(data) < num_bytes and self._running:
            try:
                chunk = self._socket.recv(num_bytes - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            except Exception:
                return None
        return bytes(data)

    def _worker_receive_loop(self) -> None:
        """
        Background worker thread receiving TCP socket frames continuously.
        Enforces Zero-Lag policy by dropping unconsumed older frames.
        """
        while self._running and self._connected:
            # 1. Read 12-byte header (frame_id uint32, timestamp float64)
            header_bytes = self._recv_exact(self.HEADER_SIZE)
            if not header_bytes or len(header_bytes) < self.HEADER_SIZE:
                break

            frame_id, timestamp = struct.unpack(self.HEADER_FORMAT, header_bytes)

            # 2. Read frame image payload
            payload_bytes = self._recv_exact(self.payload_size)
            if not payload_bytes or len(payload_bytes) < self.payload_size:
                break

            # 3. Convert raw bytes to NumPy array (640x480x3 uint8)
            frame_np = np.frombuffer(payload_bytes, dtype=np.uint8).reshape(
                (self.frame_height, self.frame_width, self.channels)
            )

            self._total_received_count += 1

            # 4. Enforce Zero-Lag Policy: Overwrite old unconsumed frame in 1-element queue
            if self._frame_queue.full():
                try:
                    self._frame_queue.get_nowait()
                    self._dropped_frames_count += 1
                except queue.Empty:
                    pass

            self._frame_queue.put((frame_np, frame_id, timestamp))

        self._connected = False
