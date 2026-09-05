"""
Telemetry Bar Component for FSOC Optical Simulator GUI.
Renders real-time telemetry metrics (FPS, TCP Connection, Latency ms, Packets count).
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


class TelemetryBarWidget(QFrame):
    """Middle panel displaying live network telemetry metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card-panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        title = QLabel("TELEMETRY")
        title.setProperty("class", "section-title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        data_layout = QHBoxLayout()
        data_layout.setSpacing(15)

        # FPS
        data_layout.addWidget(QLabel("FPS:"))
        self.lbl_fps = QLabel("60")
        self.lbl_fps.setProperty("class", "value-text")
        data_layout.addWidget(self.lbl_fps)

        data_layout.addWidget(QLabel("│"))

        # TCP Status
        data_layout.addWidget(QLabel("TCP:"))
        self.lbl_tcp = QLabel("● CONNECTED")
        self.lbl_tcp.setProperty("class", "accent-green")
        data_layout.addWidget(self.lbl_tcp)

        data_layout.addWidget(QLabel("│"))

        # Latency
        data_layout.addWidget(QLabel("Latency:"))
        self.lbl_latency = QLabel("12 ms")
        self.lbl_latency.setProperty("class", "value-text")
        data_layout.addWidget(self.lbl_latency)

        data_layout.addWidget(QLabel("│"))

        # Packets
        data_layout.addWidget(QLabel("Packets:"))
        self.lbl_packets = QLabel("0")
        self.lbl_packets.setProperty("class", "value-text")
        data_layout.addWidget(self.lbl_packets)

        layout.addLayout(data_layout)

    def update_telemetry(self, fps: float, connected: bool, latency_ms: float, packets: int):
        """Update live telemetry metrics bar."""
        self.lbl_fps.setText(f"{int(fps)}")
        if connected:
            self.lbl_tcp.setText("● CONNECTED")
            self.lbl_tcp.setProperty("class", "accent-green")
        else:
            self.lbl_tcp.setText("● DISCONNECTED")
            self.lbl_tcp.setProperty("class", "status-indicator-inactive")
        self.lbl_tcp.style().unpolish(self.lbl_tcp)
        self.lbl_tcp.style().polish(self.lbl_tcp)

        self.lbl_latency.setText(f"{latency_ms:.1f} ms")
        self.lbl_packets.setText(f"{packets}")
