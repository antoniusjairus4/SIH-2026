"""
Tracking Bar Component for FSOC Optical Simulator GUI.
Renders tracking status bar (Mode, Pan angle, Tilt angle, Angular error).
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


class TrackingBarWidget(QFrame):
    """Middle panel displaying active tracking telemetry."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card-panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        title = QLabel("TRACKING")
        title.setProperty("class", "section-title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        data_layout = QHBoxLayout()
        data_layout.setSpacing(20)

        # Mode
        data_layout.addWidget(QLabel("Mode:"))
        self.lbl_mode = QLabel("AUTO")
        self.lbl_mode.setProperty("class", "accent-blue")
        data_layout.addWidget(self.lbl_mode)

        data_layout.addStretch()

        # Pan
        data_layout.addWidget(QLabel("Pan:"))
        self.lbl_pan = QLabel("0.0°")
        self.lbl_pan.setProperty("class", "value-text")
        data_layout.addWidget(self.lbl_pan)

        data_layout.addStretch()

        # Tilt
        data_layout.addWidget(QLabel("Tilt:"))
        self.lbl_tilt = QLabel("0.0°")
        self.lbl_tilt.setProperty("class", "value-text")
        data_layout.addWidget(self.lbl_tilt)

        data_layout.addStretch()

        # Error
        data_layout.addWidget(QLabel("Error:"))
        self.lbl_error = QLabel("0.00°")
        self.lbl_error.setProperty("class", "accent-green")
        data_layout.addWidget(self.lbl_error)

        layout.addLayout(data_layout)

    def update_tracking_bar(self, mode: str, pan_deg: float, tilt_deg: float, error_deg: float):
        """Update tracking metrics bar."""
        self.lbl_mode.setText(mode.upper())
        sign_pan = "+" if pan_deg >= 0 else ""
        sign_tilt = "+" if tilt_deg >= 0 else ""
        self.lbl_pan.setText(f"{sign_pan}{pan_deg:.1f}°")
        self.lbl_tilt.setText(f"{sign_tilt}{tilt_deg:.1f}°")
        self.lbl_error.setText(f"{error_deg:.2f}°")
