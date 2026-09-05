"""
Header Widget Component for FSOC Optical Simulator GUI.
Renders title banner and simulation status pulse indicator.
"""

# pyrefly: ignore [missing-import]
from PyQt6.QtCore import Qt
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class HeaderWidget(QFrame):
    """Top header panel displaying system title and live status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("header-widget")
        self.setProperty("class", "card-panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        # Title label
        title_label = QLabel("FSOC OPTICAL SIMULATOR")
        title_label.setProperty("class", "header-title")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Subtitle / Status indicator
        self.status_label = QLabel("● SIMULATION IDLE")
        self.status_label.setProperty("class", "status-indicator-inactive")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    def set_running(self, running: bool):
        """Update live running indicator status."""
        if running:
            self.status_label.setText("● SIMULATION RUNNING")
            self.status_label.setProperty("class", "status-indicator-active")
        else:
            self.status_label.setText("● SIMULATION STOPPED")
            self.status_label.setProperty("class", "status-indicator-inactive")

        # Force style re-evaluation
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
