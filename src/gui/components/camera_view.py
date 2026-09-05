"""
Camera Viewport Component for FSOC Optical Simulator GUI.
Renders live 640x480 RGB camera frame with target crosshair overlays matching wireframe.
"""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class CameraViewWidget(QFrame):
    """Center column viewport displaying optical video feed and target markers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card-panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header Title
        title = QLabel("CAMERA VIEW")
        title.setProperty("class", "section-title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Video canvas label
        self.viewport_label = QLabel()
        self.viewport_label.setMinimumSize(480, 360)
        self.viewport_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewport_label.setStyleSheet("background-color: #0b0f19; border: 1px solid #1f2937;")
        layout.addWidget(self.viewport_label, 1)

        # Default dark viewport initialization
        self._set_placeholder_viewport()

    def _set_placeholder_viewport(self):
        """Draw default dark viewport with wireframe beacon marker."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.update_frame(img, detected=True, target_x=320.0, target_y=240.0, mode="IDLE")

    def update_frame(
        self,
        frame: np.ndarray,
        detected: bool = True,
        target_x: float = 320.0,
        target_y: float = 240.0,
        mode: str = "AUTO",
    ):
        """Renders an RGB frame onto the Qt canvas with target marker overlays."""
        if frame is None or frame.size == 0:
            return

        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw Target Beacon Marker matching ASCII wireframe (✦ BEACON)
        cx = int(target_x) if target_x is not None else w // 2
        cy = int(target_y) if target_y is not None else h // 2

        # ✦ Symbol
        painter.setPen(QColor(248, 250, 252))
        painter.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        painter.drawText(cx - 7, cy - 4, "✦")

        # BEACON Label
        painter.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        painter.drawText(cx - 24, cy + 16, "BEACON")

        painter.end()

        scaled_pixmap = pixmap.scaled(
            self.viewport_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.viewport_label.setPixmap(scaled_pixmap)
