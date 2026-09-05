"""
Real-Time Tracking Error Plotting Component for FSOC Optical Simulator GUI.
=============================================================================
Renders smooth, dark-themed rolling trajectory graph of centroid tracking errors.
"""

from collections import deque
from typing import Deque
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class ErrorPlotWidget(QFrame):
    """Real-time performance chart displaying centroiding error over time."""

    def __init__(self, max_points: int = 100, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card-panel")
        self.max_points = max_points
        self.error_deg_history: Deque[float] = deque([0.0] * max_points, maxlen=max_points)
        self.error_px_history: Deque[float] = deque([0.0] * max_points, maxlen=max_points)

        self.setMinimumHeight(130)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        # Title Header
        title = QLabel("TRACKING ERROR TRAJECTORY (LIVE)")
        title.setProperty("class", "section-title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        layout.addStretch()

    def add_error_point(self, error_deg: float, error_px: float):
        """Append latest error metric to rolling buffer and trigger canvas repaint."""
        self.error_deg_history.append(error_deg)
        self.error_px_history.append(error_px)
        self.update()

    def reset_plot(self):
        """Clear error history buffers."""
        self.error_deg_history = deque([0.0] * self.max_points, maxlen=self.max_points)
        self.error_px_history = deque([0.0] * self.max_points, maxlen=self.max_points)
        self.update()

    def paintEvent(self, event):
        """Custom QPainter draw routine for real-time error telemetry graph."""
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Chart area dimensions
        margin_left = 40
        margin_right = 15
        margin_top = 30
        margin_bottom = 20

        w = self.width() - margin_left - margin_right
        h = self.height() - margin_top - margin_bottom

        if w <= 10 or h <= 10:
            return

        # 1. Background Grid & Outer Border
        grid_pen = QPen(QColor(31, 41, 55), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        painter.drawRect(margin_left, margin_top, w, h)

        # Horizontal Gridlines (0.0°, 0.5°, 1.0°)
        max_y = max(max(self.error_deg_history, default=1.0), 1.0)
        grid_steps = 3
        for i in range(grid_steps + 1):
            y_val = (max_y / grid_steps) * i
            y_pos = margin_top + h - (i / grid_steps) * h
            painter.drawLine(margin_left, int(y_pos), margin_left + w, int(y_pos))

            # Y-Axis Text Labels
            painter.setPen(QColor(148, 163, 184))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(5, int(y_pos) + 4, f"{y_val:.2f}°")
            painter.setPen(grid_pen)

        # 2. ISRO Specification Error Limit (10 pixels ~ 0.0625 deg)
        threshold_deg = 10.0 * 0.00625
        if threshold_deg <= max_y:
            y_thresh = margin_top + h - (threshold_deg / max_y) * h
            thresh_pen = QPen(QColor(245, 158, 11, 180), 1, Qt.PenStyle.DotLine)
            painter.setPen(thresh_pen)
            painter.drawLine(margin_left, int(y_thresh), margin_left + w, int(y_thresh))

            painter.setPen(QColor(245, 158, 11))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(margin_left + 6, int(y_thresh) - 3, "10 px Limit (0.0625°)")

        # 3. Plot Curve & Gradient Fill
        points = list(self.error_deg_history)
        n = len(points)
        if n < 2:
            return

        path = QPainterPath()
        fill_path = QPainterPath()
        fill_path.moveTo(margin_left, margin_top + h)

        for i, val in enumerate(points):
            x = margin_left + (i / (self.max_points - 1)) * w
            y = margin_top + h - (min(val, max_y) / max_y) * h
            pt = QPointF(x, y)

            if i == 0:
                path.moveTo(pt)
                fill_path.lineTo(pt)
            else:
                path.lineTo(pt)
                fill_path.lineTo(pt)

        fill_path.lineTo(margin_left + w, margin_top + h)
        fill_path.closeSubpath()

        # Gradient area under curve
        gradient = QLinearGradient(0, margin_top, 0, margin_top + h)
        gradient.setColorAt(0.0, QColor(16, 185, 129, 80))
        gradient.setColorAt(1.0, QColor(16, 185, 129, 0))
        painter.fillPath(fill_path, gradient)

        # Plot Line Pen (Emerald Green)
        line_pen = QPen(QColor(16, 185, 129), 2)
        painter.setPen(line_pen)
        painter.drawPath(path)

        # 4. Current & Peak Metric Overlay text
        curr_err_deg = points[-1] if points else 0.0
        peak_err_deg = max(points) if points else 0.0
        curr_err_px = curr_err_deg / 0.00625

        painter.setPen(QColor(56, 189, 248))
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        info_str = f"Cur: {curr_err_deg:.3f}° ({curr_err_px:.1f}px) | Peak: {peak_err_deg:.3f}°"
        painter.drawText(margin_left + w - 240, margin_top - 6, info_str)
