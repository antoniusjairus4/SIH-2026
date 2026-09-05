"""
Camera Control Component for FSOC Optical Simulator GUI.
Renders Pan/Tilt angle indicators, visual arrows, and 3D camera position readouts.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout


class CameraControlWidget(QFrame):
    """Left column widget displaying camera pan/tilt indicators and 3D position."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card-panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header Title
        title = QLabel("CAMERA CONTROL")
        title.setProperty("class", "section-title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(4)

        # PAN Section
        pan_header = QLabel("PAN")
        pan_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pan_header.setProperty("class", "section-title")
        layout.addWidget(pan_header)

        pan_indicator_layout = QHBoxLayout()
        pan_indicator_layout.setSpacing(0)
        pan_left_arrow = QLabel("◄───────")
        self.pan_dot = QLabel("●")
        self.pan_dot.setProperty("class", "accent-blue")
        pan_right_arrow = QLabel("───────►")

        pan_indicator_layout.addWidget(pan_left_arrow, 0, Qt.AlignmentFlag.AlignRight)
        pan_indicator_layout.addWidget(self.pan_dot, 0, Qt.AlignmentFlag.AlignCenter)
        pan_indicator_layout.addWidget(pan_right_arrow, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(pan_indicator_layout)

        self.pan_val_label = QLabel("+12.4°")
        self.pan_val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pan_val_label.setProperty("class", "value-text")
        layout.addWidget(self.pan_val_label)

        layout.addSpacing(6)

        # TILT Section
        tilt_header = QLabel("TILT")
        tilt_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tilt_header.setProperty("class", "section-title")
        layout.addWidget(tilt_header)

        tilt_indicator_layout = QVBoxLayout()
        tilt_indicator_layout.setSpacing(2)

        tilt_up = QLabel("▲")
        tilt_up.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tilt_dot = QLabel("│ ●")
        self.tilt_dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tilt_dot.setProperty("class", "accent-blue")
        tilt_down = QLabel("▼")
        tilt_down.setAlignment(Qt.AlignmentFlag.AlignCenter)

        tilt_indicator_layout.addWidget(tilt_up)
        tilt_indicator_layout.addWidget(self.tilt_dot)
        tilt_indicator_layout.addWidget(tilt_down)
        layout.addLayout(tilt_indicator_layout)

        self.tilt_val_label = QLabel("-3.2°")
        self.tilt_val_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tilt_val_label.setProperty("class", "value-text")
        layout.addWidget(self.tilt_val_label)

        layout.addSpacing(8)

        # Position Section
        pos_header = QLabel("Position")
        pos_header.setProperty("class", "section-title")
        layout.addWidget(pos_header)

        pos_grid = QGridLayout()
        pos_grid.setSpacing(4)
        pos_grid.addWidget(QLabel("X:"), 0, 0)
        self.lbl_pos_x = QLabel("0")
        self.lbl_pos_x.setProperty("class", "value-text")
        pos_grid.addWidget(self.lbl_pos_x, 0, 1)

        pos_grid.addWidget(QLabel("Y:"), 1, 0)
        self.lbl_pos_y = QLabel("0")
        self.lbl_pos_y.setProperty("class", "value-text")
        pos_grid.addWidget(self.lbl_pos_y, 1, 1)

        pos_grid.addWidget(QLabel("Z:"), 2, 0)
        self.lbl_pos_z = QLabel("0")
        self.lbl_pos_z.setProperty("class", "value-text")
        pos_grid.addWidget(self.lbl_pos_z, 2, 1)

        layout.addLayout(pos_grid)
        layout.addStretch()

    def update_angles(self, pan_deg: float, tilt_deg: float):
        """Update Pan and Tilt values."""
        sign_pan = "+" if pan_deg >= 0 else ""
        sign_tilt = "+" if tilt_deg >= 0 else ""
        self.pan_val_label.setText(f"{sign_pan}{pan_deg:.1f}°")
        self.tilt_val_label.setText(f"{sign_tilt}{tilt_deg:.1f}°")

    def update_position(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Update 3D camera position readouts."""
        self.lbl_pos_x.setText(f"{int(x) if x == int(x) else x:.2f}")
        self.lbl_pos_y.setText(f"{int(y) if y == int(y) else y:.2f}")
        self.lbl_pos_z.setText(f"{int(z) if z == int(z) else z:.2f}")
