"""
Beacon Target Information Component for FSOC Optical Simulator GUI.
Renders target 3D world coordinates, distance, motion trajectory, speed, and lock state.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout


class BeaconTargetInfoWidget(QFrame):
    """Right column widget displaying beacon target properties and lock badge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card-panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Title
        title = QLabel("BEACON / TARGET")
        title.setProperty("class", "section-title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Target ID
        self.lbl_target_id = QLabel("Target: Beacon #01")
        self.lbl_target_id.setProperty("class", "value-text")
        layout.addWidget(self.lbl_target_id)

        # 3D Position
        pos_header = QLabel("Position:")
        pos_header.setProperty("class", "section-title")
        layout.addWidget(pos_header)

        grid = QGridLayout()
        grid.setSpacing(6)

        grid.addWidget(QLabel("X:"), 0, 0)
        self.lbl_x = QLabel("12.43 m")
        self.lbl_x.setProperty("class", "value-text")
        grid.addWidget(self.lbl_x, 0, 1)

        grid.addWidget(QLabel("Y:"), 1, 0)
        self.lbl_y = QLabel("3.21 m")
        self.lbl_y.setProperty("class", "value-text")
        grid.addWidget(self.lbl_y, 1, 1)

        grid.addWidget(QLabel("Z:"), 2, 0)
        self.lbl_z = QLabel("100.00 m")
        self.lbl_z.setProperty("class", "value-text")
        grid.addWidget(self.lbl_z, 2, 1)

        layout.addLayout(grid)

        # Distance & Motion
        dist_layout = QGridLayout()
        dist_layout.addWidget(QLabel("Distance:"), 0, 0)
        self.lbl_distance = QLabel("100.8 m")
        self.lbl_distance.setProperty("class", "value-text")
        dist_layout.addWidget(self.lbl_distance, 0, 1)

        dist_layout.addWidget(QLabel("Trajectory:"), 1, 0)
        self.lbl_trajectory = QLabel("Figure-8")
        self.lbl_trajectory.setProperty("class", "value-text")
        dist_layout.addWidget(self.lbl_trajectory, 1, 1)

        dist_layout.addWidget(QLabel("Speed:"), 2, 0)
        self.lbl_speed = QLabel("5 m/s")
        self.lbl_speed.setProperty("class", "value-text")
        dist_layout.addWidget(self.lbl_speed, 2, 1)

        layout.addLayout(dist_layout)
        layout.addStretch()

        # Target Lock Badge
        self.lbl_lock_status = QLabel("● TARGET LOCKED")
        self.lbl_lock_status.setProperty("class", "status-indicator-active")
        self.lbl_lock_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_lock_status)

    def update_target_info(
        self,
        x: float,
        y: float,
        z: float = 100.0,
        locked: bool = True,
        trajectory: str = "Figure-8",
    ):
        """Update 3D target coordinates and lock status."""
        self.lbl_x.setText(f"{x:.2f} m")
        self.lbl_y.setText(f"{y:.2f} m")
        self.lbl_z.setText(f"{z:.2f} m")

        distance = (x**2 + y**2 + z**2) ** 0.5
        self.lbl_distance.setText(f"{distance:.1f} m")
        self.lbl_trajectory.setText(trajectory)

        if locked:
            self.lbl_lock_status.setText("● TARGET LOCKED")
            self.lbl_lock_status.setProperty("class", "status-indicator-active")
        else:
            self.lbl_lock_status.setText("● LOCK SEARCHING")
            self.lbl_lock_status.setProperty("class", "status-indicator-inactive")

        self.lbl_lock_status.style().unpolish(self.lbl_lock_status)
        self.lbl_lock_status.style().polish(self.lbl_lock_status)
