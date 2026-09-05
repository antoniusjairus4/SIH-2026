"""
Settings Dialog Modal Component for FSOC Optical Simulator Desktop GUI.
========================================================================
Provides interactive controls for adjusting system settings:
  - Network TCP & Offline .mp4 Video File selection (Benchmark-2 mode)
  - PID Controller tuning parameters (Kp, Ki, Kd, Slew limits)
  - Perception & AI Fallback configurations
  - Synthetic Noise & Atmospheric Disturbance injection
"""

import os
from typing import Dict, Any, Optional
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class SettingsDialog(QDialog):
    """Configuration Dialog Modal for FSOC Simulator parameters."""

    def __init__(self, current_config: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FSOC Optical Simulator — Configuration & Settings")
        self.resize(520, 420)
        self.setModal(True)

        config = current_config or {}

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(12)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Network & Execution Source
        self._init_network_tab(config)

        # Tab 2: PID Controller Tuning
        self._init_pid_tab(config)

        # Tab 3: Perception & AI
        self._init_perception_tab(config)

        # Tab 4: Disturbances & Noise
        self._init_disturbances_tab(config)

        # Bottom Buttons [ Apply / Cancel ]
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.btn_apply = QPushButton("Apply Settings")
        self.btn_apply.setObjectName("btn-start")
        self.btn_apply.clicked.connect(self.accept)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)

        button_layout.addWidget(self.btn_cancel)
        button_layout.addWidget(self.btn_apply)

        main_layout.addLayout(button_layout)

    def _init_network_tab(self, config: Dict[str, Any]):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Execution Mode & Data Source")
        form = QFormLayout(box)

        self.combo_source = QComboBox()
        self.combo_source.addItems(["Live TCP Streaming", "Benchmark-2 Video (.mp4)"])
        mode_idx = 1 if config.get("video_file_path") else 0
        self.combo_source.setCurrentIndex(mode_idx)
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("Mode:", self.combo_source)

        self.txt_host = QLineEdit(config.get("host", "localhost"))
        form.addRow("TCP Host:", self.txt_host)

        self.spin_port = QSpinBox()
        self.spin_port.setRange(1, 65535)
        self.spin_port.setValue(config.get("port", 5005))
        form.addRow("TCP Port:", self.spin_port)

        self.chk_mock_server = QCheckBox("Run Internal Mock Unity Server (Port 5005)")
        self.chk_mock_server.setChecked(config.get("use_mock_server", True))
        form.addRow(self.chk_mock_server)

        # Video File Selection
        video_layout = QHBoxLayout()
        default_v_path = config.get("video_file_path") or os.path.abspath("data/benchmark2/sample_beacon_tracking.mp4")
        self.txt_video_path = QLineEdit(default_v_path if os.path.exists(default_v_path) else "")
        self.txt_video_path.setPlaceholderText("Select pre-recorded .mp4 file...")
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._browse_video_file)
        video_layout.addWidget(self.txt_video_path)
        video_layout.addWidget(self.btn_browse)
        form.addRow("Video Path:", video_layout)

        layout.addWidget(box)
        layout.addStretch()
        self.tab_widget.addTab(tab, "Network & Source")
        self._on_source_changed(mode_idx)

    def _on_source_changed(self, index: int):
        is_tcp = (index == 0)
        self.txt_host.setEnabled(is_tcp)
        self.spin_port.setEnabled(is_tcp)
        self.chk_mock_server.setEnabled(is_tcp)
        self.txt_video_path.setEnabled(not is_tcp)
        self.btn_browse.setEnabled(True)

    def _browse_video_file(self):
        initial_dir = os.path.abspath("data/benchmark2") if os.path.exists("data/benchmark2") else ""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select Pre-recorded MP4 Video File",
            initial_dir,
            "Video Files (*.mp4 *.avi *.mkv);;All Files (*)",
        )
        if filename:
            self.txt_video_path.setText(filename)
            self.combo_source.setCurrentIndex(1)  # Automatically switch mode to Benchmark-2 Video

    def _init_pid_tab(self, config: Dict[str, Any]):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box_pan = QGroupBox("Pan Axis PID Parameters")
        form_pan = QFormLayout(box_pan)

        self.spin_pan_kp = QDoubleSpinBox()
        self.spin_pan_kp.setRange(0.0, 50.0)
        self.spin_pan_kp.setSingleStep(0.1)
        self.spin_pan_kp.setValue(config.get("pan_kp", 1.2))
        form_pan.addRow("Kp (Proportional):", self.spin_pan_kp)

        self.spin_pan_ki = QDoubleSpinBox()
        self.spin_pan_ki.setRange(0.0, 10.0)
        self.spin_pan_ki.setSingleStep(0.01)
        self.spin_pan_ki.setValue(config.get("pan_ki", 0.05))
        form_pan.addRow("Ki (Integral):", self.spin_pan_ki)

        self.spin_pan_kd = QDoubleSpinBox()
        self.spin_pan_kd.setRange(0.0, 10.0)
        self.spin_pan_kd.setSingleStep(0.01)
        self.spin_pan_kd.setValue(config.get("pan_kd", 0.15))
        form_pan.addRow("Kd (Derivative):", self.spin_pan_kd)

        layout.addWidget(box_pan)

        box_tilt = QGroupBox("Tilt Axis & Gimbal Kinematics")
        form_tilt = QFormLayout(box_tilt)

        self.spin_tilt_kp = QDoubleSpinBox()
        self.spin_tilt_kp.setRange(0.0, 50.0)
        self.spin_tilt_kp.setSingleStep(0.1)
        self.spin_tilt_kp.setValue(config.get("tilt_kp", 1.2))
        form_tilt.addRow("Tilt Kp:", self.spin_tilt_kp)

        self.spin_max_slew = QDoubleSpinBox()
        self.spin_max_slew.setRange(0.5, 20.0)
        self.spin_max_slew.setSingleStep(0.5)
        self.spin_max_slew.setValue(config.get("max_slew_deg_s", 5.0))
        form_tilt.addRow("Max Slew Limit (°/s):", self.spin_max_slew)

        layout.addWidget(box_tilt)
        layout.addStretch()
        self.tab_widget.addTab(tab, "PID Tuning")

    def _init_perception_tab(self, config: Dict[str, Any]):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Computer Vision & AI Detection Pipeline")
        form = QFormLayout(box)

        self.spin_tophat_kernel = QSpinBox()
        self.spin_tophat_kernel.setRange(3, 31)
        self.spin_tophat_kernel.setSingleStep(2)
        self.spin_tophat_kernel.setValue(config.get("tophat_kernel", 15))
        form.addRow("Top-Hat Kernel Size (px):", self.spin_tophat_kernel)

        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(50, 255)
        self.spin_thresh.setValue(config.get("threshold", 180))
        form.addRow("Detection Threshold:", self.spin_thresh)

        self.chk_onnx = QCheckBox("Enable YOLOv8n ONNX AI Fallback Model")
        self.chk_onnx.setChecked(config.get("use_onnx_fallback", True))
        form.addRow(self.chk_onnx)

        layout.addWidget(box)
        layout.addStretch()
        self.tab_widget.addTab(tab, "Perception & AI")

    def _init_disturbances_tab(self, config: Dict[str, Any]):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Synthetic Environment Disturbances")
        form = QFormLayout(box)

        self.spin_salt_pepper = QDoubleSpinBox()
        self.spin_salt_pepper.setRange(0.0, 0.5)
        self.spin_salt_pepper.setSingleStep(0.02)
        self.spin_salt_pepper.setValue(config.get("salt_pepper_rate", 0.0))
        form.addRow("Salt & Pepper Rate:", self.spin_salt_pepper)

        self.spin_gaussian = QDoubleSpinBox()
        self.spin_gaussian.setRange(0.0, 50.0)
        self.spin_gaussian.setSingleStep(1.0)
        self.spin_gaussian.setValue(config.get("gaussian_std", 0.0))
        form.addRow("Gaussian Noise Std Dev (px):", self.spin_gaussian)

        self.spin_jitter = QDoubleSpinBox()
        self.spin_jitter.setRange(0.0, 30.0)
        self.spin_jitter.setSingleStep(1.0)
        self.spin_jitter.setValue(config.get("jitter_std", 0.0))
        form.addRow("Camera Jitter Std Dev (px):", self.spin_jitter)

        layout.addWidget(box)
        layout.addStretch()
        self.tab_widget.addTab(tab, "Disturbances")

    def get_config(self) -> Dict[str, Any]:
        """Extract user-configured settings into a dictionary."""
        is_video = (self.combo_source.currentIndex() == 1)
        video_path = self.txt_video_path.text().strip() if is_video else None

        return {
            "host": self.txt_host.text().strip(),
            "port": self.spin_port.value(),
            "use_mock_server": self.chk_mock_server.isChecked(),
            "video_file_path": video_path,
            "pan_kp": self.spin_pan_kp.value(),
            "pan_ki": self.spin_pan_ki.value(),
            "pan_kd": self.spin_pan_kd.value(),
            "tilt_kp": self.spin_tilt_kp.value(),
            "max_slew_deg_s": self.spin_max_slew.value(),
            "tophat_kernel": self.spin_tophat_kernel.value(),
            "threshold": self.spin_thresh.value(),
            "use_onnx_fallback": self.chk_onnx.isChecked(),
            "salt_pepper_rate": self.spin_salt_pepper.value(),
            "gaussian_std": self.spin_gaussian.value(),
            "jitter_std": self.spin_jitter.value(),
        }
