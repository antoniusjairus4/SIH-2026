"""
Unit tests for Module 6 PyQt6 Desktop GUI Application.
Runs in offscreen mode to verify headless GUI initialization, components, and settings dialog.
"""

import os
import sys
import numpy as np
import pytest

# pyrefly: ignore [missing-import]
from PyQt6.QtCore import Qt
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QApplication

from src.gui.app import FSOCDesktopApp
from src.gui.components import SettingsDialog
from src.metrics.schemas import TelemetryRecord, LockState


# Create single QApplication instance for offscreen testing
@pytest.fixture(scope="session")
def qapp():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_fsoc_desktop_app_initialization(qapp):
    """Verify FSOCDesktopApp initializes all wireframe component panels."""
    window = FSOCDesktopApp(use_mock_server=False)
    assert window is not None
    assert window.windowTitle() == "FSOC OPTICAL SIMULATOR — ISRO PS-169"

    # Check component presence
    assert window.header_widget is not None
    assert window.camera_control is not None
    assert window.camera_view is not None
    assert window.target_info is not None
    assert window.tracking_bar is not None
    assert window.telemetry_bar is not None
    assert window.control_footer is not None


def test_telemetry_signal_update_slot(qapp):
    """Verify GUI updates all UI components when telemetry signal is emitted."""
    window = FSOCDesktopApp(use_mock_server=False)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_telemetry = TelemetryRecord(
        frame_id=1,
        video_timestamp=0.033,
        processing_latency_ms=6.9,
        detector_status=True,
        raw_x=342.1,
        raw_y=238.4,
        confidence=0.95,
        detection_method="FastPathCV",
        lock_state=LockState.TRACK,
        filtered_x=341.8,
        filtered_y=238.6,
        is_valid_track=True,
        metadata={
            "current_pan_deg": 12.4,
            "current_tilt_deg": -3.2,
            "raw_error_x_deg": 0.46,
            "raw_error_y_deg": 0.0,
            "should_command": True,
        },
    )

    # Invoke telemetry slot
    window.on_telemetry_updated(dummy_frame, dummy_telemetry, packet_count=1248)

    # Verify visual text fields updated according to wireframe
    assert "12.4°" in window.camera_control.pan_val_label.text()
    assert "-3.2°" in window.camera_control.tilt_val_label.text()
    assert "12.4°" in window.tracking_bar.lbl_pan.text()
    assert "-3.2°" in window.tracking_bar.lbl_tilt.text()
    assert "0.46°" in window.tracking_bar.lbl_error.text()
    assert "1248" in window.telemetry_bar.lbl_packets.text()


def test_settings_dialog(qapp):
    """Verify SettingsDialog initialization and configuration extraction."""
    initial_cfg = {"host": "127.0.0.1", "port": 5006, "pan_kp": 2.5}
    dlg = SettingsDialog(initial_cfg)
    assert dlg is not None
    assert dlg.txt_host.text() == "127.0.0.1"
    assert dlg.spin_port.value() == 5006
    assert dlg.spin_pan_kp.value() == 2.5

    extracted = dlg.get_config()
    assert extracted["host"] == "127.0.0.1"
    assert extracted["port"] == 5006
    assert extracted["pan_kp"] == 2.5
