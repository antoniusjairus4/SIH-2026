"""
Control Footer Component for FSOC Optical Simulator GUI.
Renders action buttons: [ ▶ START ], [ ⏸ PAUSE ], [ ■ STOP ], [ ↻ RESET ], [ ⚙ SETTINGS ], [ 💾 EXPORT LOGS ].
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QSizePolicy


class ControlFooterWidget(QFrame):
    """Bottom control toolbar displaying operational control buttons."""

    # Custom signals for button actions
    start_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card-panel")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # Buttons
        self.btn_start = QPushButton("[ ▶ START ]")
        self.btn_start.setObjectName("btn-start")
        self.btn_start.clicked.connect(self.start_requested.emit)

        self.btn_pause = QPushButton("[ ⏸ PAUSE ]")
        self.btn_pause.clicked.connect(self.pause_requested.emit)

        self.btn_stop = QPushButton("[ ■ STOP ]")
        self.btn_stop.setObjectName("btn-stop")
        self.btn_stop.clicked.connect(self.stop_requested.emit)

        self.btn_reset = QPushButton("[ ↻ RESET ]")
        self.btn_reset.clicked.connect(self.reset_requested.emit)

        self.btn_settings = QPushButton("[ ⚙ SETTINGS ]")
        self.btn_settings.clicked.connect(self.settings_requested.emit)

        self.btn_export = QPushButton("[ 💾 EXPORT LOGS ]")
        self.btn_export.setObjectName("btn-export")
        self.btn_export.clicked.connect(self.export_requested.emit)

        # Expand buttons evenly
        for btn in (
            self.btn_start,
            self.btn_pause,
            self.btn_stop,
            self.btn_reset,
            self.btn_settings,
            self.btn_export,
        ):
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            layout.addWidget(btn)
