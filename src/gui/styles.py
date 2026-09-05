"""
Module 6 — Dark Mode QSS Stylesheet & Visual Theme for FSOC Optical Simulator GUI.
===================================================================================
Matches the exact terminal/tech aesthetic: deep space dark background, neon cyan/green
indicators, crisp white typography, and glassmorphic card borders.
"""

DARK_THEME_QSS = """
/* Global Window Styling */
QMainWindow, QDialog {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 13px;
}

/* Card Panels & Containers */
QFrame.card-panel {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 8px;
}

QFrame.card-header {
    border-bottom: 1px solid #1f2937;
    margin-bottom: 6px;
}

/* Header Banner */
QLabel.header-title {
    font-size: 18px;
    font-weight: bold;
    color: #38bdf8;
    letter-spacing: 2px;
}

QLabel.status-indicator-active {
    color: #10b981;
    font-weight: bold;
}

QLabel.status-indicator-inactive {
    color: #ef4444;
    font-weight: bold;
}

/* Section Titles */
QLabel.section-title {
    font-size: 13px;
    font-weight: bold;
    color: #94a3b8;
    letter-spacing: 1px;
    padding-bottom: 4px;
}

/* Value Labels */
QLabel.value-text {
    font-size: 13px;
    color: #f8fafc;
    font-weight: 500;
}

QLabel.accent-green {
    color: #10b981;
    font-weight: bold;
}

QLabel.accent-amber {
    color: #f59e0b;
    font-weight: bold;
}

QLabel.accent-blue {
    color: #38bdf8;
    font-weight: bold;
}

/* Controls & Sliders */
QSlider::groove:horizontal {
    border: 1px solid #374151;
    height: 6px;
    background: #1f2937;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #38bdf8;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #f8fafc;
    border: 1px solid #38bdf8;
    width: 14px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 7px;
}

QSlider::groove:vertical {
    border: 1px solid #374151;
    width: 6px;
    background: #1f2937;
    border-radius: 3px;
}

QSlider::sub-page:vertical {
    background: #38bdf8;
    border-radius: 3px;
}

QSlider::handle:vertical {
    background: #f8fafc;
    border: 1px solid #38bdf8;
    height: 14px;
    margin-left: -5px;
    margin-right: -5px;
    border-radius: 7px;
}

/* Action Buttons */
QPushButton {
    background-color: #1e293b;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #334155;
    border-color: #38bdf8;
    color: #38bdf8;
}

QPushButton:pressed {
    background-color: #0f172a;
}

QPushButton#btn-start {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
}

QPushButton#btn-start:hover {
    background-color: #047857;
    color: #ffffff;
}

QPushButton#btn-stop {
    background-color: #7f1d1d;
    color: #fca5a5;
    border: 1px solid #dc2626;
}

QPushButton#btn-stop:hover {
    background-color: #b91c1c;
    color: #ffffff;
}

QPushButton#btn-export {
    background-color: #1e1b4b;
    color: #a78bfa;
    border: 1px solid #6d28d9;
}

QPushButton#btn-export:hover {
    background-color: #4c1d95;
    color: #ffffff;
}

/* Tabs & Dialog Elements */
QTabWidget::pane {
    border: 1px solid #1f2937;
    background-color: #111827;
    border-radius: 6px;
}

QTabBar::tab {
    background-color: #0f172a;
    color: #94a3b8;
    border: 1px solid #1f2937;
    padding: 8px 14px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: bold;
}

QTabBar::tab:selected {
    background-color: #1e293b;
    color: #38bdf8;
    border-bottom-color: #38bdf8;
}

QTabBar::tab:hover {
    color: #38bdf8;
}

QGroupBox {
    border: 1px solid #1f2937;
    border-radius: 6px;
    margin-top: 12px;
    font-weight: bold;
    color: #38bdf8;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 5px 8px;
    font-family: 'Consolas', 'Monaco', monospace;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #38bdf8;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #334155;
}

QComboBox QAbstractItemView {
    background-color: #0f172a;
    color: #f8fafc;
    selection-background-color: #1e293b;
    selection-color: #38bdf8;
    border: 1px solid #334155;
}

QCheckBox {
    color: #e2e8f0;
    font-size: 13px;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #334155;
    background-color: #0f172a;
    border-radius: 3px;
}

QCheckBox::indicator:checked {
    background-color: #38bdf8;
    border-color: #38bdf8;
}
"""

