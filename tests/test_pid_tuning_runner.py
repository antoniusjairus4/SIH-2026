"""
Unit tests for Module 5 PID Grid Search & Tuning script.
"""

import os
import csv
import pytest

from examples.test_pid_tuning import main, simulate_closed_loop, generate_straight_line
from src.control import ControllerConfig


def test_simulate_closed_loop_returns_valid_metrics():
    """Verify closed-loop simulation produces valid numeric metrics."""
    cfg = ControllerConfig(kp_x=0.5, ki_x=0.05, kd_x=0.1)
    traj = generate_straight_line(num_frames=50)

    rmse_px, max_error_px, max_slew, compliance_pct = simulate_closed_loop(traj, cfg)

    assert isinstance(rmse_px, float)
    assert rmse_px > 0.0
    assert max_error_px >= rmse_px
    assert max_slew >= 0.0
    assert 0.0 <= compliance_pct <= 100.0


def test_main_grid_search_exports_csv_and_plot():
    """Verify main() grid search runs completely and outputs CSV & plot files."""
    main()

    assert os.path.exists("pid_grid_search_results.csv")
    assert os.path.exists("pid_tuning_results.png")

    with open("pid_grid_search_results.csv", "r") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) > 0
        assert "avg_rmse_px" in reader[0]
        assert "slew_compliance_pct" in reader[0]
