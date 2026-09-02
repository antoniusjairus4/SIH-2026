"""
Unit tests for GroundTruthLoader.
"""

import os
import tempfile
import pytest
from src.benchmark.ground_truth_loader import GroundTruthLoader


def test_standard_csv_loading():
    content = """frame_id,gt_x,gt_y,timestamp
0,320.5,240.0,0.000
1,322.0,241.5,0.033
2,324.5,243.0,0.066
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.is_loaded is True
        assert loader.total_records == 3
        assert loader.valid_coordinate_count == 3

        rec0 = loader.get_ground_truth(0)
        assert rec0 is not None
        assert rec0.frame_id == 0
        assert rec0.gt_x == 320.5
        assert rec0.gt_y == 240.0
        assert rec0.timestamp == 0.000

        rec1 = loader.get_ground_truth(1)
        assert rec1 is not None
        assert rec1.gt_x == 322.0

        # Non-existent frame
        assert loader.get_ground_truth(99) is None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_alias_column_names_and_delimiters():
    # Semicolon delimited with alias headers
    content = """frame;pos_x;pos_y;time_s;occluded
0;100.2;200.4;0.0;false
1;105.0;202.0;0.033;true
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.is_loaded is True
        assert loader.total_records == 2
        rec1 = loader.get_ground_truth(1)
        assert rec1 is not None
        assert rec1.gt_x == 105.0
        assert rec1.gt_y == 202.0
        assert rec1.is_occluded is True
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_headerless_csv():
    content = """0,50.0,60.0
1,55.0,65.0
2,60.0,70.0
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.is_loaded is True
        assert loader.total_records == 3
        rec2 = loader.get_ground_truth(2)
        assert rec2 is not None
        assert rec2.gt_x == 60.0
        assert rec2.gt_y == 70.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_malformed_and_missing_values():
    content = """frame_id,gt_x,gt_y
0,10.0,20.0
1,invalid,NaN
2,15.0,
3,30.0,40.0
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.is_loaded is True
        # Row 0 valid
        assert loader.get_ground_truth(0).gt_x == 10.0
        # Row 1 has None for coords
        assert loader.get_ground_truth(1).gt_x is None
        # Row 3 valid
        assert loader.get_ground_truth(3).gt_x == 30.0
        assert loader.valid_coordinate_count == 2
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_time_based_lookup():
    content = """frame_id,gt_x,gt_y,timestamp
0,100.0,100.0,0.00
1,110.0,100.0,0.50
2,120.0,100.0,1.00
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        # Exact match
        rec = loader.get_ground_truth_by_time(0.50, tolerance_s=0.05)
        assert rec is not None
        assert rec.frame_id == 1

        # Match within tolerance
        rec_near = loader.get_ground_truth_by_time(0.52, tolerance_s=0.05)
        assert rec_near is not None
        assert rec_near.frame_id == 1

        # Out of tolerance
        rec_out = loader.get_ground_truth_by_time(0.70, tolerance_s=0.05)
        assert rec_out is None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_nonexistent_file():
    loader = GroundTruthLoader("non_existent_file_path_12345.csv")
    assert loader.is_loaded is False
    assert loader.total_records == 0
    assert loader.get_ground_truth(0) is None
