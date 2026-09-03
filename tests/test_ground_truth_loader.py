"""
Unit tests and edge-case validation for GroundTruthLoader.
"""

import os
import math
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


def test_non_numeric_and_special_string_coordinates():
    content = """frame_id,gt_x,gt_y
0,null,none
1,N/A,undefined
2,?,xyz
3, ,
4,150.0,250.0
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.is_loaded is True
        for fid in range(4):
            rec = loader.get_ground_truth(fid)
            assert rec is not None
            assert rec.gt_x is None
            assert rec.gt_y is None
        rec4 = loader.get_ground_truth(4)
        assert rec4.gt_x == 150.0
        assert rec4.gt_y == 250.0
        assert loader.valid_coordinate_count == 1
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_scientific_notation_coordinates():
    content = """frame_id,gt_x,gt_y
0,1.5e2,-2.0e1
1,+3.4E1,4.5e-1
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.is_loaded is True
        rec0 = loader.get_ground_truth(0)
        assert rec0.gt_x == 150.0
        assert rec0.gt_y == -20.0
        rec1 = loader.get_ground_truth(1)
        assert rec1.gt_x == 34.0
        assert math.isclose(rec1.gt_y, 0.45, abs_tol=1e-5)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_nan_and_inf_coordinates_handling():
    content = """frame_id,gt_x,gt_y
0,nan,NaN
1,inf,-inf
2,+INF,200.0
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.is_loaded is True
        assert loader.get_ground_truth(0).gt_x is None
        assert loader.get_ground_truth(0).gt_y is None
        assert loader.get_ground_truth(1).gt_x is None
        assert loader.get_ground_truth(1).gt_y is None
        assert loader.get_ground_truth(2).gt_x is None
        assert loader.get_ground_truth(2).gt_y == 200.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_duplicate_frame_ids():
    content = """frame_id,gt_x,gt_y
0,10.0,20.0
1,30.0,40.0
1,99.0,88.0
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.is_loaded is True
        # Frame 1 should deterministically reflect last row
        rec1 = loader.get_ground_truth(1)
        assert rec1 is not None
        assert rec1.gt_x == 99.0
        assert rec1.gt_y == 88.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_missing_frame_ids_and_gaps():
    content = """frame_id,gt_x,gt_y
0,10.0,20.0
5,50.0,60.0
20,100.0,120.0
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.is_loaded is True
        assert loader.get_ground_truth(0) is not None
        assert loader.get_ground_truth(1) is None
        assert loader.get_ground_truth(4) is None
        assert loader.get_ground_truth(5) is not None
        assert loader.get_ground_truth(20) is not None
        assert loader.get_ground_truth(100) is None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_malformed_csv_delimiters_and_jagged_rows():
    content = """frame_id,gt_x,gt_y
0,10.0,20.0,extra1,extra2
1
2,30.0
3,40.0,50.0
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.is_loaded is True
        assert loader.get_ground_truth(0).gt_x == 10.0
        assert loader.get_ground_truth(1).gt_x is None
        assert loader.get_ground_truth(2).gt_y is None
        assert loader.get_ground_truth(3).gt_x == 40.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_time_based_lookup_edge_cases():
    # Unordered timestamps in CSV
    content = """frame_id,gt_x,gt_y,timestamp
1,110.0,100.0,0.50
0,100.0,100.0,0.00
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

        # Near match
        rec_near = loader.get_ground_truth_by_time(0.02, tolerance_s=0.05)
        assert rec_near is not None
        assert rec_near.frame_id == 0

        # Query before range
        assert loader.get_ground_truth_by_time(-0.5, tolerance_s=0.05) is None

        # Query after range
        assert loader.get_ground_truth_by_time(2.0, tolerance_s=0.05) is None

        # Negative tolerance
        assert loader.get_ground_truth_by_time(0.50, tolerance_s=-0.05) is None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_time_based_lookup_empty_timestamps():
    content = """frame_id,gt_x,gt_y
0,100.0,100.0
1,110.0,100.0
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(content)
        temp_path = f.name

    try:
        loader = GroundTruthLoader(temp_path)
        assert loader.get_ground_truth_by_time(0.5, tolerance_s=1.0) is None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_nonexistent_and_empty_file():
    loader = GroundTruthLoader("non_existent_file_path_12345.csv")
    assert loader.is_loaded is False
    assert loader.total_records == 0
    assert loader.get_ground_truth(0) is None

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        temp_empty = f.name

    try:
        empty_loader = GroundTruthLoader(temp_empty)
        assert empty_loader.is_loaded is False
        assert empty_loader.total_records == 0
    finally:
        if os.path.exists(temp_empty):
            os.remove(temp_empty)
