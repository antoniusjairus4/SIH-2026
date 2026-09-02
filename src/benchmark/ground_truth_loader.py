"""
Modular Ground Truth Loader for SIH 2026 Benchmark-2.
Parses companion ground-truth CSV/text files and provides indexed lookup for evaluation.
Strictly isolated from perception and tracking modules.
"""

import os
import csv
import logging
from typing import Dict, Optional, Tuple, Any

from src.metrics.schemas import GroundTruthRecord

logger = logging.getLogger(__name__)


class GroundTruthLoader:
    """
    Loads and indexes ground truth beacon coordinates from companion evaluation files.
    Supports flexible column naming, automatic delimiter detection, and safe fallback handling.
    """

    # Common candidate column header names
    FRAME_ID_ALIASES = {"frame_id", "frame", "frame_idx", "idx", "index", "id", "seq"}
    GT_X_ALIASES = {"gt_x", "x_gt", "x", "pos_x", "centroid_x", "target_x", "u"}
    GT_Y_ALIASES = {"gt_y", "y_gt", "y", "pos_y", "centroid_y", "target_y", "v"}
    TIMESTAMP_ALIASES = {"timestamp", "time", "time_s", "t", "video_time"}
    OCCLUDED_ALIASES = {"is_occluded", "occluded", "occlusion", "visible", "visibility"}

    def __init__(self, file_path: Optional[str] = None) -> None:
        self.file_path = file_path
        self._records_by_frame: Dict[int, GroundTruthRecord] = {}
        self._records_by_time: list[Tuple[float, GroundTruthRecord]] = []
        self._is_loaded = False
        self._total_rows = 0
        self._valid_coordinate_rows = 0

        if file_path:
            self.load_file(file_path)

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def total_records(self) -> int:
        return len(self._records_by_frame)

    @property
    def valid_coordinate_count(self) -> int:
        return self._valid_coordinate_rows

    def load_file(self, file_path: str) -> bool:
        """
        Parses a ground truth CSV file and populates the in-memory lookup index.

        Args:
            file_path: Absolute or relative path to companion CSV file.

        Returns:
            True if loaded successfully, False if file could not be read.
        """
        self.file_path = file_path
        self._records_by_frame.clear()
        self._records_by_time.clear()
        self._total_rows = 0
        self._valid_coordinate_rows = 0
        self._is_loaded = False

        if not os.path.exists(file_path):
            logger.warning(f"Ground truth file not found at: {file_path}")
            return False

        try:
            with open(file_path, mode="r", encoding="utf-8-sig") as f:
                # Read sample to detect delimiter
                sample = f.read(2048)
                f.seek(0)

                delimiter = ","
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
                    delimiter = dialect.delimiter
                except Exception:
                    delimiter = ","

                reader = csv.reader(f, delimiter=delimiter)
                first_row = next(reader, None)
                if not first_row:
                    logger.warning(f"Ground truth file is empty: {file_path}")
                    return False

                # Determine if first row is header
                col_map = self._map_columns(first_row)
                has_header = col_map is not None

                if not has_header:
                    # Treat first row as data (col 0: frame_id, col 1: x, col 2: y)
                    f.seek(0)
                    col_map = {"frame_id": 0, "gt_x": 1, "gt_y": 2}

                for line_num, row in enumerate(reader, start=2 if has_header else 1):
                    if not row or all(c.strip() == "" for c in row):
                        continue

                    record = self._parse_row(row, col_map, line_num)
                    if record is not None:
                        self._records_by_frame[record.frame_id] = record
                        if record.timestamp is not None:
                            self._records_by_time.append((record.timestamp, record))
                        self._total_rows += 1
                        if record.gt_x is not None and record.gt_y is not None:
                            self._valid_coordinate_rows += 1

            # Sort by timestamp for time-based lookups
            self._records_by_time.sort(key=lambda item: item[0])
            self._is_loaded = True
            logger.info(
                f"Loaded {self._total_rows} ground-truth records ({self._valid_coordinate_rows} valid) from {file_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to parse ground truth file {file_path}: {e}")
            return False

    def get_ground_truth(self, frame_id: int) -> Optional[GroundTruthRecord]:
        """
        Retrieves ground-truth record for a specific frame index.
        """
        return self._records_by_frame.get(frame_id)

    def get_ground_truth_by_time(
        self, timestamp: float, tolerance_s: float = 0.033
    ) -> Optional[GroundTruthRecord]:
        """
        Finds the closest ground-truth record matching a video timestamp within tolerance.
        """
        if not self._records_by_time:
            return None

        # Binary search for nearest timestamp
        left, right = 0, len(self._records_by_time) - 1
        best_record = None
        best_diff = float("inf")

        while left <= right:
            mid = (left + right) // 2
            t_mid, rec_mid = self._records_by_time[mid]
            diff = abs(t_mid - timestamp)

            if diff < best_diff:
                best_diff = diff
                best_record = rec_mid

            if t_mid < timestamp:
                left = mid + 1
            elif t_mid > timestamp:
                right = mid - 1
            else:
                return rec_mid

        if best_diff <= tolerance_s:
            return best_record
        return None

    def _map_columns(self, header_row: list[str]) -> Optional[Dict[str, int]]:
        """Maps column names to standard field keys."""
        normalized = [h.strip().lower().replace(" ", "_") for h in header_row]
        mapping: Dict[str, int] = {}

        for idx, name in enumerate(normalized):
            if name in self.FRAME_ID_ALIASES and "frame_id" not in mapping:
                mapping["frame_id"] = idx
            elif name in self.GT_X_ALIASES and "gt_x" not in mapping:
                mapping["gt_x"] = idx
            elif name in self.GT_Y_ALIASES and "gt_y" not in mapping:
                mapping["gt_y"] = idx
            elif name in self.TIMESTAMP_ALIASES and "timestamp" not in mapping:
                mapping["timestamp"] = idx
            elif name in self.OCCLUDED_ALIASES and "is_occluded" not in mapping:
                mapping["is_occluded"] = idx

        # Must have at least frame_id or gt_x to be considered a header
        if "gt_x" in mapping or "frame_id" in mapping:
            return mapping
        return None

    def _parse_row(
        self, row: list[str], col_map: Dict[str, int], line_num: int
    ) -> Optional[GroundTruthRecord]:
        """Parses a single row using column mapping safely."""
        try:
            # 1. Parse frame_id
            frame_id_idx = col_map.get("frame_id")
            if frame_id_idx is not None and frame_id_idx < len(row):
                raw_fid = row[frame_id_idx].strip()
                try:
                    frame_id = int(float(raw_fid)) if raw_fid else line_num - 1
                except ValueError:
                    frame_id = line_num - 1
            else:
                frame_id = line_num - 1

            # 2. Parse gt_x
            gt_x: Optional[float] = None
            gt_x_idx = col_map.get("gt_x")
            if gt_x_idx is not None and gt_x_idx < len(row):
                raw_x = row[gt_x_idx].strip()
                if raw_x and raw_x.lower() not in {"nan", "none", "null", ""}:
                    try:
                        gt_x = float(raw_x)
                    except ValueError:
                        gt_x = None

            # 3. Parse gt_y
            gt_y: Optional[float] = None
            gt_y_idx = col_map.get("gt_y")
            if gt_y_idx is not None and gt_y_idx < len(row):
                raw_y = row[gt_y_idx].strip()
                if raw_y and raw_y.lower() not in {"nan", "none", "null", ""}:
                    try:
                        gt_y = float(raw_y)
                    except ValueError:
                        gt_y = None

            # 4. Parse timestamp
            ts: Optional[float] = None
            ts_idx = col_map.get("timestamp")
            if ts_idx is not None and ts_idx < len(row):
                raw_ts = row[ts_idx].strip()
                if raw_ts and raw_ts.lower() not in {"nan", "none", "null", ""}:
                    try:
                        ts = float(raw_ts)
                    except ValueError:
                        ts = None

            # 5. Parse is_occluded
            occluded = False
            occ_idx = col_map.get("is_occluded")
            if occ_idx is not None and occ_idx < len(row):
                raw_occ = row[occ_idx].strip().lower()
                if raw_occ in {"true", "1", "yes", "occluded"}:
                    occluded = True

            return GroundTruthRecord(
                frame_id=frame_id,
                gt_x=gt_x,
                gt_y=gt_y,
                timestamp=ts,
                is_occluded=occluded,
            )
        except Exception as e:
            logger.debug(f"Skipping unparseable row {line_num}: {row} (error: {e})")
            return None
