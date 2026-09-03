from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional


class TrackerMode(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    TRACKING = "TRACKING"
    COASTING = "COASTING"
    LOST = "LOST"


@dataclass
class EstimatorResult:
    x: Optional[float]
    y: Optional[float]

    vx: Optional[float]
    vy: Optional[float]

    ax: Optional[float]
    ay: Optional[float]

    predicted_x: Optional[float]
    predicted_y: Optional[float]

    mode: TrackerMode

    prediction_only: bool
    measurement_available: bool
    measurement_rejected: bool

    confidence: float

    missing_frames: int
    coast_time: float

    inside_fov: bool

    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the estimator result to a JSON-friendly dictionary.
        """

        result = asdict(self)

        # Enum -> ordinary string
        result["mode"] = self.mode.value

        # Explicit conversion protects module/network boundaries
        # from NumPy scalar types.
        for key in (
            "x", "y",
            "vx", "vy",
            "ax", "ay",
            "predicted_x", "predicted_y",
            "confidence",
            "coast_time",
            "timestamp",
        ):
            if result[key] is not None:
                result[key] = float(result[key])

        result["missing_frames"] = int(result["missing_frames"])
        result["prediction_only"] = bool(result["prediction_only"])
        result["measurement_available"] = bool(
            result["measurement_available"]
        )
        result["measurement_rejected"] = bool(
            result["measurement_rejected"]
        )
        result["inside_fov"] = bool(result["inside_fov"])

        return result