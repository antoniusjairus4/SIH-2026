from dataclasses import dataclass, asdict
from typing import Any, Dict

from src.estimation.state import TrackerMode


@dataclass
class ControlResult:
    """
    Output of a single PID controller computation step.

    Sign convention (matches frozen socket interface spec):
        pan_delta  positive -> pan right
        tilt_delta positive -> tilt up
    """

    # Signed angular command in degrees.
    pan_delta: float
    tilt_delta: float

    # False when the estimator mode is LOST or UNINITIALIZED,
    # signaling that control authority should yield to
    # Jeevan's reacquisition logic.
    should_command: bool

    # The TrackerMode under which this control decision was made,
    # useful for logging and post-hoc analysis.
    mode_at_command_time: TrackerMode

    # Angular error before speed clamping (degrees), useful for
    # debugging and gain-tuning.
    raw_error_x_deg: float
    raw_error_y_deg: float

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the control result to a JSON-friendly dictionary.
        """

        result = asdict(self)

        # Enum -> ordinary string
        result["mode_at_command_time"] = (
            self.mode_at_command_time.value
        )

        # Explicit float conversion protects module/network
        # boundaries from NumPy scalar types.
        for key in (
            "pan_delta",
            "tilt_delta",
            "raw_error_x_deg",
            "raw_error_y_deg",
        ):
            if result[key] is not None:
                result[key] = float(result[key])

        result["should_command"] = bool(result["should_command"])

        return result
