"""
Benchmark package for SIH 2026 Virtual Camera Tracking System.
"""

from src.benchmark.pipeline_adapter import (
    BeaconDetectorProtocol,
    StateEstimatorProtocol,
    CombinedTrackingPipeline,
    NullDetector,
    NullStateEstimator,
    MockThresholdDetector,
    MockSimpleTracker,
)
from src.benchmark.ground_truth_loader import GroundTruthLoader
from src.benchmark.benchmark2_runner import Benchmark2Runner

__all__ = [
    "BeaconDetectorProtocol",
    "StateEstimatorProtocol",
    "CombinedTrackingPipeline",
    "NullDetector",
    "NullStateEstimator",
    "MockThresholdDetector",
    "MockSimpleTracker",
    "GroundTruthLoader",
    "Benchmark2Runner",
]
