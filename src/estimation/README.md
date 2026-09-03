# FSOC Beacon State Estimator

## Overview

This module implements the state-estimation stage of the AI-based virtual camera tracking system developed for coarse alignment of mobile Free Space Optical Communication (FSOC) terminals.

The estimator receives noisy beacon detections from the detection module and produces a filtered and predicted motion state for downstream pan-tilt control.

The module is intentionally independent of:

- Unity scene generation
- Camera rendering
- CNN / classical detection implementation
- PID controller implementation
- Reacquisition search patterns
- Ground-truth information during live operation

The production data flow is:

Detector → Kalman State Estimator → Controller

---

## Input Interface

Each estimator update accepts:

- `x` — detected beacon horizontal position in pixels
- `y` — detected beacon vertical position in pixels
- `confidence` — detector confidence in the range 0 to 1
- `timestamp` — frame timestamp in seconds

If the beacon is not detected:

```python
x = None
y = None
confidence = 0.0