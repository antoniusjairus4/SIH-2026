# SIH 2026 - Problem Statement 169 (ISRO)
## AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile Free Space Optical Communication (FSOC) Terminals

**Organization / Domain:** Indian Space Research Organisation (ISRO) / Department of Space — Space Technology, Smart Automation  
**Repository:** [SIH-2026](https://github.com/antoniusjairus4/SIH-2026)

---

## 📊 Evaluation Scope & Score Distribution

| Evaluation Category | Weight | Target Metric / Deliverable |
| :--- | :---: | :--- |
| **Functional Live Demo + GUI** | 20% | Closed-loop tracking demo in Unity with interactive UI |
| **Benchmark-1 (Scenarios & Error Logs)** | 30% | Performance under disturbances, CSV/JSON log generation |
| **Benchmark-2 (Direct .mp4 Centroiding)** | 30% | Offline video centroiding pipeline without active gimbal feedback |
| **Technical Architecture & Q&A** | 20% | Modular codebase, architecture design, and defense |

---

## 1. Executive Summary & Problem Formulation

In Free Space Optical Communication (FSOC), high-bandwidth data transmission uses narrow-divergence laser beams. In mobile operational scenarios (satellites, aircraft, ground terminals), platform vibration, atmospheric turbulence, and high relative velocities make direct fine-beam pointing impossible without an initial wide-field **Coarse Alignment** stage.

This project delivers an autonomous, software-driven closed-loop tracking pipeline without physical hardware dependencies. The architecture couples a high-fidelity virtual space simulation running in **Unity (C#)** with an external real-time intelligence stack running in **Python** across a decoupled TCP socket boundary (Port 5005). The system detects, identifies, filters, and centers an optical beacon under severe atmospheric turbulence, sensor noise, platform motion, and camera jitter while maintaining strict control limits and low latency.

---

## 2. High-Level System Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NOORUL: UNITY SIMULATOR (C#)                       │
│  - 3D Dark Space Scene (640x480 Monochrome FPA)                             │
│  - Target Dynamics: Linear, Circular, Figure-8, Random                      │
│  - Disturbance Injector: Salt & Pepper, Gaussian, Jitter, Fog, Rain         │
│  - Virtual PTZ Motor Model (Max Slew: 5 deg/s, >= 20Hz loop)                │
└──────────────┬──────────────────────────────────────────────▲───────────────┘
               │ RGB Frame (640x480)                          │ pan_delta
               │ frame_id + timestamp                         │ tilt_delta
               │ (TCP Port 5005 @ 30Hz)                       │ (>= 20Hz)
               ▼                                              │
┌──────────────────────────────────────────────┐              │
│       JEEVAN: NETWORK & BENCHMARK RUNNER     │              │
│  - TCP Receiver / Frame Deserialization      │              │
│  - Direct .mp4 File Mode (Benchmark-2)       │              │
└──────────────┬───────────────────────────────┘              │
               │ Raw Frame (NumPy Array)                      │
               ▼                                              │
┌──────────────────────────────────────────────┐              │
│     JAIRUS: CV / CNN DETECTION ENGINE        │              │
│  - Fast Path: Median + Top-Hat + Sub-pixel   │              │
│  - AI Path: YOLOv8n / YOLO11n ONNX Fallback  │              │
└──────────────┬───────────────────────────────┘              │
               │ Raw Coordinate (u, v) + Confidence           │
               ▼                                              │
┌──────────────────────────────────────────────┐              │
│      DHANYA: KALMAN STATE ESTIMATOR          │              │
│  - Constant Acceleration EKF / Motion Model  │              │
│  - Jitter Filtering (+/- 20 px/frame)        │              │
│  - Occlusion Dead-Reckoning                  │              │
└──────────────┬───────────────────────────────┘              │
               │ Filtered State: [x, y, vx, vy]               │
               ▼                                              │
┌──────────────────────────────────────────────┐              │
│     JAIRUS: PID CONTROL & REACQUISITION      │              │
│  - Pixel-to-Angle Mapping (4°x3° FOV)        │              │
│  - Dual-Axis PID Loop with Slew-Limit Clamping│              │
│  - Spiral Search Reacquisition State Machine │              │
└──────────────┬───────────────────────────────┘              │
               └──────────────────────────────────────────────┘
               │ Diagnostic Telemetry
               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│            JAIRUS: PERFORMANCE LOGGER & UNIFIED LAUNCHER                    │
│  - Real-time RMSE, Tracking Error, Lock Retention, Latency/FPS              │
│  - Auto-generated CSV / JSON evaluation logs                                │
│  - 1-Click Desktop Packaging (PyQt / Subprocess supervisor)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Comprehensive Module Descriptions & Team Ownership

### Module 1: Unity Simulation & Virtual Environment
* **Owner:** Noorul
* **Tech Stack:** Unity Engine, C#, HLSL Compute Shaders, .NET Sockets
* **Key Responsibilities:**
  * **Scene Geometry & Optics:** Deep-space environment; virtual Focal Plane Array (FPA) monochrome camera running at $640 \times 480$ resolution with $4.0^\circ \times 3.0^\circ$ FOV.
  * **Target Dynamics:** Configurable optical beacon ($5\times5$ to $20\times20\text{ px}$, default $10\times10\text{ px}$). 4 kinematic models: Straight Line, Circular, Figure-8, and Random Brownian Walk.
  * **Disturbance Injector:** Modular shaders for 10% Salt & Pepper, Gaussian ($\sigma \le 20\text{ px}$), Poisson noise, camera jitter ($\pm 20\text{ px/frame}$), platform drift ($\pm 20\text{ px/frame}$), and atmospheric attenuation (fog, rain, low-light).
  * **Pan-Tilt Actuator Physics:** 2-axis gimbal model with slew rate limiting (default $5^\circ/\text{s}$, adjustable to $10^\circ/\text{s}$) and angular end-stops. Loop rates: Control $\ge 20\text{ Hz}$, Camera Render $\ge 30\text{ Hz}$.
  * **Socket Server & GUI:** Non-blocking TCP server (port 5005). Interactive GUI for real-time noise/target tuning and telemetry HUD.
  * **Synthetic Dataset Exporter:** Script to export 3,000 labeled frames + ground-truth bounding box text files (`.txt`) for AI model fine-tuning.

---

### Module 2: Socket Client, Serialization & Benchmark-2 Runner
* **Owner:** Jeevan
* **Tech Stack:** Python, `socket`, `struct`, `NumPy`, OpenCV (`cv2`)
* **Key Responsibilities:**
  * **TCP Client Socket Layer:** Multi-threaded streaming client connecting to `localhost:5005`.
  * **Zero-Copy Ingestion:** Unpacks header (`frame_id`: 4-byte int, `timestamp`: 8-byte double) and raw RGB payload ($640\times 480\times 3 = 921,600\text{ bytes}$) into contiguous NumPy arrays with sub-millisecond overhead.
  * **Backpressure Management:** Ring-buffer / latest-frame drop policy to eliminate phase-lag.
  * **Command Uplink:** Serializes and sends `(pan_delta, tilt_delta)` floats back to Unity at $\ge 20\text{ Hz}$.
  * **Benchmark-2 Offline Execution Runner:** Standalone CLI script for processing pre-recorded `.mp4` video feeds (`cv2.VideoCapture`), executing detection, and logging metrics without PTZ feedback.

---

### Module 3: Detection Engine (Classical CV + CNN Fallback)
* **Owner:** Jairus
* **Tech Stack:** Python, OpenCV, NumPy, ONNX Runtime, Ultralytics (YOLOv8n / YOLO11n)
* **Key Responsibilities:**
  * **Fast Path (Classical CV - 1 to 3 ms, >300 FPS):**
    1. Convert $640\times 480$ frame to 8-bit single-channel grayscale.
    2. $3\times 3$ or $5\times 5$ Median Blur (`cv2.medianBlur`) to filter 10% Salt & Pepper noise.
    3. Morphological White Top-Hat Transform ($15\times 15$ structuring element) to remove non-uniform background haze and lighting gradients.
    4. Dynamic thresholding for bright spot isolation.
    5. Sub-pixel Center of Gravity (CoG) computation via spatial moments:
       $$C_x = \frac{M_{10}}{M_{00}}, \quad C_y = \frac{M_{01}}{M_{00}}$$
  * **Lightweight CNN Path (Degraded Weather Fallback):**
    * Triggers when heavy fog/rain breaks contour moments ($M_{00} \approx 0$).
    * Fine-tuned YOLOv8n ONNX model (trained on Noorul's 3,000 synthetic frames).
    * Sub-8ms box localization and centroid extraction.
  * **Output Contract:** `(x, y, confidence)` or `(None, None, 0.0)` during complete occlusion.

---

### Module 4: State Estimator (Kalman Filter)
* **Owner:** Dhanya
* **Tech Stack:** Python, NumPy, SciPy
* **Key Responsibilities:**
  * **Kinematic State Formulation:** Constant Acceleration (CA) Discrete Kalman Filter:
    $$\mathbf{x}_k = \begin{bmatrix} x & y & v_x & v_y & a_x & a_y \end{bmatrix}^T$$
  * **Jitter & Vibration Rejection:** Rejects $\pm 20\text{ px/frame}$ high-frequency platform jitter via tuned measurement covariance matrix $\mathbf{R}$ and process covariance matrix $\mathbf{Q}$.
  * **Occlusion Dead-Reckoning:** Pure prediction cycles ($\mathbf{x}_k = \mathbf{F}\mathbf{x}_{k-1}$) for up to $1.0\text{ s}$ when detection returns `None`.
  * **Output Contract:** Filtered state tuple `(x_est, y_est, vx, vy)`.

---

### Module 5: Pan-Tilt Control Loop & Reacquisition Engine
* **Owner:** Jairus
* **Tech Stack:** Python, Standard Math
* **Key Responsibilities:**
  * **Coordinate Transformation:** Map pixel offset from center $(320, 240)$ to angular gimbal errors:
    $$\text{Scale}_x = \frac{4.0^\circ}{640\text{ px}} = 0.00625^\circ/\text{px}, \quad \text{Scale}_y = \frac{3.0^\circ}{480\text{ px}} = 0.00625^\circ/\text{px}$$
    $$\Delta \theta_{\text{pan\_err}} = (x_{\text{est}} - 320.0) \times \text{Scale}_x, \quad \Delta \phi_{\text{tilt\_err}} = -(y_{\text{est}} - 240.0) \times \text{Scale}_y$$
  * **Dual-Axis PID Controller:** Proportional, Integral, Derivative calculation with anti-windup clamping and motor slew-rate output limiting ($\le 5^\circ/\text{s}$ or $\le 0.166^\circ/\text{step}$ at $30\text{ Hz}$).
  * **Reacquisition State Machine:** Transitions to `REACQUISITION_MODE` if lost for $>0.5\text{ s}$. Drives an Archimedean spiral search pattern outward from last known position to re-acquire target within $\le 1.0\text{ s}$.
  * **Output Contract:** Signed `(pan_delta, tilt_delta)` floats in degrees.

---

### Module 6: Performance Logger, Analytics & Unified Launcher
* **Owner:** Jairus
* **Tech Stack:** Python, PyQt6 / Tkinter, Pandas, Matplotlib
* **Key Responsibilities:**
  * **Real-Time Evaluation Metrics Engine:**
    * *Tracking Error:* Euclidean pixel offset $\sqrt{(x-320)^2 + (y-240)^2}$ ($\le 10\text{ px}$ target).
    * *Centroiding RMSE:* Evaluated against ground-truth coordinates.
    * *Acquisition Time:* Time from initiation to continuous stable lock ($\le 2\text{ s}$ target).
    * *Re-acquisition Time:* Time to re-acquire lock after loss ($\le 1\text{ s}$ target).
    * *Lock Retention Rate:* Percentage of operational frames with error $\le 10\text{ px}$ ($>95\%$ target).
    * *Frame Throughput:* Loop processing latency and FPS ($\ge 20\text{ FPS}$ target).
  * **Data Logging:** Synchronized `.csv` and summary `.json` report export for Benchmark-1 and Benchmark-2.
  * **Unified Desktop Launcher:** 1-click GUI dashboard launching Unity `.exe` as a subprocess, managing Python client threads, rendering live charts, and ensuring clean cleanup on exit.

---

## 4. Official Parameter Verification Matrix

| Parameter | Official Specification | Implementation & Handling |
| :--- | :--- | :--- |
| **Screen Resolution** | $640 \times 480\text{ px}$ (min $2000 \times 2000$ canvas) | Unity FPA monochrome camera renders at $640 \times 480$. |
| **Camera FOV** | $4.0^\circ \times 3.0^\circ$ (Configurable) | Fixed ratio $0.00625^\circ/\text{px}$ in PID error mapper. |
| **Camera Update Rate** | $\ge 30\text{ Hz}$ | Unity camera rendering loop locked to 30–60 FPS. |
| **Target Size** | $5 \times 5$ to $20 \times 20\text{ px}$ (Default: $10 \times 10\text{ px}$) | White Top-Hat filter kernel ($15 \times 15$) tuned to spot geometry. |
| **Motion Profiles** | $\ge 4$ Types (Linear, Circular, Fig-8, Random) | Selectable state classes in Unity C# scripts. |
| **Max Slew Rate** | $5\text{ to }10^\circ/\text{s}$ (Default: $5^\circ/\text{s}$) | Hard clamping bounds in PID controller output stage. |
| **Control Update Rate** | $\ge 20\text{ Hz}$ | Python loop targets $30\text{ Hz}$ ($<33\text{ ms}$ budget). |
| **Tracking Error** | $\le 10\text{ px}$ | Sub-pixel CoG + EKF achieves $\sim 1\text{ to }3\text{ px}$. |
| **Target Loss Rate** | $< 5\%$ | Dual Classical CV / CNN engine maintains $>95\%$ lock. |
| **Acquisition Time** | $\le 2\text{ seconds}$ | Fast path classical CV locks in $<3\text{ ms}$ on first frame. |
| **Re-acquisition Time** | $\le 1\text{ second}$ | High-speed Archimedean spiral search pattern sweeps FOV. |
| **Disturbances** | 10% S&P, Gaussian ($\sigma=20$), Jitter ($\pm 20\text{ px}$), Weather | Median filter + EKF process noise matrix reject disturbance. |

---

## 5. Step-by-Step Team Execution Plan

```mermaid
graph TD
    subgraph Phase 1: Zero-Dependency Modular Dev (Days 1-2)
        P1A[Jairus: Classical Detector & PID Controller]
        P1B[Noorul: Unity 3D Space Scene & TCP Server]
        P1C[Jeevan: TCP Client Socket & Unpacker]
        P1D[Dhanya: EKF State Estimator]
    end

    subgraph Phase 2: Python Brain Integration (Day 3)
        P2A[Integrate: Socket -> CV Detector -> EKF -> PID]
        P2B[Benchmark Pipeline Throughput: Target < 15ms]
    end

    subgraph Phase 3: Hardware-in-the-Loop Integration (Day 4)
        P3A[Connect Python Brain to Unity via localhost:5005]
        P3B[Live PID Tuning against 5°/s Slew Limits]
    end

    subgraph Phase 4: Disturbance Validation & AI Training (Day 5)
        P4A[Enable Shaders: Noise, Jitter, Weather in Unity]
        P4B[Export 3,000 Synthetic Frames]
        P4C[Train & Fine-tune YOLOv8n ONNX Fallback]
    end

    subgraph Phase 5: Verification & Packaging (Day 6)
        P5A[Jeevan: Test Benchmark-2 .mp4 Runner]
        P5B[Jairus: Assemble PyQt Unified Launcher & Metrics UI]
        P5C[Execute Benchmark-1 & Benchmark-2 Validation Runs]
    end

    P1A --> P2A
    P1B --> P3A
    P1C --> P2A
    P1D --> P2A
    P2A --> P2B
    P2B --> P3A
    P3A --> P3B
    P3B --> P4A
    P4A --> P4B
    P4B --> P4C
    P4C --> P5A
    P5A --> P5B
    P5B --> P5C
```
