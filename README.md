# SIH 2026 - Problem Statement 169 (ISRO)
## AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile Free Space Optical Communication (FSOC) Terminals

**Organization / Domain:** Department of Space / Indian Space Research Organisation (ISRO) — Smart Automation, Space Technology  
**Category:** Software | **Repository:** [SIH-2026](https://github.com/antoniusjairus4/SIH-2026)

---

## 📋 Official ISRO Evaluation Method and Criteria (100 Marks Total)

| Evaluation Stage | Description | Evaluation Method & Criteria | Weight (%) |
| :--- | :--- | :--- | :---: |
| **Functional Verification** | 10-15 min live demonstration of software functionality | 1. Implementation of all mandatory functions<br>2. Operational success<br>3. Interactive GUI | **20%** |
| **Benchmark Performance-1** | Execution under provided operational scenarios | 1. Scenario execution<br>2. Centroiding error logs<br>3. Automatically generated performance reports | **30%** |
| **Benchmark Performance-2** | Processing pre-recorded `.mp4` video files @ 30 FPS with noise & moving beacon spot | 1. Bypass PTZ camera & process raw `.mp4`<br>2. Centroiding error comparison<br>3. Metrics: RMSE, acquisition & re-acquisition time, lock retention rate, FPS | **30%** |
| **Technical Evaluation** | Presentation of approach, architecture, and design to ISRO evaluators | 1. Problem understanding & architecture design<br>2. Algorithm selection & AI/CV methodology<br>3. Innovation, technical documentation, Q&A | **20%** |

---

## 🎯 Official ISRO Parameters & Specifications Matrix

| Sr. No. | Parameter | Suggested Value | Remarks |
| :---: | :--- | :--- | :--- |
| **Camera Parameters** | | | |
| 1 | Screen Size (min.) | $2000 \times 2000\text{ pixels}$ | Optional: User-defined |
| 2 | Camera Type | Monochrome, Focal Plane Array (FPA) | Optional: Colour |
| 3 | Camera Resolution | $640 \times 480\text{ pixels}$ | Default: $640 \times 480$ (User-defined) |
| 4 | Camera FOV | $4.0^\circ \times 3.0^\circ$ | Default: $4^\circ \times 3^\circ$ |
| 5 | Camera Update Rate | $\ge 30\text{ Hz}$ | Minimum 30 Hz |
| 6 | Initial Camera Position | Centre of the Screen | Default starting frame |
| **Target Parameters** | | | |
| 7 | Target Type | Beacon Spot | Optical laser beacon spot |
| 8 | Number of Targets | 1 (Mandatory) | Multiple optional |
| 9 | Target Shape | Square | Default: Square (User-defined) |
| 10 | Target Size | $5 \times 5$ to $20 \times 20\text{ pixels}$ | Default: $10 \times 10\text{ pixels}$ |
| 11 | Initial Target Location | User-defined | Default: Random |
| 12 | Motion Profiles | Selectable: Linear, Circular, Figure of 8, Random | Mandatory $\ge 4$ profiles. Optional: Spiral, Sinusoidal |
| **Camera Motion Constraints** | | | |
| 13 | Max. Pan Speed | $5\text{ to }10^\circ/\text{s}$ | Default: $5^\circ/\text{s}$ (User-defined) |
| 14 | Max. Tilt Speed | $5\text{ to }10^\circ/\text{s}$ | Default: $5^\circ/\text{s}$ (User-defined) |
| 15 | Update Interval | $\ge 20\text{ Hz}$ | Target: $30\text{ Hz}$ loop |
| **Performance Specifications** | | | |
| 16 | Acquisition Time | $\le 2\text{ seconds}$ | Time to continuous lock |
| 17 | Tracking Error | $\le 10\text{ pixels}$ | Max Euclidean offset from center |
| 18 | Target Loss Rate | $< 5\%$ | $>95\%$ lock retention rate |
| 19 | Re-acquisition Time | $\le 1\text{ second}$ | Time to re-lock after occlusion |
| 20 | Processing Speed | $\ge 20\text{ FPS}$ | Real-time loop throughput |
| **Disturbances & Noise** | | | |
| 21 | Image Noise | 1. Salt & Pepper (~10%), 2. Gaussian, 3. Poisson | User Selectable (one or more) |
| 22 | Max. Noise Std Dev | $20\text{ pixels}$ | User-defined Gaussian noise |
| 23 | Max. Camera Jitter | $\pm 20\text{ pixels / frame}$ | User-defined camera vibration |
| 24 | Atmospheric Disturbance| Clear, Haze, Fog, Rain, Low Light | User-defined contrast & brightness attenuation |
| 25 | Platform Motion | $\pm 20\text{ pixels / frame (max.)}$ | Selectable (Linear default, optional circular/random) |

---

## 1. Executive Summary & Problem Formulation

In Free Space Optical Communication (FSOC), data transmission uses highly directional optical laser beams. Mobile operational platforms (satellites, UAVs, ground terminals) experience platform vibration, atmospheric turbulence, and high relative velocities. Before a fine-pointing mechanism can take over, a **Coarse Alignment stage** must observe the environment, detect the remote terminal beacon, estimate its position, and adjust the pointing direction to keep the beacon within the camera Field-of-View (FOV).

This project delivers a software-based virtual camera tracking system providing autonomous coarse alignment in a simulated environment without specialized physical hardware.

---

## 2. High-Level System Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          NOORUL: UNITY SIMULATOR (C#)                       │
│  - 3D Dark Space Scene (640x480 Monochrome FPA)                             │
│  - Target Dynamics: Linear, Circular, Figure-8, Random                      │
│  - Disturbance Injector: Salt & Pepper, Gaussian, Jitter, Fog, Rain         │
│  - Virtual PTZ Motor Model (Max Slew: 5 deg/s, >= 20Hz loop)                │
│  - Dataset Exporter: 3,000 Synthetic Frames + Bounding Box Text Labels     │
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
│   JAIRUS & AI ENGINEER: CV & CNN ENGINE      │              │
│  - Fast Path: Median + Top-Hat + Sub-pixel   │              │
│  - AI Fallback: Fine-Tuned YOLOv8n ONNX      │              │
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

## 3. Modular Responsibilities & Team Ownership

### Module 1: Unity Simulation & Virtual Environment
* **Owner:** Noorul
* **Tech Stack:** Unity Engine, C#, HLSL Compute Shaders, .NET Sockets
* **Deliverables:**
  * Virtual FPA camera ($640 \times 480$, $4.0^\circ \times 3.0^\circ$ FOV, $\ge 30\text{ Hz}$).
  * Moving target spot ($5\times5$ to $20\times20\text{ px}$) with 4 motion profiles (Linear, Circular, Figure-8, Random).
  * Disturbances: Salt & Pepper (10%), Gaussian ($\sigma \le 20\text{ px}$), Poisson, camera jitter ($\pm 20\text{ px/frame}$), atmospheric fog/rain.
  * Virtual PTZ gimbal physics ($5^\circ/\text{s}$ max slew rate limit).
  * TCP Socket Server on Port 5005 streaming raw frames and receiving gimbal commands.
  * **Synthetic Dataset Exporter:** 3,000 synthetic frames + normalized bounding box `.txt` label files for AI model training.

---

### Module 2: Network Client, Serialization & Benchmark-2 Runner
* **Owner:** Jeevan
* **Tech Stack:** Python, `socket`, `struct`, `NumPy`, OpenCV (`cv2`)
* **Deliverables:**
  * Multi-threaded TCP client ingesting frames from `localhost:5005` at $\ge 30\text{ Hz}$ without lag.
  * Command uplink sending `(pan_delta, tilt_delta)` floats back to Unity.
  * **Benchmark-2 Offline Execution Runner:** Standalone CLI script reading pre-recorded `.mp4` video files, executing detection/tracking, bypassing PTZ camera, and generating accuracy logs against ground truth.

---

### Module 3: Detection Engine & AI Model Pipeline
* **Owners:** Jairus (CV Core & Pipeline Integration) & AI Model Engineer (AI Fine-Tuning & ONNX Exporter)
* **Tech Stack:** Python, OpenCV, NumPy, ONNX Runtime, Ultralytics (YOLOv8n)
* **Deliverables:**
  * **Fast Path (Jairus - 1 to 3 ms):** Grayscale $\rightarrow$ Median Filter ($3\times3$) $\rightarrow$ Morphological White Top-Hat ($15\times15$) $\rightarrow$ Dynamic Thresholding $\rightarrow$ Sub-pixel Center of Gravity ($C_x = M_{10}/M_{00}, C_y = M_{01}/M_{00}$).
  * **AI Model Training & ONNX Export (AI Teammate):** Fine-tune YOLOv8n on Noorul's 3,000 synthetic frames under heavy fog/rain, export to `beacon_yolo.onnx`, and deliver the ONNX model for CPU fallback execution ($<8\text{ ms}$).

---

### Module 4: State Estimator (Kalman Filter)
* **Owner:** Dhanya
* **Tech Stack:** Python, NumPy, SciPy (`cv2.KalmanFilter`)
* **Deliverables:**
  * Constant Acceleration Discrete Kalman Filter: $\mathbf{x}_k = [x, y, v_x, v_y, a_x, a_y]^T$.
  * Filters high-frequency camera jitter ($\pm 20\text{ px/frame}$).
  * Occlusion Dead-Reckoning: Predicts trajectory for up to $1.0\text{ s}$ during dropped frames or cloud cover.

---

### Module 5: Pan-Tilt Control Loop & Reacquisition Engine
* **Owner:** Jairus
* **Tech Stack:** Python, Standard Math
* **Deliverables:**
  * Coordinate Transformation ($0.00625^\circ/\text{px}$ scale).
  * Dual-Axis PID controller with anti-windup clamping and motor slew clamping ($\le 5^\circ/\text{s}$).
  * Reacquisition State Machine: Triggers Archimedean spiral search pattern when target is lost $>0.5\text{ s}$, re-acquiring lock in $\le 1.0\text{ s}$.

---

### Module 6: Performance Logger, Analytics & Unified Desktop Launcher
* **Owner:** Jairus
* **Tech Stack:** Python, PyQt6 / Tkinter, Pandas, Matplotlib
* **Deliverables:**
  * Real-Time Metrics Engine (Tracking Error $\le 10\text{ px}$, RMSE, Lock Retention $>95\%$, Acquisition Time $\le 2\text{ s}$, FPS $\ge 20$).
  * Automatic `.csv` and `.json` performance log auto-generator.
  * Unified 1-Click Desktop GUI App managing Unity `.exe` subprocess and Python telemetry pipeline.

---

## 📁 Official Mandatory Deliverables Checklist

- [x] **Standalone Executable Application:** 1-Click Desktop Packaging (PyQt GUI launcher + compiled Unity executable).
- [x] **Modular Source Code:** Clean, documented Python & C# scripts.
- [x] **Technical Report (10-15 pages):** PDF report covering architecture, CV/AI algorithms, EKF formulation, and PID dynamics.
- [x] **User Manual & Demo Video:** Installation guide, UI navigation, parameter configuration, and 3-5 min video demonstration.
- [x] **Performance Logs:** Automated CSV/JSON log generator recording FPS, Acquisition Time, Average & Max Error, Lock Retention Rate, and Processing Speed.
