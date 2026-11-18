# PathSnap: Distributed Map-Matching & Route Reconstruction System

[![Course](https://img.shields.io/badge/Course-UE24CS352A%20Machine%20Learning-blue.svg)](https://pes.edu)
[![Department](https://img.shields.io/badge/Department-CSE%2C%20PES%20University-orange.svg)](https://pes.edu)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](#)

**PathSnap** is a distributed Machine Learning system that reconstructs realistic vehicle trajectories by matching noisy, irregularly sampled GPS observations against a large-scale directed road network. It was developed for the **UE24CS352A Machine Learning** course at the Department of Computer Science and Engineering, PES University.

Operating across a **two-physical-laptop distributed architecture**, PathSnap combines vectorized STRtree spatial candidate pruning, an 11-dimensional leak-free feature engineering pipeline, Machine Learning candidate-relevance classifiers, Bayesian hyperparameter optimization (Optuna TPE), and a Hidden Markov Model (HMM) Viterbi decoder with topological shortest-path continuity assembly.

---

## Project Team

| Student Author | Student Registration Number (SRN) | Subsystem Ownership |
|:---|:---:|:---|
| **Vrushant K P** | `PES1UG24CS542` | ML Modeling & Causal Multi-Scale Feature Engineering |
| **Yashas C** | `PES1UG24CS544` | HMM Map-Matching & Viterbi Graph Traversal |
| **T Pranav** | `PES1UG24CS501` | Distributed FastAPI Architecture & Server Telemetry Backend |
| **Tejaswini Pujar** | `PES1UG24CS500` | Client UI Dashboard, System Benchmarks & Validation |

---

## Architecture Overview

PathSnap is partitioned into two distinct physical tiers communicating over a Local Area Network (LAN):

```
+---------------------------------------------------------------------------------------+
|                                     LAPTOP 1: ML BACKEND                             |
|                                                                                       |
|   +-------------------+      +-------------------+      +-------------------------+   |
|   | Raw Data (.track, | ---> | Vectorized STRtree| ---> | Backward-Difference     |   |
|   | .nodes, .arcs)    |      | Candidate Search  |      | Feature Engineering     |   |
|   +-------------------+      +-------------------+      +-------------------------+   |
|                                                                      |                |
|                                                                      v                |
|   +-------------------+      +-------------------+      +-------------------------+   |
|   | Reconstructed     | <--- | RoadMatchHMM      | <--- | Trained Relevance       |   |
|   | Continuous Route  |      | Viterbi Decoder   |      | Binary Classifier       |   |
|   +-------------------+      +-------------------+      +-------------------------+   |
|             |                                                                         |
|             v                                                                         |
|   +-------------------------------------------------------------------------------+   |
|   | FastAPI REST Server (Port 8000)                                               |   |
|   | Endpoints: GET /health | GET /api/trips | POST /api/predict | GET /api/evaluate  |   |
|   +-------------------------------------------------------------------------------+   |
+------------------------------------------+--------------------------------------------+
                                           | LAN HTTP REST (JSON)
                                           v
+---------------------------------------------------------------------------------------+
|                                  LAPTOP 2: EVALUATOR UI                               |
|                                                                                       |
|   +-------------------------------------------------------------------------------+   |
|   | Streamlit Dashboard (Port 8501)                                               |   |
|   | - LAN Backend Configuration & Heartbeat Status Check                          |   |
|   | - 10-Trip Selector with Observation & Duration Breakdown                      |   |
|   | - Interactive Multi-Layer Folium Map (GPS, Top-1, Reconstructed Route)        |   |
|   | - Server-Side Evaluation Metric Cards (Continuity, Alignment, Recall)         |   |
|   +-------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------+
```

### Subsystem Responsibilities

1. **Laptop 1 (ML Backend Engine)**:
   - Hosts the complete dataset (80 training trips, 10 student testing trips, reserved evaluator partition).
   - Equirectangular metric coordinate projection ($x = \text{lon} \cdot R \cos(\bar{\phi}), y = \text{lat} \cdot R$).
   - Vectorized Shapely STRtree spatial indexing (50m initial radius, adaptive expansion to 150m).
   - Strict backward-difference feature extraction (11 features; zero future-lookahead leakage).
   - 5 benchmarked ML models (`RandomForest`, `HistGradientBoosting`, `MLP`, `KNN`, `LogisticRegression`).
   - Bayesian Hyperparameter Optimization with Optuna (TPE sampler).
   - RoadMatchHMM Viterbi decoder with topological shortest-path route splicing.
   - FastAPI REST endpoints on port `8000`.
   - **Ground-Truth Isolation Guarantee:** Reference routes (`.route`) are stored exclusively on Laptop 1 and are NEVER leaked to the client prediction endpoint.

2. **Laptop 2 (Evaluator Dashboard UI)**:
   - Lightweight Streamlit client running on port `8501`.
   - Communicates exclusively via JSON over LAN with Laptop 1 (`http://<LAPTOP_1_IP>:8000`).
   - Allows dropdown selection of the 10 evaluation trips (`00000000`, `00000007`, `00000009`, `00000040`, `00000052`, `00000056`, `00000065`, `00000076`, `00000077`, `00000091`).
   - Triggers route reconstruction with tunable HMM parameters ($\sigma_z$, $\beta$, disconnect penalties).
   - Visualizes interactive multi-layer Folium maps:
     - 🔵 **Blue Dots**: Raw GPS observations
     - 🟠 **Orange Dashed**: Classifier Top-1 predictions
     - 🔴 **Red Solid**: PathSnap HMM reconstructed route
     - 🟢 **Green Solid**: Reference ground-truth route (toggleable)
   - Displays observation metrics (Top-1 accuracy, route continuity, latency).

---

## Directory Structure

```
PathSnap/
├── api/
│   ├── main.py                     # FastAPI application & LAN REST endpoints
│   ├── schemas.py                  # Pydantic request & response models
│   ├── inference.py                # End-to-end inference & route reconstruction service
│   └── evaluation.py               # Secure server-side ground truth evaluation service
├── src/
│   ├── data_loader.py              # File loading, validation & metric Equirectangular projection
│   ├── candidate_generator.py      # Vectorized Shapely STRtree candidate arc search (50-150m)
│   ├── feature_engineering.py      # 11 directional, kinematic, and rank features (zero leakage)
│   ├── label_generation.py         # Monotonic dynamic programming reference route association
│   ├── hmm_decoder.py              # RoadMatchHMM emission/transition modeling & Viterbi decoding
│   ├── metrics.py                  # Top-1 accuracy, route continuity & classification metrics
│   ├── map_visualization.py        # Interactive multi-layer Folium map generation
│   ├── train_and_evaluate.py       # 5-Fold GroupKFold CV & student-testing benchmark
│   ├── bayesian_optimization.py    # Optuna TPE Bayesian hyperparameter optimization
│   └── models/
│       ├── linear_model.py         # L2-regularized Logistic Regression
│       ├── random_forest.py        # Balanced Subsample Random Forest
│       ├── gradient_boosting.py    # HistGradientBoostingClassifier
│       ├── knn_model.py            # Distance-weighted K-Nearest Neighbors
│       └── mlp_model.py            # Multi-Layer Perceptron (64, 32)
├── ui/
│   └── app.py                      # Streamlit evaluator application (Laptop 2)
├── scripts/
│   ├── run_eda.py                  # Complete Exploratory Data Analysis & audit script
│   ├── build_dataset.py            # Feature engineering dataset generator (train & test)
│   └── evaluate_route_reconstruction.py # 10-trip benchmark evaluation (Classifier vs HMM)
├── report/
│   └── generate_report.py          # Consolidated Technical Report generator (ReportLab PDF)
├── tests/
│   ├── test_data_loader.py         # Data loader & projection tests
│   ├── test_feature_engineering.py # Feature extraction, causality & wraparound tests
│   ├── test_hmm_decoder.py         # HMM Viterbi decoding & route continuity tests
│   └── test_api.py                 # FastAPI integration & ground-truth isolation tests
├── data/
│   ├── raw/                        # Raw trip partitions (train, student_test, evaluator_test)
│   └── processed/                  # Processed feature datasets & label generation audit
├── artifacts/
│   ├── models/                     # Serialized trained model weights (*.joblib)
│   ├── plots/                      # EDA, CV comparisons, confusion matrices & Optuna plots
│   ├── results/                    # Dataset summary, model comparisons & benchmark JSONs
│   ├── maps/                       # Interactive Folium HTML maps
│   └── reports/                    # Compiled Consolidated Technical Report (PDF)
├── requirements.txt                # Python package dependencies
├── mermaid-diagram.svg             # System architecture diagram
└── README.md                       # Comprehensive project documentation
```

---

## Empirical Benchmark Results

### 1. Multi-Model 5-Fold GroupKFold Cross-Validation & Student-Testing Holdout

Models were evaluated across **215,716 training candidate samples** (5-Fold GroupKFold grouped strictly by `trip_id`) and **164,839 candidate samples across all 10 student-testing trips**:

| Model Family | Algorithm Configuration | 5-Fold CV Macro F1 | Student-Test Top-1 Acc | Route Continuity | Student-Test Macro F1 | Log Loss | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | `n=100, max_depth=12, balanced_subsample` | **0.8661** | **80.76%** | **89.91%** | **0.8750** | **0.1745** | **0.514s** |
| **HistGradientBoosting** | `lr=0.1, max_iter=100, max_depth=8` | 0.8557 | 80.35% | 89.79% | 0.8541 | 0.1739 | 0.786s |
| **Multi-Layer Perceptron** | `layers=(64, 32), relu, adam, early_stop` | **0.8695** | 80.04% | 89.78% | **0.8813** | **0.1292** | **0.106s** |
| **K-Nearest Neighbors** | `k=9, distance-weighted, subsampled` | 0.8559 | 78.42% | 88.00% | 0.8696 | 0.2556 | 23.935s |
| **Logistic Regression** | `L2, C=1.0, balanced, scaled` | 0.7994 | 78.30% | 88.18% | 0.7965 | 0.2484 | 0.032s |

### 2. Route Reconstruction Benchmark (10 Student-Testing Trips)

Impact of coupling the ML classifier with the **RoadMatchHMM Viterbi Decoder & Continuous Route Assembly**:

- **Raw Classifier-Only Route Continuity**: 89.91%
- **PathSnap HMM Reconstructed Route Continuity**: **100.00%**
- **Observation Top-1 Accuracy**: **80.76%**
- **Ground-Truth Arc Jaccard Overlap**: **82.4%**
- **Sequence Alignment Ratio**: **88.6%**
- **Average Inference Latency**: **48.2 ms per trip**

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 (Anaconda or venv recommended)
- Git

### Installation Commands

```bash
# Clone the repository
git clone https://github.com/<your-username>/PathSnap.git
cd PathSnap

# Create and activate virtual environment
conda create -n pathsnap python=3.11 -y
conda activate pathsnap

# Install dependencies
pip install -r requirements.txt
```

---

## Step-by-Step Execution Runbook

### Step 1: Run Exploratory Data Analysis & Audit
Inspects all 80 training trips and 10 testing trips, computes sampling intervals, coordinate ranges, road network topology, and generates figures and Folium maps:
```powershell
$env:PYTHONPATH = "."
python scripts/run_eda.py
```

### Step 2: Build Feature Datasets
Executes vectorized STRtree spatial candidate generation and extracts all 11 directional/kinematic features:
```powershell
python scripts/build_dataset.py
```

### Step 3: Run Multi-Model Training & 5-Fold GroupKFold CV
Trains all 5 model families, runs GroupKFold CV, evaluates student-testing holdout, and generates confusion matrices and comparison charts:
```powershell
python src/train_and_evaluate.py
```

### Step 4: Run Bayesian Hyperparameter Optimization
Optimizes hyperparameters using Optuna TPE, generates optimization history and parameter importance plots:
```powershell
python src/bayesian_optimization.py
```

### Step 5: Evaluate 10-Trip Route Reconstruction Benchmark
Evaluates Classifier-Only vs Classifier + HMM Viterbi across all 10 student trips and produces comparison charts:
```powershell
python scripts/evaluate_route_reconstruction.py
```

### Step 6: Launch Laptop 1 — ML Backend Service
Starts the FastAPI backend listening on all LAN network interfaces on port `8000`:
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Step 7: Launch Laptop 2 — Evaluator Dashboard UI
On Laptop 2 (or in a second terminal locally), configure the backend LAN IP and launch Streamlit:
```powershell
$env:PATHSNAP_BACKEND_URL = "http://<LAPTOP_1_IP>:8000"
streamlit run ui/app.py --server.port 8501
```

### Step 8: Generate Consolidated Technical Report (PDF)
Compiles the authoritative, publication-ready PDF report covering Sections 2.2 through 7:
```powershell
python report/generate_report.py
```
Output: `artifacts/reports/PathSnap_Technical_Report.pdf`

### Step 9: Run Test Suite
Executes the full automated unit and integration test suite:
```powershell
pytest -v
```

---

## API Documentation

When Laptop 1 is running, interactive Swagger documentation is available at `http://localhost:8000/docs`:

- `GET /health`: Health status, model registry, device metadata.
- `GET /api/trips`: Lists all available evaluation trips with observation counts, bounding boxes, and duration.
- `POST /api/predict/{trip_id}`: Reconstructs route from noisy GPS observations. Returns arc sequence, polylines, and latency without leaking ground truth.
- `GET /api/evaluate/{trip_id}`: Performs secure server-side evaluation against ground truth.

---

## Requirement Traceability Matrix

| Requirement | Source Document | Implementation Module | Test / Verification Evidence | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Distributed 2-Laptop Setup** | Main Guidelines §2 | `api/main.py` + `ui/app.py` | Verified REST communication over LAN port 8000 & 8501 | **COMPLETED** |
| **Dataset Audit & EDA** | Dataset Exploration | `scripts/run_eda.py` | `artifacts/results/eda_metrics.json`, plots & summary CSV | **COMPLETED** |
| **Vectorized STRtree Candidate Pruning** | Main Guidelines §4 | `src/candidate_generator.py` | 50-150m adaptive search, 1.45% miss rate | **COMPLETED** |
| **11-Feature Extraction (Zero Leakage)** | Models Guidelines §2 | `src/feature_engineering.py` | Backward differences only, causality verified in unit tests | **COMPLETED** |
| **5 Classification Model Families** | Models Guidelines §3 | `src/models/*.py` | LogReg, RF, HistGB, KNN, MLP all trained and benchmarked | **COMPLETED** |
| **GroupKFold CV Grouped by trip_id** | Models Guidelines §4 | `src/train_and_evaluate.py` | 5-Fold GroupKFold preventing spatial cross-trip leakage | **COMPLETED** |
| **Bayesian Hyperparameter Tuning** | Models Guidelines §5 | `src/bayesian_optimization.py` | Optuna TPE tuning, history & importance plots generated | **COMPLETED** |
| **HMM Viterbi Route Reconstruction** | Models Guidelines §6 | `src/hmm_decoder.py` | Emission/transition modeling + Dijkstra gap splicing | **COMPLETED** |
| **Ground-Truth Reference Isolation** | Assessment Policy §2 | `api/inference.py` | Server-side ground truth isolation verified in `tests/test_api.py` | **COMPLETED** |
| **Consolidated Technical Report** | Deliverables §4 | `report/generate_report.py` | Publication-ready PDF covering Sections 2.2 - 7 | **COMPLETED** |

---

## License & Attribution

Developed for **UE24CS352A Machine Learning**, Department of Computer Science and Engineering, **PES University**.
Course assignment submission. All rights reserved.
