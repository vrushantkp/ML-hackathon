"""
PathSnap — Evaluator UI (Streamlit Application)
Front-end interface running on Laptop 2 (Evaluator Laptop).
Connects to Laptop 1 (ML Backend) over LAN to trigger inference, display interactive maps,
and present evaluation metrics without duplicating the ML pipeline.
"""

import os
import requests
import pandas as pd
import folium
import streamlit as st
try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False
import streamlit.components.v1 as components

# Page Configuration
st.set_page_config(
    page_title="PathSnap — Route Reconstruction System",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling (Modern Deep Charcoal / Navy Geospatial Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stApp {
        background-color: #0B0F19;
        color: #F1F5F9;
    }
    [data-testid="stSidebar"] {
        background-color: #070A12;
        border-right: 1px solid #1E293B;
    }
    .main-header {
        font-size: 24px;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.5px;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 13px;
        color: #94A3B8;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1F293D;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .status-badge-ok {
        background-color: #064E3B;
        color: #34D399;
        padding: 5px 10px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid #059669;
    }
    .status-badge-err {
        background-color: #450A0A;
        color: #F87171;
        padding: 5px 10px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid #DC2626;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "backend_url" not in st.session_state:
    st.session_state["backend_url"] = os.getenv("PATHSNAP_BACKEND_URL", "http://localhost:8000")
if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None
if "evaluation_result" not in st.session_state:
    st.session_state["evaluation_result"] = None

# Sidebar — LAN Connection & Configuration
st.sidebar.markdown("### 📡 Laptop 1 Backend Connection")
backend_input = st.sidebar.text_input("Backend LAN URL:", value=st.session_state["backend_url"])
st.session_state["backend_url"] = backend_input.rstrip("/")

# Check Health
backend_status = "offline"
backend_meta = {}
try:
    health_resp = requests.get(f"{st.session_state['backend_url']}/health", timeout=2.5)
    if health_resp.status_code == 200:
        backend_status = "online"
        backend_meta = health_resp.json()
except Exception:
    backend_status = "offline"

if backend_status == "online":
    st.sidebar.markdown('<span class="status-badge-ok">● Connected to Laptop 1</span>', unsafe_allow_html=True)
    st.sidebar.caption(f"Device: {backend_meta.get('device', 'Laptop 1')} | Default Model: `{backend_meta.get('default_model', 'N/A')}`")
else:
    st.sidebar.markdown('<span class="status-badge-err">● Offline / Unreachable</span>', unsafe_allow_html=True)
    st.sidebar.warning(f"Could not reach {st.session_state['backend_url']}. Please ensure Laptop 1 is running `uvicorn api.main:app`.")

st.sidebar.divider()

# Trip Selection
st.sidebar.markdown("### 🚘 Evaluation Trip Selection")
available_trips = []
trip_meta_dict = {}

if backend_status == "online":
    try:
        trips_resp = requests.get(f"{st.session_state['backend_url']}/api/trips", timeout=3.0)
        if trips_resp.status_code == 200:
            trips_data = trips_resp.json()
            for t in trips_data.get("trips", []):
                trip_id = t["trip_id"]
                available_trips.append(trip_id)
                trip_meta_dict[trip_id] = t
    except Exception as e:
        st.sidebar.error(f"Error fetching trips: {e}")

if not available_trips:
    # Default fallback trip IDs
    available_trips = [
        "00000000", "00000007", "00000009", "00000040", "00000052",
        "00000056", "00000065", "00000076", "00000077", "00000091"
    ]

selected_trip = st.sidebar.selectbox("Choose Trip:", options=available_trips)

if selected_trip in trip_meta_dict:
    t_info = trip_meta_dict[selected_trip]
    st.sidebar.info(
        f"Partition: `{t_info.get('partition')}`\n\n"
        f"Observations: **{t_info.get('observation_count')}**\n\n"
        f"Duration: **{t_info.get('duration_seconds')}s**\n\n"
        f"Sampling: **{t_info.get('mean_sampling_interval_s')}s**"
    )

st.sidebar.divider()

# Model and Decoding Parameters
st.sidebar.markdown("### ⚙️ Decoding Configuration")

# Clean trained models dictionary with Causal Advanced HistGB as Recommended Default
TRAINED_MODELS = {
    "histgradientboosting": "HistGradientBoosting (Causal Advanced • 91.11% Top-1 • Recommended)",
    "random_forest": "Random Forest (Tuned Ensemble • 99.98% Continuity)",
    "multi_layer_perceptron": "Multi-Layer Perceptron (Neural Network)",
    "k_nearest_neighbors": "K-Nearest Neighbors (Instance-Based)",
    "logistic_regression": "Logistic Regression (Linear)"
}

selected_model_key = st.sidebar.selectbox(
    "Classifier Model:",
    options=list(TRAINED_MODELS.keys()),
    format_func=lambda k: TRAINED_MODELS[k],
    index=0
)

use_hmm = st.sidebar.checkbox("Enable HMM Viterbi Decoding", value=True)

if use_hmm:
    st.sidebar.markdown("""
    <div style="background-color: #0F291E; border: 1px solid #059669; border-radius: 8px; padding: 10px; margin-top: 6px; font-size: 11px; color: #34D399;">
        <b>● HMM Decoder Active</b><br>
        <span style="color:#A7F3D0;">Tuned Server Config: σ<sub>z</sub> = 10.0m, β = 5.0m<br>Candidate Search: 50–150m (Spatial STRtree)</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="background-color: #2D1A1A; border: 1px solid #DC2626; border-radius: 8px; padding: 10px; margin-top: 6px; font-size: 11px; color: #FCA5A5;">
        <b>○ HMM Decoder Bypassed</b><br>
        <span style="color:#FEE2E2;">Executing raw Classifier Top-1 predictions.</span>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("### 🗺️ Map Layer Toggles")
show_gt = st.sidebar.checkbox("Actual Path (Ground Truth)", value=True, help="Emerald Green (#10B981) reference trajectory")
show_misaligned = st.sidebar.checkbox("Highlight Route Misalignment", value=True, help="Fluorescent Red (#EF4444) error segments where model deviated from actual route")
show_clf = st.sidebar.checkbox("Classifier Top-1 Path", value=False, help="Amber (#F59E0B) raw classifier predictions")

# Trigger Button
run_reconstruction = st.sidebar.button("🚀 Reconstruct Route", type="primary", use_container_width=True)

# Main Application Header
st.markdown('<div class="main-header">PathSnap — Distributed Map-Matching & Route Reconstruction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">UE24CS352A Machine Learning | Department of Computer Science & Engineering, PES University</div>', unsafe_allow_html=True)

# Execute Route Reconstruction on Button Click
if run_reconstruction:
    if backend_status != "online":
        st.error(f"Cannot execute reconstruction: Laptop 1 backend at `{st.session_state['backend_url']}` is offline.")
    else:
        with st.spinner(f"Reconstructing route for Trip {selected_trip} via Laptop 1 ML engine..."):
            req_payload = {
                "model_name": selected_model_key,
                "use_hmm": use_hmm
            }
            try:
                resp = requests.post(
                    f"{st.session_state['backend_url']}/api/predict/{selected_trip}",
                    json=req_payload,
                    timeout=30.0
                )
                if resp.status_code == 200:
                    st.session_state["prediction_result"] = resp.json()
                    st.success(f"Trip {selected_trip} reconstructed successfully in {st.session_state['prediction_result']['inference_time_ms']} ms!")
                else:
                    st.error(f"Inference error ({resp.status_code}): {resp.text}")
            except Exception as e:
                st.error(f"Failed to communicate with backend: {e}")

# Display Results
pred_data = st.session_state.get("prediction_result")

if pred_data and pred_data.get("trip_id") == selected_trip:
    # Top Metrics Row with Individual Top-1 Accuracy
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    
    with m_col1:
        st.metric("Observations", f"{pred_data['observation_count']}")
    with m_col2:
        st.metric("Candidates", f"{pred_data['candidate_count']:,}")
    with m_col3:
        top1_disp = pred_data.get('top1_accuracy_pct')
        top1_text = f"{top1_disp}%" if top1_disp is not None else "N/A"
        st.metric("Top-1 Accuracy", top1_text)
    with m_col4:
        st.metric("Route Continuity", f"{pred_data['reconstruction_continuity_pct']}%")
    with m_col5:
        st.metric("Reconstructed Arcs", f"{pred_data['reconstructed_route_length']}")
    with m_col6:
        st.metric("Inference Latency", f"{pred_data['inference_time_ms']} ms")
        
    # Main Tabs
    tab_map, tab_seq, tab_eval, tab_arch = st.tabs([
        "🗺️ Geographic Trajectory Map",
        "📋 Reconstructed Sequence & Audit",
        "🛡️ Server-Side Ground Truth Evaluation",
        "🏗️ System Architecture & Distributed State"
    ])
    
    with tab_map:
        st.markdown("#### Interactive Trajectory & Route Map")
        st.caption("Layer Palette: 🟢 **Green** = Actual Path (Ground Truth) | 🔵 **Blue** = Model Path (HMM Reconstructed) | 🔴 **Red** = Misaligned Error Segments | 🟡 **Amber** = Classifier Top-1 | ⚫ **Slate** = Raw GPS Track")
        
        # Build Folium Map
        gps_poly = pred_data.get("raw_gps_polyline", [])
        route_poly = pred_data.get("route_polyline", [])
        top1_poly = pred_data.get("classifier_top1_polyline", [])
        gt_poly = pred_data.get("ground_truth_polyline", [])
        misaligned_segs = pred_data.get("misaligned_segments", [])
        top1_arcs = pred_data.get("classifier_top1_route", [])
        
        if gps_poly:
            mean_lat = sum(p[0] for p in gps_poly) / len(gps_poly)
            mean_lon = sum(p[1] for p in gps_poly) / len(gps_poly)
        else:
            mean_lat, mean_lon = 49.85, 8.58
            
        # Sleek dark mode basemap (free, zero API key)
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=15, tiles="CartoDB dark_matter")
        
        # 1. Raw GPS Observations (Muted Slate)
        gps_layer = folium.FeatureGroup(name="Raw GPS Track", show=True)
        folium.PolyLine(gps_poly, color="#64748B", weight=2, opacity=0.45, dash_array="3, 5").add_to(gps_layer)
        if gps_poly:
            folium.Marker(gps_poly[0], popup="GPS Start", icon=folium.Icon(color="blue", icon="play")).add_to(gps_layer)
            folium.Marker(gps_poly[-1], popup="GPS End", icon=folium.Icon(color="darkblue", icon="stop")).add_to(gps_layer)
            
        step = max(1, len(gps_poly) // 120)
        for idx in range(0, len(gps_poly), step):
            folium.CircleMarker(
                location=gps_poly[idx],
                radius=3,
                color="#64748B",
                fill=True,
                fill_color="#38BDF8",
                fill_opacity=0.6,
                popup=f"Obs #{idx}<br>Lat: {gps_poly[idx][0]:.6f}<br>Lon: {gps_poly[idx][1]:.6f}"
            ).add_to(gps_layer)
        gps_layer.add_to(m)
        
        # 2. Actual Path / Ground Truth (Emerald Green - Bold & Clearly Visible)
        if gt_poly and show_gt:
            gt_layer = folium.FeatureGroup(name="Actual Path (Ground Truth)", show=True)
            folium.PolyLine(
                gt_poly,
                color="#10B981",
                weight=6,
                opacity=0.85,
                tooltip="Actual Path (Ground Truth Route)"
            ).add_to(gt_layer)
            gt_layer.add_to(m)
            
        # 3. Model Path / Reconstructed Route (Sky Blue)
        if route_poly:
            rec_layer = folium.FeatureGroup(name="Model Path (Reconstructed Route)", show=True)
            folium.PolyLine(
                route_poly,
                color="#38BDF8",
                weight=4,
                opacity=0.95,
                tooltip=f"Model Reconstructed Route ({pred_data['reconstructed_route_length']} arcs)"
            ).add_to(rec_layer)
            rec_layer.add_to(m)
            
        # 4. Misaligned Error Segments (Fluorescent Red - Model Deviations)
        if misaligned_segs and show_misaligned:
            err_layer = folium.FeatureGroup(name="Misaligned Route Segments", show=True)
            for seg in misaligned_segs:
                folium.PolyLine(
                    seg,
                    color="#EF4444",
                    weight=7,
                    opacity=0.95,
                    tooltip="Misaligned Segment: In Model Route but NOT in Actual Ground Truth!"
                ).add_to(err_layer)
            err_layer.add_to(m)
            
        # 5. Classifier Top-1 Predictions (Amber Dashed)
        if top1_poly and show_clf:
            clf_layer = folium.FeatureGroup(name="Classifier Top-1 Path", show=True)
            folium.PolyLine(
                top1_poly,
                color="#F59E0B",
                weight=3,
                opacity=0.85,
                dash_array="5, 6",
                tooltip="Raw Classifier Top-1 Route (pre-HMM)"
            ).add_to(clf_layer)
            clf_layer.add_to(m)
            
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Render map in Streamlit
        if HAS_ST_FOLIUM:
            st_folium(m, width=None, height=540)
        else:
            components.html(m._repr_html_(), height=550)
            
    with tab_seq:
        st.markdown("#### Observation-to-Arc Mapping & Candidate Audit")
        col_s1, col_s2 = st.columns([1, 1])
        with col_s1:
            st.markdown(f"**Reconstructed Arc Sequence ({pred_data['reconstructed_route_length']} arcs):**")
            rec_arcs = pred_data.get("reconstructed_route", [])
            st.text_area("Arc ID Sequence", value=str(rec_arcs), height=150)
            
        with col_s2:
            st.markdown(f"**Classifier Top-1 Arc Sequence ({len(top1_arcs)} observations):**")
            st.text_area("Top-1 Arc Sequence", value=str(top1_arcs), height=150)
            
        st.markdown("##### Trajectory Observations Sample (First 50 Points)")
        pts = pred_data.get("trajectory_points", [])
        if pts:
            pts_df = pd.DataFrame(pts[:50])
            st.dataframe(pts_df, use_container_width=True)
            
    with tab_eval:
        st.markdown("#### Server-Side Ground Truth Evaluation")
        st.caption("Secure server-side evaluation executed on Laptop 1. Ground truth reference routes remain protected on the server.")
        
        if st.button("🔍 Run Server-Side Ground Truth Audit", type="secondary"):
            with st.spinner("Fetching server-side evaluation from Laptop 1..."):
                try:
                    eval_resp = requests.get(
                        f"{st.session_state['backend_url']}/api/evaluate/{selected_trip}",
                        params={"model_name": selected_model, "use_hmm": use_hmm},
                        timeout=20.0
                    )
                    if eval_resp.status_code == 200:
                        st.session_state["evaluation_result"] = eval_resp.json()
                    else:
                        st.error(f"Evaluation request failed: {eval_resp.text}")
                except Exception as e:
                    st.error(f"Failed to fetch evaluation: {e}")
                    
        eval_data = st.session_state.get("evaluation_result")
        if eval_data and eval_data.get("trip_id") == selected_trip:
            if not eval_data.get("has_ground_truth"):
                st.info(eval_data.get("message", "No ground truth available for this trip partition."))
            else:
                em = eval_data.get("evaluation_metrics", {})
                ec1, ec2, ec3, ec4 = st.columns(4)
                with ec1:
                    st.metric("Sequence Alignment", f"{em.get('sequence_alignment_ratio_pct')}%")
                with ec2:
                    st.metric("Arc Jaccard Overlap", f"{em.get('arc_jaccard_overlap_pct')}%")
                with ec3:
                    st.metric("Arc Precision", f"{em.get('arc_precision_pct')}%")
                with ec4:
                    st.metric("Arc Recall", f"{em.get('arc_recall_pct')}%")
                    
                st.markdown("##### Detailed Metric Breakdown")
                st.json(em)
                
    with tab_arch:
        st.markdown("#### Distributed System & Network Architecture")
        st.markdown("""
        **PathSnap Architecture Overview:**
        - **Laptop 1 (ML Backend Engine)**: Hosts raw trajectory data, directed road-network STRtree spatial indices, trained binary classifiers, and HMM Viterbi decoder. Exposes FastAPI REST endpoints over LAN.
        - **Laptop 2 (Evaluator Dashboard)**: Lightweight Streamlit client communicating exclusively via JSON REST payloads over LAN. Contains zero ML prediction code or ground truth exposure.
        """)
        
        st.code("""
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
                                           | LAN HTTP (JSON)
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
        """, language="text")
else:
    st.info("👈 Select an evaluation trip from the sidebar and click **'🚀 Reconstruct Route'** to run inference.")
