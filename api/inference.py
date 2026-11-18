"""
PathSnap — Inference Pipeline Service
Orchestrates candidate generation, feature extraction, ML classification, and HMM Viterbi decoding.
"""

import os
import json
import time
import datetime
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from src.data_loader import load_trip, TripData
from src.candidate_generator import generate_candidates_for_trip
from src.feature_engineering import (
    extract_features_for_trip,
    FEATURE_COLUMNS,
    extract_causal_features_for_trip,
    CAUSAL_FEATURE_COLUMNS
)
from src.hmm_decoder import build_road_graph, RoadMatchHMM
from src.metrics import compute_trip_route_continuity
from src.map_visualization import extract_arc_coordinates

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(BASE_DIR, "artifacts", "configurations")
HMM_CONFIG_PATH = os.path.join(CONFIGS_DIR, "hmm_config.json")

def load_hmm_config() -> Dict[str, Any]:
    """
    Loads saved HMM configuration from artifacts/configurations/hmm_config.json.
    Ensures parameters are selected through development experimentation, not arbitrary runtime overrides.
    """
    default_config = {
        "configuration_name": "PathSnap Tuned Production HMM Configuration",
        "parameters": {
            "sigma_z": 10.0,
            "beta": 5.0,
            "disconnect_penalty": -35.0,
            "initial_search_radius_m": 50.0,
            "max_search_radius_m": 150.0
        }
    }
    if os.path.exists(HMM_CONFIG_PATH):
        try:
            with open(HMM_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"[INFO] Successfully loaded HMM configuration from {HMM_CONFIG_PATH}", flush=True)
                return data
        except Exception as e:
            print(f"[WARNING] Failed to parse {HMM_CONFIG_PATH}: {e}. Using tuned defaults.", flush=True)
            return default_config
    else:
        print(f"[WARNING] {HMM_CONFIG_PATH} not found. Using tuned defaults.", flush=True)
        return default_config

ACTIVE_HMM_CONFIG = load_hmm_config()

def extract_arc_segments(arc_ids: List[int], nodes_df: pd.DataFrame, arcs_df: pd.DataFrame) -> List[List[List[float]]]:
    """
    Extracts individual 2-point line segments [[u_lat, u_lon], [v_lat, v_lon]] for each arc ID.
    Enables Leaflet to render disconnected error segments without artifact connecting lines.
    """
    segments = []
    node_lats = nodes_df['lat'].values
    node_lons = nodes_df['lon'].values
    arc_u = arcs_df['from_node_id'].values
    arc_v = arcs_df['to_node_id'].values
    n_arcs = len(arc_u)
    
    for a in arc_ids:
        if 0 <= a < n_arcs:
            u, v = int(arc_u[a]), int(arc_v[a])
            segments.append([
                [float(node_lats[u]), float(node_lons[u])],
                [float(node_lats[v]), float(node_lons[v])]
            ])
    return segments

MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

# Model Cache & Aliases
LOADED_MODELS: Dict[str, Any] = {}
GRAPH_CACHE: Dict[str, Tuple[Any, Dict[Tuple[int, int], int]]] = {}

MODEL_ALIASES = {
    "best_model": "random_forest",
    "tuned_best_model": "random_forest",
    "tuned_random_forest": "random_forest",
    "causal_histgradientboosting": "histgradientboosting"
}

def get_model(model_name: str):
    """Loads and caches trained ML model."""
    clean_name = model_name.lower().replace(" ", "_").replace("-", "_")
    if clean_name.endswith(".joblib"):
        clean_name = clean_name[:-7]
        
    if clean_name in MODEL_ALIASES:
        clean_name = MODEL_ALIASES[clean_name]
        
    if clean_name in LOADED_MODELS:
        return LOADED_MODELS[clean_name]
        
    target_path = os.path.join(MODELS_DIR, f"{clean_name}.joblib")
    if not os.path.exists(target_path):
        # Fallback search
        available = [f[:-7] for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
        if "random_forest" in available:
            target_path = os.path.join(MODELS_DIR, "random_forest.joblib")
            clean_name = "random_forest"
        elif available:
            target_path = os.path.join(MODELS_DIR, f"{available[0]}.joblib")
            clean_name = available[0]
        else:
            raise FileNotFoundError(f"No trained models found in {MODELS_DIR}")
            
    model = joblib.load(target_path)
    LOADED_MODELS[clean_name] = model
    return model

def run_route_reconstruction(
    trip_id: str,
    partition: Optional[str] = None,
    model_name: str = "histgradientboosting",
    use_hmm: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Executes the end-to-end map matching and route reconstruction pipeline for a given trip.
    Strictly loads and uses the tuned HMM configuration from artifacts/configurations/hmm_config.json.
    Client/evaluator requests cannot override sigma_z or beta.
    Does NOT leak ground truth reference routes to the output!
    """
    t0_start = time.time()
    
    # 1. Load parameters strictly from saved HMM configuration
    hmm_params = ACTIVE_HMM_CONFIG.get("parameters", {})
    cfg_sigma_z = float(hmm_params.get("sigma_z", 10.0))
    cfg_beta = float(hmm_params.get("beta", 5.0))
    cfg_disc_pen = float(hmm_params.get("disconnect_penalty", -35.0))
    
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    inf_log = f"[{timestamp_str}] [INFERENCE] Trip {trip_id} | Model: {model_name} | HMM: {use_hmm} (Server Config: σz={cfg_sigma_z}m, β={cfg_beta}m, penalty={cfg_disc_pen})"
    print(inf_log, flush=True)
    try:
        log_fpath = os.path.join(BASE_DIR, "logs", "api_access.log")
        with open(log_fpath, "a", encoding="utf-8") as lf:
            lf.write(inf_log + "\n")
    except Exception:
        pass

    # 1. Load Trip Data
    trip_data = load_trip(trip_id, RAW_DIR, partition=partition)
    
    # 2. Candidate Generation
    cands_df = generate_candidates_for_trip(trip_data, initial_radius=50.0, max_radius=150.0)
    
    # 3. Model Loading & Feature Extraction
    model = get_model(model_name)
    if isinstance(model, dict):
        estimator = model.get('estimator', model)
        feature_cols = model.get('feature_columns', FEATURE_COLUMNS)
        if feature_cols == CAUSAL_FEATURE_COLUMNS or 'heading_cosine_1' in feature_cols:
            features_df = extract_causal_features_for_trip(trip_data, cands_df)
        else:
            features_df = extract_features_for_trip(trip_data, cands_df)
        X = features_df[feature_cols].values
        probs = estimator.predict_proba(X)[:, 1]
    else:
        features_df = extract_features_for_trip(trip_data, cands_df)
        X = features_df[FEATURE_COLUMNS].values
        probs = model.predict_proba(X)[:, 1]
    
    features_df['prob_pos'] = probs
    
    # 5. Extract Classifier Top-1 per observation
    idx_top1 = features_df.groupby('obs_idx')['prob_pos'].idxmax()
    top1_df = features_df.loc[idx_top1].sort_values('obs_idx')
    top1_arcs = top1_df['arc_id'].tolist()
    
    # 6. Reconstruct Route (HMM Viterbi or Top-1)
    if use_hmm:
        # Build or retrieve cached graph
        cache_key = trip_data.trip_id
        if cache_key not in GRAPH_CACHE:
            G, edge_to_arc = build_road_graph(trip_data.arcs_df)
            GRAPH_CACHE[cache_key] = (G, edge_to_arc)
        else:
            G, edge_to_arc = GRAPH_CACHE[cache_key]
            
        hmm = RoadMatchHMM(
            G=G,
            arcs_df=trip_data.arcs_df,
            sigma_z=cfg_sigma_z,
            beta=cfg_beta,
            disconnect_penalty=cfg_disc_pen
        )
        
        # Group candidates by observation index
        cands_by_obs = []
        for obs_i, group in features_df.groupby('obs_idx'):
            cand_records = group.to_dict('records')
            cands_by_obs.append(cand_records)
            
        chosen_obs_arcs = hmm.decode(
            candidates_by_obs=cands_by_obs,
            track_xy=trip_data.track_xy,
            timestamps=trip_data.track_df['timestamp'].values
        )
        
        final_reconstructed_route = hmm.reconstruct_continuous_route(chosen_obs_arcs)
    else:
        chosen_obs_arcs = top1_arcs
        # Simple compression of consecutive duplicates
        final_reconstructed_route = []
        for a in top1_arcs:
            if not final_reconstructed_route or a != final_reconstructed_route[-1]:
                final_reconstructed_route.append(a)
                
    # Continuity computation on reconstructed route
    continuity_pct = compute_trip_route_continuity(final_reconstructed_route, trip_data.arcs_df)
    
    # 7. Coordinates & Polylines for UI
    route_polyline = extract_arc_coordinates(final_reconstructed_route, trip_data.nodes_df, trip_data.arcs_df)
    raw_gps_polyline = trip_data.track_df[['lat', 'lon']].values.tolist()
    classifier_top1_polyline = extract_arc_coordinates(top1_arcs, trip_data.nodes_df, trip_data.arcs_df)
    
    gt_polyline = []
    misaligned_segments = []
    missed_gt_segments = []
    misaligned_arc_count = 0
    misalignment_rate_pct = 0.0
    top1_accuracy_pct = None
    hmm_observation_accuracy_pct = None

    if trip_data.route_df is not None:
        gt_arcs = trip_data.route_df['arc_id'].tolist()
        gt_set = set(gt_arcs)
        gt_polyline = extract_arc_coordinates(gt_arcs, trip_data.nodes_df, trip_data.arcs_df)

        # Individual Top-1 Observation Accuracy
        top1_correct = sum(1 for a in top1_arcs if a in gt_set)
        top1_accuracy_pct = round((top1_correct / max(1, len(top1_arcs))) * 100.0, 2)
        hmm_obs_correct = sum(1 for a in chosen_obs_arcs if a in gt_set)
        hmm_observation_accuracy_pct = round((hmm_obs_correct / max(1, len(chosen_obs_arcs))) * 100.0, 2)

        # Misalignment: arcs in reconstructed route NOT present in ground truth
        dev_arcs = [a for a in final_reconstructed_route if a not in gt_set]
        misaligned_arc_count = len(dev_arcs)
        misalignment_rate_pct = round((misaligned_arc_count / max(1, len(final_reconstructed_route))) * 100.0, 2)
        misaligned_segments = extract_arc_segments(dev_arcs, trip_data.nodes_df, trip_data.arcs_df)

        # Arcs in ground truth that model failed to traverse
        rec_set = set(final_reconstructed_route)
        missed_arcs = [a for a in gt_arcs if a not in rec_set]
        missed_gt_segments = extract_arc_segments(missed_arcs, trip_data.nodes_df, trip_data.arcs_df)
    
    # Trajectory Points summary
    trajectory_points = []
    track_df = trip_data.track_df
    for i in range(len(track_df)):
        chosen_a = chosen_obs_arcs[i] if i < len(chosen_obs_arcs) else top1_arcs[i]
        top1_prob = float(top1_df.iloc[i]['prob_pos']) if i < len(top1_df) else 0.5
        trajectory_points.append({
            'obs_idx': i,
            'timestamp': float(track_df['timestamp'].iloc[i]),
            'lat': float(track_df['lat'].iloc[i]),
            'lon': float(track_df['lon'].iloc[i]),
            'chosen_arc_id': int(chosen_a),
            'confidence': round(top1_prob, 4)
        })
        
    duration_ms = round((time.time() - t0_start) * 1000.0, 2)
    
    return {
        'trip_id': trip_data.trip_id,
        'partition': trip_data.partition,
        'model_used': model_name,
        'use_hmm': use_hmm,
        'observation_count': len(trip_data.track_df),
        'candidate_count': len(features_df),
        'reconstructed_route': final_reconstructed_route,
        'reconstructed_route_length': len(final_reconstructed_route),
        'classifier_top1_route': top1_arcs,
        'trajectory_points': trajectory_points,
        'route_polyline': route_polyline,
        'raw_gps_polyline': raw_gps_polyline,
        'ground_truth_polyline': gt_polyline,
        'classifier_top1_polyline': classifier_top1_polyline,
        'misaligned_segments': misaligned_segments,
        'missed_gt_segments': missed_gt_segments,
        'misaligned_arc_count': misaligned_arc_count,
        'misalignment_rate_pct': misalignment_rate_pct,
        'top1_accuracy_pct': top1_accuracy_pct,
        'hmm_observation_accuracy_pct': hmm_observation_accuracy_pct,
        'hmm_config_used': {
            'configuration_name': ACTIVE_HMM_CONFIG.get('configuration_name', 'PathSnap Tuned Production HMM Configuration'),
            'sigma_z': cfg_sigma_z,
            'beta': cfg_beta,
            'disconnect_penalty': cfg_disc_pen,
            'status': 'ACTIVE' if use_hmm else 'BYPASSED'
        },
        'inference_time_ms': duration_ms,
        'reconstruction_continuity_pct': round(continuity_pct, 2)
    }
