"""
PathSnap — Causal Advanced HistGradientBoosting Trainer & Evaluator
Implements multi-scale backward heading smoothing, backward motion straightness,
candidate delta proximity, and backward trajectory smoothing.
STRICTLY CAUSAL: ZERO FUTURE OBSERVATIONS USED.
"""

import os
import sys
import math
import time
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, log_loss

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_trip, project_coordinates, TripData
from src.candidate_generator import generate_candidates_for_trip
from src.metrics import compute_trip_route_continuity
from src.hmm_decoder import build_road_graph, RoadMatchHMM

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "artifacts", "models")
os.makedirs(OUT_DIR, exist_ok=True)

CAUSAL_FEATURE_COLUMNS = [
    'distance_m',
    'closest_distance_delta_m',
    'candidate_rank',
    'speed_mps',
    'speed_available',
    'heading_cosine_1',
    'heading_cosine_2',
    'heading_cosine_4',
    'heading_cosine_8',
    'heading_diff_1',
    'motion_straightness_backward_3',
    'motion_straightness_backward_5',
    'arc_length_m',
    'arc_offset_fraction',
    'endpoint_proximity',
    'distance_to_prev_gps_m',
    'distance_to_backward_smooth_3_m',
    'distance_to_backward_smooth_5_m'
]

def extract_causal_features_for_trip(trip: TripData, cands: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts strictly causal, multi-scale kinematic & directional features.
    Zero future observation lookahead!
    """
    track_df = trip.track_df
    n = len(track_df)
    x = trip.track_xy[:, 0]
    y = trip.track_xy[:, 1]
    t = track_df['timestamp'].values

    dt = np.maximum(np.diff(t), 0.01)
    dx = np.diff(x)
    dy = np.diff(y)
    step_disp = np.hypot(dx, dy)

    speeds = np.zeros(n, dtype=np.float64)
    speeds[1:] = step_disp / dt
    speeds[0] = speeds[1] if n > 1 else 0.0

    def get_h(dx_, dy_):
        return (np.degrees(np.arctan2(dx_, dy_)) + 360.0) % 360.0

    h1 = np.zeros(n, dtype=np.float64)
    h1[1:] = get_h(dx, dy)
    h1[0] = h1[1] if n > 1 else 0.0

    # Multi-scale backward headings (2, 4, 8 steps back)
    def get_backward_h(k):
        h = np.zeros(n, dtype=np.float64)
        for i in range(n):
            i0 = max(0, i - k)
            h[i] = get_h(x[i] - x[i0], y[i] - y[i0]) if i != i0 else h1[i]
        return h

    h2 = get_backward_h(2)
    h4 = get_backward_h(4)
    h8 = get_backward_h(8)

    # Backward motion straightness (displacement / cumulative path distance)
    def get_backward_straightness(k):
        s = np.ones(n, dtype=np.float64)
        for i in range(1, n):
            i0 = max(0, i - k)
            net_d = np.hypot(x[i] - x[i0], y[i] - y[i0])
            gross_d = np.sum(step_disp[i0:i])
            s[i] = net_d / max(1.0, gross_d)
        return s

    ms3 = get_backward_straightness(3)
    ms5 = get_backward_straightness(5)

    # Causal backward rolling averages (historical only, center=False)
    smooth3_x = pd.Series(x).rolling(window=3, min_periods=1).mean().values
    smooth3_y = pd.Series(y).rolling(window=3, min_periods=1).mean().values
    smooth5_x = pd.Series(x).rolling(window=5, min_periods=1).mean().values
    smooth5_y = pd.Series(y).rolling(window=5, min_periods=1).mean().values

    # Project candidates onto metric XY plane
    nodes_df = trip.nodes_df
    arcs_df = trip.arcs_df
    res_coords = project_coordinates(nodes_df['lat'].values, nodes_df['lon'].values)
    node_x, node_y = res_coords[0], res_coords[1]

    u_idx = arcs_df['from_node_id'].values[cands['arc_id'].values]
    v_idx = arcs_df['to_node_id'].values[cands['arc_id'].values]
    u_x, u_y = node_x[u_idx], node_y[u_idx]
    v_x, v_y = node_x[v_idx], node_y[v_idx]
    proj_ratio = cands['projection_ratio'].values
    proj_x = u_x + proj_ratio * (v_x - u_x)
    proj_y = u_y + proj_ratio * (v_y - u_y)

    cands_copy = cands.copy()
    cands_copy['proj_x'] = proj_x
    cands_copy['proj_y'] = proj_y

    feats = []
    for obs_i, group in cands_copy.groupby('obs_idx'):
        min_d = group['perp_distance'].min()
        for _, row in group.iterrows():
            b = row['arc_bearing']
            
            def circ_cos(h):
                diff = abs(h - b) % 360.0
                d_ = 360.0 - diff if diff > 180.0 else diff
                return math.cos(math.radians(d_))
                
            def circ_deg(h):
                diff = abs(h - b) % 360.0
                return 360.0 - diff if diff > 180.0 else diff

            h_cos1 = circ_cos(h1[obs_i])
            h_cos2 = circ_cos(h2[obs_i])
            h_cos4 = circ_cos(h4[obs_i])
            h_cos8 = circ_cos(h8[obs_i])
            h_diff1 = circ_deg(h1[obs_i])

            i_prev = max(0, obs_i - 1)
            d_prev = np.hypot(row['proj_x'] - x[i_prev], row['proj_y'] - y[i_prev])
            d_sm3 = np.hypot(row['proj_x'] - smooth3_x[obs_i], row['proj_y'] - smooth3_y[obs_i])
            d_sm5 = np.hypot(row['proj_x'] - smooth5_x[obs_i], row['proj_y'] - smooth5_y[obs_i])
            
            pr = row['projection_ratio']
            end_prox = min(pr, 1.0 - pr)

            feats.append({
                'obs_idx': obs_i,
                'arc_id': int(row['arc_id']),
                'distance_m': row['perp_distance'],
                'closest_distance_delta_m': row['perp_distance'] - min_d,
                'candidate_rank': row['candidate_rank'],
                'speed_mps': speeds[obs_i],
                'speed_available': 1.0 if speeds[obs_i] > 0.5 else 0.0,
                'heading_cosine_1': h_cos1,
                'heading_cosine_2': h_cos2,
                'heading_cosine_4': h_cos4,
                'heading_cosine_8': h_cos8,
                'heading_diff_1': h_diff1,
                'motion_straightness_backward_3': ms3[obs_i],
                'motion_straightness_backward_5': ms5[obs_i],
                'arc_length_m': row['arc_length'],
                'arc_offset_fraction': pr,
                'endpoint_proximity': end_prox,
                'distance_to_prev_gps_m': d_prev,
                'distance_to_backward_smooth_3_m': d_sm3,
                'distance_to_backward_smooth_5_m': d_sm5
            })

    return pd.DataFrame(feats)

def main():
    print("=================================================================")
    print(" PathSnap — Training Causal Advanced HistGradientBoosting Model")
    print("=================================================================")
    
    # 1. Load Training Trips and Extract Causal Features
    train_dir = os.path.join(RAW_DIR, "train")
    all_train_trips = sorted(os.listdir(train_dir))[:30] # 30 training trips
    
    print(f"Extracting causal features from {len(all_train_trips)} training trips...")
    train_dfs = []
    
    t0 = time.time()
    for idx, tid in enumerate(all_train_trips):
        trip = load_trip(tid, RAW_DIR, partition="train")
        cands = generate_candidates_for_trip(trip, initial_radius=50.0, max_radius=150.0)
        feats_df = extract_causal_features_for_trip(trip, cands)
        
        # Ground truth route mapping
        gt_arcs = set(trip.route_df['arc_id'])
        feats_df['label'] = feats_df['arc_id'].apply(lambda a: 1 if a in gt_arcs else 0)
        feats_df['trip_id'] = tid
        train_dfs.append(feats_df)
        if (idx + 1) % 5 == 0 or idx == len(all_train_trips) - 1:
            print(f"[{idx+1}/{len(all_train_trips)}] Processed {tid} ({len(feats_df)} candidate pairs)")
            
    train_data = pd.concat(train_dfs, ignore_index=True)
    print(f"Total training samples: {len(train_data):,} in {time.time()-t0:.1f}s")
    print(f"Positive samples: {train_data['label'].sum():,} ({train_data['label'].mean()*100:.2f}%)")
    
    # 2. Train HistGradientBoosting with Friend's Tuned Regularization (min_samples_leaf=100)
    print("\nTraining HistGradientBoostingClassifier (min_samples_leaf=100, learning_rate=0.1)...")
    X_train = train_data[CAUSAL_FEATURE_COLUMNS].values
    y_train = train_data['label'].values
    
    clf = HistGradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.1,
        max_iter=100,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        random_state=42
    )
    
    t_train = time.time()
    clf.fit(X_train, y_train)
    print(f"Model trained in {time.time()-t_train:.2f} seconds!")
    
    # Save the trained model package
    model_package = {
        'estimator': clf,
        'feature_columns': CAUSAL_FEATURE_COLUMNS,
        'model_name': 'Causal Advanced HistGradientBoosting',
        'causality': 'Strictly Causal (Backward-only, zero future observations)'
    }
    save_path = os.path.join(OUT_DIR, "causal_histgradientboosting.joblib")
    joblib.dump(model_package, save_path)
    print(f"Saved model package to: {save_path}")
    
    # 3. Evaluate on 10 Student-Testing Trips
    test_trips = ['00000000', '00000007', '00000009', '00000040', '00000052', '00000056', '00000065', '00000076', '00000077', '00000091']
    print("\n--- Evaluating Causal Advanced HGBoost on 10 Student-Testing Trips ---")
    
    eval_results = []
    
    for tid in test_trips:
        trip = load_trip(tid, RAW_DIR)
        cands = generate_candidates_for_trip(trip, initial_radius=50.0, max_radius=150.0)
        feats_df = extract_causal_features_for_trip(trip, cands)
        
        X_test = feats_df[CAUSAL_FEATURE_COLUMNS].values
        probs = clf.predict_proba(X_test)[:, 1]
        feats_df['prob_pos'] = probs
        
        idx_top1 = feats_df.groupby('obs_idx')['prob_pos'].idxmax()
        top1_df = feats_df.loc[idx_top1].sort_values('obs_idx')
        top1_arcs = top1_df['arc_id'].tolist()
        
        gt_arcs = set(trip.route_df['arc_id'])
        top1_in_gt = sum(1 for a in top1_arcs if a in gt_arcs) / len(top1_arcs) * 100.0
        cont_clf = compute_trip_route_continuity(top1_arcs, trip.arcs_df)
        
        # HMM Viterbi decoding
        G, edge_to_arc = build_road_graph(trip.arcs_df)
        hmm = RoadMatchHMM(G=G, arcs_df=trip.arcs_df, sigma_z=10.0, beta=5.0)
        cands_by_obs = [g.to_dict('records') for _, g in feats_df.groupby('obs_idx')]
        chosen_obs = hmm.decode(cands_by_obs, trip.track_xy, trip.track_df['timestamp'].values)
        final_route = hmm.reconstruct_continuous_route(chosen_obs)
        
        hmm_obs_acc = sum(1 for a in chosen_obs if a in gt_arcs) / len(chosen_obs) * 100.0
        hmm_cont = compute_trip_route_continuity(final_route, trip.arcs_df)
        jaccard = len(gt_arcs & set(final_route)) / max(1, len(gt_arcs | set(final_route))) * 100.0
        recall = len(gt_arcs & set(final_route)) / max(1, len(gt_arcs)) * 100.0
        
        print(f"Trip {tid}: Top-1 Clf={top1_in_gt:.2f}%, HMM Obs={hmm_obs_acc:.2f}%, Continuity={hmm_cont:.2f}%, Jaccard={jaccard:.2f}%, Recall={recall:.2f}%")
        
        eval_results.append({
            'trip_id': tid,
            'top1_clf': top1_in_gt,
            'hmm_obs': hmm_obs_acc,
            'continuity': hmm_cont,
            'jaccard': jaccard,
            'recall': recall
        })
        
    res_df = pd.DataFrame(eval_results)
    print("\n=================================================================")
    print(" 10-TRIP AVERAGE BENCHMARK RESULTS (CAUSAL ADVANCED HGBOOST)")
    print("=================================================================")
    mean_stats = res_df.mean(numeric_only=True)
    for col, val in mean_stats.items():
        print(f"{col:15s}: {val:.2f}%")
        
if __name__ == "__main__":
    main()
