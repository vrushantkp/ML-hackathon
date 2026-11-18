"""
PathSnap — Feature Engineering Module
Extracts spatial, kinematic, directional, and candidate-rank features for every (GPS observation, candidate road arc) pair.
Ensures zero future-observation lookahead and handles circular wraparound correctly.
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, List, Any, Optional
from src.data_loader import project_coordinates

FEATURE_COLUMNS = [
    'perp_distance',
    'heading_diff',
    'heading_cos',
    'projection_ratio',
    'endpoint_proximity',
    'arc_length',
    'vehicle_speed',
    'displacement_m',
    'dt_s',
    'candidate_rank',
    'distance_ratio'
]

def compute_circular_heading_difference(theta1: float, theta2: float) -> float:
    """
    Computes absolute angular difference between two bearings in degrees [0, 180]
    with proper circular wraparound across 0/360.
    """
    raw_diff = abs(theta1 - theta2) % 360.0
    return float(min(raw_diff, 360.0 - raw_diff))

def compute_kinematics(track_xy: np.ndarray, times: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes instantaneous heading, speed, step displacement, and dt for each GPS observation.
    Uses causal backward differences (current vs previous step) to avoid future-lookahead leakage.
    At t=0, forward difference is used as an initialization.
    """
    N = len(track_xy)
    headings = np.zeros(N, dtype=np.float64)
    speeds = np.zeros(N, dtype=np.float64)
    displacements = np.zeros(N, dtype=np.float64)
    dts = np.ones(N, dtype=np.float64)
    
    if N < 2:
        return headings, speeds, displacements, dts
        
    dx = np.diff(track_xy[:, 0])
    dy = np.diff(track_xy[:, 1])
    dt = np.diff(times)
    dt_safe = np.maximum(dt, 0.01)
    
    step_dists = np.sqrt(dx**2 + dy**2)
    step_speeds = step_dists / dt_safe
    step_headings = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    
    # Fill step 1 to N-1 using backward step
    displacements[1:] = step_dists
    dts[1:] = dt
    speeds[1:] = step_speeds
    headings[1:] = step_headings
    
    # Step 0 initialized with first forward step
    displacements[0] = step_dists[0]
    dts[0] = dt[0]
    speeds[0] = step_speeds[0]
    headings[0] = step_headings[0]
    
    return headings, speeds, displacements, dts

def extract_candidate_features(cand_df: pd.DataFrame, track_xy: np.ndarray, track_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches candidate DataFrame with all engineering features.
    cand_df columns expected: ['obs_idx', 'arc_id', 'perp_distance', 'projection_ratio', 'arc_length', 'arc_bearing', 'candidate_rank']
    """
    times = track_df['time'].values
    headings, speeds, displacements, dts = compute_kinematics(track_xy, times)
    
    obs_idx = cand_df['obs_idx'].values
    
    # Kinematics mapped to candidates
    cand_heading = headings[obs_idx]
    cand_speed = speeds[obs_idx]
    cand_disp = displacements[obs_idx]
    cand_dt = dts[obs_idx]
    
    # Angular difference with proper circular wraparound [0, 180] degrees
    arc_bearing = cand_df['arc_bearing'].values
    raw_diff = np.abs(arc_bearing - cand_heading) % 360.0
    heading_diff = np.minimum(raw_diff, 360.0 - raw_diff)
    
    # For very low speeds (< 0.5 m/s, vehicle stationary), heading is noisy/arbitrary -> neutral 90 deg
    stationary_mask = cand_speed < 0.5
    heading_diff[stationary_mask] = 90.0
    
    # Cosine of heading difference: +1 = aligned, -1 = reverse direction, 0 = perpendicular
    heading_cos = np.cos(np.radians(heading_diff))
    
    # Distance ratio to closest candidate (rank 1 candidate for that observation)
    min_dists = cand_df.groupby('obs_idx')['perp_distance'].transform('min').values
    distance_ratio = cand_df['perp_distance'].values / np.maximum(min_dists, 0.1)
    
    # Proximity to segment endpoints: 0 at ends, 0.5 at midpoint
    t_proj = cand_df['projection_ratio'].values
    endpoint_proximity = np.minimum(t_proj, 1.0 - t_proj)
    
    df_feat = cand_df.copy()
    df_feat['heading_diff'] = heading_diff
    df_feat['heading_cos'] = heading_cos
    df_feat['vehicle_speed'] = cand_speed
    df_feat['displacement_m'] = cand_disp
    df_feat['dt_s'] = cand_dt
    df_feat['distance_ratio'] = distance_ratio
    df_feat['endpoint_proximity'] = endpoint_proximity
    
    return df_feat

def extract_features_for_trip(trip_data: Any, cand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience wrapper to extract features from a TripData object and candidates DataFrame.
    """
    return extract_candidate_features(cand_df, trip_data.track_xy, trip_data.track_df)

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

def extract_causal_features_for_trip(trip: Any, cands: pd.DataFrame) -> pd.DataFrame:
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

    def get_backward_h(k):
        h = np.zeros(n, dtype=np.float64)
        for i in range(n):
            i0 = max(0, i - k)
            h[i] = get_h(x[i] - x[i0], y[i] - y[i0]) if i != i0 else h1[i]
        return h

    h2 = get_backward_h(2)
    h4 = get_backward_h(4)
    h8 = get_backward_h(8)

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

    smooth3_x = pd.Series(x).rolling(window=3, min_periods=1).mean().values
    smooth3_y = pd.Series(y).rolling(window=3, min_periods=1).mean().values
    smooth5_x = pd.Series(x).rolling(window=5, min_periods=1).mean().values
    smooth5_y = pd.Series(y).rolling(window=5, min_periods=1).mean().values

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
                'perp_distance': row['perp_distance'],
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


