"""
PathSnap — Label Generation Module
Associates ordered reference route with GPS trajectory using monotonic progression alignment.
Generates binary candidate classification labels (1 = matching reference road arc, 0 = non-matching).
Audits candidate hit rate, miss rate, and positive/negative balance.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from src.data_loader import TripData, load_trip
from src.candidate_generator import CandidateGenerator
from src.feature_engineering import extract_candidate_features, FEATURE_COLUMNS

def align_route_to_trajectory(track_xy: np.ndarray, nodes_xy: np.ndarray, arcs_df: pd.DataFrame, route_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Monotonically aligns the ordered ground truth route sequence to the GPS trajectory.
    Returns matched_arc_ids: Array of shape (N,) containing reference arc_id for each GPS observation.
    """
    route_arcs = route_df['arc_id'].values
    K = len(route_arcs)
    N = len(track_xy)
    
    if K == 0:
        return np.zeros(N, dtype=np.int64), {'route_arcs': 0}
        
    u_idx = arcs_df.iloc[route_arcs]['u'].values
    v_idx = arcs_df.iloc[route_arcs]['v'].values
    
    u_coords = nodes_xy[u_idx]
    v_coords = nodes_xy[v_idx]
    ab = v_coords - u_coords
    ab_len_sq = np.maximum(np.sum(ab**2, axis=-1), 1e-4)
    
    k_curr = 0
    matched_arcs = np.zeros(N, dtype=np.int64)
    matched_k_indices = np.zeros(N, dtype=np.int64)
    
    for t in range(N):
        p = track_xy[t]
        k_window = list(range(k_curr, min(K, k_curr + 8)))
        
        sub_a = u_coords[k_window]
        sub_ab = ab[k_window]
        sub_len_sq = ab_len_sq[k_window]
        
        ap = p - sub_a
        t_ratio = np.clip(np.sum(ap * sub_ab, axis=-1) / sub_len_sq, 0.0, 1.0)
        proj = sub_a + t_ratio[:, None] * sub_ab
        dists = np.sqrt(np.sum((p - proj)**2, axis=-1))
        
        best_local_idx = np.argmin(dists)
        k_curr = k_window[best_local_idx]
        
        matched_k_indices[t] = k_curr
        matched_arcs[t] = route_arcs[k_curr]
        
    alignment_stats = {
        'total_route_arcs': K,
        'unique_route_arcs_matched': int(len(np.unique(matched_arcs))),
        'route_coverage_pct': round(float(len(np.unique(matched_arcs)) / K * 100), 2)
    }
    
    return matched_arcs, alignment_stats

def generate_candidate_labels(cand_df: pd.DataFrame, matched_arcs: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Attaches binary target labels to candidate records.
    label = 1 if cand_df['arc_id'] == matched_arcs[obs_idx] else 0.
    """
    obs_indices = cand_df['obs_idx'].values
    true_arc_for_obs = matched_arcs[obs_indices]
    
    labels = (cand_df['arc_id'].values == true_arc_for_obs).astype(np.int32)
    
    cand_with_labels = cand_df.copy()
    cand_with_labels['label'] = labels
    
    # Audit hit and miss rates
    pos_by_obs = cand_with_labels.groupby('obs_idx')['label'].sum()
    n_total_obs = len(matched_arcs)
    obs_with_positive = int((pos_by_obs > 0).sum())
    candidate_misses = n_total_obs - obs_with_positive
    miss_rate_pct = (candidate_misses / n_total_obs) * 100.0
    
    audit = {
        'n_gps_observations': n_total_obs,
        'n_candidate_samples': len(cand_with_labels),
        'positive_samples': int(labels.sum()),
        'negative_samples': int((labels == 0).sum()),
        'candidate_hits': obs_with_positive,
        'candidate_misses': candidate_misses,
        'candidate_miss_rate_pct': round(miss_rate_pct, 4),
        'imbalance_ratio': round(float((labels == 0).sum() / max(1, labels.sum())), 2)
    }
    
    return cand_with_labels, audit

def process_trip_pipeline(trip_data: TripData, generator: Optional[CandidateGenerator] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the full spatial pipeline for a single trip:
    1. Candidate generation
    2. Feature engineering
    3. Monotonic route alignment and label generation (if route_df is available)
    """
    if generator is None:
        generator = CandidateGenerator(trip_data.nodes_xy, trip_data.arcs_df)
        
    cand_df, cand_stats = generator.generate_candidates_for_points(trip_data.track_xy)
    feat_df = extract_candidate_features(cand_df, trip_data.track_xy, trip_data.track_df)
    feat_df['trip_id'] = trip_data.trip_id
    feat_df['partition'] = trip_data.partition
    
    audit_info = {
        'trip_id': trip_data.trip_id,
        'candidate_stats': cand_stats
    }
    
    if trip_data.route_df is not None:
        matched_arcs, align_stats = align_route_to_trajectory(
            trip_data.track_xy, trip_data.nodes_xy, trip_data.arcs_df, trip_data.route_df
        )
        labeled_df, label_audit = generate_candidate_labels(feat_df, matched_arcs)
        audit_info['alignment_stats'] = align_stats
        audit_info['label_audit'] = label_audit
        return labeled_df, audit_info
    else:
        feat_df['label'] = -1
        return feat_df, audit_info
