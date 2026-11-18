"""
PathSnap — Metrics Evaluation Module
Calculates standard classification metrics, observation-level Top-1 Accuracy, Log Loss, and Route Continuity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Computes candidate-level classification metrics.
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'binary_f1': float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
    }
    if y_prob is not None:
        try:
            # Clip probabilities for numerical stability in log loss
            eps = 1e-15
            y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
            metrics['log_loss'] = float(log_loss(y_true, y_prob_clipped))
        except Exception:
            metrics['log_loss'] = None
    else:
        metrics['log_loss'] = None
        
    return metrics

def compute_observation_top1_accuracy(df: pd.DataFrame, prob_col: str = 'prob_pos', label_col: str = 'label') -> Dict[str, float]:
    """
    Observation-level Top-1 Accuracy:
    For each GPS observation (trip_id, obs_idx), select the candidate with highest predicted probability.
    Compare with reference ground truth (label == 1).
    """
    # Find row with max probability per (trip_id, obs_idx)
    idx_top1 = df.groupby(['trip_id', 'obs_idx'])[prob_col].idxmax()
    top1_df = df.loc[idx_top1]
    
    total_obs = len(top1_df)
    correct_obs = int((top1_df[label_col] == 1).sum())
    top1_accuracy = float(correct_obs / max(1, total_obs))
    
    return {
        'total_observations': total_obs,
        'correct_top1_observations': correct_obs,
        'top1_accuracy': round(top1_accuracy, 4),
        'top1_accuracy_pct': round(top1_accuracy * 100.0, 2)
    }

def compute_trip_route_continuity(predicted_arcs: np.ndarray, arcs_df: pd.DataFrame) -> float:
    """
    Route Continuity:
    Percentage of consecutive predicted arcs that are topologically connected in the directed road graph.
    Consecutive arcs a_t and a_{t+1} are connected if:
    1. a_t == a_{t+1} (vehicle remains on same segment)
    2. v(a_t) == u(a_{t+1}) (direct downstream transition)
    """
    if len(predicted_arcs) <= 1:
        return 100.0
        
    u_col = 'u' if 'u' in arcs_df.columns else 'from_node_id'
    v_col = 'v' if 'v' in arcs_df.columns else 'to_node_id'
    u_vals = arcs_df[u_col].values
    v_vals = arcs_df[v_col].values
    
    connected_steps = 0
    total_steps = len(predicted_arcs) - 1
    
    for t in range(total_steps):
        a1 = predicted_arcs[t]
        a2 = predicted_arcs[t+1]
        
        if a1 == a2:
            connected_steps += 1
        elif v_vals[a1] == u_vals[a2]:
            connected_steps += 1
            
    return float(connected_steps / total_steps * 100.0)
