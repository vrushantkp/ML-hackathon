"""
PathSnap — Server-Side Evaluation Service
Securely evaluates reconstructed routes against ground truth routes on Laptop 1.
Ensures ground truth labels remain server-side and are never leaked to client predictions.
"""

import os
import difflib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from src.data_loader import load_trip, TripData
from src.metrics import (
    compute_trip_route_continuity,
    compute_classification_metrics,
    compute_observation_top1_accuracy
)
from api.inference import run_route_reconstruction

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

def evaluate_trip_server_side(
    trip_id: str,
    partition: Optional[str] = None,
    model_name: str = "histgradientboosting",
    use_hmm: bool = True
) -> Dict[str, Any]:
    """
    Evaluates model reconstruction on Laptop 1 against ground truth.
    """
    trip_data = load_trip(trip_id, RAW_DIR, partition=partition)
    
    if trip_data.route_df is None:
        return {
            'trip_id': trip_data.trip_id,
            'partition': trip_data.partition,
            'has_ground_truth': False,
            'evaluation_metrics': None,
            'message': f"Trip {trip_data.trip_id} belongs to a reserved partition without ground truth reference route."
        }
        
    gt_route_arcs = trip_data.route_df['arc_id'].tolist()
    
    # Run prediction
    pred_res = run_route_reconstruction(
        trip_id=trip_id,
        partition=partition,
        model_name=model_name,
        use_hmm=use_hmm
    )
    
    reconstructed_arcs = pred_res['reconstructed_route']
    classifier_top1_arcs = pred_res['classifier_top1_route']
    
    # 1. Route Continuity
    reconstructed_cont = compute_trip_route_continuity(reconstructed_arcs, trip_data.arcs_df)
    clf_cont = compute_trip_route_continuity(classifier_top1_arcs, trip_data.arcs_df)
    gt_cont = compute_trip_route_continuity(gt_route_arcs, trip_data.arcs_df)
    
    # 2. Sequence similarity / Jaccard Overlap
    set_gt = set(gt_route_arcs)
    set_rec = set(reconstructed_arcs)
    jaccard_similarity = len(set_gt & set_rec) / max(1, len(set_gt | set_rec))
    
    # Sequence Match Ratio (Gestalt Pattern Matching via difflib)
    matcher = difflib.SequenceMatcher(None, gt_route_arcs, reconstructed_arcs)
    sequence_match_ratio = matcher.ratio()
    
    # Arc Coverage / Recall
    arc_recall = len(set_gt & set_rec) / max(1, len(set_gt))
    arc_precision = len(set_gt & set_rec) / max(1, len(set_rec))
    
    top1_acc = pred_res.get('top1_accuracy_pct', 0.0)
    hmm_obs_acc = pred_res.get('hmm_observation_accuracy_pct', 0.0)
    
    eval_metrics = {
        'ground_truth_arc_count': len(gt_route_arcs),
        'reconstructed_arc_count': len(reconstructed_arcs),
        'classifier_top1_arc_count': len(classifier_top1_arcs),
        'top1_accuracy_pct': top1_acc,
        'hmm_observation_accuracy_pct': hmm_obs_acc,
        'reconstructed_route_continuity_pct': round(reconstructed_cont, 2),
        'classifier_route_continuity_pct': round(clf_cont, 2),
        'ground_truth_continuity_pct': round(gt_cont, 2),
        'arc_jaccard_overlap_pct': round(jaccard_similarity * 100.0, 2),
        'sequence_alignment_ratio_pct': round(sequence_match_ratio * 100.0, 2),
        'arc_precision_pct': round(arc_precision * 100.0, 2),
        'arc_recall_pct': round(arc_recall * 100.0, 2),
        'inference_latency_ms': pred_res['inference_time_ms']
    }
    
    return {
        'trip_id': trip_data.trip_id,
        'partition': trip_data.partition,
        'has_ground_truth': True,
        'evaluation_metrics': eval_metrics,
        'message': f"Server-side evaluation successful for trip {trip_data.trip_id}."
    }
