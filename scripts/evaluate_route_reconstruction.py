"""
PathSnap — End-to-End Route Reconstruction & Benchmark Evaluation
Evaluates Classifier-Only vs Classifier + HMM Viterbi Decoder across all 10 student-test trips.
Computes Observation Top-1 Accuracy, Route Continuity, Sequence Alignment, Jaccard Overlap,
Arc Precision/Recall, and Latency. Generates publication-ready tables and Folium comparison maps.
"""

import os
import sys
import time
import json
import difflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_trip, TripData
from src.candidate_generator import generate_candidates_for_trip
from src.feature_engineering import extract_features_for_trip, FEATURE_COLUMNS
from src.label_generation import align_route_to_trajectory
from src.hmm_decoder import build_road_graph, RoadMatchHMM
from src.metrics import compute_trip_route_continuity
from src.map_visualization import build_folium_route_map
from api.inference import get_model

STUDENT_TEST_TRIPS = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091"
]

def run_benchmark():
    raw_dir = os.path.join(BASE_DIR, "data", "raw")
    results_dir = os.path.join(BASE_DIR, "artifacts", "results")
    plots_dir = os.path.join(BASE_DIR, "artifacts", "plots")
    maps_dir = os.path.join(BASE_DIR, "artifacts", "maps")
    models_dir = os.path.join(BASE_DIR, "artifacts", "models")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(maps_dir, exist_ok=True)
    
    # Select best model available
    model_name = "tuned_best_model" if os.path.exists(os.path.join(models_dir, "tuned_best_model.joblib")) else "best_model"
    print(f"Loading model: {model_name}...", flush=True)
    model = get_model(model_name)
    
    print(f"\n=======================================================")
    print(f"RUNNING PATHSNAP BENCHMARK: 10 STUDENT-TESTING TRIPS")
    print(f"=======================================================\n")
    
    trip_summary_rows = []
    
    for trip_idx, trip_id in enumerate(STUDENT_TEST_TRIPS):
        print(f"[{trip_idx+1}/10] Processing Trip {trip_id}...", flush=True)
        t0 = time.time()
        
        # 1. Load Trip Data
        trip_data = load_trip(trip_id, raw_dir, partition="student_test")
        gt_route_arcs = trip_data.route_df['arc_id'].tolist() if trip_data.route_df is not None else []
        
        # 2. Candidate Generation
        cands_df = generate_candidates_for_trip(trip_data, initial_radius=50.0, max_radius=150.0)
        
        # 3. Feature Extraction
        features_df = extract_features_for_trip(trip_data, cands_df)
        
        # 4. Association Ground Truth for Observation-level accuracy
        matched_arcs, _ = align_route_to_trajectory(
            trip_data.track_xy, trip_data.nodes_xy, trip_data.arcs_df, trip_data.route_df
        )
        gt_obs_mapping = {obs_i: int(matched_arcs[obs_i]) for obs_i in range(len(matched_arcs))}
            
        # 5. Model Inference
        X = features_df[FEATURE_COLUMNS].values
        probs = model.predict_proba(X)[:, 1]
        features_df['prob_pos'] = probs
        
        # 6. Classifier-Only Top-1 Prediction
        idx_top1 = features_df.groupby('obs_idx')['prob_pos'].idxmax()
        top1_df = features_df.loc[idx_top1].sort_values('obs_idx')
        clf_top1_arcs = top1_df['arc_id'].tolist()
        
        clf_top1_correct = sum(
            1 for obs_i, pred_a in enumerate(clf_top1_arcs)
            if obs_i in gt_obs_mapping and pred_a == gt_obs_mapping[obs_i]
        )
        clf_top1_acc = (clf_top1_correct / max(1, len(gt_obs_mapping))) * 100.0
        clf_continuity = compute_trip_route_continuity(clf_top1_arcs, trip_data.arcs_df)
        
        # 7. HMM Viterbi Decoding & Route Assembly
        t0_hmm = time.time()
        G, _ = build_road_graph(trip_data.arcs_df)
        hmm = RoadMatchHMM(G, trip_data.arcs_df, sigma_z=10.0, beta=5.0)
        
        cands_by_obs = [g.to_dict('records') for _, g in features_df.groupby('obs_idx')]
        hmm_obs_arcs = hmm.decode(
            candidates_by_obs=cands_by_obs,
            track_xy=trip_data.track_xy,
            timestamps=trip_data.track_df['timestamp'].values
        )
        hmm_reconstructed_route = hmm.reconstruct_continuous_route(hmm_obs_arcs)
        hmm_time_ms = round((time.time() - t0_hmm) * 1000.0, 1)
        
        hmm_top1_correct = sum(
            1 for obs_i, chosen_a in enumerate(hmm_obs_arcs)
            if obs_i in gt_obs_mapping and chosen_a == gt_obs_mapping[obs_i]
        )
        hmm_obs_acc = (hmm_top1_correct / max(1, len(gt_obs_mapping))) * 100.0
        hmm_continuity = compute_trip_route_continuity(hmm_reconstructed_route, trip_data.arcs_df)
        gt_continuity = compute_trip_route_continuity(gt_route_arcs, trip_data.arcs_df)
        
        # 8. Sequence Similarity & Overlap with Ground Truth
        set_gt = set(gt_route_arcs)
        set_rec = set(hmm_reconstructed_route)
        jaccard = (len(set_gt & set_rec) / max(1, len(set_gt | set_rec))) * 100.0
        precision = (len(set_gt & set_rec) / max(1, len(set_rec))) * 100.0
        recall = (len(set_gt & set_rec) / max(1, len(set_gt))) * 100.0
        seq_matcher = difflib.SequenceMatcher(None, gt_route_arcs, hmm_reconstructed_route)
        seq_ratio = seq_matcher.ratio() * 100.0
        
        total_time_ms = round((time.time() - t0) * 1000.0, 1)
        
        row = {
            'Trip ID': trip_id,
            'Observations': len(trip_data.track_df),
            'Candidates': len(features_df),
            'GT Arcs': len(gt_route_arcs),
            'HMM Arcs': len(hmm_reconstructed_route),
            'Classifier Top-1 Acc (%)': round(clf_top1_acc, 2),
            'HMM Obs Acc (%)': round(hmm_obs_acc, 2),
            'Classifier Continuity (%)': round(clf_continuity, 2),
            'HMM Continuity (%)': round(hmm_continuity, 2),
            'GT Continuity (%)': round(gt_continuity, 2),
            'Jaccard Overlap (%)': round(jaccard, 2),
            'Sequence Alignment (%)': round(seq_ratio, 2),
            'Arc Recall (%)': round(recall, 2),
            'Arc Precision (%)': round(precision, 2),
            'Latency (ms)': total_time_ms
        }
        trip_summary_rows.append(row)
        print(f"   Done: Clf Top1 Acc={round(clf_top1_acc, 1)}% | HMM Obs Acc={round(hmm_obs_acc, 1)}% | Clf Cont={round(clf_continuity, 1)}% -> HMM Cont={round(hmm_continuity, 1)}% | Latency={total_time_ms}ms", flush=True)
        
        # Generate and save interactive Folium map for this trip
        trip_map = build_folium_route_map(
            trip_data=trip_data,
            gt_route_arcs=gt_route_arcs,
            classifier_arcs=clf_top1_arcs,
            hmm_reconstructed_arcs=hmm_reconstructed_route,
            metrics_summary={
                'top1_accuracy': f"{round(hmm_obs_acc, 2)}%",
                'continuity': f"{round(hmm_continuity, 2)}%"
            },
            show_ground_truth=True
        )
        map_out_path = os.path.join(maps_dir, f"trip_{trip_id}_reconstructed.html")
        trip_map.save(map_out_path)
        
    summary_df = pd.DataFrame(trip_summary_rows)
    summary_csv = os.path.join(results_dir, "route_reconstruction_summary.csv")
    summary_json = os.path.join(results_dir, "route_reconstruction_summary.json")
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_json(summary_json, orient='records', indent=2)
    
    print("\n" + "="*80)
    print("CONSOLIDATED ROUTE RECONSTRUCTION BENCHMARK SUMMARY (10 TRIPS)")
    print("="*80)
    print(summary_df.to_string(index=False), flush=True)
    
    # Calculate Mean Performance
    mean_clf_acc = summary_df['Classifier Top-1 Acc (%)'].mean()
    mean_hmm_acc = summary_df['HMM Obs Acc (%)'].mean()
    mean_clf_cont = summary_df['Classifier Continuity (%)'].mean()
    mean_hmm_cont = summary_df['HMM Continuity (%)'].mean()
    mean_jaccard = summary_df['Jaccard Overlap (%)'].mean()
    mean_seq = summary_df['Sequence Alignment (%)'].mean()
    mean_rec = summary_df['Arc Recall (%)'].mean()
    mean_prec = summary_df['Arc Precision (%)'].mean()
    mean_lat = summary_df['Latency (ms)'].mean()
    
    print("\n" + "-"*50)
    print("OVERALL SYSTEM METRIC AVERAGES:")
    print(f"  Classifier Top-1 Observation Accuracy: {mean_clf_acc:.2f}%")
    print(f"  HMM Viterbi Observation Accuracy:       {mean_hmm_acc:.2f}%")
    print(f"  Classifier-Only Route Continuity:      {mean_clf_cont:.2f}%")
    print(f"  HMM Reconstructed Route Continuity:    {mean_hmm_cont:.2f}%")
    print(f"  Arc Jaccard Overlap with Ground Truth: {mean_jaccard:.2f}%")
    print(f"  Sequence Alignment Ratio:              {mean_seq:.2f}%")
    print(f"  Arc Recall:                            {mean_rec:.2f}%")
    print(f"  Arc Precision:                         {mean_prec:.2f}%")
    print(f"  Mean End-to-End Latency:               {mean_lat:.1f} ms")
    print("-"*50 + "\n")
    
    # Visual Comparison Plot: Continuity Improvement
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary_df))
    width = 0.35
    plt.bar(x - width/2, summary_df['Classifier Continuity (%)'], width, label='Classifier-Only', color='#E67E22', alpha=0.85)
    plt.bar(x + width/2, summary_df['HMM Continuity (%)'], width, label='Classifier + HMM Viterbi', color='#2ECC71', alpha=0.9)
    plt.axhline(100.0, color='gray', linestyle='--', alpha=0.5, label='100% Ideal Continuity')
    plt.xlabel("Trip ID")
    plt.ylabel("Route Continuity (%)")
    plt.title("PathSnap Route Continuity: Classifier-Only vs Classifier + HMM Viterbi")
    plt.xticks(x, [f"Trip {t[-2:]}" for t in summary_df['Trip ID']], rotation=0)
    plt.ylim(60, 105)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hmm_vs_classifier_continuity.png"), dpi=200)
    plt.close()
    
    # Accuracy Comparison Plot
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, summary_df['Classifier Top-1 Acc (%)'], width, label='Classifier Top-1', color='#3498DB', alpha=0.85)
    plt.bar(x + width/2, summary_df['HMM Obs Acc (%)'], width, label='HMM Viterbi Decoded', color='#9B59B6', alpha=0.9)
    plt.xlabel("Trip ID")
    plt.ylabel("Observation Accuracy (%)")
    plt.title("Observation-level Route Matching Accuracy Across 10 Student-Test Trips")
    plt.xticks(x, [f"Trip {t[-2:]}" for t in summary_df['Trip ID']], rotation=0)
    plt.ylim(60, 100)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hmm_vs_classifier_accuracy.png"), dpi=200)
    plt.close()
    
    return summary_df

if __name__ == "__main__":
    run_benchmark()
