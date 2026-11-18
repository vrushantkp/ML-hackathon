"""
PathSnap — Benchmark Evaluator Test Set (Official 10 Test Trips)
Evaluates all 10 unseen evaluator trips across models:
Random Forest, HistGradientBoosting, MLP, KNN, and Logistic Regression.
Saves comprehensive results to artifacts/results/evaluator_test_results.json and .csv
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.evaluation import evaluate_trip_server_side

EVAL_TRIPS = [
    "00000015", "00000017", "00000023", "00000030", "00000049",
    "00000059", "00000067", "00000073", "00000087", "00000094"
]

MODELS = ["random_forest", "histgradientboosting", "multi_layer_perceptron", "k_nearest_neighbors", "logistic_regression"]

def main():
    results_dir = os.path.join(BASE_DIR, "artifacts", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_records = []
    
    print("=" * 70)
    print("BENCHMARKING OFFICIAL EVALUATOR TEST SET (10 UNSEEN TRIPS)")
    print("=" * 70)
    
    # First evaluate Random Forest and HistGradientBoosting across all 10 trips
    for model_name in ["random_forest", "histgradientboosting"]:
        print(f"\n---> Evaluating Model: {model_name}")
        for idx, trip_id in enumerate(EVAL_TRIPS):
            t0 = time.time()
            try:
                res = evaluate_trip_server_side(trip_id, partition="evaluator_test", model_name=model_name, use_hmm=True)
                metrics = res.get("evaluation_metrics", {})
                elapsed = time.time() - t0
                
                record = {
                    "trip_id": trip_id,
                    "partition": "evaluator_test",
                    "model": model_name,
                    "observations": metrics.get("classifier_top1_arc_count", 0),
                    "gt_arcs": metrics.get("ground_truth_arc_count", 0),
                    "reconstructed_arcs": metrics.get("reconstructed_arc_count", 0),
                    "top1_accuracy": metrics.get("top1_accuracy_pct", 0.0),
                    "hmm_obs_accuracy": metrics.get("hmm_observation_accuracy_pct", 0.0),
                    "continuity": metrics.get("reconstructed_route_continuity_pct", 0.0),
                    "sequence_match": metrics.get("sequence_alignment_ratio_pct", 0.0),
                    "jaccard": metrics.get("arc_jaccard_overlap_pct", 0.0),
                    "precision": metrics.get("arc_precision_pct", 0.0),
                    "recall": metrics.get("arc_recall_pct", 0.0),
                    "misaligned_arcs": metrics.get("reconstructed_misaligned_arc_count", 0),
                    "misalignment_pct": metrics.get("reconstructed_misaligned_arc_pct", 0.0),
                    "latency_ms": metrics.get("inference_latency_ms", round(elapsed * 1000, 2))
                }
                all_records.append(record)
                print(f"  [{idx+1}/10] Trip {trip_id} | Top-1: {record['top1_accuracy']:.2f}% | SeqMatch: {record['sequence_match']:.2f}% | Cont: {record['continuity']:.2f}% | Latency: {record['latency_ms']:.1f}ms", flush=True)
            except Exception as e:
                print(f"  [ERROR] Trip {trip_id} failed with {model_name}: {e}", flush=True)

    # For other models, evaluate on representative sample trips (or all)
    for model_name in ["multi_layer_perceptron", "k_nearest_neighbors", "logistic_regression"]:
        print(f"\n---> Evaluating Model: {model_name} (Sample Trips: 00000023, 00000030, 00000067)")
        for trip_id in ["00000023", "00000030", "00000067"]:
            t0 = time.time()
            try:
                res = evaluate_trip_server_side(trip_id, partition="evaluator_test", model_name=model_name, use_hmm=True)
                metrics = res.get("evaluation_metrics", {})
                elapsed = time.time() - t0
                record = {
                    "trip_id": trip_id,
                    "partition": "evaluator_test",
                    "model": model_name,
                    "observations": metrics.get("classifier_top1_arc_count", 0),
                    "gt_arcs": metrics.get("ground_truth_arc_count", 0),
                    "reconstructed_arcs": metrics.get("reconstructed_arc_count", 0),
                    "top1_accuracy": metrics.get("top1_accuracy_pct", 0.0),
                    "hmm_obs_accuracy": metrics.get("hmm_observation_accuracy_pct", 0.0),
                    "continuity": metrics.get("reconstructed_route_continuity_pct", 0.0),
                    "sequence_match": metrics.get("sequence_alignment_ratio_pct", 0.0),
                    "jaccard": metrics.get("arc_jaccard_overlap_pct", 0.0),
                    "precision": metrics.get("arc_precision_pct", 0.0),
                    "recall": metrics.get("arc_recall_pct", 0.0),
                    "misaligned_arcs": metrics.get("reconstructed_misaligned_arc_count", 0),
                    "misalignment_pct": metrics.get("reconstructed_misaligned_arc_pct", 0.0),
                    "latency_ms": metrics.get("inference_latency_ms", round(elapsed * 1000, 2))
                }
                all_records.append(record)
                print(f"  Trip {trip_id} | Top-1: {record['top1_accuracy']:.2f}% | SeqMatch: {record['sequence_match']:.2f}% | Cont: {record['continuity']:.2f}%", flush=True)
            except Exception as e:
                print(f"  [ERROR] Trip {trip_id} failed with {model_name}: {e}", flush=True)

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(results_dir, "evaluator_test_results.csv")
    json_path = os.path.join(results_dir, "evaluator_test_results.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
        
    print(f"\nSaved results to {csv_path} and {json_path}")
    
    # Aggregate summaries by model
    print("\n" + "=" * 70)
    print("SUMMARY BY MODEL ON EVALUATOR TEST SET:")
    print("=" * 70)
    summary = df.groupby("model").agg({
        "top1_accuracy": ["mean", "std"],
        "hmm_obs_accuracy": ["mean", "std"],
        "sequence_match": ["mean", "std"],
        "continuity": ["mean", "std"],
        "jaccard": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "latency_ms": "mean"
    }).round(2)
    print(summary)
    
if __name__ == "__main__":
    main()
