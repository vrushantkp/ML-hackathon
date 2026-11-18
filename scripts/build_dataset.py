"""
PathSnap — Dataset Builder Script
Processes training and student_test trips through the spatial pipeline:
loads raw files, generates candidate arcs, extracts features, and creates binary labels.
Saves processed feature datasets and detailed label-generation audits.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from src.data_loader import load_trip
from src.label_generation import process_trip_pipeline

def build_dataset(base_raw_dir: str, out_dir: str, n_train_trips: int = 25, train_step: int = 2):
    os.makedirs(out_dir, exist_ok=True)
    
    train_dir = os.path.join(base_raw_dir, "train")
    test_dir = os.path.join(base_raw_dir, "student_test")
    
    all_train_trips = sorted(os.listdir(train_dir))
    all_test_trips = sorted(os.listdir(test_dir))
    
    selected_train_trips = all_train_trips[:n_train_trips]
    print(f"Building dataset: {len(selected_train_trips)} training trips (step={train_step}), {len(all_test_trips)} student_test trips (full)")
    
    train_dfs = []
    train_audits = []
    
    print("\n--- Processing Training Trips ---")
    for i, tid in enumerate(selected_train_trips):
        try:
            trip_data = load_trip(tid, base_raw_dir, partition="train")
            # Subsample observations if step > 1 to manage training size
            if train_step > 1:
                sub_indices = np.arange(0, len(trip_data.track_df), train_step)
                trip_data.track_df = trip_data.track_df.iloc[sub_indices].reset_index(drop=True)
                trip_data.track_xy = trip_data.track_xy[sub_indices]
                
            labeled_df, audit = process_trip_pipeline(trip_data)
            train_dfs.append(labeled_df)
            train_audits.append(audit)
            print(f"[{i+1}/{len(selected_train_trips)}] Trip {tid}: {len(labeled_df)} samples, misses={audit['label_audit']['candidate_misses']}")
        except Exception as e:
            print(f"Error processing train trip {tid}: {e}")
            
    test_dfs = []
    test_audits = []
    print("\n--- Processing Student-Testing Trips (Full Resolution) ---")
    for i, tid in enumerate(all_test_trips):
        try:
            trip_data = load_trip(tid, base_raw_dir, partition="student_test")
            labeled_df, audit = process_trip_pipeline(trip_data)
            test_dfs.append(labeled_df)
            test_audits.append(audit)
            print(f"[{i+1}/{len(all_test_trips)}] Test Trip {tid}: {len(labeled_df)} samples, misses={audit['label_audit']['candidate_misses']}")
        except Exception as e:
            print(f"Error processing test trip {tid}: {e}")
            
    train_full_df = pd.concat(train_dfs, ignore_index=True)
    test_full_df = pd.concat(test_dfs, ignore_index=True)
    
    train_csv = os.path.join(out_dir, "train_features.csv")
    test_csv = os.path.join(out_dir, "test_features.csv")
    
    train_full_df.to_csv(train_csv, index=False)
    test_full_df.to_csv(test_csv, index=False)
    print(f"\nSaved train features to {train_csv} ({len(train_full_df)} samples)")
    print(f"Saved test features to {test_csv} ({len(test_full_df)} samples)")
    
    # Global label audit summary
    total_train_obs = sum(a['label_audit']['n_gps_observations'] for a in train_audits)
    total_train_pos = sum(a['label_audit']['positive_samples'] for a in train_audits)
    total_train_neg = sum(a['label_audit']['negative_samples'] for a in train_audits)
    total_train_misses = sum(a['label_audit']['candidate_misses'] for a in train_audits)
    
    total_test_obs = sum(a['label_audit']['n_gps_observations'] for a in test_audits)
    total_test_pos = sum(a['label_audit']['positive_samples'] for a in test_audits)
    total_test_neg = sum(a['label_audit']['negative_samples'] for a in test_audits)
    total_test_misses = sum(a['label_audit']['candidate_misses'] for a in test_audits)
    
    audit_summary = {
        'training_partition': {
            'n_trips': len(train_audits),
            'n_gps_observations': total_train_obs,
            'n_candidate_samples': len(train_full_df),
            'positive_samples': total_train_pos,
            'negative_samples': total_train_neg,
            'candidate_misses': total_train_misses,
            'candidate_miss_rate_pct': round(total_train_misses / total_train_obs * 100, 4),
            'imbalance_ratio': round(total_train_neg / max(1, total_train_pos), 2)
        },
        'student_test_partition': {
            'n_trips': len(test_audits),
            'n_gps_observations': total_test_obs,
            'n_candidate_samples': len(test_full_df),
            'positive_samples': total_test_pos,
            'negative_samples': total_test_neg,
            'candidate_misses': total_test_misses,
            'candidate_miss_rate_pct': round(total_test_misses / total_test_obs * 100, 4),
            'imbalance_ratio': round(total_test_neg / max(1, total_test_pos), 2)
        }
    }
    
    audit_file = os.path.join(out_dir, "label_generation_audit.json")
    with open(audit_file, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Saved audit summary to {audit_file}:")
    print(json.dumps(audit_summary, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"))
    parser.add_argument("--n-train", type=int, default=25)
    parser.add_argument("--train-step", type=int, default=2)
    args = parser.parse_args()
    
    build_dataset(args.raw_dir, args.out_dir, args.n_train, args.train_step)
