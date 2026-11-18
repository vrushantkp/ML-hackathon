"""
PathSnap — Multi-Model Training and Evaluation Pipeline
Trains and benchmarks five distinct machine learning algorithm families using GroupKFold (grouped by trip_id).
Evaluates on the student-testing partition: Macro F1, Top-1 Accuracy, Route Continuity, Log Loss, and Latency.
Saves model artifacts, comparison tables, and diagnostic plots.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix

from src.feature_engineering import FEATURE_COLUMNS
from src.metrics import compute_classification_metrics, compute_observation_top1_accuracy, compute_trip_route_continuity
from src.data_loader import load_trip
from src.models.linear_model import create_logistic_regression
from src.models.random_forest import create_random_forest
from src.models.gradient_boosting import create_gradient_boosting
from src.models.knn_model import create_knn
from src.models.mlp_model import create_mlp

def get_models() -> Dict[str, Dict[str, Any]]:
    return {
        'Logistic Regression': {
            'family': 'Linear / Probabilistic',
            'config': 'L2, C=1.0, balanced, lbfgs, scaled',
            'factory': lambda: create_logistic_regression(C=1.0)
        },
        'Random Forest': {
            'family': 'Ensemble / Bagging',
            'config': 'n=100, max_depth=12, balanced_subsample',
            'factory': lambda: create_random_forest(n_estimators=100, max_depth=12)
        },
        'HistGradientBoosting': {
            'family': 'Boosting',
            'config': 'lr=0.1, max_iter=100, max_depth=8, balanced',
            'factory': lambda: create_gradient_boosting(learning_rate=0.1, max_iter=100)
        },
        'K-Nearest Neighbors': {
            'family': 'Instance-based',
            'config': 'k=9, distance-weighted, scaled, subsampled',
            'factory': lambda: create_knn(n_neighbors=9)
        },
        'Multi-Layer Perceptron': {
            'family': 'Neural Network',
            'config': 'layers=(64, 32), relu, adam, early_stop, scaled',
            'factory': lambda: create_mlp(hidden_layer_sizes=(64, 32))
        }
    }

def evaluate_route_continuity_for_preds(test_df: pd.DataFrame, base_raw_dir: str) -> float:
    """
    Computes average route continuity across all student-test trips for given predictions.
    """
    trip_continuities = []
    idx_top1 = test_df.groupby(['trip_id', 'obs_idx'])['prob_pos'].idxmax()
    top1_df = test_df.loc[idx_top1]
    
    for trip_id, group in top1_df.groupby('trip_id'):
        sorted_group = group.sort_values('obs_idx')
        pred_arcs = sorted_group['arc_id'].values
        
        str_trip_id = str(trip_id).zfill(8)
        trip_data = load_trip(str_trip_id, base_raw_dir, partition="student_test")
        cont = compute_trip_route_continuity(pred_arcs, trip_data.arcs_df)
        trip_continuities.append(cont)
        
    return float(np.mean(trip_continuities)) if trip_continuities else 0.0

def run_experiments():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    raw_dir = os.path.join(base_dir, "data", "raw")
    plots_dir = os.path.join(base_dir, "artifacts", "plots")
    models_dir = os.path.join(base_dir, "artifacts", "models")
    results_dir = os.path.join(base_dir, "artifacts", "results")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading processed features...", flush=True)
    train_df = pd.read_csv(os.path.join(data_dir, "train_features.csv"), dtype={'trip_id': str})
    test_df = pd.read_csv(os.path.join(data_dir, "test_features.csv"), dtype={'trip_id': str})
    
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df['label'].values
    groups_train = train_df['trip_id'].values
    
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df['label'].values
    
    models = get_models()
    comparison_rows = []
    fitted_models = {}
    
    # 1. 5-Fold GroupKFold Cross Validation
    print("\n" + "="*50, flush=True)
    print("STEP 1: 5-Fold GroupKFold Cross-Validation (Grouped by trip_id)", flush=True)
    print("="*50, flush=True)
    gkf = GroupKFold(n_splits=5)
    
    cv_scores = {}
    for name, m_info in models.items():
        print(f"\n--- Running 5-Fold GroupKFold for: {name} ---", flush=True)
        fold_f1s = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
            model = m_info['factory']()
            if name == 'K-Nearest Neighbors':
                sub_tr = np.random.choice(train_idx, min(10000, len(train_idx)), replace=False)
                sub_val = np.random.choice(val_idx, min(2500, len(val_idx)), replace=False)
                model.fit(X_train[sub_tr], y_train[sub_tr])
                val_preds = model.predict(X_train[sub_val])
                val_f1 = compute_classification_metrics(y_train[sub_val], val_preds)['macro_f1']
            else:
                model.fit(X_train[train_idx], y_train[train_idx])
                val_preds = model.predict(X_train[val_idx])
                val_f1 = compute_classification_metrics(y_train[val_idx], val_preds)['macro_f1']
                
            fold_f1s.append(val_f1)
            print(f"  Fold {fold+1}: Macro F1 = {val_f1:.4f}", flush=True)
            
        mean_cv_f1 = float(np.mean(fold_f1s))
        std_cv_f1 = float(np.std(fold_f1s))
        cv_scores[name] = {'mean_macro_f1': mean_cv_f1, 'std_macro_f1': std_cv_f1}
        print(f"==> {name} Mean CV Macro F1: {mean_cv_f1:.4f} (+/- {std_cv_f1:.4f})", flush=True)
        
    # 2. Train on Full Training Set and Evaluate on Student-Testing
    print("\n" + "="*50, flush=True)
    print("STEP 2: Full Training & Student-Testing Holdout Evaluation", flush=True)
    print("="*50, flush=True)
    
    for name, m_info in models.items():
        print(f"\nTraining {name} on full dataset ({len(X_train)} samples)...", flush=True)
        model = m_info['factory']()
        
        start_train = time.time()
        if name == 'K-Nearest Neighbors':
            sub_tr = np.random.choice(len(X_train), 15000, replace=False)
            model.fit(X_train[sub_tr], y_train[sub_tr])
        else:
            model.fit(X_train, y_train)
        train_time = round(time.time() - start_train, 2)
        
        # Inference on Student-Testing
        start_inf = time.time()
        if name == 'K-Nearest Neighbors':
            # Fast batch predict
            test_probs = np.zeros(len(X_test), dtype=np.float32)
            batch_size = 20000
            for b in range(0, len(X_test), batch_size):
                end_b = min(b + batch_size, len(X_test))
                test_probs[b:end_b] = model.predict_proba(X_test[b:end_b])[:, 1]
        else:
            test_probs = model.predict_proba(X_test)[:, 1]
            
        test_preds = (test_probs >= 0.5).astype(int)
        inf_time = round(time.time() - start_inf, 3)
        
        test_eval_df = test_df.copy()
        test_eval_df['prob_pos'] = test_probs
        test_eval_df['pred_label'] = test_preds
        
        # Metrics
        cand_metrics = compute_classification_metrics(y_test, test_preds, test_probs)
        top1_metrics = compute_observation_top1_accuracy(test_eval_df)
        route_continuity = evaluate_route_continuity_for_preds(test_eval_df, raw_dir)
        
        fitted_models[name] = model
        
        model_filename = name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
        joblib.dump(model, os.path.join(models_dir, model_filename))
        
        cm = confusion_matrix(y_test, test_preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        plt.title(f"Confusion Matrix: {name}")
        plt.ylabel("Ground Truth")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"cm_{name.lower().replace(' ', '_').replace('-', '_')}.png"), dpi=200)
        plt.close()
        
        row = {
            'Model': name,
            'Algorithm Family': m_info['family'],
            'Key Configuration': m_info['config'],
            'CV Macro F1': round(cv_scores[name]['mean_macro_f1'], 4),
            'Top-1 Accuracy': f"{top1_metrics['top1_accuracy_pct']}%",
            'Route Continuity': f"{round(route_continuity, 2)}%",
            'Student-Test Macro F1': round(cand_metrics['macro_f1'], 4),
            'Log Loss': round(cand_metrics['log_loss'], 4) if cand_metrics['log_loss'] is not None else "N/A",
            'Training Time (s)': train_time,
            'Inference Time (s)': inf_time
        }
        comparison_rows.append(row)
        print(f"Results for {name}:", flush=True)
        for k, v in row.items():
            print(f"  {k}: {v}", flush=True)
            
    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = os.path.join(results_dir, "model_comparison.csv")
    comp_json = os.path.join(results_dir, "model_comparison.json")
    comp_df.to_csv(comp_csv, index=False)
    comp_df.to_json(comp_json, orient='records', indent=2)
    
    print("\n" + "="*50, flush=True)
    print("CONSOLIDATED MODEL COMPARISON TABLE", flush=True)
    print("="*50, flush=True)
    print(comp_df.to_string(index=False), flush=True)
    
    # 3. Comparison Plots
    sns.set_theme(style="whitegrid")
    
    # CV Macro F1 Plot
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=comp_df, x="Model", y="CV Macro F1", hue="Model", palette="Blues_d", legend=False)
    plt.title("5-Fold Cross-Validation Macro F1 Across Model Families")
    plt.ylabel("CV Macro F1")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "cv_macro_f1_comparison.png"), dpi=200)
    plt.close()
    
    # Student Test Top-1 Accuracy Plot
    top1_vals = [float(r['Top-1 Accuracy'].replace('%', '')) for r in comparison_rows]
    plt.figure(figsize=(9, 4.5))
    plt.bar(comp_df['Model'], top1_vals, color='#2ECC71')
    plt.title("Observation-level Top-1 Accuracy on Student-Testing Partition")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "test_top1_accuracy_comparison.png"), dpi=200)
    plt.close()
    
    # Student Test Macro F1 Plot
    plt.figure(figsize=(9, 4.5))
    sns.barplot(data=comp_df, x="Model", y="Student-Test Macro F1", hue="Model", palette="Purples_d", legend=False)
    plt.title("Student-Testing Candidate-Level Macro F1-Score")
    plt.ylabel("Macro F1-Score")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "test_macro_f1_comparison.png"), dpi=200)
    plt.close()
    
    # Select Best Model based on Student-Test Top-1 and Macro F1
    best_row = max(comparison_rows, key=lambda r: float(r['Top-1 Accuracy'].replace('%', '')) * 0.5 + r['Student-Test Macro F1'] * 50)
    best_model_name = best_row['Model']
    print(f"\n>>> Best Performing Model Selected: {best_model_name} (Top-1 Accuracy: {best_row['Top-1 Accuracy']}, CV Macro F1: {best_row['CV Macro F1']})", flush=True)
    
    joblib.dump(fitted_models[best_model_name], os.path.join(models_dir, "best_model.joblib"))
    print(f"Saved {best_model_name} as artifacts/models/best_model.joblib", flush=True)

if __name__ == "__main__":
    run_experiments()
