"""
PathSnap — Bayesian Hyperparameter Optimization Module
Uses Optuna with Tree-structured Parzen Estimator (TPE) to tune key hyperparameters
of candidate relevance classifiers using GroupKFold cross-validation.
"""

import os
import time
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from src.feature_engineering import FEATURE_COLUMNS
from src.metrics import compute_classification_metrics, compute_observation_top1_accuracy, compute_trip_route_continuity
from src.data_loader import load_trip

optuna.logging.set_verbosity(optuna.logging.WARNING)

def evaluate_route_continuity_for_preds(test_df: pd.DataFrame, base_raw_dir: str) -> float:
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

def run_bayesian_optimization(n_trials: int = 25, model_type: str = "random_forest"):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    raw_dir = os.path.join(base_dir, "data", "raw")
    plots_dir = os.path.join(base_dir, "artifacts", "plots")
    models_dir = os.path.join(base_dir, "artifacts", "models")
    results_dir = os.path.join(base_dir, "artifacts", "results")
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading features for Bayesian Optimization ({model_type})...", flush=True)
    train_df = pd.read_csv(os.path.join(data_dir, "train_features.csv"), dtype={'trip_id': str})
    test_df = pd.read_csv(os.path.join(data_dir, "test_features.csv"), dtype={'trip_id': str})
    
    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df['label'].values
    groups_train = train_df['trip_id'].values
    
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df['label'].values
    
    # Use 3-fold GroupKFold for Bayesian optimization speed
    gkf = GroupKFold(n_splits=3)
    splits = list(gkf.split(X_train, y_train, groups=groups_train))
    
    print(f"Defining Optuna study with TPE sampler ({n_trials} trials)...", flush=True)
    
    def objective(trial):
        if model_type == "random_forest":
            n_estimators = trial.suggest_int("n_estimators", 40, 80, step=20)
            max_depth = trial.suggest_int("max_depth", 10, 16)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 6)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 3)
            
            fold_scores = []
            for train_idx, val_idx in splits:
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced_subsample",
                    n_jobs=4,
                    random_state=42
                )
                clf.fit(X_train[train_idx], y_train[train_idx])
                y_val_pred = clf.predict(X_train[val_idx])
                fold_scores.append(f1_score(y_train[val_idx], y_val_pred, average="macro"))
                
            mean_score = float(np.mean(fold_scores))
            print(f"  [Trial #{trial.number:02d}] Macro F1 = {mean_score:.4f} | depth={max_depth}, n_est={n_estimators}, split={min_samples_split}, leaf={min_samples_leaf}", flush=True)
            return mean_score
            
        elif model_type == "hist_gradient_boosting":
            learning_rate = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
            max_iter = trial.suggest_int("max_iter", 50, 130, step=20)
            max_depth = trial.suggest_int("max_depth", 6, 14)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 40)
            l2_regularization = trial.suggest_float("l2_regularization", 1e-3, 5.0, log=True)
            
            fold_scores = []
            for train_idx, val_idx in splits:
                clf = HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    l2_regularization=l2_regularization,
                    class_weight="balanced",
                    random_state=42
                )
                clf.fit(X_train[train_idx], y_train[train_idx])
                y_val_pred = clf.predict(X_train[val_idx])
                fold_scores.append(f1_score(y_train[val_idx], y_val_pred, average="macro"))
                
            return np.mean(fold_scores)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    start_opt = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    opt_duration = time.time() - start_opt
    
    print(f"\nOptimization completed in {round(opt_duration, 2)}s across {len(study.trials)} trials.", flush=True)
    print(f"Best Trial #{study.best_trial.number}: Mean CV Macro F1 = {round(study.best_value, 4)}", flush=True)
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    # Save best parameters
    with open(os.path.join(results_dir, "best_hyperparameters.json"), "w") as f:
        json.dump({
            "model_type": model_type,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": n_trials,
            "optimization_time_s": round(opt_duration, 2)
        }, f, indent=2)
        
    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(results_dir, "optuna_trials.csv"), index=False)
    
    # 1. Plot Optimization History
    plt.figure(figsize=(8, 4.5))
    trial_numbers = [t.number for t in study.trials]
    trial_values = [t.value for t in study.trials]
    best_so_far = np.maximum.accumulate(trial_values)
    plt.plot(trial_numbers, trial_values, 'o-', color='#3498DB', alpha=0.7, label='Trial Macro F1')
    plt.plot(trial_numbers, best_so_far, 'r--', linewidth=2, label='Best Macro F1 So Far')
    plt.title(f"Optuna Bayesian Optimization History ({model_type})")
    plt.xlabel("Trial Number")
    plt.ylabel("Validation Macro F1 (3-Fold CV)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "optuna_optimization_history.png"), dpi=200)
    plt.close()
    
    # 2. Plot Parameter Importances
    try:
        importances = optuna.importance.get_param_importances(study)
        plt.figure(figsize=(7, 4))
        params = list(importances.keys())
        scores = list(importances.values())
        y_pos = np.arange(len(params))
        plt.barh(y_pos, scores, align='center', color='#9B59B6')
        plt.yticks(y_pos, params)
        plt.gca().invert_yaxis()
        plt.xlabel("Relative Importance")
        plt.title(f"Hyperparameter Importance ({model_type})")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "optuna_param_importances.png"), dpi=200)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not plot parameter importances: {e}")
        
    # Retrain Best Model on full training dataset
    print("\nRetraining tuned model on full training set (215,716 samples)...", flush=True)
    if model_type == "random_forest":
        tuned_model = RandomForestClassifier(
            **study.best_params,
            class_weight="balanced_subsample",
            n_jobs=4,
            random_state=42
        )
    else:
        tuned_model = HistGradientBoostingClassifier(
            **study.best_params,
            class_weight="balanced",
            random_state=42
        )
        
    t0_fit = time.time()
    tuned_model.fit(X_train, y_train)
    fit_time = time.time() - t0_fit
    
    t0_inf = time.time()
    test_probs = tuned_model.predict_proba(X_test)[:, 1]
    inf_time = time.time() - t0_inf
    test_preds = (test_probs >= 0.5).astype(int)
    
    test_eval_df = test_df.copy()
    test_eval_df['prob_pos'] = test_probs
    test_eval_df['pred_label'] = test_preds
    
    tuned_cand_metrics = compute_classification_metrics(y_test, test_preds, test_probs)
    tuned_top1_metrics = compute_observation_top1_accuracy(test_eval_df)
    tuned_route_continuity = evaluate_route_continuity_for_preds(test_eval_df, raw_dir)
    
    # Save tuned model
    joblib.dump(tuned_model, os.path.join(models_dir, "tuned_best_model.joblib"))
    print(f"Saved tuned model to {os.path.join(models_dir, 'tuned_best_model.joblib')}", flush=True)
    
    # Load untuned model baseline for comparison
    untuned_path = os.path.join(models_dir, f"{model_type.lower().replace('-', '_')}.joblib")
    untuned_row = None
    if os.path.exists(untuned_path):
        untuned_model = joblib.load(untuned_path)
        u_t0_inf = time.time()
        u_probs = untuned_model.predict_proba(X_test)[:, 1]
        u_inf_time = time.time() - u_t0_inf
        u_preds = (u_probs >= 0.5).astype(int)
        
        u_test_df = test_df.copy()
        u_test_df['prob_pos'] = u_probs
        u_metrics = compute_classification_metrics(y_test, u_preds, u_probs)
        u_top1 = compute_observation_top1_accuracy(u_test_df)
        u_continuity = evaluate_route_continuity_for_preds(u_test_df, raw_dir)
        
        untuned_row = {
            "Configuration": "Untuned Baseline",
            "Student-Test Top-1 Accuracy": f"{u_top1['top1_accuracy_pct']}%",
            "Route Continuity": f"{round(u_continuity, 2)}%",
            "Candidate Macro F1": round(u_metrics['macro_f1'], 4),
            "Log Loss": round(u_metrics['log_loss'], 4) if u_metrics['log_loss'] is not None else "N/A",
            "Inference Latency (s)": round(u_inf_time, 3)
        }
        
    tuned_row = {
        "Configuration": "Optuna Tuned Model",
        "Student-Test Top-1 Accuracy": f"{tuned_top1_metrics['top1_accuracy_pct']}%",
        "Route Continuity": f"{round(tuned_route_continuity, 2)}%",
        "Candidate Macro F1": round(tuned_cand_metrics['macro_f1'], 4),
        "Log Loss": round(tuned_cand_metrics['log_loss'], 4) if tuned_cand_metrics['log_loss'] is not None else "N/A",
        "Inference Latency (s)": round(inf_time, 3)
    }
    
    comp_list = [untuned_row, tuned_row] if untuned_row else [tuned_row]
    tuning_comp_df = pd.DataFrame(comp_list)
    tuning_comp_df.to_csv(os.path.join(results_dir, "bayesian_tuning_comparison.csv"), index=False)
    
    print("\n" + "="*50, flush=True)
    print("HYPERPARAMETER OPTIMIZATION COMPARISON", flush=True)
    print("="*50, flush=True)
    print(tuning_comp_df.to_string(index=False), flush=True)
    
    return study.best_params

if __name__ == "__main__":
    run_bayesian_optimization(n_trials=12, model_type="random_forest")
