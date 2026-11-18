"""
PathSnap — Build & Execute Jupyter Notebook
Generates and executes experiments/model_evaluation_and_benchmarks.ipynb
with full code outputs, model score tables, and performance plots.
"""

import os
import sys
import nbformat as nbf
from nbclient import NotebookClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(BASE_DIR, "experiments", "model_evaluation_and_benchmarks.ipynb")

def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    }
    
    cells = []
    
    # Cell 1: Markdown Header
    cells.append(nbf.v4.new_markdown_cell("""# PathSnap — Model Evaluation, Benchmarking & Route Reconstruction
**UE24CS352A Machine Learning | Department of Computer Science & Engineering, PES University**

This notebook contains the authoritative empirical evaluation, cross-validation metrics, Optuna Bayesian hyperparameter tuning results, and end-to-end HMM Viterbi route reconstruction benchmarks across all trained model families."""))

    # Cell 2: Code Imports & Setup
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print("PathSnap environment loaded successfully!")
print(f"Working Directory: {BASE_DIR}")"""))

    # Cell 3: Markdown Section 1
    cells.append(nbf.v4.new_markdown_cell("""## 1. Model Registry & Artifact Inspection
Inspecting the trained models in `artifacts/models`. Notice that redundant duplicates have been eliminated, leaving exactly **1 model per algorithm family**:
1. **Ensemble / Bagging**: Random Forest (Bayesian Tuned)
2. **Boosting**: HistGradientBoosting (Causal Multi-Scale Advanced)
3. **Neural Network**: Multi-Layer Perceptron (64, 32)
4. **Instance-Based**: K-Nearest Neighbors ($k=9$)
5. **Linear / Probabilistic**: Logistic Regression ($L_2$)"""))

    # Cell 4: Code Model Registry
    cells.append(nbf.v4.new_code_cell("""models_dir = os.path.join(BASE_DIR, "artifacts", "models")
model_files = [f for f in os.listdir(models_dir) if f.endswith(".joblib")]

registry = []
for mf in sorted(model_files):
    fpath = os.path.join(models_dir, mf)
    size_kb = round(os.path.getsize(fpath) / 1024, 1)
    obj = joblib.load(fpath)
    obj_type = str(type(obj['estimator'])) if isinstance(obj, dict) else str(type(obj))
    registry.append({
        "Model File": mf,
        "File Size (KB)": size_kb,
        "Classifier Class": obj_type.split("'")[1] if "'" in obj_type else obj_type
    })

pd.DataFrame(registry)"""))

    # Cell 5: Markdown Section 2
    cells.append(nbf.v4.new_markdown_cell("""## 2. Master Model Benchmarking Table
Evaluating all models across:
* **5-Fold Cross Validation** (CV Macro F1 & Log Loss)
* **Student-Testing Holdout Partition** (Macro F1, Top-1 Point Accuracy, Route Continuity, Inference Latency)"""))

    # Cell 6: Code Benchmark Table
    cells.append(nbf.v4.new_code_cell("""results_dir = os.path.join(BASE_DIR, "artifacts", "results")
comp_df = pd.read_csv(os.path.join(results_dir, "model_comparison.csv"))
comp_df.sort_values(by="CV Macro F1", ascending=False)"""))

    # Cell 7: Markdown Section 3
    cells.append(nbf.v4.new_markdown_cell("""## 3. Optuna Bayesian Hyperparameter Optimization
A 12-trial optimization using Tree-structured Parzen Estimator (TPE) targeting candidate Macro F1."""))

    # Cell 8: Code Optuna
    cells.append(nbf.v4.new_code_cell("""optuna_csv = os.path.join(results_dir, "optuna_trials.csv")
if os.path.exists(optuna_csv):
    opt_df = pd.read_csv(optuna_csv)
    print("Top 5 Optimization Trials:")
    display(opt_df.sort_values(by="value", ascending=False).head(5))

best_params_path = os.path.join(results_dir, "best_hyperparameters.json")
if os.path.exists(best_params_path):
    with open(best_params_path, "r") as f:
        print('\\nOptimal Hyperparameters:', json.dumps(json.load(f), indent=2))"""))

    # Cell 9: Markdown Section 4
    cells.append(nbf.v4.new_markdown_cell("""## 4. End-to-End Route Reconstruction & HMM Viterbi Evaluation
Evaluating the complete reconstructed routes across the **10 student-testing evaluation trips** comparing:
* **Top-1 Observation Accuracy**
* **HMM Viterbi Observation Accuracy**
* **Topological Route Continuity** (Eliminating broken hops)
* **Arc Jaccard Overlap & Sequence Alignment**"""))

    # Cell 10: Code Reconstruction Summary
    cells.append(nbf.v4.new_code_cell("""recon_csv = os.path.join(results_dir, "route_reconstruction_summary.csv")
if os.path.exists(recon_csv):
    recon_df = pd.read_csv(recon_csv)
    display(recon_df)
    
    print('\\n--- 10-TRIP MACRO AVERAGES ---')
    print(f"Mean Classifier Top-1 Accuracy : {recon_df['Classifier Top-1 Acc (%)'].mean():.2f}%")
    print(f"Mean HMM Observation Accuracy  : {recon_df['HMM Obs Acc (%)'].mean():.2f}%")
    print(f"Mean Reconstructed Continuity  : {recon_df['HMM Continuity (%)'].mean():.2f}%")
    print(f"Mean Arc Jaccard Overlap       : {recon_df['Jaccard Overlap (%)'].mean():.2f}%")
    print(f"Mean Sequence Alignment        : {recon_df['Sequence Alignment (%)'].mean():.2f}%")
    print(f"Mean Arc Recall                : {recon_df['Arc Recall (%)'].mean():.2f}%")"""))

    # Cell 11: Markdown Section 5
    cells.append(nbf.v4.new_markdown_cell("""## 5. Performance Visualizations
Generating comparison plots for the technical report."""))

    # Cell 12: Code Plots
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(10, 4), dpi=120)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

plt.subplot(1, 2, 1)
models_plot = comp_df['Model'].apply(lambda x: x.split('(')[0].strip())
accs = comp_df['Top-1 Accuracy'].str.rstrip('%').astype(float)
colors = ['#00B4D8' if 'Hist' in m else ('#10B981' if 'Tuned' in m else '#94A3B8') for m in comp_df['Model']]
plt.barh(models_plot, accs, color=colors)
plt.xlabel("Top-1 Accuracy (%)", fontweight='bold')
plt.title("Top-1 Classifier Accuracy by Model", fontweight='bold')
plt.xlim(70, 95)

plt.subplot(1, 2, 2)
conts = comp_df['Route Continuity'].str.rstrip('%').astype(float)
plt.barh(models_plot, conts, color=colors)
plt.xlabel("Route Continuity (%)", fontweight='bold')
plt.title("Route Continuity by Model", fontweight='bold')
plt.xlim(80, 102)

plt.tight_layout()
plt.show()"""))

    # Cell 9: Markdown Section 5 - Evaluator Test Set
    cells.append(nbf.v4.new_markdown_cell("""## 5. Official Evaluator Test Suite (10 Unseen Trajectories)
Here we evaluate generalization on the 10 official unseen evaluation trips (`00000015` through `00000094`, comprising **19,674 raw GPS observations** across challenging multi-km urban routes).
This empirical test confirms whether our models overfit or generalize smoothly to entirely unseen road graphs."""))

    # Cell 10: Evaluator Test Set Summary
    cells.append(nbf.v4.new_code_cell("""eval_csv = os.path.join(BASE_DIR, "artifacts", "results", "evaluator_test_results.csv")
eval_df = pd.read_csv(eval_csv)
print(f"Total Evaluator Test Evaluation Records: {len(eval_df)}")

eval_summary = eval_df.groupby("model")[['top1_accuracy', 'hmm_obs_accuracy', 'sequence_match', 'continuity', 'jaccard', 'recall', 'latency_ms']].mean().round(2)
eval_summary.columns = ['Top-1 Acc (%)', 'HMM Snapped Acc (%)', 'Seq Match (%)', 'Continuity (%)', 'Jaccard (%)', 'Arc Recall (%)', 'Latency (ms)']
print("\\n=== MODEL PERFORMANCE SUMMARY ON 10 UNSEEN EVALUATOR TRIPS ===")
display(eval_summary)"""))

    # Cell 11: Trip-by-Trip Evaluator Breakdown
    cells.append(nbf.v4.new_code_cell("""# Trip-by-trip breakdown for Random Forest (Our Recommended Production Trajectory Model)
rf_eval_df = eval_df[eval_df['model'] == 'random_forest'][['trip_id', 'observations', 'top1_accuracy', 'hmm_obs_accuracy', 'sequence_match', 'continuity', 'recall', 'latency_ms']]
rf_eval_df.columns = ['Trip ID', 'Observations', 'Top-1 Acc (%)', 'HMM Acc (%)', 'Seq Match (%)', 'Continuity (%)', 'Arc Recall (%)', 'Latency (ms)']
print("Random Forest Detailed Breakdown across all 10 Evaluator Trips:")
display(rf_eval_df)"""))

    # Cell 12: Evaluator Benchmark Plots
    cells.append(nbf.v4.new_code_cell("""# Comparative Plots across 10 Unseen Trips
plt.figure(figsize=(13, 5))

plt.subplot(1, 2, 1)
plt.plot(rf_eval_df['Trip ID'].astype(str), rf_eval_df['Seq Match (%)'], marker='o', color='#10B981', linewidth=2, label='Random Forest (Seq Match)')
hgb_eval_df = eval_df[eval_df['model'] == 'histgradientboosting']
plt.plot(hgb_eval_df['trip_id'].astype(str), hgb_eval_df['sequence_match'], marker='s', color='#00B4D8', linewidth=2, label='HistGB (Seq Match)')
plt.xlabel("Trip ID", fontweight='bold')
plt.ylabel("Sequence Alignment Ratio (%)", fontweight='bold')
plt.title("Sequence Match across 10 Unseen Evaluator Trips", fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(rf_eval_df['Trip ID'].astype(str), rf_eval_df['Top-1 Acc (%)'], marker='o', color='#10B981', linewidth=2, label='Random Forest (Top-1)')
plt.plot(hgb_eval_df['trip_id'].astype(str), hgb_eval_df['top1_accuracy'], marker='s', color='#00B4D8', linewidth=2, label='HistGB (Top-1)')
plt.xlabel("Trip ID", fontweight='bold')
plt.ylabel("Top-1 Classifier Accuracy (%)", fontweight='bold')
plt.title("Top-1 Point Accuracy across 10 Unseen Evaluator Trips", fontweight='bold')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

    nb.cells = cells
    
    # Save notebook
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook written to {NB_PATH}")
    
    # Execute notebook
    print("Executing notebook with nbclient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved: {NB_PATH} ({os.path.getsize(NB_PATH)/1024:.1f} KB)")

if __name__ == "__main__":
    build_notebook()
