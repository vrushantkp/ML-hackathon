"""
PathSnap — Ensemble / Bagging Model Family
Random Forest Classifier with bootstrap aggregation and balanced sub-sampling.
"""

from sklearn.ensemble import RandomForestClassifier

def create_random_forest(n_estimators: int = 100, max_depth: int = 12, random_state: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=random_state
    )
