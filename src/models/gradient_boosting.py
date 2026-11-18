"""
PathSnap — Boosting Model Family
Histogram-based Gradient Boosting Classifier with balanced class weights.
"""

from sklearn.ensemble import HistGradientBoostingClassifier

def create_gradient_boosting(learning_rate: float = 0.1, max_iter: int = 100, max_depth: int = 8, random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_depth=max_depth,
        class_weight='balanced',
        l2_regularization=1.0,
        random_state=random_state
    )
