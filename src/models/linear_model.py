"""
PathSnap — Linear / Probabilistic Model Family
Regularized Logistic Regression with feature scaling and class weighting.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_logistic_regression(C: float = 1.0, max_iter: int = 500, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=C,
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced',
            max_iter=max_iter,
            random_state=random_state
        ))
    ])
