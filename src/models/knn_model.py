"""
PathSnap — Instance-Based Model Family
Distance-weighted K-Nearest Neighbors Classifier with feature scaling.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_knn(n_neighbors: int = 9, metric: str = 'minkowski') -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',
            metric=metric,
            n_jobs=-1
        ))
    ])
