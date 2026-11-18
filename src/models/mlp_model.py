"""
PathSnap — Neural Network Model Family
Multi-Layer Perceptron (MLP) Classifier with feature scaling and Adam optimizer.
"""

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def create_mlp(hidden_layer_sizes=(64, 32), alpha: float = 1e-4, max_iter: int = 60, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            batch_size=256,
            learning_rate_init=0.005,
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=max_iter,
            random_state=random_state
        ))
    ])
