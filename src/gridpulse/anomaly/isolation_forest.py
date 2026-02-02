"""Anomaly detection: isolation forest."""
from __future__ import annotations
from sklearn.ensemble import IsolationForest

def fit_iforest(X, contamination: float = 0.01, random_state: int = 42):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    return model

def predict_iforest(model, X):
    # -1 anomaly, 1 normal
    return model.predict(X)
