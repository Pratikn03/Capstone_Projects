"""Anomaly detection: detection."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class MultivariateAnomalyDetector:
    """
    Layer 3: Isolation Forest for multivariate anomaly detection.
    
    Detects physically unlikely combinations (e.g., high load but low temperature,
    or high solar generation at night) by learning the joint distribution of features.
    """
    def __init__(self, contamination: float = 0.01, n_estimators: int = 100, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.feature_cols: List[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: List[str]) -> "MultivariateAnomalyDetector":
        """
        Fit the Isolation Forest on historical data.
        
        Args:
            df: DataFrame containing training data.
            feature_cols: List of column names to use for detection (e.g. load, temp, wind_speed).
        """
        self.feature_cols = feature_cols
        # Fill NaNs with 0 or handle them before calling fit.
        X = df[self.feature_cols].fillna(0).values
        self.model.fit(X)
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in new data.
        
        Returns:
            DataFrame with original index and new columns:
            - 'anomaly_score': Mean anomaly score (lower is more anomalous).
            - 'is_anomaly': Boolean flag (True if anomalous).
        """
        if not self.feature_cols:
            raise ValueError("Model has not been fitted yet.")

        X = df[self.feature_cols].fillna(0).values
        
        # decision_function: Average anomaly score of X of the base classifiers.
        # The lower, the more abnormal.
        scores = self.model.decision_function(X)
        
        # predict: -1 for outliers and 1 for inliers.
        labels = self.model.predict(X) 
        
        result = df.copy()
        result["anomaly_score"] = scores
        result["is_anomaly"] = (labels == -1)
        
        return result

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "MultivariateAnomalyDetector":
        with open(path, "rb") as f:
            return pickle.load(f)
