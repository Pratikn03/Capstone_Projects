"""Anomaly detection: train."""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd

from gridpulse.anomaly.detection import MultivariateAnomalyDetector

# Default features if config is missing
DEFAULT_FEATURES = [
    "load_mw",
    "wind_mw", 
    "solar_mw",
    "hour",
    "dayofweek"
]

def main():
    # Key: flag anomalies from residuals or isolation forest signals
    parser = argparse.ArgumentParser(description="Train Isolation Forest for Anomaly Detection")
    parser.add_argument("--train-data", type=str, default="data/processed/splits/train.parquet", help="Path to training data parquet")
    parser.add_argument("--config", type=str, default="configs/anomaly.yaml", help="Path to configuration YAML")
    parser.add_argument("--out", type=str, default="artifacts/models/anomaly_detector.pkl", help="Path to save trained model")
    args = parser.parse_args()

    train_path = Path(args.train_data)
    if not train_path.exists():
        # Fallback to full features if splits don't exist
        fallback = Path("data/processed/features.parquet")
        if fallback.exists():
            print(f"Training data {train_path} not found. Falling back to {fallback}")
            train_path = fallback
        else:
            raise FileNotFoundError(f"Training data not found at {train_path}")

    print(f"Loading data from {train_path}...")
    df = pd.read_parquet(train_path)

    # Load config
    config_path = Path(args.config)
    params = {}
    features = DEFAULT_FEATURES
    
    if config_path.exists():
        print(f"Loading config from {config_path}...")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
            if "isolation_forest" in cfg:
                params = cfg["isolation_forest"]
            if "features" in cfg:
                features = cfg["features"]
    else:
        print("Config file not found. Using default parameters.")

    # Validate features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"The following features are missing from the dataset: {missing}")

    print(f"Training Isolation Forest with features: {features}")
    print(f"Model parameters: {params}")

    detector = MultivariateAnomalyDetector(
        contamination=params.get("contamination", 0.01),
        n_estimators=params.get("n_estimators", 100),
        random_state=params.get("random_state", 42)
    )

    detector.fit(df, feature_cols=features)

    out_path = Path(args.out)
    print(f"Saving model to {out_path}...")
    detector.save(out_path)

    # Quick sanity check
    results = detector.detect(df)
    n_anomalies = results["is_anomaly"].sum()
    pct = (n_anomalies / len(df)) * 100
    print(f"Sanity check on training data: {n_anomalies} anomalies detected ({pct:.2f}%)")

if __name__ == "__main__":
    main()
