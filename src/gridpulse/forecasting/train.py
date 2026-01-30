from __future__ import annotations

import argparse
import json
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

from gridpulse.forecasting.ml_gbm import train_gbm

def evaluate(model, X, y, split_name="val"):
    """Calculate MSE, RMSE, MAE for a given split."""
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mse)
    
    print(f"[{split_name.upper()}] RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    return {
        f"{split_name}_mse": float(mse),
        f"{split_name}_mae": float(mae),
        f"{split_name}_rmse": float(rmse)
    }

def main():
    parser = argparse.ArgumentParser(description="Train Forecasting Model (GBM)")
    parser.add_argument("--config", default="configs/train_forecast.yaml", help="Path to config")
    parser.add_argument("--train-path", default="data/processed/splits/train.parquet")
    parser.add_argument("--val-path", default="data/processed/splits/val.parquet")
    parser.add_argument("--out-dir", default="artifacts/models", help="Directory to save model")
    args = parser.parse_args()

    # 1. Load Config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    target = cfg.get("target", "load_mw")
    features = cfg.get("features", [])
    model_params = cfg.get("model_params", {})

    print(f"Target: {target}")
    print(f"Features ({len(features)}): {features}")

    # 2. Load Data
    print("Loading data...")
    train_df = pd.read_parquet(args.train_path)
    val_df = pd.read_parquet(args.val_path)

    # Check features exist
    missing = [f for f in features if f not in train_df.columns]
    if missing:
        raise ValueError(f"Missing features in training data: {missing}")

    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]

    # 3. Train
    print(f"Training model with params: {model_params}...")
    model_type, model = train_gbm(X_train, y_train, params=model_params)
    print(f"Trained using backend: {model_type}")

    # 4. Evaluate
    metrics = {}
    metrics.update(evaluate(model, X_train, y_train, "train"))
    metrics.update(evaluate(model, X_val, y_val, "val"))

    # 5. Save
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = out_dir / f"{target}_gbm.pkl"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    metrics_path = out_dir / f"{target}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    main()