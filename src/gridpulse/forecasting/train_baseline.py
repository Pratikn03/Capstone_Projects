"""Forecasting: train baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gridpulse.utils.metrics import rmse, mape
from gridpulse.utils.seed import set_seed
from gridpulse.forecasting.baselines import persistence_24h, moving_average
from gridpulse.forecasting.ml_gbm import train_gbm, predict_gbm

TARGETS = ["load_mw", "wind_mw", "solar_mw"]

def make_xy(df: pd.DataFrame, target: str, drop_cols=("timestamp",)) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in df.columns if c not in drop_cols and c not in TARGETS]
    X = df[cols].to_numpy()
    y = df[target].to_numpy()
    return X, y, cols

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/processed/features.parquet")
    p.add_argument("--splits", default="data/processed/splits")
    p.add_argument("--out-dir", default="artifacts/backtests")
    p.add_argument("--target", default="load_mw", choices=TARGETS)
    args = p.parse_args()
    set_seed(42)

    features_path = Path(args.features)
    splits_dir = Path(args.splits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (splits_dir / "train.parquet").exists():
        train_df = pd.read_parquet(splits_dir / "train.parquet")
        val_df = pd.read_parquet(splits_dir / "val.parquet")
        test_df = pd.read_parquet(splits_dir / "test.parquet")
    else:
        # fallback: do internal split
        df = pd.read_parquet(features_path).sort_values("timestamp")
        n = len(df)
        train_df = df.iloc[: int(n * 0.7)]
        val_df = df.iloc[int(n * 0.7): int(n * 0.85)]
        test_df = df.iloc[int(n * 0.85):]

    # Baselines
    def eval_baseline(name: str, y_true: np.ndarray, y_pred: np.ndarray):
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        return {
            "name": name,
            "rmse": rmse(y_true[mask], y_pred[mask]),
            "mape": mape(y_true[mask], y_pred[mask]),
            "n": int(mask.sum()),
        }

    # persistence and moving average evaluated on test window
    y_test = test_df[args.target].to_numpy()
    pers = persistence_24h(test_df, args.target)
    ma = moving_average(test_df, args.target, 24)

    baseline_metrics = [
        eval_baseline("persistence_24h", y_test, pers),
        eval_baseline("moving_average_24h", y_test, ma),
    ]

    # GBM model
    X_train, y_train, feat_cols = make_xy(train_df, args.target)
    X_val, y_val, _ = make_xy(val_df, args.target)
    X_test, y_test, _ = make_xy(test_df, args.target)

    model_kind, model = train_gbm(
        X_train,
        y_train,
        params={"n_estimators": 500, "learning_rate": 0.05, "random_state": 42},
    )
    pred_test = predict_gbm(model, X_test)

    gbm_metrics = {
        "name": f"gbm_{model_kind}",
        "rmse": rmse(y_test, pred_test),
        "mape": mape(y_test, pred_test),
        "n": int(len(y_test)),
        "n_features": int(len(feat_cols)),
    }

    payload = {
        "target": args.target,
        "baselines": baseline_metrics,
        "gbm": gbm_metrics,
        "feature_columns": feat_cols[:50],  # preview
    }

    out_json = out_dir / f"week1_baseline_{args.target}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out_json}")

if __name__ == "__main__":
    main()
