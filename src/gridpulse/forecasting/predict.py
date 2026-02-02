"""Forecasting: predict."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch

from gridpulse.forecasting.dl_lstm import LSTMForecaster
from gridpulse.forecasting.dl_tcn import TCNForecaster
from gridpulse.utils.scaler import StandardScaler


def load_model_bundle(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".pkl":
        import pickle
        with open(path, "rb") as f:
            bundle = pickle.load(f)
    elif path.suffix == ".pt":
        bundle = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")
    bundle["_path"] = str(path)
    return bundle


def _build_torch_model(bundle: Dict[str, Any]):
    model_type = bundle.get("model_type")
    feat_cols = bundle.get("feature_cols", [])
    params = bundle.get("model_params", {})
    if model_type == "lstm":
        model = LSTMForecaster(
            n_features=len(feat_cols),
            hidden_size=int(params.get("hidden_size", 128)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.1)),
            horizon=int(bundle.get("horizon", params.get("horizon", 24))),
        )
    elif model_type == "tcn":
        model = TCNForecaster(
            n_features=len(feat_cols),
            num_channels=list(params.get("num_channels", [32, 32, 32])),
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unknown torch model_type: {model_type}")

    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def _maybe_scale_X(X: np.ndarray, bundle: Dict[str, Any]) -> np.ndarray:
    scaler = StandardScaler.from_dict(bundle.get("x_scaler"))
    return scaler.transform(X) if scaler is not None else X


def _maybe_inverse_y(y: np.ndarray, bundle: Dict[str, Any]) -> np.ndarray:
    scaler = StandardScaler.from_dict(bundle.get("y_scaler"))
    if scaler is None:
        return y
    return scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)


def predict_next_24h(features_df: pd.DataFrame, model_bundle: Dict[str, Any], horizon: int | None = None) -> Dict[str, Any]:
    df = features_df.sort_values("timestamp").reset_index(drop=True)
    feat_cols = model_bundle.get("feature_cols", [])
    if not feat_cols:
        raise ValueError("model_bundle missing feature_cols")
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"features_df missing columns: {missing[:5]}" + (" ..." if len(missing) > 5 else ""))

    model_type = model_bundle.get("model_type")
    horizon = int(horizon or model_bundle.get("horizon", 24))

    last_ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True, errors="coerce")
    if pd.isna(last_ts):
        raise ValueError("Invalid timestamp in features_df")
    timestamps = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="h", tz=last_ts.tz)

    if model_type == "gbm":
        X = df[feat_cols].to_numpy()
        if len(X) < horizon:
            raise ValueError("Not enough rows in features_df for GBM horizon")
        pred = model_bundle["model"].predict(X[-horizon:])
    elif model_type in {"lstm", "tcn"}:
        lookback = int(model_bundle.get("lookback", 168))
        X = df[feat_cols].to_numpy()
        if len(X) < lookback:
            raise ValueError("Not enough rows in features_df for sequence lookback")
        X = _maybe_scale_X(X, model_bundle)
        model = _build_torch_model(model_bundle)
        with torch.no_grad():
            xb = torch.from_numpy(X[-lookback:].astype(np.float32)).unsqueeze(0)
            pred_seq = model(xb).numpy()[0]
            pred = pred_seq[-horizon:]
        pred = _maybe_inverse_y(pred, model_bundle)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    quantiles = model_bundle.get("residual_quantiles", {})
    quantile_preds = {str(q): (pred + float(delta)).tolist() for q, delta in quantiles.items()}

    return {
        "target": model_bundle.get("target"),
        "model": model_type,
        "timestamp": [t.isoformat() for t in timestamps],
        "forecast": pred.tolist(),
        "quantiles": quantile_preds,
    }


def predict_multi_target(features_df: pd.DataFrame, bundles: Dict[str, Dict[str, Any]], horizon: int | None = None) -> Dict[str, Any]:
    out = {}
    for target, bundle in bundles.items():
        out[target] = predict_next_24h(features_df, bundle, horizon=horizon)
    return out
