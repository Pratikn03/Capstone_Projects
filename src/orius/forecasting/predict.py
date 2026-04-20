"""Forecasting: prediction helpers for trained model bundles."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

from orius.utils.scaler import StandardScaler

# ── Lazy torch / DL imports ──
_torch = None
_LSTMForecaster = None
_NBEATSForecaster = None
_PatchTSTForecaster = None
_TFTForecaster = None
_TCNForecaster = None

def _ensure_torch():
    global _torch, _LSTMForecaster, _NBEATSForecaster, _PatchTSTForecaster, _TFTForecaster, _TCNForecaster
    if _torch is not None:
        return
    import torch as _t
    from orius.forecasting.dl_lstm import LSTMForecaster
    from orius.forecasting.dl_nbeats import NBEATSForecaster
    from orius.forecasting.dl_patchtst import PatchTSTForecaster
    from orius.forecasting.dl_tft import TFTForecaster
    from orius.forecasting.dl_tcn import TCNForecaster
    _torch = _t
    _LSTMForecaster = LSTMForecaster
    _NBEATSForecaster = NBEATSForecaster
    _PatchTSTForecaster = PatchTSTForecaster
    _TFTForecaster = TFTForecaster
    _TCNForecaster = TCNForecaster


def _normalize_quantile_levels(quantiles: Any) -> list[float]:
    if isinstance(quantiles, list):
        return [float(q) for q in quantiles]
    return []


def _median_quantile_index(quantile_levels: list[float]) -> int:
    if not quantile_levels:
        return 0
    arr = np.asarray(quantile_levels, dtype=float)
    return int(np.argmin(np.abs(arr - 0.5)))


def _extract_prediction_outputs(
    pred_seq: np.ndarray,
    *,
    horizon: int,
    quantile_levels: list[float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    arr = np.asarray(pred_seq, dtype=float)
    n_quantiles = len(quantile_levels)
    if n_quantiles > 1:
        if arr.ndim == 2:
            if arr.shape[1] % n_quantiles != 0:
                raise ValueError(f"Cannot reshape prediction array with shape {arr.shape} into quantiles")
            arr = arr.reshape(arr.shape[0], -1, n_quantiles)
        elif arr.ndim != 3:
            raise ValueError(f"Expected 2D/3D quantile prediction array, got {arr.ndim}D")
        if arr.shape[-1] != n_quantiles:
            if arr.shape[1] == n_quantiles:
                arr = np.transpose(arr, (0, 2, 1))
            else:
                raise ValueError(f"Quantile prediction array shape {arr.shape} does not match {n_quantiles} quantiles")
        arr = np.sort(arr[:, -horizon:, :], axis=-1)
        median_idx = _median_quantile_index(quantile_levels)
        point = arr[:, :, median_idx]
        quantile_map = {str(q): arr[:, :, idx] for idx, q in enumerate(quantile_levels)}
        return point, quantile_map

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    point = arr[:, -horizon:]
    return point, {}


def load_model_bundle(path: str | Path) -> Dict[str, Any]:
    """Load a serialized model bundle (GBM pickle or Torch checkpoint)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".pkl":
        import pickle
        with open(path, "rb") as f:
            bundle = pickle.load(f)
    elif path.suffix == ".pt":
        _ensure_torch()
        bundle = _torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")
    bundle["_path"] = str(path)
    return bundle


def _build_torch_model(bundle: Dict[str, Any]):
    """Instantiate a Torch model from a saved bundle."""
    _ensure_torch()
    model_type = bundle.get("model_type")
    feat_cols = bundle.get("feature_cols", [])
    params = bundle.get("model_params", {})
    quantile_levels = _normalize_quantile_levels(bundle.get("quantile_levels"))
    n_quantiles = max(1, int(bundle.get("n_quantiles", len(quantile_levels) or 1)))
    if model_type == "lstm":
        model = _LSTMForecaster(
            n_features=len(feat_cols),
            hidden_size=int(params.get("hidden_size", 128)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.1)),
            horizon=int(bundle.get("horizon", params.get("horizon", 24))),
            bidirectional=bool(params.get("bidirectional", False)),
            attention=bool(params.get("attention", False)),
            residual=bool(params.get("residual", False)),
            n_quantiles=n_quantiles,
        )
    elif model_type == "tcn":
        model = _TCNForecaster(
            n_features=len(feat_cols),
            num_channels=list(params.get("num_channels", [32, 32, 32])),
            kernel_size=int(params.get("kernel_size", 3)),
            dropout=float(params.get("dropout", 0.1)),
            horizon=int(bundle.get("horizon", params.get("horizon", 24))),
            dilation_base=int(params.get("dilation_base", 2)),
            use_skip_connections=bool(params.get("use_skip_connections", False)),
            n_quantiles=n_quantiles,
        )
    elif model_type == "nbeats":
        model = _NBEATSForecaster(
            n_features=len(feat_cols),
            lookback=int(bundle.get("lookback", params.get("lookback", 168))),
            horizon=int(bundle.get("horizon", params.get("horizon", 24))),
            hidden_dim=int(params.get("hidden_dim", 512)),
            num_blocks=int(params.get("num_blocks", 4)),
            dropout=float(params.get("dropout", 0.1)),
            n_quantiles=n_quantiles,
        )
    elif model_type == "tft":
        model = _TFTForecaster(
            n_features=len(feat_cols),
            horizon=int(bundle.get("horizon", params.get("horizon", 24))),
            d_model=int(params.get("d_model", 128)),
            n_heads=int(params.get("n_heads", 4)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.1)),
            dim_feedforward=int(params.get("dim_feedforward", 256)),
            n_quantiles=n_quantiles,
        )
    elif model_type == "patchtst":
        model = _PatchTSTForecaster(
            n_features=len(feat_cols),
            lookback=int(bundle.get("lookback", params.get("lookback", 168))),
            horizon=int(bundle.get("horizon", params.get("horizon", 24))),
            patch_len=int(params.get("patch_len", 24)),
            stride=int(params.get("stride", 12)),
            d_model=int(params.get("d_model", 128)),
            n_heads=int(params.get("n_heads", 4)),
            num_layers=int(params.get("num_layers", 2)),
            dropout=float(params.get("dropout", 0.1)),
            dim_feedforward=int(params.get("dim_feedforward", 256)),
            n_quantiles=n_quantiles,
        )
    else:
        raise ValueError(f"Unknown torch model_type: {model_type}")

    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def _maybe_scale_X(X: np.ndarray, bundle: Dict[str, Any]) -> np.ndarray:
    """Apply the saved feature scaler when present."""
    scaler = StandardScaler.from_dict(bundle.get("x_scaler"))
    return scaler.transform(X) if scaler is not None else X


def _maybe_inverse_y(y: np.ndarray, bundle: Dict[str, Any]) -> np.ndarray:
    """Undo target scaling when a scaler is present."""
    scaler = StandardScaler.from_dict(bundle.get("y_scaler"))
    if scaler is None:
        return y
    return scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)


def predict_next_24h(features_df: pd.DataFrame, model_bundle: Dict[str, Any], horizon: int | None = None) -> Dict[str, Any]:
    """Generate the next-horizon forecast and (optional) quantiles."""
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
        ensemble_models = model_bundle.get("ensemble_models")
        if ensemble_models:
            preds = [m.predict(X[-horizon:]) for m in ensemble_models]
            pred = np.mean(np.vstack(preds), axis=0)
        else:
            pred = model_bundle["model"].predict(X[-horizon:])
    elif model_type in {"lstm", "tcn", "nbeats", "tft", "patchtst"}:
        lookback = int(model_bundle.get("lookback", 168))
        X = df[feat_cols].to_numpy()
        if len(X) < lookback:
            raise ValueError("Not enough rows in features_df for sequence lookback")
        X = _maybe_scale_X(X, model_bundle)
        model = _build_torch_model(model_bundle)
        quantile_levels = _normalize_quantile_levels(model_bundle.get("quantile_levels"))
        with _torch.no_grad():
            xb = _torch.from_numpy(X[-lookback:].astype(np.float32)).unsqueeze(0)
            pred_seq = model(xb).numpy()
            pred_arr, quantile_arrs = _extract_prediction_outputs(
                pred_seq,
                horizon=horizon,
                quantile_levels=quantile_levels,
            )
            pred = pred_arr[0]
        pred = _maybe_inverse_y(pred, model_bundle)
        if quantile_arrs:
            quantile_preds = {
                str(q): _maybe_inverse_y(values[0], model_bundle).tolist()
                for q, values in quantile_arrs.items()
            }
        else:
            quantile_preds = {}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Prefer explicit quantile models when available; otherwise use residual offsets.
    if model_type == "gbm":
        quantile_preds = {}
        quantile_models = model_bundle.get("quantile_models")
        if quantile_models:
            for q, q_model in quantile_models.items():
                q_pred = q_model.predict(X[-horizon:])
                quantile_preds[str(q)] = q_pred.tolist()
        else:
            quantiles = model_bundle.get("residual_quantiles", {})
            quantile_preds = {str(q): (pred + float(delta)).tolist() for q, delta in quantiles.items()}
    elif not quantile_preds:
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
