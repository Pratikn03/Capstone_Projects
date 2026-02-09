"""Forecasting evaluation: time-series cross-validation and metrics."""
from __future__ import annotations

from typing import Any, Callable
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from gridpulse.utils.metrics import rmse, mae, mape, smape, daylight_mape, r2_score


def time_series_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    train_fn: Callable[[np.ndarray, np.ndarray], Any],
    predict_fn: Callable[[Any, np.ndarray], np.ndarray],
    n_splits: int = 5,
    target: str = "load_mw",
) -> dict:
    """
    Perform forward-chaining time-series cross-validation.
    
    Args:
        X: Features array (n_samples, n_features)
        y: Target array (n_samples,)
        train_fn: Function that trains a model given (X_train, y_train) and returns model
        predict_fn: Function that predicts given (model, X_val) and returns predictions
        n_splits: Number of forward-chaining folds
        target: Target name for specialized metrics (e.g., solar/wind daylight_mape)
    
    Returns:
        Dictionary with per-fold metrics and aggregated statistics
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_metrics = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model on this fold
        model = train_fn(X_train, y_train)
        
        # Predict on validation fold
        y_pred = predict_fn(model, X_val)
        
        # Compute metrics
        metrics = {
            "fold": fold_idx,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "rmse": rmse(y_val, y_pred),
            "mae": mae(y_val, y_pred),
            "mape": mape(y_val, y_pred),
            "smape": smape(y_val, y_pred),
            "r2": r2_score(y_val, y_pred),
        }
        
        # Add specialized metrics for renewables
        if target in ("solar_mw", "wind_mw"):
            metrics["daylight_mape"] = daylight_mape(y_val, y_pred)
        
        fold_metrics.append(metrics)
    
    # Aggregate statistics across folds
    metric_names = ["rmse", "mae", "mape", "smape", "r2"]
    if target in ("solar_mw", "wind_mw"):
        metric_names.append("daylight_mape")
    
    aggregated = {}
    for metric_name in metric_names:
        values = [f[metric_name] for f in fold_metrics]
        aggregated[f"{metric_name}_mean"] = float(np.mean(values))
        aggregated[f"{metric_name}_std"] = float(np.std(values))
        aggregated[f"{metric_name}_min"] = float(np.min(values))
        aggregated[f"{metric_name}_max"] = float(np.max(values))
    
    return {
        "n_splits": n_splits,
        "target": target,
        "fold_metrics": fold_metrics,
        "aggregated": aggregated,
    }


def multi_horizon_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    train_fn: Callable,
    predict_fn: Callable,
    horizon: int,
    n_splits: int = 5,
    target: str = "load_mw",
) -> dict:
    """
    Time-series CV with per-horizon metrics decomposition.
    
    Assumes y_pred is sequential forecasts that can be grouped by horizon step.
    
    Args:
        X: Features
        y: True values (sequential)
        train_fn: Training function
        predict_fn: Prediction function
        horizon: Forecast horizon length
        n_splits: Number of CV folds
        target: Target name
    
    Returns:
        CV results with per-horizon breakdown
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = train_fn(X_train, y_train)
        y_pred = predict_fn(model, X_val)
        
        # Overall fold metrics
        overall = {
            "rmse": rmse(y_val, y_pred),
            "mae": mae(y_val, y_pred),
            "r2": r2_score(y_val, y_pred),
        }
        
        # Per-horizon step metrics
        n = min(len(y_val), len(y_pred))
        per_horizon = {}
        for h in range(horizon):
            idx = np.arange(h, n, horizon)
            if len(idx) == 0:
                continue
            yt = y_val[idx]
            yp = y_pred[idx]
            per_horizon[f"h{h+1}"] = {
                "rmse": rmse(yt, yp),
                "mae": mae(yt, yp),
                "r2": r2_score(yt, yp),
            }
        
        fold_results.append({
            "fold": fold_idx,
            "overall": overall,
            "per_horizon": per_horizon,
        })
    
    return {
        "n_splits": n_splits,
        "horizon": horizon,
        "target": target,
        "fold_results": fold_results,
    }


def evaluate_model_cv(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    train_fn: Callable,
    predict_fn: Callable,
    n_splits: int = 5,
    target: str = "load_mw",
    horizon: int | None = None,
) -> dict:
    """
    Comprehensive CV evaluation wrapper.
    
    Returns both standard CV metrics and optional per-horizon decomposition.
    """
    # Standard time-series CV
    cv_results = time_series_cv_score(
        X, y, train_fn, predict_fn, n_splits=n_splits, target=target
    )
    
    results = {
        "model_type": model_type,
        "cv_standard": cv_results,
    }
    
    # Optional: per-horizon CV (if horizon specified)
    if horizon is not None and horizon > 1:
        results["cv_multi_horizon"] = multi_horizon_cv_score(
            X, y, train_fn, predict_fn, horizon=horizon, n_splits=n_splits, target=target
        )
    
    return results
