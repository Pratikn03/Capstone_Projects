"""
Forecasting: Walk-Forward Backtesting for Multi-Horizon Evaluation.

This module implements walk-forward backtesting, which simulates how a
forecasting model would perform in production by evaluating predictions
at each horizon step.

Why Walk-Forward Backtesting?
    Traditional train/test splits give a single aggregate metric. Walk-forward
    evaluation shows how error grows with forecast horizon - crucial for:
    
    - Understanding model degradation over time
    - Setting horizon-specific confidence intervals
    - Deciding when to trigger re-forecasting

Example:
    For a 24-hour forecast horizon:
    - Step 1 (h=1): Next hour error (typically lowest)
    - Step 12 (h=12): Half-day ahead error
    - Step 24 (h=24): Day-ahead error (typically highest)

Usage:
    >>> from gridpulse.forecasting.backtest import walk_forward_horizon_metrics
    >>> metrics = walk_forward_horizon_metrics(y_true, y_pred, horizon=24, target='load_mw')
    >>> print(metrics['per_horizon']['1']['rmse'])  # Hour 1 RMSE
    >>> print(metrics['per_horizon']['24']['rmse']) # Hour 24 RMSE
"""
from __future__ import annotations

import numpy as np

from gridpulse.utils.metrics import rmse, mae, mape, smape, daylight_mape


def walk_forward_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizon: int, target: str) -> dict:
    """
    Compute per-step metrics for walk-forward forecast evaluation.
    
    This function breaks down forecast performance by horizon step, showing
    how prediction accuracy degrades as we forecast further ahead.
    
    Args:
        y_true: Actual values (flattened across all windows)
        y_pred: Predicted values (same shape as y_true)
        horizon: Forecast horizon length (e.g., 24 for day-ahead)
        target: Target variable name (used for special metrics like daylight_mape)
        
    Returns:
        Dictionary with:
        - per_horizon: Dict mapping step '1', '2', ... to metric dicts
        - horizon: The horizon value for reference
        
    Example:
        >>> metrics = walk_forward_horizon_metrics(y, y_hat, 24, 'load_mw')
        >>> metrics['per_horizon']['1']  # {'rmse': 5.2, 'mae': 4.1, ...}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    # For each step in the horizon, evaluate that step across the series.
    per_step = {}
    for h in range(horizon):
        idx = np.arange(h, n, horizon)
        yt = y_true[idx]
        yp = y_pred[idx]
        if len(yt) == 0:
            continue
        metrics = {
            "rmse": rmse(yt, yp),
            "mae": mae(yt, yp),
            "mape": mape(yt, yp),
            "smape": smape(yt, yp),
        }
        if target == "solar_mw":
            metrics["daylight_mape"] = daylight_mape(yt, yp)
        per_step[str(h + 1)] = metrics  # 1-indexed horizon step

    return {"per_horizon": per_step, "horizon": horizon}


def _mean_metric(per_step: dict, key: str) -> float | None:
    """Compute the mean metric across all horizon steps."""
    vals = [m.get(key) for m in per_step.values() if m.get(key) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def multi_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizons: list[int], target: str) -> dict:
    """Compute aggregate metrics across multiple horizon lengths."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    results: dict[str, dict] = {}
    for horizon in horizons:
        if horizon <= 0 or n < horizon:
            continue
        per = walk_forward_horizon_metrics(y_true, y_pred, horizon, target).get("per_horizon", {})
        summary = {
            "rmse": _mean_metric(per, "rmse"),
            "mae": _mean_metric(per, "mae"),
            "mape": _mean_metric(per, "mape"),
            "smape": _mean_metric(per, "smape"),
        }
        if target == "solar_mw":
            summary["daylight_mape"] = _mean_metric(per, "daylight_mape")
        results[str(horizon)] = {"summary": summary, "per_horizon": per}

    return {"results": results}
