"""Forecasting: backtest."""
from __future__ import annotations

import numpy as np

from gridpulse.utils.metrics import rmse, mae, mape, smape, daylight_mape


def walk_forward_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizon: int, target: str) -> dict:
    # Key: prepare features/targets and train or evaluate models
    """
    Compute per-horizon step metrics for walk-forward style evaluation.
    y_true and y_pred are 1D arrays aligned in time.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]

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


def multi_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizons: list[int], target: str) -> dict:
    """Compute walk-forward metrics for multiple horizons and summarize mean per metric."""
    out = {}
    for h in horizons:
        per = walk_forward_horizon_metrics(y_true, y_pred, horizon=h, target=target)
        per_h = per.get("per_horizon", {})
        summary = {}
        if per_h:
            metric_keys = list(next(iter(per_h.values())).keys())
            for k in metric_keys:
                vals = [v.get(k) for v in per_h.values() if v.get(k) is not None]
                summary[k] = float(np.mean(vals)) if vals else None
        out[str(h)] = {"summary": summary, "per_horizon": per_h}
    return {"horizons": horizons, "results": out}
