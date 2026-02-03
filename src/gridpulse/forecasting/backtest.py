from __future__ import annotations

import numpy as np

from gridpulse.utils.metrics import rmse, mae, mape, smape, daylight_mape


def walk_forward_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizon: int, target: str) -> dict:
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


def _mean_metric(per_step: dict, key: str) -> float | None:
    vals = [m.get(key) for m in per_step.values() if m.get(key) is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def multi_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray, horizons: list[int], target: str) -> dict:
    """
    Compute aggregate metrics across multiple horizons.
    Expects y_true/y_pred to be 1D arrays aligned in time.
    """
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
