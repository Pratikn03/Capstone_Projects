"""Monitoring: retraining decisions and drift checks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from orius.monitoring.data_drift import ks_drift
from orius.monitoring.model_drift import metric_drift
from orius.utils.metrics import rmse, mape


@dataclass
class RetrainingDecision:
    retrain: bool
    reasons: List[str]
    last_trained_days_ago: int | None


def load_monitoring_config(path: str | Path = "configs/monitoring.yaml") -> dict:
    """Load monitoring configuration with sensible defaults."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {
            "data_drift": {"p_value_threshold": 0.01},
            "model_drift": {"metric": "mape", "degradation_threshold": 0.15},
            "retraining": {"cadence_days": 30, "min_new_data_days": 14},
            "dc3s_health": {
                "enabled": True,
                "lookback_hours": 24,
                "min_commands": 50,
                "intervention_rate_threshold": 0.30,
                "low_reliability_w_threshold": 0.60,
                "low_reliability_rate_threshold": 0.25,
                "drift_flag_rate_threshold": 0.10,
                "inflation_p95_threshold": 2.0,
                "sustained_windows": 3,
            },
        }
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def days_since(path: Path) -> int | None:
    """Return days since a file was modified (or None if missing)."""
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).days


def compute_data_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: List[str], p_value_threshold: float) -> Dict[str, Any]:
    """Run KS drift tests per feature and summarize whether any drifted."""
    drifted = {}
    for col in feature_cols:
        ref = reference_df[col].dropna().to_numpy()
        cur = current_df[col].dropna().to_numpy()
        if len(ref) == 0 or len(cur) == 0:
            continue
        out = ks_drift(ref, cur, p_value_threshold=p_value_threshold)
        drifted[col] = out
    any_drift = any(v.get("drift") for v in drifted.values()) if drifted else False
    return {"columns": drifted, "drift": any_drift}


def compute_model_metrics_gbm(bundle: dict, df: pd.DataFrame, target: str) -> Dict[str, float]:
    """Compute simple metrics for a GBM bundle on a given dataframe."""
    feat_cols = bundle.get("feature_cols", [])
    if not feat_cols:
        raise ValueError("Model bundle missing feature_cols")
    # Keep monitoring robust across schema evolution by aligning missing columns.
    aligned = df.copy()
    for col in feat_cols:
        if col not in aligned.columns:
            aligned[col] = np.nan
    X = aligned[feat_cols].to_numpy()
    y = df[target].to_numpy()
    pred = bundle["model"].predict(X)
    return {"rmse": rmse(y, pred), "mape": mape(y, pred)}


def evaluate_model_drift(baseline_metric: float | None, current_metric: float | None, threshold: float) -> Dict[str, Any]:
    """Decide if model performance degraded beyond a threshold."""
    if baseline_metric is None or current_metric is None:
        return {"drift": False, "ratio": None, "note": "missing metrics"}
    return metric_drift(current_metric, baseline_metric, degradation_threshold=threshold)


def retraining_decision(
    cfg: dict,
    data_drift: bool,
    model_drift: bool,
    last_trained_path: Path | None,
    dc3s_health: dict | None = None,
) -> RetrainingDecision:
    """Combine drift checks and cadence rules into a retrain decision."""
    retrain_cfg = cfg.get("retraining", {})
    cadence_days = int(retrain_cfg.get("cadence_days", 30))
    last_days = days_since(last_trained_path) if last_trained_path else None

    reasons = []
    if data_drift:
        reasons.append("data_drift")
    if model_drift:
        reasons.append("model_drift")
    if isinstance(dc3s_health, dict) and bool(dc3s_health.get("triggered", False)):
        flags = set(dc3s_health.get("triggered_flags", []) or [])
        if "intervention_rate" in flags:
            reasons.append("dc3s_intervention_spike")
        if "low_reliability_rate" in flags:
            reasons.append("dc3s_reliability_degradation")
        if "drift_flag_rate" in flags:
            reasons.append("dc3s_drift_persistence")
    if last_days is not None and last_days >= cadence_days:
        reasons.append("scheduled_cadence")

    return RetrainingDecision(retrain=bool(reasons), reasons=reasons, last_trained_days_ago=last_days)
