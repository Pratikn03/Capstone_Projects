"""API router: monitor."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import pandas as pd

from fastapi import APIRouter

from gridpulse.forecasting.predict import load_model_bundle
from gridpulse.monitoring.retraining import (
    load_monitoring_config,
    compute_data_drift,
    compute_model_metrics_gbm,
    evaluate_model_drift,
    retraining_decision,
)
from gridpulse.monitoring.report import write_monitoring_report

router = APIRouter()


def _load_week2_metrics() -> dict | None:
    path = Path("reports/week2_metrics.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("")
def monitor() -> Dict[str, Any]:
    cfg = load_monitoring_config()

    features_path = Path("data/processed/features.parquet")
    train_path = Path("data/processed/splits/train.parquet")

    if not features_path.exists() or not train_path.exists():
        return {"drift": False, "note": "missing processed data or splits"}

    df = pd.read_parquet(features_path).sort_values("timestamp")
    train_df = pd.read_parquet(train_path)

    # current window: last 7 days
    current_df = df.tail(7 * 24)

    feature_cols = [c for c in df.columns if c not in {"timestamp", "load_mw", "wind_mw", "solar_mw"}]
    data_drift = compute_data_drift(
        train_df,
        current_df,
        feature_cols,
        p_value_threshold=float(cfg.get("data_drift", {}).get("p_value_threshold", 0.01)),
    )

    # model drift (load_mw GBM)
    model_drift = {"drift": False, "ratio": None, "note": "missing model or metrics"}
    metrics = _load_week2_metrics()
    baseline_mape = None
    if metrics:
        tgt = metrics.get("targets", {}).get("load_mw", {})
        baseline_mape = tgt.get("gbm", {}).get("mape") if tgt else None

    gbm_path = None
    for p in Path("artifacts/models").glob("gbm_*_load_mw.pkl"):
        gbm_path = p
        break

    if gbm_path and gbm_path.exists() and baseline_mape is not None:
        bundle = load_model_bundle(gbm_path)
        cur_metrics = compute_model_metrics_gbm(bundle, current_df, "load_mw")
        model_drift = evaluate_model_drift(
            baseline_mape,
            cur_metrics.get("mape"),
            float(cfg.get("model_drift", {}).get("degradation_threshold", 0.15)),
        )
        model_drift["current_mape"] = cur_metrics.get("mape")
        model_drift["baseline_mape"] = baseline_mape

    decision = retraining_decision(
        cfg,
        data_drift.get("drift", False),
        model_drift.get("drift", False),
        last_trained_path=Path("reports/week2_metrics.json") if Path("reports/week2_metrics.json").exists() else None,
    )

    payload = {
        "data_drift": data_drift,
        "model_drift": model_drift,
        "retraining": {
            "retrain": decision.retrain,
            "reasons": decision.reasons,
            "last_trained_days_ago": decision.last_trained_days_ago,
        },
    }

    write_monitoring_report(payload, out_path="reports/monitoring_report.md")
    return payload
