"""Generate a lightweight monitoring report (data drift + model drift)."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.monitoring.report import write_monitoring_report
from gridpulse.monitoring.alerts import send_webhook
from gridpulse.utils.logging import setup_logging
from gridpulse.monitoring.retraining import (
    load_monitoring_config,
    compute_data_drift,
    evaluate_model_drift,
    retraining_decision,
    compute_model_metrics_gbm,
)
from gridpulse.forecasting.predict import load_model_bundle


def _load_split() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    splits_dir = Path("data/processed/splits")
    if (splits_dir / "train.parquet").exists() and (splits_dir / "test.parquet").exists():
        train_df = pd.read_parquet(splits_dir / "train.parquet")
        test_df = pd.read_parquet(splits_dir / "test.parquet")
        return train_df, test_df
    features_path = Path("data/processed/features.parquet")
    if features_path.exists():
        df = pd.read_parquet(features_path).sort_values("timestamp")
        n = len(df)
        return df.iloc[: int(n * 0.7)], df.iloc[int(n * 0.85):]
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alert-webhook", default=None, help="Override GRIDPULSE_ALERT_WEBHOOK for alerts")
    parser.add_argument("--disable-alerts", action="store_true", help="Disable alerting even if webhook is set")
    args = parser.parse_args()

    setup_logging()
    if args.alert_webhook:
        os.environ["GRIDPULSE_ALERT_WEBHOOK"] = args.alert_webhook
    if args.disable_alerts:
        os.environ.pop("GRIDPULSE_ALERT_WEBHOOK", None)
    cfg = load_monitoring_config()
    train_df, test_df = _load_split()
    payload: dict = {"data_drift": None, "model_drift": None, "retraining": None}

    if train_df is None or test_df is None or train_df.empty or test_df.empty:
        payload["note"] = "Missing splits/features; monitoring skipped."
        write_monitoring_report(payload)
        _write_summary(payload)
        print("Monitoring report written (no data).")
        return

    feature_cols = [c for c in train_df.columns if c not in {"timestamp", "load_mw", "wind_mw", "solar_mw"}]
    p_thresh = float(cfg.get("data_drift", {}).get("p_value_threshold", 0.01))
    recent = test_df.tail(7 * 24)
    data_drift = compute_data_drift(train_df, recent, feature_cols, p_thresh)

    # Model drift: compare a baseline metric to current GBM metric when available.
    baseline_metric = None
    metrics_path = Path("reports/week2_metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        baseline_metric = metrics.get("targets", {}).get("load_mw", {}).get("gbm", {}).get("mape")

    model_drift = {"note": "model drift not computed"}
    gbm_path = None
    for p in Path("artifacts/models").glob("gbm_*_load_mw.pkl"):
        gbm_path = p
        break
    if gbm_path and gbm_path.exists():
        bundle = load_model_bundle(gbm_path)
        current_metrics = compute_model_metrics_gbm(bundle, test_df, "load_mw")
        drift_cfg = cfg.get("model_drift", {})
        thresh = float(drift_cfg.get("degradation_threshold", 0.15))
        model_drift = {
            "current": current_metrics,
            "baseline_mape": baseline_metric,
            "decision": evaluate_model_drift(baseline_metric, current_metrics.get("mape"), thresh),
        }

    retraining = retraining_decision(cfg, data_drift.get("drift", False), model_drift.get("decision", {}).get("drift", False), gbm_path)

    payload["data_drift"] = data_drift
    payload["model_drift"] = model_drift
    payload["retraining"] = {
        "retrain": retraining.retrain,
        "reasons": retraining.reasons,
        "last_trained_days_ago": retraining.last_trained_days_ago,
    }

    write_monitoring_report(payload)
    _write_summary(payload)
    _maybe_alert(payload)
    print("Monitoring report written.")


def _write_summary(payload: dict, out_path: str = "reports/monitoring_summary.json") -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _maybe_alert(payload: dict) -> None:
    webhook = os.getenv("GRIDPULSE_ALERT_WEBHOOK")
    if not webhook:
        return
    data_drift = payload.get("data_drift", {}) or {}
    model_drift = payload.get("model_drift", {}) or {}
    retraining = payload.get("retraining", {}) or {}

    should_alert = bool(data_drift.get("drift")) or bool(model_drift.get("decision", {}).get("drift")) or bool(
        retraining.get("retrain")
    )
    if not should_alert:
        return
    send_webhook(webhook, payload)


if __name__ == "__main__":
    main()
