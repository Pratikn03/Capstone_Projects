"""API router: monitor."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import pandas as pd

from fastapi import APIRouter

from gridpulse.forecasting.predict import load_model_bundle
from gridpulse.monitoring.dc3s_health import compute_dc3s_health, load_dc3s_audit_config, load_dc3s_health_config
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
    # Key: API endpoint handler
    path = Path("reports/week2_metrics.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_registry_latest(path: Path = Path("artifacts/registry/models.json")) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    latest = payload.get("latest")
    return latest if isinstance(latest, dict) else None


def _to_builtin(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _load_latest_research_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "row_type" in df.columns:
        summary = df[df["row_type"].astype(str).isin(["run_summary", "summary_mean"])]
        row = summary.iloc[-1] if not summary.empty else df.iloc[-1]
    else:
        row = df.iloc[-1]
    payload = {k: _to_builtin(v) for k, v in row.to_dict().items()}
    payload["source_csv"] = str(path)
    return payload


def _load_frozen_metrics_snapshot(path: Path = Path("reports/frozen_metrics_snapshot.json")) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _compute_dc3s_health_block(update_state: bool = False) -> Dict[str, Any] | None:
    cfg = load_dc3s_health_config()
    if not bool(cfg.get("enabled", True)):
        return None
    audit_cfg = load_dc3s_audit_config()
    return compute_dc3s_health(
        window_hours=int(cfg.get("lookback_hours", 24)),
        min_commands=int(cfg.get("min_commands", 50)),
        thresholds=cfg,
        duckdb_path=str(audit_cfg.get("duckdb_path", "data/audit/dc3s_audit.duckdb")),
        table_name=str(audit_cfg.get("table_name", "dispatch_certificates")),
        sustained_windows=int(cfg.get("sustained_windows", 3)),
        state_path="reports/monitoring_state.json",
        update_state=update_state,
    )


@router.get("")
def monitor() -> Dict[str, Any]:
    cfg = load_monitoring_config()
    dc3s_health = _compute_dc3s_health_block(update_state=False)

    features_path = Path("data/processed/features.parquet")
    train_path = Path("data/processed/splits/train.parquet")

    if not features_path.exists() or not train_path.exists():
        decision = retraining_decision(cfg, False, False, None, dc3s_health=dc3s_health)
        payload = {
            "drift": False,
            "note": "missing processed data or splits",
            "dc3s_health": dc3s_health,
            "retraining": {
                "retrain": decision.retrain,
                "reasons": decision.reasons,
                "last_trained_days_ago": decision.last_trained_days_ago,
            },
        }
        write_monitoring_report(payload, out_path="reports/monitoring_report.md")
        return payload

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
        dc3s_health=dc3s_health,
    )

    payload = {
        "data_drift": data_drift,
        "model_drift": model_drift,
        "dc3s_health": dc3s_health,
        "retraining": {
            "retrain": decision.retrain,
            "reasons": decision.reasons,
            "last_trained_days_ago": decision.last_trained_days_ago,
        },
    }

    write_monitoring_report(payload, out_path="reports/monitoring_report.md")
    return payload


@router.get("/dc3s")
def monitor_dc3s() -> Dict[str, Any]:
    block = _compute_dc3s_health_block(update_state=False)
    return block or {"enabled": False}


@router.get("/research-metrics")
def research_metrics() -> Dict[str, Any]:
    de = _load_latest_research_summary(Path("reports/research_metrics_de.csv"))
    us = _load_latest_research_summary(Path("reports/research_metrics_us.csv"))
    frozen = _load_frozen_metrics_snapshot()

    return {
        "available": bool(de or us or frozen),
        "datasets": {
            "de": de,
            "us": us,
        },
        "frozen_metrics_snapshot": frozen,
    }


@router.get("/model-info")
def model_info() -> Dict[str, Any]:
    metrics = _load_week2_metrics()
    registry_latest = _load_registry_latest()

    notes: list[str] = []
    if metrics is None:
        notes.append("Missing or invalid reports/week2_metrics.json")
    if registry_latest is None:
        notes.append("Missing or invalid artifacts/registry/models.json latest block")

    return {
        "available": bool(metrics or registry_latest),
        "metrics_source": "reports/week2_metrics.json" if metrics else None,
        "registry_source": "artifacts/registry/models.json" if registry_latest else None,
        "generated_at": (
            registry_latest.get("generated_at")
            if registry_latest and registry_latest.get("generated_at")
            else (metrics.get("generated_at") if metrics else None)
        ),
        "device": metrics.get("device") if metrics else None,
        "quantiles": metrics.get("quantiles") if metrics else None,
        "targets": metrics.get("targets") if metrics else None,
        "registry": {
            "run_id": registry_latest.get("run_id") if registry_latest else None,
            "models_dir": registry_latest.get("models_dir") if registry_latest else None,
            "model_count": len(registry_latest.get("models", [])) if registry_latest else 0,
            "models": registry_latest.get("models", []) if registry_latest else [],
        },
        "dc3s_shield": _load_dc3s_shield_summary(),
        "notes": notes,
    }


def _load_dc3s_calibrated_params() -> dict | None:
    """Load data-fitted k_quality/k_drift/infl_max from calibrate_dc3s_params.py output."""
    path = Path("configs/dc3s_calibrated.yaml")
    if not path.exists():
        return None
    try:
        import yaml  # type: ignore[import]
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return raw.get("dc3s") or None
    except Exception:
        return None


def _load_dc3s_base_params() -> dict:
    """Load base DC³S config (k_quality, k_drift, infl_max, detector type)."""
    path = Path("configs/dc3s.yaml")
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore[import]
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        dc3s = raw.get("dc3s", raw)
        return {
            "k_quality": dc3s.get("k_quality"),
            "k_drift":   dc3s.get("k_drift"),
            "infl_max":  dc3s.get("infl_max"),
            "detector":  dc3s.get("drift", {}).get("detector", "page_hinkley"),
            "alpha":     dc3s.get("alpha", 0.10),
        }
    except Exception:
        return {}


def _load_ablation_top5() -> list[dict] | None:
    """Load top-5 ablation configs (narrowest width at PICP>=88%) from ablation_summary.csv."""
    path = Path("reports/publication/ablation_summary.csv")
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if df.empty:
            return None
        # Filter dc3s_wrapped rows with PICP >= 88%
        picp_col = next(
            (c for c in ["picp_90_mean", "picp_90", "mean_picp_90"] if c in df.columns),
            None,
        )
        width_col = next(
            (c for c in ["mean_interval_width_mean", "mean_interval_width", "adaptive_width_mean_mean"]
             if c in df.columns),
            None,
        )
        ctrl_col = "controller" if "controller" in df.columns else None
        if ctrl_col:
            df = df[df[ctrl_col] == "dc3s_wrapped"]
        if picp_col and df[picp_col].notna().any():
            meets = df[pd.to_numeric(df[picp_col], errors="coerce") >= 0.88]
            df = meets if not meets.empty else df
        if width_col and df[width_col].notna().any():
            df = df.sort_values(width_col)
        return df[["k_quality", "k_drift", "infl_max", *(c for c in [picp_col, width_col] if c)]
                  ].head(5).to_dict(orient="records")
    except Exception:
        return None


def _load_wilcoxon_results() -> dict | None:
    """Load Wilcoxon signed-rank test results from wilcoxon_tests.json."""
    path = Path("reports/publication/wilcoxon_tests.json")
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_dc3s_shield_summary() -> Dict[str, Any]:
    """
    Aggregate DC³S formal guarantee and parameter metadata for the /model-info endpoint.

    Surfaces:
    - Coverage theorem availability (formal guarantee from coverage_theorem.py)
    - Calibrated parameters (from calibrate_dc3s_params.py output)
    - Base config (k_quality, k_drift, detector type)
    - Top ablation configs
    - Statistical significance (Wilcoxon test results)
    - Cross-region transfer summary
    """
    calibrated = _load_dc3s_calibrated_params()
    base = _load_dc3s_base_params()
    wilcoxon = _load_wilcoxon_results()
    ablation_top5 = _load_ablation_top5()

    # Cross-region transfer summary
    transfer_summary: dict | None = None
    transfer_path = Path("reports/publication/cross_region_transfer.csv")
    if transfer_path.exists():
        try:
            df_t = pd.read_csv(transfer_path)
            dc3s_t = df_t[df_t["controller"] == "dc3s_wrapped"] if "controller" in df_t.columns else df_t
            metric_col = next(
                (c for c in ["true_soc_violation_rate", "violation_rate"] if c in dc3s_t.columns), None
            )
            transfer_summary = {
                "available": True,
                "n_rows": int(len(df_t)),
                "regions": list(df_t["region"].unique()) if "region" in df_t.columns else [],
            }
            if metric_col and not dc3s_t.empty:
                for region in df_t.get("region", pd.Series()).unique() if "region" in df_t.columns else []:
                    reg_data = dc3s_t[dc3s_t["region"] == region]
                    transfer_summary.setdefault("dc3s_by_region", {})[str(region)] = {
                        "mean_violation_rate": float(reg_data[metric_col].mean()),
                    }
        except Exception:
            transfer_summary = {"available": False}

    # Coverage theorem: check if module exists and can verify
    coverage_theorem_available = False
    try:
        from gridpulse.dc3s.coverage_theorem import verify_inflation_geq_one  # noqa: F401
        coverage_theorem_available = True
    except ImportError:
        pass

    return {
        "coverage_theorem_implemented": coverage_theorem_available,
        "coverage_guarantee": "Marginal coverage: P(y ∈ [ŷ - λ·q, ŷ + λ·q]) ≥ 1 - α (Proposition 1, Theorem 3.1)",
        "detector_type": base.get("detector", "page_hinkley"),
        "supported_detectors": ["page_hinkley", "adwin"],
        "base_params": base,
        "calibrated_params": calibrated,
        "calibration_source": "configs/dc3s_calibrated.yaml" if calibrated else None,
        "ablation_top5_configs": ablation_top5,
        "statistical_significance": {
            "test": "Wilcoxon signed-rank (paired, non-parametric)",
            "comparison": "dc3s_wrapped vs deterministic_lp",
            "results": wilcoxon,
        } if wilcoxon else None,
        "cross_region_transfer": transfer_summary,
    }


@router.get("/dc3s-params")
def dc3s_params() -> Dict[str, Any]:
    """
    Detailed DC³S parameter and ablation endpoint.

    Returns full ablation summary, calibrated params, and statistical test results
    for use in the frontend model info panel and research dashboard.
    """
    calibrated = _load_dc3s_calibrated_params()
    base = _load_dc3s_base_params()
    wilcoxon = _load_wilcoxon_results()
    ablation_top5 = _load_ablation_top5()

    # Full ablation table (max 50 rows for API response size)
    ablation_full: list[dict] | None = None
    ablation_path = Path("reports/publication/ablation_summary.csv")
    if ablation_path.exists():
        try:
            df = pd.read_csv(ablation_path)
            ablation_full = df.head(50).to_dict(orient="records")
        except Exception:
            pass

    # Cross-region transfer stats
    transfer_stats: dict | None = None
    xr_path = Path("reports/publication/cross_region_transfer_summary.csv")
    if xr_path.exists():
        try:
            df_xr = pd.read_csv(xr_path)
            transfer_stats = df_xr.to_dict(orient="records")
        except Exception:
            pass

    return {
        "available": True,
        "base_params": base,
        "calibrated_params": calibrated,
        "calibration_source": "configs/dc3s_calibrated.yaml" if calibrated else None,
        "coverage_theorem": {
            "implemented": True,
            "guarantee": "P(y ∈ inflated interval) ≥ 1 − α",
            "proof": "Monotonicity of inflation factor λ ≥ 1 under Prop.1 conditions",
            "runtime_assertions": "verify_inflation_geq_one() + assert_coverage_guarantee()",
        },
        "ablation": {
            "top5": ablation_top5,
            "full_summary": ablation_full,
            "source": str(ablation_path) if ablation_path.exists() else None,
        },
        "statistical_tests": {
            "wilcoxon": wilcoxon,
            "source": "reports/publication/wilcoxon_tests.json",
        },
        "cross_region_transfer": {
            "summary": transfer_stats,
            "source": str(xr_path) if xr_path.exists() else None,
        },
    }
