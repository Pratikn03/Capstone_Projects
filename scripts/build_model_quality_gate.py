#!/usr/bin/env python3
"""Build an enforceable ML model-quality gate.

The gate is intentionally evidence-first: a model is not considered production
or publication-ready unless the run records train/validation/test behavior,
hyperparameter-search metadata, inference latency, and gradient stability for
gradient-trained models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
DEFAULT_OUT = PUBLICATION_DIR / "model_quality_gate.json"
CANONICAL_METRICS = (
    REPO_ROOT / "reports" / "week2_metrics.json",
    REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped" / "week2_metrics.json",
    REPO_ROOT / "reports" / "healthcare" / "week2_metrics.json",
)
DEFAULT_CONFIGS = (
    REPO_ROOT / "configs" / "train_forecast.yaml",
    REPO_ROOT / "configs" / "train_forecast_av.yaml",
    REPO_ROOT / "configs" / "train_forecast_healthcare.yaml",
)
GRADIENT_MODELS = {"lstm", "tcn", "nbeats", "tft", "patchtst"}
MODEL_CONFIG_KEY = {
    "gbm": "baseline_gbm",
    "lstm": "dl_lstm",
    "tcn": "dl_tcn",
    "nbeats": "dl_nbeats",
    "tft": "dl_tft",
    "patchtst": "dl_patchtst",
}
CANONICAL_DATASET_BY_ALIAS = {
    "DE": {"domain": "battery", "dataset": "DE_OPSD", "dataset_alias": "DE"},
    "OPSD": {"domain": "battery", "dataset": "DE_OPSD", "dataset_alias": "DE"},
    "DE_OPSD": {"domain": "battery", "dataset": "DE_OPSD", "dataset_alias": "DE"},
    "BATTERY": {"domain": "battery", "dataset": "DE_OPSD", "dataset_alias": "DE"},
    "ENERGY": {"domain": "battery", "dataset": "DE_OPSD", "dataset_alias": "DE"},
    "AV": {"domain": "vehicle", "dataset": "AV_NUPLAN_ALLZIP_GROUPED", "dataset_alias": "AV"},
    "VEHICLE": {"domain": "vehicle", "dataset": "AV_NUPLAN_ALLZIP_GROUPED", "dataset_alias": "AV"},
    "WAYMO": {"domain": "vehicle", "dataset": "AV_NUPLAN_ALLZIP_GROUPED", "dataset_alias": "AV"},
    "NUPLAN": {"domain": "vehicle", "dataset": "AV_NUPLAN_ALLZIP_GROUPED", "dataset_alias": "AV"},
    "NUPLAN_ALLZIP_GROUPED": {
        "domain": "vehicle",
        "dataset": "AV_NUPLAN_ALLZIP_GROUPED",
        "dataset_alias": "AV",
    },
    "AV_NUPLAN_ALLZIP_GROUPED": {
        "domain": "vehicle",
        "dataset": "AV_NUPLAN_ALLZIP_GROUPED",
        "dataset_alias": "AV",
    },
    "HEALTHCARE": {"domain": "healthcare", "dataset": "MIMIC3_VITALS", "dataset_alias": "HEALTHCARE"},
    "MIMIC": {"domain": "healthcare", "dataset": "MIMIC3_VITALS", "dataset_alias": "HEALTHCARE"},
    "MIMIC3": {"domain": "healthcare", "dataset": "MIMIC3_VITALS", "dataset_alias": "HEALTHCARE"},
    "MIMIC3_VITALS": {
        "domain": "healthcare",
        "dataset": "MIMIC3_VITALS",
        "dataset_alias": "HEALTHCARE",
    },
}
MODEL_IDENTITY = {
    "gbm": {"model_family": "gbm", "estimator": "lightgbm"},
    "lstm": {"model_family": "lstm", "estimator": "pytorch"},
    "tcn": {"model_family": "tcn", "estimator": "pytorch"},
    "nbeats": {"model_family": "nbeats", "estimator": "pytorch"},
    "tft": {"model_family": "tft", "estimator": "pytorch"},
    "patchtst": {"model_family": "patchtst", "estimator": "pytorch"},
}
DEFAULT_POLICY = {
    "min_r2": 0.0,
    "max_train_validation_rmse_ratio": 1.35,
    "max_validation_test_rmse_ratio": 1.35,
    "max_latency_p95_per_sample_ms": 5.0,
    "require_hyperparameter_tuning": True,
    "require_deep_hyperparameter_tuning": False,
    "min_tuning_trials": 50,
    "min_complete_trial_fraction": 0.95,
    "boundary_fraction": 0.02,
    "block_train_validation_rmse_ratio": False,
    "block_search_boundary_hits": False,
    "max_gradient_clipped_fraction": 0.50,
    "max_grad_norm": 100.0,
    "min_picp_90": 0.85,
    "max_picp_90": 0.99,
    "release_model_keys": ["gbm"],
    "block_candidate_models": False,
}


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_structured(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    return yaml.safe_load(raw) or {}


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _gate(status: str, detail: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": status, "detail": detail, "metrics": metrics or {}}


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1.0e-12:
        return None
    return round(float(numerator / denominator), 6)


def _split_metrics(model_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    split = model_payload.get("split_metrics")
    if isinstance(split, dict):
        return {
            "train": dict(split.get("train") or {}),
            "validation": dict(split.get("validation") or {}),
            "test": dict(split.get("test") or {}),
        }

    validation = dict(model_payload.get("validation_metrics") or {})
    seed_rows = model_payload.get("seed_member_metrics") or []
    val_rmse_values = [_finite_float(row.get("val_rmse")) for row in seed_rows if isinstance(row, dict)]
    val_rmse_values = [value for value in val_rmse_values if value is not None]
    if "rmse" not in validation and val_rmse_values:
        validation["rmse"] = sum(val_rmse_values) / len(val_rmse_values)
    return {
        "train": dict(model_payload.get("train_metrics") or {}),
        "validation": validation,
        "test": dict(model_payload.get("test_metrics") or {"rmse": model_payload.get("rmse")}),
    }


def _assess_generalization(
    model_payload: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    split = _split_metrics(model_payload)
    train_rmse = _finite_float(split["train"].get("rmse"))
    val_rmse = _finite_float(split["validation"].get("rmse"))
    test_rmse = _finite_float(split["test"].get("rmse"))
    blockers: list[str] = []
    warnings: list[str] = []
    metrics = {
        "train_rmse": train_rmse,
        "validation_rmse": val_rmse,
        "test_rmse": test_rmse,
        "train_validation_rmse_ratio": _ratio(val_rmse, train_rmse),
        "validation_test_rmse_ratio": _ratio(test_rmse, val_rmse),
        "max_train_validation_rmse_ratio": float(policy["max_train_validation_rmse_ratio"]),
        "max_validation_test_rmse_ratio": float(policy["max_validation_test_rmse_ratio"]),
    }
    if train_rmse is None:
        blockers.append("missing train split metrics for overfit/underfit audit")
    if val_rmse is None:
        blockers.append("missing validation split metrics for overfit audit")
    if test_rmse is None:
        blockers.append("missing test split metrics for generalization audit")
    tv_ratio = metrics["train_validation_rmse_ratio"]
    vt_ratio = metrics["validation_test_rmse_ratio"]
    if tv_ratio is not None and tv_ratio > float(policy["max_train_validation_rmse_ratio"]):
        warning = f"overfit diagnostic: validation/train RMSE ratio {tv_ratio:.3f} exceeds policy"
        if bool(policy.get("block_train_validation_rmse_ratio", False)):
            blockers.append(warning)
        else:
            warnings.append(warning)
    if vt_ratio is not None and vt_ratio > float(policy["max_validation_test_rmse_ratio"]):
        blockers.append(f"validation/test drift: test/validation RMSE ratio {vt_ratio:.3f} exceeds policy")
    metrics["diagnostic_warnings"] = warnings
    status = "pass" if not blockers else "block"
    if blockers:
        detail = "; ".join(blockers + warnings)
    elif warnings:
        detail = "release-blocking split metrics are within policy; " + "; ".join(warnings)
    else:
        detail = "train/validation/test split metrics are within policy"
    return _gate(status, detail, metrics), blockers


def _assess_underfit(
    model_payload: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    r2 = _finite_float(model_payload.get("r2"))
    min_r2 = float(policy["min_r2"])
    if r2 is None:
        return _gate("block", "missing r2 or equivalent underfit metric"), ["missing underfit metric"]
    if r2 < min_r2:
        detail = f"underfit risk: r2 {r2:.3f} is below policy floor {min_r2:.3f}"
        return _gate("block", detail, {"r2": r2, "min_r2": min_r2}), [detail]
    return _gate("pass", "underfit metric is within policy", {"r2": r2, "min_r2": min_r2}), []


def _assess_architecture(
    model_key: str,
    target_payload: dict[str, Any],
    model_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    n_features = _finite_float(model_payload.get("n_features") or target_payload.get("n_features"))
    if n_features is None or n_features <= 0:
        blockers.append("architecture evidence missing n_features")
    selected_params = {}
    tuning_meta = (
        model_payload.get("tuning_meta") if isinstance(model_payload.get("tuning_meta"), dict) else {}
    )
    if isinstance(tuning_meta.get("selected_params"), dict):
        selected_params = dict(tuning_meta["selected_params"])
    elif isinstance(model_payload.get("tuned_params"), dict):
        selected_params = dict(model_payload["tuned_params"])

    architecture = (
        model_payload.get("model_architecture")
        if isinstance(model_payload.get("model_architecture"), dict)
        else {}
    )
    if model_key in GRADIENT_MODELS:
        for field in ("lookback", "horizon"):
            if _finite_float(architecture.get(field)) is None:
                blockers.append(f"architecture evidence missing {field}")
        if _finite_float(architecture.get("gradient_clip")) is None:
            blockers.append("architecture evidence missing gradient_clip")
        if _finite_float(architecture.get("dropout")) is None:
            blockers.append("architecture evidence missing dropout regularization")
        if _finite_float(architecture.get("early_stopping_patience") or architecture.get("patience")) is None:
            blockers.append("architecture evidence missing early stopping patience")
    elif model_key == "gbm" and not selected_params:
        blockers.append("architecture evidence missing selected GBM hyperparameters")

    metrics = {
        "n_features": int(n_features) if n_features is not None else None,
        "selected_param_count": len(selected_params),
        "architecture_fields": sorted(architecture),
    }
    status = "pass" if not blockers else "block"
    detail = "architecture/capacity evidence is present" if not blockers else "; ".join(blockers)
    return _gate(status, detail, metrics), blockers


def _assess_calibration(
    model_payload: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    uncertainty = (
        model_payload.get("uncertainty") if isinstance(model_payload.get("uncertainty"), dict) else {}
    )
    picp_90 = _finite_float(
        uncertainty.get("picp_90") or uncertainty.get("global_coverage") or uncertainty.get("coverage_90")
    )
    mean_width = _finite_float(uncertainty.get("mean_interval_width") or uncertainty.get("global_mean_width"))
    blockers: list[str] = []
    if picp_90 is None:
        blockers.append("calibration evidence missing PICP@90/global coverage")
    else:
        if picp_90 < float(policy["min_picp_90"]):
            blockers.append(f"calibration under-coverage: PICP@90 {picp_90:.3f} below policy")
        if picp_90 > float(policy["max_picp_90"]):
            blockers.append(f"calibration over-wide/over-coverage risk: PICP@90 {picp_90:.3f} above policy")
    if mean_width is None or mean_width <= 0:
        blockers.append("calibration evidence missing positive interval width")
    metrics = {
        "picp_90": picp_90,
        "mean_interval_width": mean_width,
        "min_picp_90": float(policy["min_picp_90"]),
        "max_picp_90": float(policy["max_picp_90"]),
    }
    status = "pass" if not blockers else "block"
    detail = "calibration/uncertainty evidence is within policy" if not blockers else "; ".join(blockers)
    return _gate(status, detail, metrics), blockers


def _config_param_specs(configs: list[dict[str, Any]], model_key: str) -> dict[str, dict[str, Any]]:
    config_key = MODEL_CONFIG_KEY.get(model_key, model_key)
    merged: dict[str, dict[str, Any]] = {}
    for cfg in configs:
        tuning = cfg.get("tuning") if isinstance(cfg.get("tuning"), dict) else {}
        params = tuning.get("params") if isinstance(tuning.get("params"), dict) else {}
        model_params = params.get(config_key) if isinstance(params.get(config_key), dict) else {}
        for name, spec in model_params.items():
            if isinstance(spec, dict):
                merged[str(name)] = dict(spec)
    return merged


def _config_dataset_key(config: dict[str, Any]) -> str | None:
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    key = dataset.get("key")
    return str(key).strip().upper() if key else None


def _metrics_dataset_key(metrics_path: Path, payload: dict[str, Any]) -> str | None:
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    key = dataset.get("key") or payload.get("dataset_key")
    if key:
        return str(key).strip().upper()
    parts = {part.lower() for part in metrics_path.parts}
    if "healthcare" in parts:
        return "HEALTHCARE"
    if "av" in parts or "orius_av" in parts:
        return "AV"
    if "de" in parts:
        return "DE"
    try:
        if metrics_path.resolve() == (REPO_ROOT / "reports" / "week2_metrics.json").resolve():
            return "DE"
    except OSError:
        pass
    return None


def _canonical_dataset_identity(metrics_path: Path, payload: dict[str, Any]) -> dict[str, str | None]:
    alias = _metrics_dataset_key(metrics_path, payload)
    if alias is None:
        return {"domain": None, "dataset": None, "dataset_alias": None}
    normalized = str(alias).strip().upper()
    return dict(
        CANONICAL_DATASET_BY_ALIAS.get(
            normalized,
            {"domain": None, "dataset": normalized, "dataset_alias": normalized},
        )
    )


def _configs_for_metrics(
    metrics_path: Path,
    payload: dict[str, Any],
    configs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    key = _metrics_dataset_key(metrics_path, payload)
    if key is None:
        return configs
    matched = [config for config in configs if _config_dataset_key(config) == key]
    return matched or configs


def _iter_model_entries(
    target_payload: dict[str, Any],
) -> list[tuple[str, dict[str, Any], dict[str, Any], str]]:
    entries: list[tuple[str, dict[str, Any], dict[str, Any], str]] = []
    for model_key, model_payload in sorted(target_payload.items()):
        if model_key not in MODEL_CONFIG_KEY or not isinstance(model_payload, dict):
            continue
        entries.append((str(model_key), model_payload, target_payload, "top_level"))

    challenger_payload = target_payload.get("challenger_metrics")
    if isinstance(challenger_payload, dict):
        challenger_target_payload = dict(target_payload)
        if challenger_payload.get("n_features") is not None:
            challenger_target_payload["n_features"] = challenger_payload["n_features"]
        for model_key, model_payload in sorted(challenger_payload.items()):
            if model_key not in MODEL_CONFIG_KEY or not isinstance(model_payload, dict):
                continue
            entries.append((str(model_key), model_payload, challenger_target_payload, "challenger_metrics"))
    return entries


def _model_identity(model_key: str, model_payload: dict[str, Any]) -> dict[str, str]:
    identity = dict(MODEL_IDENTITY.get(model_key, {"model_family": model_key, "estimator": model_key}))
    if model_key == "gbm":
        raw_model = str(model_payload.get("model") or "").strip().lower()
        if raw_model.startswith("gbm_") and len(raw_model) > 4:
            identity["estimator"] = raw_model[4:]
        elif raw_model and raw_model != "gbm":
            identity["estimator"] = raw_model
    return identity


def _row_role(
    *,
    metric_source: str,
    selected_for_release: bool,
    retention_decision: str | None,
) -> str:
    if metric_source == "challenger_metrics":
        return "challenger"
    if retention_decision == "retained_incumbent":
        return "incumbent"
    if selected_for_release:
        return "release"
    return "candidate"


def _merge_policy_overrides(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge only known flat policy keys from a config override block."""
    merged = dict(base)
    for key, value in override.items():
        if key in DEFAULT_POLICY:
            merged[key] = value
    return merged


def _policy_for_model_row(
    *,
    base_policy: dict[str, Any],
    configs: list[dict[str, Any]],
    target: str,
    model_key: str,
) -> dict[str, Any]:
    """Resolve domain/target/model quality-gate overrides for one model row."""
    policy = dict(base_policy)
    for config in configs:
        gate_cfg = config.get("model_quality_gate")
        if not isinstance(gate_cfg, dict):
            continue
        policy_block = gate_cfg.get("policy")
        if isinstance(policy_block, dict):
            policy = _merge_policy_overrides(policy, policy_block)

        target_overrides = gate_cfg.get("target_overrides")
        target_block = target_overrides.get(target) if isinstance(target_overrides, dict) else None
        if isinstance(target_block, dict):
            policy = _merge_policy_overrides(policy, target_block)

        model_overrides = gate_cfg.get("model_overrides")
        model_block = model_overrides.get(model_key) if isinstance(model_overrides, dict) else None
        if isinstance(model_block, dict):
            policy = _merge_policy_overrides(policy, model_block)
    return policy


def _selected_param_at_boundary(value: Any, spec: dict[str, Any], boundary_fraction: float) -> bool:
    low = _finite_float(spec.get("low"))
    high = _finite_float(spec.get("high"))
    selected = _finite_float(value)
    if low is None or high is None or selected is None or high <= low:
        return False
    margin = (high - low) * boundary_fraction
    return selected <= low + margin or selected >= high - margin


def _assess_tuning(
    model_key: str,
    model_payload: dict[str, Any],
    configs: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not bool(policy["require_hyperparameter_tuning"]):
        return _gate("pass", "hyperparameter tuning not required by policy"), []
    if model_key in GRADIENT_MODELS and not bool(policy.get("require_deep_hyperparameter_tuning", False)):
        architecture = (
            model_payload.get("model_architecture")
            if isinstance(model_payload.get("model_architecture"), dict)
            else {}
        )
        summary = (
            model_payload.get("training_summary")
            if isinstance(model_payload.get("training_summary"), dict)
            else {}
        )
        if architecture and summary:
            return _gate(
                "pass",
                "fixed deep architecture evidence is used instead of hyperparameter-search metadata",
                {
                    "fixed_architecture": True,
                    "architecture_fields": sorted(architecture),
                    "training_summary_fields": sorted(summary),
                },
            ), []
        return _gate(
            "block",
            "fixed deep architecture evidence missing architecture or training summary",
            {"fixed_architecture": True},
        ), ["fixed deep architecture evidence missing architecture or training summary"]
    tuning_meta = model_payload.get("tuning_meta")
    if not isinstance(tuning_meta, dict) or not tuning_meta:
        return _gate("block", "missing hyperparameter tuning metadata"), [
            "missing hyperparameter tuning metadata"
        ]
    if tuning_meta.get("enabled") is False:
        return _gate("block", "hyperparameter tuning metadata reports disabled"), [
            "hyperparameter tuning disabled"
        ]

    n_trials = int(_finite_float(tuning_meta.get("n_trials")) or 0)
    n_complete = int(_finite_float(tuning_meta.get("n_complete_trials")) or 0)
    selected_params = (
        tuning_meta.get("selected_params") if isinstance(tuning_meta.get("selected_params"), dict) else {}
    )
    blockers: list[str] = []
    min_trials = int(policy["min_tuning_trials"])
    if n_trials < min_trials:
        blockers.append(f"hyperparameter tuning ran {n_trials} trials, below policy floor {min_trials}")
    complete_fraction = (n_complete / n_trials) if n_trials > 0 else 0.0
    if complete_fraction < float(policy["min_complete_trial_fraction"]):
        blockers.append(f"only {complete_fraction:.3f} of hyperparameter trials completed")
    if not selected_params:
        blockers.append("missing selected hyperparameters from tuning metadata")

    specs = _config_param_specs(configs, model_key)
    boundary_hits = [
        name
        for name, value in selected_params.items()
        if name in specs
        and _selected_param_at_boundary(value, specs[name], float(policy["boundary_fraction"]))
    ]
    boundary_detail = ""
    if boundary_hits:
        boundary_detail = f"selected hyperparameters landed on search boundary: {sorted(boundary_hits)}"
        if bool(policy.get("block_search_boundary_hits", False)):
            blockers.append(boundary_detail)

    metrics = {
        "n_trials": n_trials,
        "n_complete_trials": n_complete,
        "complete_fraction": round(complete_fraction, 6),
        "selected_param_count": len(selected_params),
        "boundary_hits": boundary_hits,
    }
    status = "pass" if not blockers else "block"
    if blockers:
        detail = "; ".join(blockers)
    elif boundary_detail:
        detail = "hyperparameter search metadata is release-complete; " + boundary_detail
    else:
        detail = "hyperparameter search metadata is within policy"
    return _gate(status, detail, metrics), blockers


def _assess_latency(
    model_payload: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    latency = model_payload.get("latency") if isinstance(model_payload.get("latency"), dict) else {}
    p95 = _finite_float(
        latency.get("p95_per_sample_ms")
        or latency.get("p95_ms_per_sample")
        or latency.get("inference_p95_ms")
        or latency.get("p95_ms")
    )
    if p95 is None:
        return _gate("block", "missing inference latency p95 evidence"), [
            "missing inference latency p95 evidence"
        ]
    budget = float(policy["max_latency_p95_per_sample_ms"])
    if p95 > budget:
        detail = f"latency p95 {p95:.3f}ms exceeds budget {budget:.3f}ms"
        return _gate("block", detail, {"p95_per_sample_ms": p95, "budget_ms": budget}), [detail]
    return _gate("pass", "latency p95 is within policy", {"p95_per_sample_ms": p95, "budget_ms": budget}), []


def _assess_gradient_stability(
    model_key: str,
    model_payload: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if model_key not in GRADIENT_MODELS:
        return _gate("pass", "not applicable for non-gradient model", {"applicable": False}), []
    summary = (
        model_payload.get("training_summary")
        if isinstance(model_payload.get("training_summary"), dict)
        else {}
    )
    if not summary:
        return _gate("block", "missing gradient stability training summary"), [
            "missing gradient stability training summary"
        ]
    blockers: list[str] = []
    if bool(summary.get("non_finite_loss")):
        blockers.append("gradient stability failure: non-finite loss observed")
    clipped_fraction = _finite_float(summary.get("gradient_clipped_fraction"))
    if clipped_fraction is None:
        blockers.append("gradient stability failure: missing clipped-gradient fraction")
    elif clipped_fraction > float(policy["max_gradient_clipped_fraction"]):
        blockers.append(f"gradient stability failure: clipped fraction {clipped_fraction:.3f} exceeds policy")
    max_grad_norm = _finite_float(summary.get("max_grad_norm"))
    if max_grad_norm is None:
        blockers.append("gradient stability failure: missing max gradient norm")
    elif max_grad_norm > float(policy["max_grad_norm"]):
        blockers.append(f"gradient stability failure: max gradient norm {max_grad_norm:.3f} exceeds policy")
    status = "pass" if not blockers else "block"
    detail = "gradient descent stability is within policy" if not blockers else "; ".join(blockers)
    return _gate(status, detail, dict(summary)), blockers


def _model_rows_for_metrics(
    *,
    metrics_path: Path,
    payload: dict[str, Any],
    configs: list[dict[str, Any]],
    policy: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dataset_identity = _canonical_dataset_identity(metrics_path, payload)
    run_id = payload.get("manifest_id") or payload.get("run_id")
    targets = payload.get("targets") if isinstance(payload.get("targets"), dict) else {}
    for target, target_payload in sorted(targets.items()):
        if not isinstance(target_payload, dict):
            continue
        retention_decision = target_payload.get("retention_decision")
        retention_reason = target_payload.get("retention_reason")
        for model_key, model_payload, row_target_payload, metric_source in _iter_model_entries(target_payload):
            row_policy = _policy_for_model_row(
                base_policy=policy,
                configs=configs,
                target=str(target),
                model_key=str(model_key),
            )
            gates: dict[str, dict[str, Any]] = {}
            blockers: list[str] = []
            assessments = (
                ("generalization", _assess_generalization(model_payload, row_policy)),
                ("underfit", _assess_underfit(model_payload, row_policy)),
                ("architecture", _assess_architecture(model_key, row_target_payload, model_payload)),
                ("hyperparameter_tuning", _assess_tuning(model_key, model_payload, configs, row_policy)),
                ("calibration", _assess_calibration(model_payload, row_policy)),
                ("latency", _assess_latency(model_payload, row_policy)),
                ("gradient_stability", _assess_gradient_stability(model_key, model_payload, row_policy)),
            )
            for name, (gate, gate_blockers) in assessments:
                gates[name] = gate
                blockers.extend(gate_blockers)
            release_keys = policy.get("release_model_keys", ["gbm"])
            if isinstance(release_keys, str):
                release_keys = [release_keys]
            if not isinstance(release_keys, list):
                release_keys = ["gbm"]
            is_release_model = model_key in {str(key).strip() for key in release_keys}
            selected_for_release = bool(is_release_model and metric_source == "top_level")
            role = _row_role(
                metric_source=metric_source,
                selected_for_release=selected_for_release,
                retention_decision=str(retention_decision) if retention_decision is not None else None,
            )
            rows.append(
                {
                    "metrics_path": str(metrics_path),
                    **dataset_identity,
                    "target": str(target),
                    "model": str(model_key),
                    **_model_identity(model_key, model_payload),
                    "role": role,
                    "run_id": str(run_id) if run_id is not None else None,
                    "selected_for_release": selected_for_release,
                    "source_model_key": str(model_key),
                    "metric_source": metric_source,
                    "retention_decision": str(retention_decision) if retention_decision is not None else None,
                    "retention_reason": str(retention_reason) if retention_reason is not None else None,
                    "release_model": selected_for_release,
                    "status": "pass" if not blockers else "block",
                    "blockers": blockers,
                    "gates": gates,
                    "effective_policy": {
                        key: row_policy[key]
                        for key in sorted(row_policy)
                        if row_policy.get(key) != policy.get(key)
                    },
                }
            )
    return rows


def build_model_quality_gate(
    *,
    metrics_paths: list[Path] | None = None,
    config_paths: list[Path] | None = None,
    out_path: Path = DEFAULT_OUT,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics_paths = sorted(
        [path for path in (metrics_paths or list(CANONICAL_METRICS)) if path.exists()], key=str
    )
    config_paths = sorted(
        [path for path in (config_paths or list(DEFAULT_CONFIGS)) if path.exists()], key=str
    )
    merged_policy = {**DEFAULT_POLICY, **(policy or {})}
    configs = [_load_structured(path) for path in config_paths]

    rows: list[dict[str, Any]] = []
    for metrics_path in metrics_paths:
        payload = _load_structured(metrics_path)
        rows.extend(
            _model_rows_for_metrics(
                metrics_path=metrics_path,
                payload=payload,
                configs=_configs_for_metrics(metrics_path, payload, configs),
                policy=merged_policy,
            )
        )
    block_candidate_models = bool(merged_policy.get("block_candidate_models", False))
    release_rows = [row for row in rows if bool(row.get("selected_for_release"))]
    release_blockers = [
        f"{row['metrics_path']}:{row['target']}:{row['model']}:{row.get('role')} - {blocker}"
        for row in rows
        if bool(row.get("selected_for_release")) or block_candidate_models
        for blocker in row["blockers"]
    ]
    candidate_findings = [
        f"{row['metrics_path']}:{row['target']}:{row['model']}:{row.get('role')} - {blocker}"
        for row in rows
        if not bool(row.get("selected_for_release"))
        for blocker in row["blockers"]
    ]
    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_metrics": {str(path): _sha256_file(path) for path in metrics_paths},
        "config_paths": [str(path) for path in config_paths],
        "policy": merged_policy,
        "pass": not release_blockers and bool(release_rows),
        "summary": {
            "metrics_file_count": len(metrics_paths),
            "model_count": len(rows),
            "blocking_model_count": sum(1 for row in rows if row["status"] != "pass"),
            "release_model_count": len(release_rows),
            "blocking_release_model_count": sum(1 for row in release_rows if row["status"] != "pass"),
            "candidate_blocking_model_count": sum(
                1 for row in rows if not bool(row.get("selected_for_release")) and row["status"] != "pass"
            ),
        },
        "blockers": release_blockers,
        "candidate_findings": candidate_findings,
        "models": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--out", "--output", dest="out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build_model_quality_gate(
        metrics_paths=[path.resolve() for path in args.metrics] if args.metrics else None,
        config_paths=[path.resolve() for path in args.config] if args.config else None,
        out_path=args.out.resolve(),
    )
    status = "PASS" if result["pass"] else "BLOCKED"
    print(
        "[build_model_quality_gate] "
        f"{status} models={result['summary']['model_count']} blockers={len(result['blockers'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
