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
    REPO_ROOT / "reports" / "av" / "week2_metrics.json",
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
DEFAULT_POLICY = {
    "min_r2": 0.0,
    "max_train_validation_rmse_ratio": 1.35,
    "max_validation_test_rmse_ratio": 1.35,
    "max_latency_p95_per_sample_ms": 5.0,
    "require_hyperparameter_tuning": True,
    "min_tuning_trials": 50,
    "min_complete_trial_fraction": 0.95,
    "boundary_fraction": 0.02,
    "max_gradient_clipped_fraction": 0.50,
    "max_grad_norm": 100.0,
    "min_picp_90": 0.85,
    "max_picp_90": 0.99,
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
    metrics = {
        "train_rmse": train_rmse,
        "validation_rmse": val_rmse,
        "test_rmse": test_rmse,
        "train_validation_rmse_ratio": _ratio(val_rmse, train_rmse),
        "validation_test_rmse_ratio": _ratio(test_rmse, val_rmse),
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
        blockers.append(f"overfit risk: validation/train RMSE ratio {tv_ratio:.3f} exceeds policy")
    if vt_ratio is not None and vt_ratio > float(policy["max_validation_test_rmse_ratio"]):
        blockers.append(f"validation/test drift: test/validation RMSE ratio {vt_ratio:.3f} exceeds policy")
    status = "pass" if not blockers else "block"
    detail = "train/validation/test split metrics are within policy" if not blockers else "; ".join(blockers)
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
    if boundary_hits:
        blockers.append(f"selected hyperparameters landed on search boundary: {sorted(boundary_hits)}")

    metrics = {
        "n_trials": n_trials,
        "n_complete_trials": n_complete,
        "complete_fraction": round(complete_fraction, 6),
        "selected_param_count": len(selected_params),
        "boundary_hits": boundary_hits,
    }
    status = "pass" if not blockers else "block"
    detail = "hyperparameter search metadata is within policy" if not blockers else "; ".join(blockers)
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
    targets = payload.get("targets") if isinstance(payload.get("targets"), dict) else {}
    for target, target_payload in sorted(targets.items()):
        if not isinstance(target_payload, dict):
            continue
        for model_key, model_payload in sorted(target_payload.items()):
            if not isinstance(model_payload, dict):
                continue
            gates: dict[str, dict[str, Any]] = {}
            blockers: list[str] = []
            assessments = (
                ("generalization", _assess_generalization(model_payload, policy)),
                ("underfit", _assess_underfit(model_payload, policy)),
                ("architecture", _assess_architecture(model_key, target_payload, model_payload)),
                ("hyperparameter_tuning", _assess_tuning(model_key, model_payload, configs, policy)),
                ("calibration", _assess_calibration(model_payload, policy)),
                ("latency", _assess_latency(model_payload, policy)),
                ("gradient_stability", _assess_gradient_stability(model_key, model_payload, policy)),
            )
            for name, (gate, gate_blockers) in assessments:
                gates[name] = gate
                blockers.extend(gate_blockers)
            rows.append(
                {
                    "metrics_path": str(metrics_path),
                    "target": str(target),
                    "model": str(model_key),
                    "status": "pass" if not blockers else "block",
                    "blockers": blockers,
                    "gates": gates,
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
        rows.extend(
            _model_rows_for_metrics(
                metrics_path=metrics_path,
                payload=_load_structured(metrics_path),
                configs=configs,
                policy=merged_policy,
            )
        )
    blockers = [
        f"{row['metrics_path']}:{row['target']}:{row['model']} - {blocker}"
        for row in rows
        for blocker in row["blockers"]
    ]
    result = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_metrics": {str(path): _sha256_file(path) for path in metrics_paths},
        "config_paths": [str(path) for path in config_paths],
        "policy": merged_policy,
        "pass": not blockers and bool(rows),
        "summary": {
            "metrics_file_count": len(metrics_paths),
            "model_count": len(rows),
            "blocking_model_count": sum(1 for row in rows if row["status"] != "pass"),
        },
        "blockers": blockers,
        "models": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
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
