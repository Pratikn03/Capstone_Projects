#!/usr/bin/env python3
"""Full offline predeployment training freeze for Battery, AV, and Healthcare."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _dataset_registry import DATASET_REGISTRY, MAX_QUALITY_DEFAULTS

PYTHON_BIN = sys.executable or ".venv/bin/python"
MODEL_REQUEST = "gbm,lstm,tcn,nbeats,tft,patchtst"
FULL_AV_GATE_NAME = "nuplan_full_av_gate.json"
FREEZE_STATUS = "predeployment_not_deployed"
NUPLAN_SURFACE = "nuplan_allzip_grouped_runtime_replay_surrogate"
AV_RUNTIME_DIR = (
    REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
)
PROMOTED_RUNTIME_MAX_TSVR = 1e-3
CLAIM_BOUNDARY = (
    "Predeployment offline freeze only: not road deployed, not live clinical deployed, "
    "and not unrestricted field deployed."
)


@dataclass(frozen=True)
class FreezeDomain:
    dataset: str
    domain_key: str
    domain_label: str
    baseline_metrics: Path
    canonical_models_dir: Path
    canonical_uncertainty_dir: Path
    canonical_backtests_dir: Path
    canonical_reports_dir: Path
    splits_dir: Path


FREEZE_DOMAINS: tuple[FreezeDomain, ...] = (
    FreezeDomain(
        dataset="DE",
        domain_key="battery",
        domain_label="Battery Energy Storage",
        baseline_metrics=REPO_ROOT / "reports" / "week2_metrics.json",
        canonical_models_dir=REPO_ROOT / "artifacts" / "models",
        canonical_uncertainty_dir=REPO_ROOT / "artifacts" / "uncertainty",
        canonical_backtests_dir=REPO_ROOT / "artifacts" / "backtests",
        canonical_reports_dir=REPO_ROOT / "reports",
        splits_dir=REPO_ROOT / "data" / "processed" / "splits",
    ),
    FreezeDomain(
        dataset="AV",
        domain_key="av",
        domain_label="Autonomous Vehicles",
        baseline_metrics=REPO_ROOT
        / "reports"
        / "orius_av"
        / "nuplan_allzip_grouped"
        / "training_summary.csv",
        canonical_models_dir=REPO_ROOT / "artifacts" / "models_orius_av_nuplan_allzip_grouped",
        canonical_uncertainty_dir=REPO_ROOT / "artifacts" / "uncertainty" / "orius_av_nuplan_allzip_grouped",
        canonical_backtests_dir=REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped",
        canonical_reports_dir=REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped",
        splits_dir=REPO_ROOT / "data" / "orius_av" / "av" / "processed_nuplan_allzip_grouped" / "splits",
    ),
    FreezeDomain(
        dataset="HEALTHCARE",
        domain_key="healthcare",
        domain_label="Medical and Healthcare Monitoring",
        baseline_metrics=REPO_ROOT / "reports" / "healthcare" / "week2_metrics.json",
        canonical_models_dir=REPO_ROOT / "artifacts" / "models_healthcare",
        canonical_uncertainty_dir=REPO_ROOT / "artifacts" / "uncertainty" / "healthcare",
        canonical_backtests_dir=REPO_ROOT / "artifacts" / "backtests" / "healthcare",
        canonical_reports_dir=REPO_ROOT / "reports" / "healthcare",
        splits_dir=REPO_ROOT / "data" / "healthcare" / "processed" / "splits",
    ),
)


def generate_release_id() -> str:
    return "PREDEPLOY_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _remove_appledouble_files(root: Path) -> None:
    for path in root.rglob("._*"):
        if path.is_file():
            with suppress(OSError):
                path.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run(cmd: list[str], *, log_path: Path) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else src_path
    env.setdefault("MPLBACKEND", "Agg")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env, text=True, stdout=handle, stderr=subprocess.STDOUT
        )
    return result.returncode == 0


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _dirty_worktree() -> bool:
    try:
        output = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True)
        return bool(output.strip())
    except Exception:
        return True


def candidate_root(domain: FreezeDomain, release_id: str) -> Path:
    return REPO_ROOT / "artifacts" / "runs" / domain.dataset.lower() / release_id


def candidate_reports_dir(domain: FreezeDomain, release_id: str) -> Path:
    return REPO_ROOT / "reports" / "runs" / domain.dataset.lower() / release_id


def candidate_manifest_path(domain: FreezeDomain, release_id: str) -> Path:
    return candidate_root(domain, release_id) / "registry" / "run_manifest.json"


def selection_summary_path(domain: FreezeDomain, release_id: str) -> Path:
    return candidate_root(domain, release_id) / "registry" / f"tuning_summary_{domain.dataset.lower()}.json"


def nuplan_full_gate_path(freeze_dir: Path) -> Path:
    return freeze_dir / FULL_AV_GATE_NAME


def _runtime_cap_hours(max_runtime_hours: float | None) -> float | None:
    if max_runtime_hours is None or float(max_runtime_hours) <= 0.0:
        return None
    return float(max_runtime_hours)


def _qmax_deep_cap_args(profile: str, n_trials: int | None) -> list[str]:
    """Record the intentionally short deep-model schedule for QMAX smoke runs."""
    if profile != "production-max-fast":
        return []
    if n_trials is None or int(n_trials) > 10 or int(n_trials) <= 0:
        return []
    return [
        "--max-deep-epochs",
        "2",
        "--deep-patience",
        "1",
        "--deep-warmup-epochs",
        "1",
    ]


def training_command(
    domain: FreezeDomain,
    release_id: str,
    *,
    profile: str,
    max_runtime_hours: float | None,
    n_trials: int | None = None,
    freeze_dir: Path | None = None,
    model_request: str = MODEL_REQUEST,
) -> list[str]:
    if domain.dataset == "AV":
        freeze_dir = freeze_dir or (REPO_ROOT / "reports" / "predeployment_freeze" / release_id)
        return [
            PYTHON_BIN,
            "scripts/validate_nuplan_freeze_gate.py",
            "--summary",
            "reports/predeployment_external_validation/nuplan_closed_loop_summary.csv",
            "--traces",
            "reports/predeployment_external_validation/nuplan_closed_loop_traces.csv",
            "--manifest",
            "reports/predeployment_external_validation/nuplan_closed_loop_manifest.json",
            "--manifest-out",
            str(nuplan_full_gate_path(freeze_dir)),
        ]
    cmd = [
        PYTHON_BIN,
        "scripts/train_dataset.py",
        "--dataset",
        domain.dataset,
        "--candidate-run",
        "--run-id",
        release_id,
        "--models",
        model_request,
        "--profile",
        profile,
        "--tune",
        "--rebuild",
        "--target-metrics-file",
        str(domain.baseline_metrics),
    ]
    cap_hours = _runtime_cap_hours(max_runtime_hours)
    if cap_hours is not None:
        cmd.extend(["--max-runtime-hours", str(cap_hours)])
    if n_trials is not None and int(n_trials) > 0:
        cmd.extend(["--n-trials", str(int(n_trials))])
    cmd.extend(_qmax_deep_cap_args(profile, n_trials))
    return cmd


def av_offline_training_command(
    release_id: str,
    *,
    profile: str,
    max_runtime_hours: float | None,
    n_trials: int | None = None,
    model_request: str = MODEL_REQUEST,
) -> list[str]:
    """Train the AV nuPlan offline forecasting surface without replacing the runtime gate."""
    domain = next(item for item in FREEZE_DOMAINS if item.dataset == "AV")
    cmd = [
        PYTHON_BIN,
        "scripts/train_dataset.py",
        "--dataset",
        "AV",
        "--candidate-run",
        "--run-id",
        release_id,
        "--models",
        model_request,
        "--profile",
        profile,
        "--tune",
        "--target-metrics-file",
        str(domain.baseline_metrics),
    ]
    cap_hours = _runtime_cap_hours(max_runtime_hours)
    if cap_hours is not None:
        cmd.extend(["--max-runtime-hours", str(cap_hours)])
    if n_trials is not None and int(n_trials) > 0:
        cmd.extend(["--n-trials", str(int(n_trials))])
    cmd.extend(_qmax_deep_cap_args(profile, n_trials))
    return cmd


def promotion_command(
    domain: FreezeDomain, release_id: str, *, model_request: str = MODEL_REQUEST
) -> list[str]:
    if domain.dataset == "AV":
        return [
            PYTHON_BIN,
            "-c",
            "print('nuPlan AV freeze profile: no generic AV candidate promotion for AV')",
        ]
    return [
        PYTHON_BIN,
        "scripts/train_dataset.py",
        "--dataset",
        domain.dataset,
        "--candidate-run",
        "--run-id",
        release_id,
        "--reports-only",
        "--models",
        model_request,
        "--target-metrics-file",
        str(domain.baseline_metrics),
        "--promote-on-accept",
    ]


def downstream_commands(release_id: str, freeze_dir: Path) -> list[list[str]]:
    return [
        [PYTHON_BIN, "scripts/build_healthcare_runtime_artifacts.py"],
        [PYTHON_BIN, "scripts/build_nuplan_closed_loop_artifacts.py"],
        [PYTHON_BIN, "scripts/build_domain_runtime_contract_artifacts.py"],
        [PYTHON_BIN, "scripts/validate_universal_contract_manifest.py"],
        [
            PYTHON_BIN,
            "scripts/run_universal_orius_validation.py",
            "--out",
            "reports/universal_orius_validation",
        ],
        [PYTHON_BIN, "scripts/build_equal_domain_artifact_discipline.py"],
        [PYTHON_BIN, "scripts/build_three_domain_ml_artifacts.py"],
        [
            PYTHON_BIN,
            "scripts/run_predeployment_external_validation.py",
            "--out",
            "reports/predeployment_external_validation",
        ],
        [
            PYTHON_BIN,
            "scripts/build_three_domain_runtime_stress_artifacts.py",
            "--out",
            str(freeze_dir / "runtime_stress"),
        ],
        [
            PYTHON_BIN,
            "scripts/run_universal_training_audit.py",
            "--out",
            "reports/orius_framework_proof/training_audit",
        ],
        [PYTHON_BIN, "scripts/run_universal_training_audit.py", "--out", "reports/universal_training_audit"],
        [PYTHON_BIN, "scripts/validate_equal_domain_artifact_discipline.py"],
        [PYTHON_BIN, "scripts/validate_theorem_surface.py"],
        [PYTHON_BIN, "scripts/sync_impact_from_manifest.py"],
        [PYTHON_BIN, "scripts/validate_paper_claims.py"],
    ]


def planned_commands(
    release_id: str,
    freeze_dir: Path,
    *,
    profile: str,
    max_runtime_hours: float | None,
    n_trials: int | None = None,
    model_request: str = MODEL_REQUEST,
    parallel_training: bool | None = None,
) -> dict[str, Any]:
    cap_hours = _runtime_cap_hours(max_runtime_hours)
    resolved_parallel_training = (
        profile == "production-max-fast" if parallel_training is None else bool(parallel_training)
    )
    av_offline_training = (
        av_offline_training_command(
            release_id,
            profile=profile,
            max_runtime_hours=max_runtime_hours,
            n_trials=n_trials,
            model_request=model_request,
        )
        if profile == "production-max-fast"
        else None
    )
    return {
        "release_id": release_id,
        "profile": profile,
        "model_request": model_request,
        "runtime_cap_enabled": cap_hours is not None,
        "runtime_cap_hours": cap_hours,
        "n_trials_override": int(n_trials) if n_trials is not None and int(n_trials) > 0 else None,
        "parallel_training_enabled": resolved_parallel_training,
        "parallel_training_datasets": [domain.dataset for domain in FREEZE_DOMAINS]
        if resolved_parallel_training
        else [],
        "nuplan_full_gate": str(nuplan_full_gate_path(freeze_dir)),
        "av_offline_training": av_offline_training,
        "training": {
            domain.dataset: training_command(
                domain,
                release_id,
                profile=profile,
                max_runtime_hours=max_runtime_hours,
                n_trials=n_trials,
                freeze_dir=freeze_dir,
                model_request=model_request,
            )
            for domain in FREEZE_DOMAINS
        },
        "promotion": {
            domain.dataset: promotion_command(domain, release_id, model_request=model_request)
            for domain in FREEZE_DOMAINS
        },
        "downstream": downstream_commands(release_id, freeze_dir),
    }


def _read_csv_first_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            return dict(row)
    return {}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _parquet_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        import pyarrow.parquet as pq  # type: ignore

        parquet = pq.ParquetFile(path)
        return {
            "exists": True,
            "rows": int(parquet.metadata.num_rows),
            "columns": list(parquet.schema_arrow.names),
        }
    except Exception:
        frame = pd.read_parquet(path)
        return {"exists": True, "rows": int(len(frame)), "columns": list(frame.columns)}


def _active_nuplan_writers() -> list[str]:
    try:
        output = subprocess.check_output(["ps", "-axo", "pid,ppid,etime,pcpu,pmem,command"], text=True)
    except Exception:
        return ["unable to inspect process table"]
    editor_markers = (
        "/Applications/PyCharm.app/",
        "/Applications/Visual Studio Code.app/",
        "Code Helper",
        "Cursor.app",
    )
    tokens = (
        "build_nuplan_av_surface",
        "processed_nuplan",
        "replay_windows.parquet.tmp",
        "step_features.parquet.tmp",
        "nuplan-v1.0",
    )
    rows: list[str] = []
    for line in output.splitlines():
        if "ps -axo" in line or " rg " in line:
            continue
        if any(marker in line for marker in editor_markers):
            continue
        if any(token in line for token in tokens):
            rows.append(line.strip())
    return rows


def _training_config_preflight(dataset: str, *, model_request: str = MODEL_REQUEST) -> dict[str, Any]:
    cfg = DATASET_REGISTRY.get(dataset)
    if cfg is None:
        return {
            "dataset": dataset,
            "exists": False,
            "blockers": [f"{dataset}: missing dataset registry entry"],
        }
    train_cfg = _read_yaml(REPO_ROOT / cfg.config_file)
    task_cfg = train_cfg.get("task", {}) if isinstance(train_cfg.get("task"), dict) else {}
    models_cfg = train_cfg.get("models", {}) if isinstance(train_cfg.get("models"), dict) else {}
    tuning_cfg = train_cfg.get("tuning", {}) if isinstance(train_cfg.get("tuning"), dict) else {}
    profile_defaults = MAX_QUALITY_DEFAULTS.get(cfg.name, {})
    blockers: list[str] = []
    features_meta = _parquet_meta(REPO_ROOT / cfg.features_path)
    if not features_meta.get("exists"):
        blockers.append(f"{dataset}: missing features parquet {cfg.features_path}")
    if not (REPO_ROOT / cfg.config_file).exists():
        blockers.append(f"{dataset}: missing training config {cfg.config_file}")
    if not domain_by_dataset(dataset).baseline_metrics.exists():
        blockers.append(f"{dataset}: missing baseline metrics {domain_by_dataset(dataset).baseline_metrics}")
    split_meta: dict[str, Any] = {}
    for split_name in ("train", "calibration", "val", "test"):
        split_path = REPO_ROOT / cfg.splits_path / f"{split_name}.parquet"
        split_meta[split_name] = _parquet_meta(split_path)
        if not split_meta[split_name].get("exists"):
            blockers.append(f"{dataset}: missing split {split_path}")
    enabled_models = {
        name: bool(config.get("enabled", True))
        for name, config in models_cfg.items()
        if isinstance(config, dict)
    }
    return {
        "dataset": dataset,
        "display_name": cfg.display_name,
        "config_file": cfg.config_file,
        "features_path": cfg.features_path,
        "features": features_meta,
        "splits_path": cfg.splits_path,
        "splits": split_meta,
        "targets": task_cfg.get("targets", []),
        "models_requested": model_request.split(","),
        "models_enabled_in_config": enabled_models,
        "tuning_config": {
            "enabled": tuning_cfg.get("enabled"),
            "engine": tuning_cfg.get("engine"),
            "configured_n_trials": tuning_cfg.get("n_trials"),
            "selection_mode": tuning_cfg.get("selection_mode"),
            "select_top_pct": tuning_cfg.get("select_top_pct"),
        },
        "max_profile_effective_defaults": profile_defaults,
        "baseline_metrics": _repo_rel(domain_by_dataset(dataset).baseline_metrics),
        "blockers": blockers,
    }


def domain_by_dataset(dataset: str) -> FreezeDomain:
    for domain in FREEZE_DOMAINS:
        if domain.dataset == dataset:
            return domain
    raise KeyError(dataset)


def _nuplan_preflight() -> dict[str, Any]:
    required_paths = {
        "closed_loop_summary": REPO_ROOT
        / "reports"
        / "predeployment_external_validation"
        / "nuplan_closed_loop_summary.csv",
        "closed_loop_traces": REPO_ROOT
        / "reports"
        / "predeployment_external_validation"
        / "nuplan_closed_loop_traces.csv",
        "closed_loop_manifest": REPO_ROOT
        / "reports"
        / "predeployment_external_validation"
        / "nuplan_closed_loop_manifest.json",
        "runtime_summary": AV_RUNTIME_DIR / "runtime_summary.csv",
        "runtime_traces": AV_RUNTIME_DIR / "runtime_traces.csv",
        "runtime_governance": AV_RUNTIME_DIR / "runtime_governance_summary.csv",
    }
    blockers = [
        f"AV: missing {name} at {_repo_rel(path)}"
        for name, path in required_paths.items()
        if not path.exists()
    ]
    summary = _read_csv_first_row(required_paths["closed_loop_summary"])
    if summary:
        if summary.get("validation_surface") != NUPLAN_SURFACE:
            blockers.append(f"AV: closed-loop summary is not {NUPLAN_SURFACE}")
        if summary.get("status") != "completed_bounded_replay_not_carla":
            blockers.append("AV: closed-loop summary status is not completed_bounded_replay_not_carla")
        for field in ("road_deployed", "full_autonomous_driving_closure_claimed", "carla_completed"):
            if str(summary.get(field, "")).strip().lower() == "true":
                blockers.append(f"AV: closed-loop summary overclaims {field}=True")
    active_writers = _active_nuplan_writers()
    if active_writers:
        blockers.append("AV: active nuPlan writer detected; wait for it to finish before training")
    return {
        "dataset": "AV",
        "display_name": "Autonomous Vehicles",
        "training_surface": NUPLAN_SURFACE,
        "required_paths": {name: _repo_rel(path) for name, path in required_paths.items()},
        "summary_row": summary,
        "active_nuplan_writers": active_writers,
        "max_profile_effective_defaults": "not used: AV full freeze is a nuPlan runtime-contract gate, not generic AV model training",
        "blockers": blockers,
    }


def write_preflight_manifest(
    *,
    release_id: str,
    freeze_dir: Path,
    plans: Mapping[str, Any],
    profile: str,
    max_runtime_hours: float | None,
    n_trials: int | None = None,
    model_request: str = MODEL_REQUEST,
) -> dict[str, Any]:
    cap_hours = _runtime_cap_hours(max_runtime_hours)
    domains = [
        _training_config_preflight("DE", model_request=model_request),
        _nuplan_preflight(),
        _training_config_preflight("HEALTHCARE", model_request=model_request),
    ]
    blockers = [finding for domain in domains for finding in domain.get("blockers", [])]
    manifest = {
        "release_id": release_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "preflight_passed" if not blockers else "preflight_blocked",
        "pass": not blockers,
        "profile": profile,
        "runtime_cap_enabled": cap_hours is not None,
        "runtime_cap_hours": cap_hours,
        "runtime_cap_policy": "uncapped" if cap_hours is None else f"{cap_hours} hours",
        "n_trials_override": int(n_trials) if n_trials is not None and int(n_trials) > 0 else None,
        "model_families_requested": model_request.split(","),
        "planned_training_commands": plans["training"],
        "planned_promotion_commands": plans["promotion"],
        "planned_downstream_commands": plans["downstream"],
        "domains": domains,
        "blockers": blockers,
    }
    path = freeze_dir / "preflight_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _remove_appledouble_files(freeze_dir)
    return manifest


def _nuplan_full_training_gate(domain: FreezeDomain, freeze_dir: Path) -> dict[str, Any]:
    manifest_path = nuplan_full_gate_path(freeze_dir)
    manifest = _load_json(manifest_path)
    return {
        "domain": domain.domain_label,
        "dataset": domain.dataset,
        "accepted": bool(manifest.get("pass", False)),
        "manifest_exists": bool(manifest),
        "selection_summary": _repo_rel(manifest_path),
        "run_manifest": _repo_rel(manifest_path),
        "primary_target": NUPLAN_SURFACE,
        "rmse": None,
        "picp_90": None,
        "baseline_picp_90": None,
        "picp_gate": True,
        "nuplan_source_dataset": manifest.get("source_dataset"),
        "nuplan_validation_surface": manifest.get("validation_surface"),
        "nuplan_runtime_rows": manifest.get("orius_runtime_rows"),
        "nuplan_trace_rows": manifest.get("trace_rows"),
        "nuplan_tsvr": manifest.get("orius_tsvr"),
        "pass": bool(manifest.get("pass", False)),
    }


def _primary_target_metrics(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    targets = payload.get("targets", {}) if isinstance(payload.get("targets"), dict) else {}
    if not targets:
        return {}
    target_name = next(iter(targets))
    gbm = targets.get(target_name, {}).get("gbm", {}) if isinstance(targets.get(target_name), dict) else {}
    uncertainty = gbm.get("uncertainty", {}) if isinstance(gbm.get("uncertainty"), dict) else {}
    return {
        "primary_target": target_name,
        "rmse": gbm.get("rmse"),
        "mae": gbm.get("mae"),
        "picp_90": uncertainty.get("picp_90"),
        "mean_interval_width": uncertainty.get("mean_interval_width"),
    }


def training_gate(domain: FreezeDomain, release_id: str, *, freeze_dir: Path | None = None) -> dict[str, Any]:
    if domain.dataset == "AV":
        if freeze_dir is None:
            freeze_dir = REPO_ROOT / "reports" / "predeployment_freeze" / release_id
        return _nuplan_full_training_gate(domain, freeze_dir)

    selection = _load_json(selection_summary_path(domain, release_id))
    manifest = _load_json(candidate_manifest_path(domain, release_id))
    metrics = _primary_target_metrics(candidate_reports_dir(domain, release_id) / "week2_metrics.json")
    baseline_metrics = _primary_target_metrics(domain.baseline_metrics)
    accepted = bool(selection.get("accepted", False))
    picp = metrics.get("picp_90")
    baseline_picp = baseline_metrics.get("picp_90")
    picp_gate = True
    if domain.dataset in {"AV", "HEALTHCARE"} and picp is not None:
        picp_gate = float(picp) >= 0.90
    if domain.dataset == "DE" and picp is not None and baseline_picp is not None:
        picp_gate = float(picp) >= max(0.0, float(baseline_picp) - 0.02)
    return {
        "domain": domain.domain_label,
        "dataset": domain.dataset,
        "accepted": accepted,
        "manifest_exists": bool(manifest),
        "selection_summary": _repo_rel(selection_summary_path(domain, release_id)),
        "run_manifest": _repo_rel(candidate_manifest_path(domain, release_id)),
        "primary_target": metrics.get("primary_target"),
        "rmse": metrics.get("rmse"),
        "picp_90": picp,
        "baseline_picp_90": baseline_picp,
        "picp_gate": picp_gate,
        "pass": bool(accepted and manifest and picp_gate),
    }


def runtime_gate() -> dict[str, Any]:
    benchmark = pd.read_csv(REPO_ROOT / "reports" / "publication" / "three_domain_ml_benchmark.csv")
    equal = pd.read_csv(REPO_ROOT / "reports" / "publication" / "equal_domain_artifact_discipline.csv")
    external = pd.read_csv(
        REPO_ROOT / "reports" / "predeployment_external_validation" / "external_validation_summary.csv"
    )
    strict = bool(benchmark["strict_runtime_gate"].astype(bool).all())
    equal_pass = bool(equal["artifact_discipline_gate"].astype(bool).all())
    external_pass = bool(external["pass"].astype(bool).all())
    av = benchmark[benchmark["domain"] == "Autonomous Vehicles"].iloc[0].to_dict()
    healthcare = benchmark[benchmark["domain"] == "Medical and Healthcare Monitoring"].iloc[0].to_dict()
    return {
        "strict_runtime_gate": strict,
        "equal_domain_gate": equal_pass,
        "external_predeployment_gate": external_pass,
        "av_orius_tsvr": float(av["orius_tsvr_mean"]),
        "av_fallback_activation_rate": float(av["fallback_activation_rate"]),
        "healthcare_orius_tsvr": float(healthcare["orius_tsvr_mean"]),
        "healthcare_fallback_activation_rate": float(healthcare["fallback_activation_rate"]),
        "pass": bool(
            strict
            and equal_pass
            and external_pass
            and float(av["orius_tsvr_mean"]) <= PROMOTED_RUNTIME_MAX_TSVR
            and float(av["fallback_activation_rate"]) <= 0.50
            and float(healthcare["orius_tsvr_mean"]) == 0.0
            and float(healthcare["fallback_activation_rate"]) <= 0.50
        ),
    }


def runtime_stress_gate(freeze_dir: Path) -> dict[str, Any]:
    manifest = _load_json(freeze_dir / "runtime_stress" / "runtime_stress_manifest.json")
    summary_path = freeze_dir / "runtime_stress" / "runtime_stress_summary.csv"
    if not manifest or not summary_path.exists():
        return {"pass": False, "error": "missing runtime stress artifacts"}
    summary = pd.read_csv(summary_path)
    expected_domains = {domain.domain_label for domain in FREEZE_DOMAINS}
    return {
        "manifest": str(summary_path.parent / "runtime_stress_manifest.json"),
        "all_passed": bool(manifest.get("all_passed", False)),
        "domains": sorted(summary["domain"].unique().tolist()),
        "stress_families": sorted(summary["stress_family"].unique().tolist()),
        "synthetic_source_count": int(summary["synthetic_source"].astype(bool).sum())
        if "synthetic_source" in summary
        else -1,
        "proxy_source_count": int(summary["proxy_source"].astype(bool).sum())
        if "proxy_source" in summary
        else -1,
        "pass": bool(
            manifest.get("all_passed", False)
            and set(summary["domain"]) == expected_domains
            and not summary.empty
            and not summary["synthetic_source"].astype(bool).any()
            and not summary["proxy_source"].astype(bool).any()
            and not summary["validation_harness_source"].astype(bool).any()
            and summary["stress_gate_pass"].astype(bool).all()
        ),
    }


def _iter_files(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if not path.exists():
            continue
        if path.is_file() and not path.name.startswith("._"):
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and not child.name.startswith("._"):
                    yield child


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_hash_rows(release_id: str, freeze_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_paths = [
        REPO_ROOT / "reports" / "publication" / "orius_universal_contract_manifest.json",
        REPO_ROOT / "reports" / "publication" / "domain_runtime_contract_summary.json",
        REPO_ROOT / "reports" / "publication" / "domain_runtime_contract_witnesses.csv",
        REPO_ROOT / "reports" / "publication" / "three_domain_ml_benchmark.csv",
        REPO_ROOT / "reports" / "publication" / "equal_domain_artifact_discipline.csv",
        REPO_ROOT / "reports" / "universal_orius_validation",
        REPO_ROOT / "reports" / "predeployment_external_validation",
        freeze_dir / "runtime_stress",
    ]
    for domain in FREEZE_DOMAINS:
        domain_paths = [
            candidate_root(domain, release_id),
            candidate_reports_dir(domain, release_id),
            domain.canonical_models_dir,
            domain.canonical_uncertainty_dir,
            domain.canonical_backtests_dir,
            domain.canonical_reports_dir / "week2_metrics.json",
            domain.canonical_reports_dir / "formal_evaluation_report.md",
            domain.splits_dir,
        ]
        if domain.domain_key == "battery":
            domain_paths.append(REPO_ROOT / "reports" / "battery_av" / "battery")
        elif domain.domain_key == "av":
            domain_paths.append(AV_RUNTIME_DIR)
            domain_paths.append(nuplan_full_gate_path(freeze_dir))
        elif domain.domain_key == "healthcare":
            domain_paths.append(REPO_ROOT / "reports" / "healthcare")
        for path in _iter_files(domain_paths):
            stat = path.stat()
            rows.append(
                {
                    "domain": domain.domain_label,
                    "artifact_class": "training_runtime_or_split",
                    "path": _repo_rel(path),
                    "sha256": _sha256(path),
                    "size_bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                }
            )
    for path in _iter_files(common_paths):
        stat = path.stat()
        rows.append(
            {
                "domain": "three_domain",
                "artifact_class": "cross_domain_freeze_gate",
                "path": _repo_rel(path),
                "sha256": _sha256(path),
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
            }
        )
    return rows


def write_hash_lock(freeze_dir: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    csv_path = freeze_dir / "frozen_artifact_hashes.csv"
    json_path = freeze_dir / "frozen_artifact_hashes.json"
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"artifacts": rows}, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}


def write_release_manifest(
    *,
    release_id: str,
    freeze_dir: Path,
    plans: Mapping[str, Any],
    training_gates: list[dict[str, Any]],
    runtime: dict[str, Any],
    stress: dict[str, Any],
    hash_paths: dict[str, str],
) -> dict[str, Any]:
    manifest = {
        "release_id": release_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": FREEZE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "git_commit": _git_commit(),
        "dirty_worktree": _dirty_worktree(),
        "domains": [domain.domain_label for domain in FREEZE_DOMAINS],
        "datasets": [domain.dataset for domain in FREEZE_DOMAINS],
        "model_families_requested": str(plans.get("model_request", MODEL_REQUEST)).split(","),
        "av_freeze_surface": NUPLAN_SURFACE,
        "universal_contract_manifest": "reports/publication/orius_universal_contract_manifest.json",
        "n_trials_override": plans.get("n_trials_override"),
        "nuplan_full_gate": plans.get("nuplan_full_gate"),
        "train_commands": plans["training"],
        "av_offline_training_command": plans.get("av_offline_training"),
        "promotion_commands": plans["promotion"],
        "downstream_commands": plans["downstream"],
        "promoted_domains": [row["domain"] for row in training_gates if row["pass"]],
        "training_gates": training_gates,
        "runtime_gates": runtime,
        "stress_gates": stress,
        "hash_artifacts": hash_paths,
        "predeployment_not_deployed_reasons": [
            "not road deployed",
            "not live clinical deployed",
            "not unrestricted field deployed",
            "runtime stress is replay/HIL/monitoring evidence only, not synthetic corruption",
        ],
        "all_passed": bool(all(row["pass"] for row in training_gates) and runtime["pass"] and stress["pass"]),
    }
    path = freeze_dir / "predeployment_release_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _remove_appledouble_files(freeze_dir)
    return manifest


def _run_training_stage(plans: Mapping[str, Any], logs_dir: Path) -> dict[str, bool]:
    if not bool(plans.get("parallel_training_enabled", False)):
        results: dict[str, bool] = {}
        for domain in FREEZE_DOMAINS:
            results[domain.dataset] = _run(
                plans["training"][domain.dataset],
                log_path=logs_dir / f"train_{domain.dataset.lower()}.log",
            )
        if plans.get("av_offline_training"):
            results["AV_OFFLINE"] = _run(
                list(plans["av_offline_training"]),
                log_path=logs_dir / "train_av_offline.log",
            )
        return results

    results = {}
    with ThreadPoolExecutor(
        max_workers=len(FREEZE_DOMAINS) + int(bool(plans.get("av_offline_training")))
    ) as executor:
        future_to_dataset = {
            executor.submit(
                _run,
                plans["training"][domain.dataset],
                log_path=logs_dir / f"train_{domain.dataset.lower()}.log",
            ): domain.dataset
            for domain in FREEZE_DOMAINS
        }
        if plans.get("av_offline_training"):
            future_to_dataset[
                executor.submit(
                    _run,
                    list(plans["av_offline_training"]),
                    log_path=logs_dir / "train_av_offline.log",
                )
            ] = "AV_OFFLINE"
        for future in as_completed(future_to_dataset):
            dataset = future_to_dataset[future]
            results[dataset] = bool(future.result())
    return results


def _blocking_training_results_passed(results: Mapping[str, bool]) -> bool:
    """Only promoted freeze domains block release; AV_OFFLINE is auxiliary evidence."""
    return all(bool(results.get(domain.dataset, False)) for domain in FREEZE_DOMAINS)


def run_freeze(
    *,
    release_id: str,
    out_dir: Path,
    profile: str,
    max_runtime_hours: float | None,
    n_trials: int | None = None,
    model_request: str = MODEL_REQUEST,
    parallel_training: bool | None = None,
    dry_run: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    freeze_dir = out_dir / release_id
    freeze_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = freeze_dir / "logs"
    plans = planned_commands(
        release_id,
        freeze_dir,
        profile=profile,
        max_runtime_hours=max_runtime_hours,
        n_trials=n_trials,
        model_request=model_request,
        parallel_training=parallel_training,
    )
    (freeze_dir / "planned_commands.json").write_text(json.dumps(plans, indent=2), encoding="utf-8")
    preflight = write_preflight_manifest(
        release_id=release_id,
        freeze_dir=freeze_dir,
        plans=plans,
        profile=profile,
        max_runtime_hours=max_runtime_hours,
        n_trials=n_trials,
        model_request=model_request,
    )
    _remove_appledouble_files(freeze_dir)
    if dry_run or preflight_only:
        return {
            "release_id": release_id,
            "dry_run": True,
            "preflight_pass": preflight["pass"],
            "planned_commands": str(freeze_dir / "planned_commands.json"),
            "preflight_manifest": str(freeze_dir / "preflight_manifest.json"),
        }
    if not preflight["pass"]:
        return {
            "release_id": release_id,
            "all_passed": False,
            "stage": "preflight",
            "preflight_manifest": str(freeze_dir / "preflight_manifest.json"),
            "blockers": preflight["blockers"],
        }

    train_results = _run_training_stage(plans, logs_dir)
    if not _blocking_training_results_passed(train_results):
        return {"release_id": release_id, "all_passed": False, "stage": "training", "results": train_results}

    training_gates = [training_gate(domain, release_id, freeze_dir=freeze_dir) for domain in FREEZE_DOMAINS]
    if not all(row["pass"] for row in training_gates):
        return {
            "release_id": release_id,
            "all_passed": False,
            "stage": "training_gates",
            "training_gates": training_gates,
        }

    for domain in FREEZE_DOMAINS:
        ok = _run(
            plans["promotion"][domain.dataset], log_path=logs_dir / f"promote_{domain.dataset.lower()}.log"
        )
        if not ok:
            return {
                "release_id": release_id,
                "all_passed": False,
                "stage": "promotion",
                "domain": domain.dataset,
            }

    downstream_results: list[dict[str, Any]] = []
    for index, cmd in enumerate(plans["downstream"]):
        ok = _run(cmd, log_path=logs_dir / f"downstream_{index:02d}.log")
        downstream_results.append({"command": cmd, "pass": ok})
        if not ok:
            return {
                "release_id": release_id,
                "all_passed": False,
                "stage": "downstream",
                "downstream": downstream_results,
            }

    runtime = runtime_gate()
    stress = runtime_stress_gate(freeze_dir)
    if not runtime["pass"] or not stress["pass"]:
        return {
            "release_id": release_id,
            "all_passed": False,
            "stage": "final_gates",
            "runtime": runtime,
            "stress": stress,
        }

    hash_rows = collect_hash_rows(release_id, freeze_dir)
    hash_paths = write_hash_lock(freeze_dir, hash_rows)
    manifest = write_release_manifest(
        release_id=release_id,
        freeze_dir=freeze_dir,
        plans=plans,
        training_gates=training_gates,
        runtime=runtime,
        stress=stress,
        hash_paths=hash_paths,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full three-domain offline predeployment freeze.")
    parser.add_argument("--release-id", default=None)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "predeployment_freeze")
    parser.add_argument(
        "--profile",
        choices=["standard", "aggressive", "max", "production-max-fast"],
        default="aggressive",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=0.0,
        help="Optional timeout for each model-training invocation. Use 0 or omit for no timeout.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Optional Optuna trial override for trainable datasets. AV nuPlan gate remains a separate validation gate.",
    )
    parser.add_argument(
        "--models",
        default=MODEL_REQUEST,
        help="Comma-separated model families for trainable datasets and AV offline training.",
    )
    parser.add_argument(
        "--sequential-training",
        action="store_true",
        help="Disable production-max-fast parallel training. Useful for laptop smoke runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight-only", action="store_true", help="Write planned commands and preflight manifest only"
    )
    args = parser.parse_args()

    release_id = args.release_id or generate_release_id()
    result = run_freeze(
        release_id=release_id,
        out_dir=args.out,
        profile=args.profile,
        max_runtime_hours=args.max_runtime_hours,
        n_trials=args.n_trials,
        model_request=args.models,
        parallel_training=False if args.sequential_training else None,
        dry_run=args.dry_run,
        preflight_only=args.preflight_only,
    )
    print(json.dumps(result, indent=2))
    if args.preflight_only:
        return 0 if bool(result.get("preflight_pass", False)) else 1
    return 0 if bool(result.get("all_passed", result.get("dry_run", False))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
