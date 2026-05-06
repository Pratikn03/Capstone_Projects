#!/usr/bin/env python3
"""
Unified Dataset Training Script.

Single entry point to train any registered dataset with consistent logic.
Supports German (OPSD), US (EIA-930), and easily extensible for new datasets.

Usage:
    # Train a specific dataset
    python scripts/train_dataset.py --dataset DE
    python scripts/train_dataset.py --dataset US

    # Train all registered datasets
    python scripts/train_dataset.py --all

    # Train with hyperparameter tuning
    python scripts/train_dataset.py --dataset DE --tune

    # Generate reports + conformal coverage
    python scripts/train_dataset.py --dataset DE --reports

Adding New Datasets:
    1. Create config file: configs/train_forecast_{name}.yaml
    2. Add entry to DATASET_REGISTRY below with paths
    3. Implement feature pipeline if needed (or reuse existing)

Author: ORIUS Team
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Support both `python scripts/train_dataset.py` (cwd=scripts/) and
# `import scripts.train_dataset` (cwd=repo root, used by tests).
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from _dataset_registry import (
    AGGRESSIVE_DEFAULTS,
    DATASET_REGISTRY,
    MAX_QUALITY_DEFAULTS,
    PRODUCTION_MAX_FAST_DEFAULTS,
    REPO_ROOT,
    DatasetConfig,
    RunLayout,
)
from _dataset_registry import (
    iter_trainable_dataset_keys as _iter_trainable_dataset_keys,
)

PYTHON_BIN = sys.executable or "python3"
TRAINING_PROFILES = ("standard", "aggressive", "max", "production-max-fast")
REPORT_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
    "TORCH_NUM_INTEROP_THREADS": "1",
}


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================


def _load_publish_audit_cfg(path: str = "configs/publish_audit.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    return payload.get("publish_audit", {}) if isinstance(payload.get("publish_audit"), dict) else {}


def _load_training_cfg(cfg: DatasetConfig) -> dict[str, Any]:
    cfg_path = REPO_ROOT / cfg.config_file
    if not cfg_path.exists():
        return {}
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _configured_targets(train_cfg: dict[str, Any]) -> list[str]:
    task_cfg = train_cfg.get("task", {}) if isinstance(train_cfg.get("task"), dict) else {}
    targets = task_cfg.get("targets", [])
    if isinstance(targets, list):
        return [str(target).strip() for target in targets if str(target).strip()]
    return ["load_mw"]


def _configured_uncertainty_targets(path: Path = REPO_ROOT / "configs" / "uncertainty.yaml") -> list[str]:
    if not path.exists():
        return []
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return []
    targets = payload.get("targets", [])
    if isinstance(targets, list):
        return [str(target).strip() for target in targets if str(target).strip()]
    if isinstance(targets, str):
        return [part.strip() for part in targets.split(",") if part.strip()]
    return []


def _resolved_uncertainty_targets(
    cfg: DatasetConfig,
    train_cfg: dict[str, Any] | None = None,
    *,
    uncertainty_cfg_path: Path = REPO_ROOT / "configs" / "uncertainty.yaml",
) -> list[str]:
    """Resolve the uncertainty-target contract for a dataset.

    Multi-domain datasets use every configured training target for uncertainty
    artifacts. Energy/battery datasets follow ``configs/uncertainty.yaml`` and
    only fall back to all training targets if that config is empty.
    """
    train_cfg = train_cfg or _load_training_cfg(cfg)
    targets = _configured_targets(train_cfg)
    dataset_cfg = train_cfg.get("dataset", {}) if isinstance(train_cfg.get("dataset"), dict) else {}
    is_multi_domain = str(dataset_cfg.get("region_group", "") or "") == "multi-domain"
    if is_multi_domain:
        return targets
    uncertainty_targets = _configured_uncertainty_targets(uncertainty_cfg_path)
    return uncertainty_targets or targets


def _configured_model_types(train_cfg: dict[str, Any], requested_models: set[str] | None = None) -> list[str]:
    models_cfg = train_cfg.get("models", {}) if isinstance(train_cfg.get("models"), dict) else {}
    model_types: list[str] = []

    gbm_cfg = models_cfg.get("baseline_gbm", {}) if isinstance(models_cfg.get("baseline_gbm"), dict) else {}
    if gbm_cfg.get("enabled", True) and (requested_models is None or "gbm" in requested_models):
        kind = str(gbm_cfg.get("kind", "lightgbm")).strip().lower() or "lightgbm"
        model_types.append(f"gbm_{kind}")

    lstm_cfg = models_cfg.get("dl_lstm", {}) if isinstance(models_cfg.get("dl_lstm"), dict) else {}
    if lstm_cfg.get("enabled", True) and (requested_models is None or "lstm" in requested_models):
        model_types.append("lstm")

    tcn_cfg = models_cfg.get("dl_tcn", {}) if isinstance(models_cfg.get("dl_tcn"), dict) else {}
    if tcn_cfg.get("enabled", False) and (requested_models is None or "tcn" in requested_models):
        model_types.append("tcn")

    nbeats_cfg = models_cfg.get("dl_nbeats", {}) if isinstance(models_cfg.get("dl_nbeats"), dict) else {}
    if nbeats_cfg.get("enabled", False) and (requested_models is None or "nbeats" in requested_models):
        model_types.append("nbeats")

    tft_cfg = models_cfg.get("dl_tft", {}) if isinstance(models_cfg.get("dl_tft"), dict) else {}
    if tft_cfg.get("enabled", False) and (requested_models is None or "tft" in requested_models):
        model_types.append("tft")

    patchtst_cfg = (
        models_cfg.get("dl_patchtst", {}) if isinstance(models_cfg.get("dl_patchtst"), dict) else {}
    )
    if patchtst_cfg.get("enabled", False) and (requested_models is None or "patchtst" in requested_models):
        model_types.append("patchtst")

    return model_types


def _resolve_run_layout(cfg: DatasetConfig, *, candidate_run: bool, run_id: str | None) -> RunLayout:
    normalized_dataset = cfg.name.lower()
    if candidate_run:
        resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        artifacts_root = REPO_ROOT / "artifacts" / "runs" / normalized_dataset / resolved_run_id
        reports_root = REPO_ROOT / "reports" / "runs" / normalized_dataset / resolved_run_id
        return RunLayout(
            mode="candidate",
            run_id=resolved_run_id,
            dataset=cfg.name,
            artifacts_root=artifacts_root,
            models_dir=artifacts_root / "models",
            uncertainty_dir=artifacts_root / "uncertainty",
            backtests_dir=artifacts_root / "backtests",
            registry_dir=artifacts_root / "registry",
            reports_dir=reports_root,
            publication_dir=reports_root / "publication",
            validation_report=reports_root / "data_quality_report_features.md",
            data_manifest_output=artifacts_root / "registry" / "data_manifest.json",
            walk_forward_report=reports_root / "walk_forward_report.json",
            selection_output_dir=artifacts_root / "registry",
        )

    return RunLayout(
        mode="canonical",
        run_id=run_id or "canonical",
        dataset=cfg.name,
        artifacts_root=REPO_ROOT / "artifacts",
        models_dir=REPO_ROOT / cfg.models_dir,
        uncertainty_dir=REPO_ROOT / cfg.uncertainty_dir,
        backtests_dir=REPO_ROOT / cfg.backtests_dir,
        registry_dir=REPO_ROOT / "artifacts" / "registry",
        reports_dir=REPO_ROOT / cfg.reports_dir,
        publication_dir=REPO_ROOT / "reports" / "publication",
        validation_report=REPO_ROOT / "reports" / f"data_quality_report_{cfg.name.lower()}_features.md",
        data_manifest_output=REPO_ROOT / "paper" / "assets" / "data" / "data_manifest.json",
        walk_forward_report=REPO_ROOT / cfg.reports_dir / "walk_forward_report.json",
        selection_output_dir=REPO_ROOT / "reports" / "publish",
    )


def _safe_iso(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _preflight_order_columns(train_cfg: dict[str, Any], frame: pd.DataFrame) -> list[str]:
    data_cfg = train_cfg.get("data", {}) if isinstance(train_cfg.get("data"), dict) else {}
    configured_order = data_cfg.get("order_cols")
    if isinstance(configured_order, list):
        order_cols = [str(col).strip() for col in configured_order if str(col).strip()]
        existing = [col for col in order_cols if col in frame.columns]
        if existing:
            return existing
    timestamp_col = str(data_cfg.get("timestamp_col") or "timestamp").strip()
    if timestamp_col and timestamp_col in frame.columns:
        return [timestamp_col]
    if "timestamp" in frame.columns:
        return ["timestamp"]
    return []


def _sort_preflight_frame(train_cfg: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    order_cols = _preflight_order_columns(train_cfg, frame)
    if not order_cols:
        return frame.reset_index(drop=True)
    return frame.sort_values(order_cols).reset_index(drop=True)


def _build_preflight_analysis(
    cfg: DatasetConfig,
    run_layout: RunLayout,
    *,
    profile: str,
    models: str | None = None,
) -> dict[str, Any]:
    train_cfg = _load_training_cfg(cfg)
    features_path = REPO_ROOT / cfg.features_path
    df = _sort_preflight_frame(train_cfg, pd.read_parquet(features_path))
    expected_targets = _configured_targets(train_cfg)
    requested_models = (
        {part.strip().lower() for part in models.split(",") if part.strip()} if models else None
    )
    expected_models = _configured_model_types(train_cfg, requested_models=requested_models)

    split_sizes: dict[str, Any] = {}
    split_ranges: dict[str, Any] = {}
    for split_name in ("train", "calibration", "val", "test"):
        split_path = REPO_ROOT / cfg.splits_path / f"{split_name}.parquet"
        if not split_path.exists():
            continue
        split_df = pd.read_parquet(split_path)
        split_sizes[split_name] = int(len(split_df))
        if "timestamp" in split_df.columns and not split_df.empty:
            split_ranges[split_name] = {
                "start": _safe_iso(split_df["timestamp"].min()),
                "end": _safe_iso(split_df["timestamp"].max()),
            }

    target_presence: dict[str, Any] = {}
    for target in expected_targets:
        present = target in df.columns
        null_ratio = float(df[target].isna().mean()) if present else None
        target_presence[target] = {
            "present": present,
            "non_null_rows": int(df[target].notna().sum()) if present else 0,
            "null_ratio": null_ratio,
        }

    null_ratios = {
        column: float(df[column].isna().mean())
        for column in df.columns
        if float(df[column].isna().mean()) > 0.0
    }

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset": cfg.name,
        "profile": profile,
        "features_path": cfg.features_path,
        "splits_path": cfg.splits_path,
        "row_count": int(len(df)),
        "date_range": {
            "start": _safe_iso(df["timestamp"].min()) if "timestamp" in df.columns and not df.empty else None,
            "end": _safe_iso(df["timestamp"].max()) if "timestamp" in df.columns and not df.empty else None,
        },
        "expected_targets": expected_targets,
        "expected_model_types": expected_models,
        "target_presence": target_presence,
        "null_ratios": null_ratios,
        "split_sizes": split_sizes,
        "split_ranges": split_ranges,
        "gap_hours": int(
            (
                (train_cfg.get("splits") or train_cfg.get("split") or {})
                if isinstance(train_cfg.get("splits") or train_cfg.get("split") or {}, dict)
                else {}
            ).get("gap_hours", 0)
            or 0
        ),
        "expected_target_model_matrix": [
            {"target": target, "models": expected_models} for target in expected_targets
        ],
        "output_layout": {
            "mode": run_layout.mode,
            "run_id": run_layout.run_id,
            "models_dir": str(run_layout.models_dir.relative_to(REPO_ROOT)),
            "reports_dir": str(run_layout.reports_dir.relative_to(REPO_ROOT)),
            "uncertainty_dir": str(run_layout.uncertainty_dir.relative_to(REPO_ROOT)),
            "backtests_dir": str(run_layout.backtests_dir.relative_to(REPO_ROOT)),
        },
    }


def _write_preflight_analysis(
    cfg: DatasetConfig,
    run_layout: RunLayout,
    *,
    profile: str,
    models: str | None = None,
) -> dict[str, Any]:
    payload = _build_preflight_analysis(cfg, run_layout, profile=profile, models=models)
    run_layout.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_layout.reports_dir / "preflight_dataset_analysis.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _write_run_context(
    *,
    cfg: DatasetConfig,
    run_layout: RunLayout,
    preflight: dict[str, Any],
    profile: str,
    models: str | None = None,
) -> None:
    run_layout.registry_dir.mkdir(parents=True, exist_ok=True)
    release_id = _derive_release_id(run_layout.run_id)
    context = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "release_id": release_id,
        "dataset": cfg.name,
        "display_name": cfg.display_name,
        "profile": profile,
        "requested_models": models,
        "mode": run_layout.mode,
        "run_mode": run_layout.mode,
        "run_id": run_layout.run_id,
        "git_commit": _git_commit(),
        "config_path": cfg.config_file,
        "config_hash": _config_hash(cfg),
        "features_path": cfg.features_path,
        "splits_path": cfg.splits_path,
        "models_dir": str(run_layout.models_dir),
        "reports_dir": str(run_layout.reports_dir),
        "uncertainty_dir": str(run_layout.uncertainty_dir),
        "backtests_dir": str(run_layout.backtests_dir),
        "expected_targets": preflight.get("expected_targets", []),
        "expected_model_types": preflight.get("expected_model_types", []),
        "preflight_analysis": str(run_layout.reports_dir / "preflight_dataset_analysis.json"),
        "preflight_path": str(run_layout.reports_dir / "preflight_dataset_analysis.json"),
        "feature_manifest_path": str(run_layout.data_manifest_output),
        "selection_summary_path": str(
            run_layout.selection_output_dir / f"tuning_summary_{cfg.name.lower()}.json"
        ),
        "artifacts": {
            "models_dir": str(run_layout.models_dir),
            "reports_dir": str(run_layout.reports_dir),
            "uncertainty_dir": str(run_layout.uncertainty_dir),
            "backtests_dir": str(run_layout.backtests_dir),
            "publication_dir": str(run_layout.publication_dir),
        },
        "targets": preflight.get("expected_targets", []),
        "accepted": None,
        "promoted_at": None,
    }
    (run_layout.registry_dir / "run_context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    _write_run_manifest(run_layout.registry_dir / "run_manifest.json", context)


def _write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_run_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _config_hash(cfg: DatasetConfig) -> str:
    config_path = REPO_ROOT / cfg.config_file
    if not config_path.exists():
        return "missing"
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _derive_release_id(run_id: str) -> str:
    return run_id[:-5] if run_id.endswith("_diag") else run_id


def _update_run_manifest(
    cfg: DatasetConfig,
    run_layout: RunLayout,
    *,
    accepted: bool | None = None,
    promoted_at: str | None = None,
) -> None:
    manifest_path = run_layout.registry_dir / "run_manifest.json"
    payload = _read_run_manifest(manifest_path)
    if not payload:
        return
    payload.setdefault("release_id", _derive_release_id(run_layout.run_id))
    payload.setdefault("run_id", run_layout.run_id)
    payload.setdefault("dataset", cfg.name)
    payload.setdefault("mode", run_layout.mode)
    payload.setdefault("profile", payload.get("profile"))
    payload.setdefault("git_commit", _git_commit())
    payload.setdefault("config_path", cfg.config_file)
    payload.setdefault("config_hash", _config_hash(cfg))
    payload.setdefault("feature_manifest_path", str(run_layout.data_manifest_output))
    payload.setdefault("preflight_path", str(run_layout.reports_dir / "preflight_dataset_analysis.json"))
    payload.setdefault(
        "artifacts",
        {
            "models_dir": str(run_layout.models_dir),
            "reports_dir": str(run_layout.reports_dir),
            "uncertainty_dir": str(run_layout.uncertainty_dir),
            "backtests_dir": str(run_layout.backtests_dir),
            "publication_dir": str(run_layout.publication_dir),
        },
    )
    payload.setdefault("targets", payload.get("expected_targets", []))
    payload["selection_summary_path"] = str(
        run_layout.selection_output_dir / f"tuning_summary_{cfg.name.lower()}.json"
    )
    if accepted is not None:
        payload["accepted"] = bool(accepted)
    if promoted_at is not None:
        payload["promoted_at"] = promoted_at
    _write_run_manifest(manifest_path, payload)


def run_command(cmd: list[str], description: str, timeout_seconds: float | None = None) -> bool:
    """
    Execute a subprocess command with logging.

    Args:
        cmd: Command and arguments
        description: Human-readable description

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"📌 {description}")
    print(f"{'=' * 60}")
    print(f"   Command: {' '.join(cmd)}")
    print()

    cmd_env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing_pythonpath = cmd_env.get("PYTHONPATH", "")
    cmd_env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path
    )
    cmd_env.setdefault("MPLBACKEND", "Agg")
    cmd_env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
    cmd_env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache-orius")
    cmd_env.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    if "scripts/build_reports.py" in cmd:
        cmd_env.update(REPORT_THREAD_ENV)

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            timeout=timeout_seconds,
            cwd=str(REPO_ROOT),
            env=cmd_env,
        )
        print(f"✅ {description} - completed successfully")
        return True
    except subprocess.TimeoutExpired:
        print(f"❌ {description} - timed out after {timeout_seconds} seconds")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - failed with exit code {e.returncode}")
        return False


def build_features(cfg: DatasetConfig, force: bool = False) -> bool:
    """
    Build engineered features for a dataset.

    Args:
        cfg: Dataset configuration
        force: Rebuild even if features exist

    Returns:
        True if successful
    """
    features_path = Path(cfg.features_path)

    if features_path.exists() and not force:
        print(f"ℹ️  Features already exist: {features_path}")
        return True

    if cfg.feature_module == "orius.data_pipeline.build_features_healthcare" and str(
        cfg.raw_data_path
    ).endswith("mimic3_healthcare_orius.csv"):
        # The promoted healthcare lane uses bridge-schema inputs that must go
        # through the max-input builder instead of the legacy single-source path.
        healthcare_inputs = []
        for candidate in (
            Path("data/healthcare/processed/healthcare_bidmc_orius.csv"),
            Path(cfg.raw_data_path),
        ):
            if candidate.exists():
                healthcare_inputs.append(candidate)
        if not healthcare_inputs:
            healthcare_inputs.append(Path(cfg.raw_data_path))

        cmd = [PYTHON_BIN, "-m", cfg.feature_module, "--max-input"]
        for input_path in healthcare_inputs:
            cmd.extend(["--in", str(input_path)])
        cmd.extend(["--out", str(features_path.parent)])
    else:
        # Build command based on dataset type
        cmd = [
            PYTHON_BIN,
            "-m",
            cfg.feature_module,
            "--in",
            cfg.raw_data_path,
            "--out",
            str(features_path.parent),
        ]

    # Add balancing authority for US data
    if cfg.ba_code:
        cmd.extend(["--ba", cfg.ba_code])

    # Add date filters if specified
    if cfg.start_date:
        cmd.extend(["--start", cfg.start_date])
    if cfg.end_date:
        cmd.extend(["--end", cfg.end_date])

    return run_command(cmd, f"Building features for {cfg.display_name}")


def create_splits(cfg: DatasetConfig, force: bool = False) -> bool:
    """
    Create temporal train/val/test splits.

    Args:
        cfg: Dataset configuration
        force: Recreate even if splits exist

    Returns:
        True if successful
    """
    splits_path = Path(cfg.splits_path)

    if cfg.feature_module == "orius.data_pipeline.build_features_healthcare" and splits_path.exists():
        summary_path = splits_path / "SPLIT_SUMMARY.md"
        if summary_path.exists():
            print(f"ℹ️  Healthcare feature builder already produced patient-disjoint splits: {splits_path}")
            return True

    if splits_path.exists() and not force:
        print(f"ℹ️  Splits already exist: {splits_path}")
        return True

    split_cfg = {}
    cfg_path = Path(cfg.config_file)
    if cfg_path.exists():
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            candidate = payload.get("splits") or payload.get("split") or {}
            if isinstance(candidate, dict):
                split_cfg = candidate

    cmd = [
        PYTHON_BIN,
        "-m",
        "orius.data_pipeline.split_time_series",
        "--in",
        cfg.features_path,
        "--out",
        cfg.splits_path,
        "--train-ratio",
        str(float(split_cfg.get("train_ratio", 0.70))),
        "--calibration-ratio",
        str(float(split_cfg.get("calibration_ratio", 0.0) or 0.0)),
        "--val-ratio",
        str(float(split_cfg.get("val_ratio", 0.15))),
        "--gap-hours",
        str(int(split_cfg.get("gap_hours", 0) or 0)),
    ]

    return run_command(cmd, f"Creating time series splits for {cfg.display_name}")


def validate_features_schema(cfg: DatasetConfig, report_path: Path | None = None) -> bool:
    """Validate processed features schema before training."""
    target_report = report_path or (
        REPO_ROOT / "reports" / f"data_quality_report_{cfg.name.lower()}_features.md"
    )
    target_report.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN,
        "-m",
        "orius.data_pipeline.validate_schema",
        "--in",
        cfg.features_path,
        "--report",
        str(target_report),
    ]
    # Multi-domain datasets use different target columns
    train_cfg = _load_training_cfg(cfg)
    targets = _configured_targets(train_cfg)
    if targets and cfg.name in ("AV", "INDUSTRIAL", "HEALTHCARE", "AEROSPACE", "NAVIGATION"):
        cmd.extend(["--required-cols", ",".join(targets)])
    data_cfg = train_cfg.get("data", {}) if isinstance(train_cfg.get("data"), dict) else {}
    order_cols = data_cfg.get("order_cols")
    if isinstance(order_cols, list) and order_cols:
        cmd.extend(["--order-cols", ",".join(str(col).strip() for col in order_cols if str(col).strip())])
    elif data_cfg.get("timestamp_col"):
        cmd.extend(["--timestamp-col", str(data_cfg["timestamp_col"])])
    return run_command(cmd, f"Validating features schema for {cfg.display_name}")


def refresh_data_manifest(cfg: DatasetConfig, output_path: Path | None = None) -> bool:
    """Build deterministic data manifest before model training."""
    manifest_path = output_path or (REPO_ROOT / "paper" / "assets" / "data" / "data_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON_BIN,
        "scripts/build_data_manifest.py",
        "--dataset",
        cfg.name,
        "--output",
        str(manifest_path),
    ]
    return run_command(cmd, f"Building data manifest for {cfg.display_name}")


def train_models(
    cfg: DatasetConfig,
    run_layout: RunLayout | None = None,
    models: str | None = None,
    tune: bool = False,
    no_tune: bool = False,
    no_cv: bool = False,
    ensemble: bool = False,
    max_seeds: int | None = None,
    n_trials: int | None = None,
    top_pct: float | None = None,
    tuning_n_jobs: int | None = None,
    gbm_threads: int | None = None,
    reuse_best_gbm_from: str | None = None,
    max_deep_epochs: int | None = None,
    deep_patience: int | None = None,
    deep_warmup_epochs: int | None = None,
    profile: str = "standard",
    max_runtime_hours: float | None = None,
    target_metrics_file: str | None = None,
) -> bool:
    """
    Train forecasting models.

    Args:
        cfg: Dataset configuration
        tune: Enable hyperparameter tuning with Optuna

    Returns:
        True if successful
    """
    effective_tune = bool(tune)
    effective_ensemble = bool(ensemble)
    effective_max_seeds = max_seeds
    effective_n_trials = n_trials
    effective_top_pct = top_pct
    effective_tuning_n_jobs = tuning_n_jobs
    effective_gbm_threads = gbm_threads
    effective_reuse_best_gbm = reuse_best_gbm_from
    effective_max_deep_epochs = max_deep_epochs
    effective_deep_patience = deep_patience
    effective_deep_warmup_epochs = deep_warmup_epochs

    profile_defaults = None
    if profile == "aggressive":
        profile_defaults = AGGRESSIVE_DEFAULTS
    elif profile == "max":
        profile_defaults = MAX_QUALITY_DEFAULTS
    elif profile == "production-max-fast":
        profile_defaults = PRODUCTION_MAX_FAST_DEFAULTS

    if profile_defaults is not None:
        defaults = profile_defaults.get(cfg.name, profile_defaults["DE"])
        effective_tune = True
        effective_ensemble = True
        if effective_max_seeds is None:
            effective_max_seeds = int(defaults["max_seeds"])
        if effective_n_trials is None:
            effective_n_trials = int(defaults["n_trials"])
        if effective_top_pct is None:
            effective_top_pct = float(defaults["top_pct"])
        if effective_tuning_n_jobs is None and "tuning_n_jobs" in defaults:
            effective_tuning_n_jobs = int(defaults["tuning_n_jobs"])
        if effective_gbm_threads is None and "gbm_threads" in defaults:
            effective_gbm_threads = int(defaults["gbm_threads"])
        if effective_reuse_best_gbm is None and defaults.get("reuse_best_gbm") and target_metrics_file:
            effective_reuse_best_gbm = target_metrics_file
        if effective_max_deep_epochs is None and "max_deep_epochs" in defaults:
            effective_max_deep_epochs = int(defaults["max_deep_epochs"])
        if effective_deep_patience is None and "deep_patience" in defaults:
            effective_deep_patience = int(defaults["deep_patience"])
        if effective_deep_warmup_epochs is None and "deep_warmup_epochs" in defaults:
            effective_deep_warmup_epochs = int(defaults["deep_warmup_epochs"])

    if (
        profile == "production-max-fast"
        and n_trials is not None
        and int(n_trials) <= 10
        and max_deep_epochs is None
    ):
        effective_max_deep_epochs = 2
        effective_deep_patience = 1 if deep_patience is None else effective_deep_patience
        effective_deep_warmup_epochs = 1 if deep_warmup_epochs is None else effective_deep_warmup_epochs

    cmd = [
        PYTHON_BIN,
        "-m",
        "orius.forecasting.train",
        "--config",
        cfg.config_file,
    ]
    if models:
        cmd.extend(["--models", models])
    if run_layout is not None:
        cmd.extend(
            [
                "--artifacts-dir",
                str(run_layout.models_dir),
                "--reports-dir",
                str(run_layout.reports_dir),
                "--uncertainty-artifacts-dir",
                str(run_layout.uncertainty_dir),
                "--backtests-dir",
                str(run_layout.backtests_dir),
                "--walk-forward-report",
                str(run_layout.walk_forward_report),
                "--validation-report",
                str(run_layout.validation_report),
                "--data-manifest-output",
                str(run_layout.data_manifest_output),
            ]
        )

    if effective_tune:
        cmd.append("--tune")
    if no_tune:
        cmd.append("--no-tune")
    if no_cv:
        cmd.append("--no-cv")
    if effective_ensemble:
        cmd.append("--ensemble")
    if effective_max_seeds is not None and effective_max_seeds > 0:
        cmd.extend(["--max-seeds", str(int(effective_max_seeds))])
    if effective_n_trials is not None and effective_n_trials > 0:
        cmd.extend(["--n-trials", str(int(effective_n_trials))])
    if effective_top_pct is not None:
        cmd.extend(["--top-pct", str(float(effective_top_pct))])
    if effective_tuning_n_jobs is not None and effective_tuning_n_jobs > 0:
        cmd.extend(["--tuning-n-jobs", str(int(effective_tuning_n_jobs))])
    if effective_gbm_threads is not None and effective_gbm_threads > 0:
        cmd.extend(["--gbm-threads", str(int(effective_gbm_threads))])
    if effective_reuse_best_gbm:
        cmd.extend(["--reuse-best-gbm-from", str(effective_reuse_best_gbm)])
    if effective_max_deep_epochs is not None and effective_max_deep_epochs > 0:
        cmd.extend(["--max-deep-epochs", str(int(effective_max_deep_epochs))])
    if effective_deep_patience is not None and effective_deep_patience > 0:
        cmd.extend(["--deep-patience", str(int(effective_deep_patience))])
    if effective_deep_warmup_epochs is not None and effective_deep_warmup_epochs > 0:
        cmd.extend(["--deep-warmup-epochs", str(int(effective_deep_warmup_epochs))])

    timeout_seconds = None
    if max_runtime_hours is not None and max_runtime_hours > 0:
        timeout_seconds = float(max_runtime_hours) * 3600.0

    return run_command(cmd, f"Training models for {cfg.display_name}", timeout_seconds=timeout_seconds)


def generate_reports(cfg: DatasetConfig, run_layout: RunLayout | None = None) -> bool:
    """
    Generate evaluation reports including conformal coverage.

    Args:
        cfg: Dataset configuration

    Returns:
        True if successful
    """
    cmd = [
        PYTHON_BIN,
        "scripts/build_reports.py",
        "--features",
        cfg.features_path,
        "--splits",
        cfg.splits_path,
        "--models-dir",
        cfg.models_dir if run_layout is None else str(run_layout.models_dir),
        "--reports-dir",
        cfg.reports_dir if run_layout is None else str(run_layout.reports_dir),
    ]
    if run_layout is not None:
        cmd.extend(
            [
                "--publication-dir",
                str(run_layout.publication_dir),
                "--uncertainty-artifacts-dir",
                str(run_layout.uncertainty_dir),
                "--backtests-dir",
                str(run_layout.backtests_dir),
                "--current-dataset",
                cfg.name,
            ]
        )
        if cfg.name in ("AV", "INDUSTRIAL", "HEALTHCARE", "AEROSPACE", "NAVIGATION"):
            train_cfg = _load_training_cfg(cfg)
            targets = _configured_targets(train_cfg)
            if targets:
                cmd.extend(["--targets", ",".join(targets)])

    return run_command(cmd, f"Generating reports for {cfg.display_name}")


def run_conformal_intervals(cfg: DatasetConfig) -> bool:
    """
    Compute conformal prediction intervals for uncertainty quantification.

    Args:
        cfg: Dataset configuration

    Returns:
        True if successful
    """
    # Check if conformal script exists
    conformal_script = Path("scripts/compute_conformal.py")
    if not conformal_script.exists():
        print("⚠️  Conformal intervals script not found, using build_reports")
        return True  # Reports include conformal if configured

    cmd = [
        PYTHON_BIN,
        str(conformal_script),
        "--config",
        cfg.config_file,
    ]

    return run_command(cmd, f"Computing conformal intervals for {cfg.display_name}")


def verify_training_outputs(cfg: DatasetConfig, run_layout: RunLayout, *, models: str | None = None) -> bool:
    """Verify that a run produced the expected artifacts and per-target metrics."""
    train_cfg = _load_training_cfg(cfg)
    targets = _configured_targets(train_cfg)
    uncertainty_targets = _resolved_uncertainty_targets(cfg, train_cfg)
    requested_models = (
        {part.strip().lower() for part in models.split(",") if part.strip()} if models else None
    )
    model_types = _configured_model_types(train_cfg, requested_models=requested_models)
    cmd = [
        PYTHON_BIN,
        "scripts/verify_training_outputs.py",
        "--models-dir",
        str(run_layout.models_dir),
        "--reports-dir",
        str(run_layout.reports_dir),
        "--artifacts-dir",
        str(run_layout.artifacts_root),
        "--uncertainty-dir",
        str(run_layout.uncertainty_dir),
        "--backtests-dir",
        str(run_layout.backtests_dir),
        "--targets",
        *targets,
        "--uncertainty-targets",
        *uncertainty_targets,
        "--model-types",
        *model_types,
    ]
    return run_command(cmd, f"Verifying training outputs for {cfg.display_name}")


def _copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _promote_candidate_run(cfg: DatasetConfig, run_layout: RunLayout, evaluation: dict[str, Any]) -> None:
    canonical_models_dir = REPO_ROOT / cfg.models_dir
    canonical_reports_dir = REPO_ROOT / cfg.reports_dir
    canonical_uncertainty_dir = REPO_ROOT / cfg.uncertainty_dir
    canonical_backtests_dir = REPO_ROOT / cfg.backtests_dir

    _copy_tree_contents(run_layout.models_dir, canonical_models_dir)
    _copy_tree_contents(run_layout.uncertainty_dir, canonical_uncertainty_dir)
    _copy_tree_contents(run_layout.backtests_dir, canonical_backtests_dir)

    canonical_reports_dir.mkdir(parents=True, exist_ok=True)
    for item in run_layout.reports_dir.iterdir():
        if item.name == "publication":
            continue
        target = canonical_reports_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    publish_dir = REPO_ROOT / "reports" / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)
    promoted_at = datetime.now(UTC).isoformat()
    promotion_record = {
        "generated_at_utc": promoted_at,
        "dataset": cfg.name,
        "release_id": _derive_release_id(run_layout.run_id),
        "source_run_id": run_layout.run_id,
        "source_reports_dir": str(run_layout.reports_dir),
        "source_models_dir": str(run_layout.models_dir),
        "promoted_to": {
            "models_dir": str(canonical_models_dir),
            "reports_dir": str(canonical_reports_dir),
            "uncertainty_dir": str(canonical_uncertainty_dir),
            "backtests_dir": str(canonical_backtests_dir),
        },
        "accepted": bool(evaluation.get("accepted", False)),
        "promoted_at": promoted_at,
    }
    (run_layout.registry_dir / "promotion_record.json").write_text(
        json.dumps(promotion_record, indent=2),
        encoding="utf-8",
    )
    (publish_dir / f"promotion_{cfg.name.lower()}_{run_layout.run_id}.json").write_text(
        json.dumps(promotion_record, indent=2),
        encoding="utf-8",
    )
    _update_run_manifest(
        cfg, run_layout, accepted=bool(evaluation.get("accepted", False)), promoted_at=promoted_at
    )


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _evaluate_against_baseline(
    *,
    dataset_name: str,
    reports_dir: Path,
    target_metrics_file: Path | None,
    publish_cfg: dict,
    profile: str,
) -> dict:
    current_metrics = _load_json(reports_dir / "week2_metrics.json")
    baseline_metrics = _load_json(target_metrics_file) if target_metrics_file else {}

    acc_cfg = (
        publish_cfg.get("retraining_acceptance", {})
        if isinstance(publish_cfg.get("retraining_acceptance"), dict)
        else {}
    )
    profile_overrides = acc_cfg.get("profile_overrides", {})
    profile_cfg = profile_overrides.get(profile, {}) if isinstance(profile_overrides, dict) else {}
    if not isinstance(profile_cfg, dict):
        profile_cfg = {}
    merged_acc_cfg = {**acc_cfg, **profile_cfg}
    metric_name = str(merged_acc_cfg.get("metric", acc_cfg.get("metric", "mape")))
    metric_by_target = merged_acc_cfg.get("metric_by_target", acc_cfg.get("metric_by_target", {}))
    if not isinstance(metric_by_target, dict):
        metric_by_target = {}
    require_non_reg = bool(
        merged_acc_cfg.get("require_non_regression", acc_cfg.get("require_non_regression", True))
    )
    min_impr = merged_acc_cfg.get("min_improvement_by_target", acc_cfg.get("min_improvement_by_target", {}))
    if not isinstance(min_impr, dict):
        min_impr = {}
    abs_tolerance = merged_acc_cfg.get(
        "absolute_regression_tolerance_by_target",
        acc_cfg.get("absolute_regression_tolerance_by_target", {}),
    )
    if not isinstance(abs_tolerance, dict):
        abs_tolerance = {}
    max_cv_std = float(merged_acc_cfg.get("max_cv_rmse_std", acc_cfg.get("max_cv_rmse_std", 1e9)))

    current_targets = (
        current_metrics.get("targets", {}) if isinstance(current_metrics.get("targets"), dict) else {}
    )
    baseline_targets = (
        baseline_metrics.get("targets", {}) if isinstance(baseline_metrics.get("targets"), dict) else {}
    )

    target_rows: list[dict] = []
    overall_pass = True
    for target, target_payload in current_targets.items():
        if not isinstance(target_payload, dict):
            continue
        cur_gbm = target_payload.get("gbm", {}) if isinstance(target_payload.get("gbm"), dict) else {}
        target_metric_name = str(metric_by_target.get(target, metric_name))
        cur_metric = cur_gbm.get(target_metric_name)
        baseline_payload = (
            baseline_targets.get(target, {}) if isinstance(baseline_targets.get(target), dict) else {}
        )
        base_gbm = baseline_payload.get("gbm", {}) if isinstance(baseline_payload.get("gbm"), dict) else {}
        base_metric = base_gbm.get(target_metric_name)

        improvement = None
        regression_delta = None
        if isinstance(base_metric, int | float) and isinstance(cur_metric, int | float) and base_metric != 0:
            improvement = (float(base_metric) - float(cur_metric)) / abs(float(base_metric))
            regression_delta = float(cur_metric) - float(base_metric)

        min_req = float(min_impr.get(target, 0.0))
        abs_tol = max(0.0, float(abs_tolerance.get(target, 0.0)))
        non_regression_pass = True
        if require_non_reg and isinstance(base_metric, int | float) and isinstance(cur_metric, int | float):
            non_regression_pass = float(cur_metric) <= float(base_metric) + abs_tol
        improvement_pass = True
        if improvement is not None:
            improvement_pass = float(improvement) >= min_req
            if not improvement_pass and min_req <= 0.0 and regression_delta is not None:
                improvement_pass = float(regression_delta) <= abs_tol

        cv_std = None
        if isinstance(cur_gbm.get("cv_results"), dict):
            cv_std = cur_gbm["cv_results"].get("rmse_std")
        cv_std_pass = True
        if isinstance(cv_std, int | float):
            cv_std_pass = float(cv_std) <= max_cv_std

        retained_incumbent = target_payload.get("retention_decision") == "retained_incumbent"
        if retained_incumbent and non_regression_pass:
            improvement_pass = True

        row_pass = bool(non_regression_pass and improvement_pass and cv_std_pass)
        if not row_pass:
            overall_pass = False

        target_rows.append(
            {
                "target": target,
                "metric": target_metric_name,
                "current_metric": cur_metric,
                "baseline_metric": base_metric,
                "improvement": improvement,
                "min_improvement_required": min_req,
                "absolute_regression_tolerance": abs_tol,
                "regression_delta": regression_delta,
                "non_regression_pass": non_regression_pass,
                "improvement_pass": improvement_pass,
                "cv_rmse_std": cv_std,
                "cv_std_pass": cv_std_pass,
                "accepted": row_pass,
                "decision": target_payload.get("retention_decision", "candidate"),
            }
        )

    if not target_rows:
        overall_pass = False

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset_name,
        "profile": profile,
        "metric": metric_name,
        "metric_by_target": metric_by_target,
        "target_metrics_file": str(target_metrics_file) if target_metrics_file else None,
        "targets": target_rows,
        "accepted": overall_pass,
    }


def _configured_retention_targets(publish_cfg: dict[str, Any]) -> set[str]:
    acc_cfg = (
        publish_cfg.get("retraining_acceptance", {})
        if isinstance(publish_cfg.get("retraining_acceptance"), dict)
        else {}
    )
    raw = acc_cfg.get("retain_incumbent_on_regression_targets", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return set()
    return {str(target).strip() for target in raw if str(target).strip()}


def _copy_target_artifacts(src_dir: Path, dst_dir: Path, target: str) -> list[str]:
    copied: list[str] = []
    if not src_dir.exists():
        return copied
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.name.startswith("._") or target not in item.name:
            continue
        dst = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
        copied.append(str(dst))
    return copied


def _copy_target_report_card(src_reports_dir: Path, dst_reports_dir: Path, target: str) -> str | None:
    src = src_reports_dir / "model_cards" / f"{target}.md"
    if not src.exists():
        return None
    dst = dst_reports_dir / "model_cards" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _apply_incumbent_retention(
    *,
    cfg: DatasetConfig,
    run_layout: RunLayout,
    evaluation: dict[str, Any],
    publish_cfg: dict[str, Any],
    baseline_metrics_file: Path | None,
    canonical_models_dir: Path | None = None,
    canonical_uncertainty_dir: Path | None = None,
    canonical_backtests_dir: Path | None = None,
    canonical_reports_dir: Path | None = None,
) -> list[str]:
    """Replace configured failed target outputs with the current incumbent."""
    retention_targets = _configured_retention_targets(publish_cfg)
    if not retention_targets or baseline_metrics_file is None or not baseline_metrics_file.exists():
        return []

    candidate_metrics_path = run_layout.reports_dir / "week2_metrics.json"
    candidate_metrics = _load_json(candidate_metrics_path)
    baseline_metrics = _load_json(baseline_metrics_file)
    candidate_targets = (
        candidate_metrics.get("targets", {}) if isinstance(candidate_metrics.get("targets"), dict) else {}
    )
    baseline_targets = (
        baseline_metrics.get("targets", {}) if isinstance(baseline_metrics.get("targets"), dict) else {}
    )

    canonical_models_dir = canonical_models_dir or (REPO_ROOT / cfg.models_dir)
    canonical_uncertainty_dir = canonical_uncertainty_dir or (REPO_ROOT / cfg.uncertainty_dir)
    canonical_backtests_dir = canonical_backtests_dir or (REPO_ROOT / cfg.backtests_dir)
    canonical_reports_dir = canonical_reports_dir or (REPO_ROOT / cfg.reports_dir)

    retained: list[str] = []
    records: list[dict[str, Any]] = []
    for row in evaluation.get("targets", []):
        if not isinstance(row, dict):
            continue
        target = str(row.get("target", "")).strip()
        regression_delta = row.get("regression_delta")
        regressed = bool(row.get("non_regression_pass") is False)
        if isinstance(regression_delta, int | float):
            regressed = regressed or float(regression_delta) > 0.0
        failed_acceptance = row.get("accepted") is False
        if not failed_acceptance:
            failed_acceptance = any(
                row.get(flag) is False for flag in ("non_regression_pass", "improvement_pass", "cv_std_pass")
            )
        if target not in retention_targets or not failed_acceptance:
            continue
        baseline_target = baseline_targets.get(target)
        if not isinstance(baseline_target, dict):
            continue

        reason = "challenger_regressed_against_baseline" if regressed else "challenger_missed_acceptance_gate"
        replacement = dict(baseline_target)
        replacement["retention_decision"] = "retained_incumbent"
        replacement["retention_reason"] = reason
        replacement["challenger_metrics"] = candidate_targets.get(target, {})
        candidate_targets[target] = replacement

        copied = []
        copied.extend(_copy_target_artifacts(canonical_models_dir, run_layout.models_dir, target))
        copied.extend(_copy_target_artifacts(canonical_uncertainty_dir, run_layout.uncertainty_dir, target))
        copied.extend(_copy_target_artifacts(canonical_backtests_dir, run_layout.backtests_dir, target))
        copied_card = _copy_target_report_card(canonical_reports_dir, run_layout.reports_dir, target)
        if copied_card is not None:
            copied.append(copied_card)

        retained.append(target)
        records.append(
            {
                "target": target,
                "reason": reason,
                "baseline_metrics_file": str(baseline_metrics_file),
                "copied_artifacts": copied,
            }
        )

    if not retained:
        return []

    candidate_metrics["targets"] = candidate_targets
    candidate_metrics_path.write_text(json.dumps(candidate_metrics, indent=2), encoding="utf-8")
    run_layout.registry_dir.mkdir(parents=True, exist_ok=True)
    (run_layout.registry_dir / "incumbent_retention.json").write_text(
        json.dumps({"retained_targets": records}, indent=2),
        encoding="utf-8",
    )
    return retained


def _persist_selection_artifacts(
    *,
    cfg: DatasetConfig,
    evaluation: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"tuning_summary_{cfg.name.lower()}.json"
    summary_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    decision_path = output_dir / "model_selection_decisions.md"
    lines: list[str] = []
    if decision_path.exists():
        lines.append(decision_path.read_text(encoding="utf-8").rstrip())
        lines.append("")
    lines.append(f"## {cfg.name} - {datetime.now(UTC).isoformat()}")
    lines.append(f"- Accepted: **{evaluation.get('accepted')}**")
    lines.append(f"- Profile: `{evaluation.get('profile')}`")
    lines.append("")
    lines.append("| Target | Current | Baseline | Improvement | Accepted |")
    lines.append("|---|---:|---:|---:|:---:|")
    for row in evaluation.get("targets", []):
        lines.append(
            f"| {row.get('target')} | {row.get('current_metric')} | {row.get('baseline_metric')} | "
            f"{row.get('improvement')} | {row.get('accepted')} |"
        )
    decision_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def train_dataset(
    dataset_name: str,
    models: str | None = None,
    tune: bool = False,
    no_tune: bool = False,
    no_cv: bool = False,
    ensemble: bool = False,
    max_seeds: int | None = None,
    n_trials: int | None = None,
    top_pct: float | None = None,
    tuning_n_jobs: int | None = None,
    gbm_threads: int | None = None,
    reuse_best_gbm_from: str | None = None,
    max_deep_epochs: int | None = None,
    deep_patience: int | None = None,
    deep_warmup_epochs: int | None = None,
    profile: str = "standard",
    max_runtime_hours: float | None = None,
    target_metrics_file: str | None = None,
    reports: bool = True,
    rebuild_features: bool = False,
    skip_training: bool = False,
    candidate_run: bool = False,
    run_id: str | None = None,
    promote_on_accept: bool = False,
) -> bool:
    """
    Full training pipeline for a single dataset.

    Args:
        dataset_name: Dataset key from DATASET_REGISTRY (DE, US, etc.)
        tune: Enable hyperparameter tuning
        reports: Generate evaluation reports
        rebuild_features: Force rebuild features
        skip_training: Skip model training (for reports only)

    Returns:
        True if all steps successful
    """
    if dataset_name not in DATASET_REGISTRY:
        print(f"❌ Unknown dataset: {dataset_name}")
        print(f"   Available datasets: {list(DATASET_REGISTRY.keys())}")
        return False

    cfg = DATASET_REGISTRY[dataset_name]
    run_layout = _resolve_run_layout(cfg, candidate_run=candidate_run, run_id=run_id)

    print(f"\n{'#' * 60}")
    print(f"#  TRAINING PIPELINE: {cfg.display_name}")
    print(f"#  Config: {cfg.config_file}")
    print(f"#  Output mode: {run_layout.mode} ({run_layout.run_id})")
    print(f"{'#' * 60}")

    # Step 1: Build features
    if not build_features(cfg, force=rebuild_features):
        return False

    # Step 2: Create splits
    if not create_splits(cfg, force=rebuild_features):
        return False

    # Step 2.5: Enforce schema + data identity contracts before training.
    if not validate_features_schema(cfg, report_path=run_layout.validation_report):
        return False
    if not refresh_data_manifest(cfg, output_path=run_layout.data_manifest_output):
        return False

    preflight = _write_preflight_analysis(cfg, run_layout, profile=profile, models=models)
    _write_run_context(cfg=cfg, run_layout=run_layout, preflight=preflight, profile=profile, models=models)

    # Step 3: Train models
    if not skip_training and not train_models(
        cfg,
        run_layout=run_layout,
        models=models,
        tune=tune,
        no_tune=no_tune,
        no_cv=no_cv,
        ensemble=ensemble,
        max_seeds=max_seeds,
        n_trials=n_trials,
        top_pct=top_pct,
        tuning_n_jobs=tuning_n_jobs,
        gbm_threads=gbm_threads,
        reuse_best_gbm_from=reuse_best_gbm_from,
        max_deep_epochs=max_deep_epochs,
        deep_patience=deep_patience,
        deep_warmup_epochs=deep_warmup_epochs,
        profile=profile,
        max_runtime_hours=max_runtime_hours,
        target_metrics_file=target_metrics_file,
    ):
        return False

    # Step 4: Generate reports (includes conformal coverage)
    if reports and not generate_reports(cfg, run_layout=run_layout):
        print("⚠️  Reports generation failed, continuing...")

    if not verify_training_outputs(cfg, run_layout, models=models):
        print(
            f"❌ Verification failed for {cfg.display_name}. Candidate outputs kept at {run_layout.reports_dir}."
        )
        return False

    publish_cfg = _load_publish_audit_cfg()
    target_metrics_path = Path(target_metrics_file) if target_metrics_file else None
    evaluation = _evaluate_against_baseline(
        dataset_name=cfg.name,
        reports_dir=run_layout.reports_dir,
        target_metrics_file=target_metrics_path,
        publish_cfg=publish_cfg,
        profile=profile,
    )
    retained_targets = _apply_incumbent_retention(
        cfg=cfg,
        run_layout=run_layout,
        evaluation=evaluation,
        publish_cfg=publish_cfg,
        baseline_metrics_file=target_metrics_path,
    )
    if retained_targets:
        evaluation = _evaluate_against_baseline(
            dataset_name=cfg.name,
            reports_dir=run_layout.reports_dir,
            target_metrics_file=target_metrics_path,
            publish_cfg=publish_cfg,
            profile=profile,
        )
        evaluation["retained_incumbent_targets"] = retained_targets
    _persist_selection_artifacts(
        cfg=cfg,
        evaluation=evaluation,
        output_dir=run_layout.selection_output_dir,
    )
    _update_run_manifest(cfg, run_layout, accepted=bool(evaluation.get("accepted", False)))
    if not bool(evaluation.get("accepted", False)):
        print(f"⚠️  Acceptance gates not met for {cfg.display_name}. See {run_layout.selection_output_dir}.")
    elif run_layout.is_candidate and promote_on_accept:
        _promote_candidate_run(cfg, run_layout, evaluation)
        print(f"📦 Promoted accepted candidate run {run_layout.run_id} into canonical paths.")

    print(f"\n✅ Pipeline completed for {cfg.display_name}")
    return True


def train_all_datasets(**kwargs) -> bool:
    """Train all registered datasets with the same settings."""
    trainable_keys = _iter_trainable_dataset_keys()
    print(f"\n{'=' * 60}")
    print(f"  TRAINING ALL DATASETS: {trainable_keys}")
    print(f"{'=' * 60}")

    results = {}
    for name in trainable_keys:
        results[name] = train_dataset(name, **kwargs)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {name}: {DATASET_REGISTRY[name].display_name}")

    return all(results.values())


def list_datasets() -> None:
    """Display available datasets."""
    print("\n📊 Registered Datasets:")
    print("-" * 60)
    for key, cfg in DATASET_REGISTRY.items():
        print(f"   {key:8s} - {cfg.display_name}")
        if cfg.alias_of:
            print(f"            Alias of: {cfg.alias_of}")
        print(f"            Config: {cfg.config_file}")
        print(f"            Features: {cfg.features_path}")
        print()


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for train_dataset."""
    parser = argparse.ArgumentParser(
        description="Unified training script for ORIUS datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_dataset.py --dataset DE        # Train German data
  python scripts/train_dataset.py --dataset US        # Train US data
  python scripts/train_dataset.py --all               # Train all datasets
  python scripts/train_dataset.py --dataset DE --tune # With hyperparameter tuning
  python scripts/train_dataset.py --dataset AV --profile max --candidate-run --run-id max_av
  python scripts/train_dataset.py --list              # Show available datasets
        """,
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=[*list(DATASET_REGISTRY.keys()), "ALL"],
        help="Dataset to train (DE, US, etc.)",
    )
    parser.add_argument("--all", "-a", action="store_true", help="Train all registered datasets")
    parser.add_argument("--list", "-l", action="store_true", help="List available datasets")
    parser.add_argument("--tune", "-t", action="store_true", help="Enable Optuna hyperparameter tuning")
    parser.add_argument(
        "--models", default=None, help="Optional comma-separated model filter, e.g. gbm or gbm,lstm"
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="Disable tuning even if YAML has tuning.enabled=true"
    )
    parser.add_argument(
        "--no-cv", action="store_true", help="Disable cross-validation even if YAML enables it"
    )
    parser.add_argument(
        "--ensemble", action="store_true", help="Train multi-seed GBM ensembles using config.seeds"
    )
    parser.add_argument("--max-seeds", type=int, default=None, help="Optional cap for ensemble seed count")
    parser.add_argument("--n-trials", type=int, default=None, help="Override Optuna trial count for this run")
    parser.add_argument(
        "--top-pct", type=float, default=None, help="Use top percent of trials for param aggregation"
    )
    parser.add_argument(
        "--tuning-n-jobs", type=int, default=None, help="Parallel Optuna trials for GBM tuning"
    )
    parser.add_argument("--gbm-threads", type=int, default=None, help="Threads per LightGBM model/trial")
    parser.add_argument(
        "--reuse-best-gbm-from",
        default=None,
        help="Seed Optuna with tuned GBM params from a prior metrics JSON",
    )
    parser.add_argument("--max-deep-epochs", type=int, default=None, help="Cap epochs for deep models")
    parser.add_argument(
        "--deep-patience", type=int, default=None, help="Cap early-stopping patience for deep models"
    )
    parser.add_argument(
        "--deep-warmup-epochs",
        type=int,
        default=None,
        help="Cap learning-rate warmup epochs for deep models",
    )
    parser.add_argument(
        "--profile", choices=list(TRAINING_PROFILES), default="standard", help="Training profile"
    )
    parser.add_argument(
        "--max-runtime-hours", type=float, default=None, help="Optional timeout for each training invocation"
    )
    parser.add_argument(
        "--target-metrics-file", default=None, help="Optional baseline metrics JSON for acceptance comparison"
    )
    parser.add_argument(
        "--reports",
        "-r",
        action="store_true",
        default=True,
        help="Generate evaluation reports (default: True)",
    )
    parser.add_argument("--no-reports", action="store_true", help="Skip report generation")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild features even if they exist")
    parser.add_argument("--reports-only", action="store_true", help="Only generate reports (skip training)")
    parser.add_argument(
        "--candidate-run",
        action="store_true",
        help="Write outputs into isolated artifacts/runs/... and reports/runs/... directories",
    )
    parser.add_argument("--run-id", default=None, help="Optional run identifier for candidate outputs")
    parser.add_argument(
        "--promote-on-accept",
        action="store_true",
        help="Promote candidate outputs into canonical paths only after acceptance gates pass",
    )
    return parser


def main() -> int:
    """Main entry point."""
    args = _build_parser().parse_args()

    # Handle --list
    if args.list:
        list_datasets()
        return 0

    # Handle --no-reports
    reports = args.reports and not args.no_reports

    # Require either --dataset or --all
    if not args.dataset and not args.all:
        _build_parser().print_help()
        print("\n❌ Error: Specify --dataset <NAME> or --all")
        return 1

    # Run training
    run_all = bool(args.all or args.dataset == "ALL")
    if run_all:
        success = train_all_datasets(
            tune=args.tune,
            models=args.models,
            no_tune=args.no_tune,
            no_cv=args.no_cv,
            ensemble=args.ensemble,
            max_seeds=args.max_seeds,
            n_trials=args.n_trials,
            top_pct=args.top_pct,
            tuning_n_jobs=args.tuning_n_jobs,
            gbm_threads=args.gbm_threads,
            reuse_best_gbm_from=args.reuse_best_gbm_from,
            max_deep_epochs=args.max_deep_epochs,
            deep_patience=args.deep_patience,
            deep_warmup_epochs=args.deep_warmup_epochs,
            profile=args.profile,
            max_runtime_hours=args.max_runtime_hours,
            target_metrics_file=args.target_metrics_file,
            reports=reports,
            rebuild_features=args.rebuild,
            skip_training=args.reports_only,
            candidate_run=args.candidate_run,
            run_id=args.run_id,
            promote_on_accept=args.promote_on_accept,
        )
    else:
        success = train_dataset(
            args.dataset,
            models=args.models,
            tune=args.tune,
            no_tune=args.no_tune,
            no_cv=args.no_cv,
            ensemble=args.ensemble,
            max_seeds=args.max_seeds,
            n_trials=args.n_trials,
            top_pct=args.top_pct,
            tuning_n_jobs=args.tuning_n_jobs,
            gbm_threads=args.gbm_threads,
            reuse_best_gbm_from=args.reuse_best_gbm_from,
            max_deep_epochs=args.max_deep_epochs,
            deep_patience=args.deep_patience,
            deep_warmup_epochs=args.deep_warmup_epochs,
            profile=args.profile,
            max_runtime_hours=args.max_runtime_hours,
            target_metrics_file=args.target_metrics_file,
            reports=reports,
            rebuild_features=args.rebuild,
            skip_training=args.reports_only,
            candidate_run=args.candidate_run,
            run_id=args.run_id,
            promote_on_accept=args.promote_on_accept,
        )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
