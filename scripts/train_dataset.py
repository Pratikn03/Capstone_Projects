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

Author: GridPulse Team
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

PYTHON_BIN = sys.executable or "python3"
REPO_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# DATASET REGISTRY - Add new datasets here
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a registered dataset."""
    name: str                           # Short name (DE, US, etc.)
    display_name: str                   # Full name for logging
    config_file: str                    # Path to training config YAML
    features_path: str                  # Path to features.parquet
    splits_path: str                    # Path to splits directory
    models_dir: str                     # Path to model artifacts directory
    reports_dir: str                    # Path to reports output
    
    # Feature pipeline settings
    raw_data_path: str                  # Path to raw data
    feature_module: str                 # Module to build features
    
    # Optional settings
    ba_code: Optional[str] = None       # Balancing authority (for US)
    start_date: Optional[str] = None    # Date filter start
    end_date: Optional[str] = None      # Date filter end


# Dataset registry - single source of truth for all datasets
DATASET_REGISTRY: dict[str, DatasetConfig] = {
    "DE": DatasetConfig(
        name="DE",
        display_name="German OPSD",
        config_file="configs/train_forecast.yaml",
        features_path="data/processed/features.parquet",
        splits_path="data/processed/splits",
        models_dir="artifacts/models",
        reports_dir="reports",
        raw_data_path="data/raw",
        feature_module="gridpulse.data_pipeline.build_features",
    ),
    "US": DatasetConfig(
        name="US", 
        display_name="US EIA-930 (MISO)",
        config_file="configs/train_forecast_eia930.yaml",
        features_path="data/processed/us_eia930/features.parquet",
        splits_path="data/processed/us_eia930/splits",
        models_dir="artifacts/models_eia930",
        reports_dir="reports/eia930",
        raw_data_path="data/raw/us_eia930",
        feature_module="gridpulse.data_pipeline.build_features_eia930",
        ba_code="MISO",
    ),
}

AGGRESSIVE_DEFAULTS = {
    "DE": {"n_trials": 220, "top_pct": 0.20, "max_seeds": 8},
    "US": {"n_trials": 260, "top_pct": 0.20, "max_seeds": 5},
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


def run_command(cmd: list[str], description: str, timeout_seconds: float | None = None) -> bool:
    """
    Execute a subprocess command with logging.
    
    Args:
        cmd: Command and arguments
        description: Human-readable description
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    cmd_env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    existing_pythonpath = cmd_env.get("PYTHONPATH", "")
    cmd_env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path

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
    
    # Build command based on dataset type
    cmd = [
        PYTHON_BIN, "-m", cfg.feature_module,
        "--in", cfg.raw_data_path,
        "--out", str(features_path.parent),
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
    
    if splits_path.exists() and not force:
        print(f"ℹ️  Splits already exist: {splits_path}")
        return True
    
    cmd = [
        PYTHON_BIN, "-m", "gridpulse.data_pipeline.split_time_series",
        "--in", cfg.features_path,
        "--out", cfg.splits_path,
    ]
    
    return run_command(cmd, f"Creating time series splits for {cfg.display_name}")


def train_models(
    cfg: DatasetConfig,
    tune: bool = False,
    no_tune: bool = False,
    ensemble: bool = False,
    max_seeds: Optional[int] = None,
    n_trials: Optional[int] = None,
    top_pct: Optional[float] = None,
    profile: str = "standard",
    max_runtime_hours: Optional[float] = None,
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

    if profile == "aggressive":
        defaults = AGGRESSIVE_DEFAULTS.get(cfg.name, AGGRESSIVE_DEFAULTS["DE"])
        effective_tune = True
        effective_ensemble = True
        if effective_max_seeds is None:
            effective_max_seeds = int(defaults["max_seeds"])
        if effective_n_trials is None:
            effective_n_trials = int(defaults["n_trials"])
        if effective_top_pct is None:
            effective_top_pct = float(defaults["top_pct"])

    cmd = [
        PYTHON_BIN, "-m", "gridpulse.forecasting.train",
        "--config", cfg.config_file,
    ]
    
    if effective_tune:
        cmd.append("--tune")
    if no_tune:
        cmd.append("--no-tune")
    if effective_ensemble:
        cmd.append("--ensemble")
    if effective_max_seeds is not None and effective_max_seeds > 0:
        cmd.extend(["--max-seeds", str(int(effective_max_seeds))])
    if effective_n_trials is not None and effective_n_trials > 0:
        cmd.extend(["--n-trials", str(int(effective_n_trials))])
    if effective_top_pct is not None:
        cmd.extend(["--top-pct", str(float(effective_top_pct))])
    
    timeout_seconds = None
    if max_runtime_hours is not None and max_runtime_hours > 0:
        timeout_seconds = float(max_runtime_hours) * 3600.0

    return run_command(cmd, f"Training models for {cfg.display_name}", timeout_seconds=timeout_seconds)


def generate_reports(cfg: DatasetConfig) -> bool:
    """
    Generate evaluation reports including conformal coverage.
    
    Args:
        cfg: Dataset configuration
        
    Returns:
        True if successful
    """
    cmd = [
        PYTHON_BIN, "scripts/build_reports.py",
        "--features", cfg.features_path,
        "--splits", cfg.splits_path,
        "--models-dir", cfg.models_dir,
        "--reports-dir", cfg.reports_dir,
    ]
    
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
        print(f"⚠️  Conformal intervals script not found, using build_reports")
        return True  # Reports include conformal if configured
    
    cmd = [
        PYTHON_BIN, str(conformal_script),
        "--config", cfg.config_file,
    ]
    
    return run_command(cmd, f"Computing conformal intervals for {cfg.display_name}")


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

    acc_cfg = publish_cfg.get("retraining_acceptance", {}) if isinstance(publish_cfg.get("retraining_acceptance"), dict) else {}
    metric_name = str(acc_cfg.get("metric", "mape"))
    require_non_reg = bool(acc_cfg.get("require_non_regression", True))
    min_impr = acc_cfg.get("min_improvement_by_target", {})
    if not isinstance(min_impr, dict):
        min_impr = {}
    max_cv_std = float(acc_cfg.get("max_cv_rmse_std", 1e9))

    current_targets = current_metrics.get("targets", {}) if isinstance(current_metrics.get("targets"), dict) else {}
    baseline_targets = baseline_metrics.get("targets", {}) if isinstance(baseline_metrics.get("targets"), dict) else {}

    target_rows: list[dict] = []
    overall_pass = True
    for target, target_payload in current_targets.items():
        if not isinstance(target_payload, dict):
            continue
        cur_gbm = target_payload.get("gbm", {}) if isinstance(target_payload.get("gbm"), dict) else {}
        cur_metric = cur_gbm.get(metric_name)
        baseline_payload = baseline_targets.get(target, {}) if isinstance(baseline_targets.get(target), dict) else {}
        base_gbm = baseline_payload.get("gbm", {}) if isinstance(baseline_payload.get("gbm"), dict) else {}
        base_metric = base_gbm.get(metric_name)

        improvement = None
        if isinstance(base_metric, (int, float)) and isinstance(cur_metric, (int, float)) and base_metric != 0:
            improvement = (float(base_metric) - float(cur_metric)) / abs(float(base_metric))

        min_req = float(min_impr.get(target, 0.0))
        non_regression_pass = True
        if require_non_reg and isinstance(base_metric, (int, float)) and isinstance(cur_metric, (int, float)):
            non_regression_pass = float(cur_metric) <= float(base_metric)
        improvement_pass = True
        if improvement is not None:
            improvement_pass = float(improvement) >= min_req

        cv_std = None
        if isinstance(cur_gbm.get("cv_results"), dict):
            cv_std = cur_gbm["cv_results"].get("rmse_std")
        cv_std_pass = True
        if isinstance(cv_std, (int, float)):
            cv_std_pass = float(cv_std) <= max_cv_std

        row_pass = bool(non_regression_pass and improvement_pass and cv_std_pass)
        if not row_pass:
            overall_pass = False

        target_rows.append(
            {
                "target": target,
                "metric": metric_name,
                "current_metric": cur_metric,
                "baseline_metric": base_metric,
                "improvement": improvement,
                "min_improvement_required": min_req,
                "non_regression_pass": non_regression_pass,
                "improvement_pass": improvement_pass,
                "cv_rmse_std": cv_std,
                "cv_std_pass": cv_std_pass,
                "accepted": row_pass,
            }
        )

    if not target_rows:
        overall_pass = False

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "profile": profile,
        "metric": metric_name,
        "target_metrics_file": str(target_metrics_file) if target_metrics_file else None,
        "targets": target_rows,
        "accepted": overall_pass,
    }


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
    lines.append(f"## {cfg.name} - {datetime.now(timezone.utc).isoformat()}")
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
    tune: bool = False,
    no_tune: bool = False,
    ensemble: bool = False,
    max_seeds: Optional[int] = None,
    n_trials: Optional[int] = None,
    top_pct: Optional[float] = None,
    profile: str = "standard",
    max_runtime_hours: Optional[float] = None,
    target_metrics_file: Optional[str] = None,
    reports: bool = True,
    rebuild_features: bool = False,
    skip_training: bool = False,
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
    
    print(f"\n{'#'*60}")
    print(f"#  TRAINING PIPELINE: {cfg.display_name}")
    print(f"#  Config: {cfg.config_file}")
    print(f"{'#'*60}")
    
    # Step 1: Build features
    if not build_features(cfg, force=rebuild_features):
        return False
    
    # Step 2: Create splits
    if not create_splits(cfg, force=rebuild_features):
        return False
    
    # Step 3: Train models
    if not skip_training:
        if not train_models(
            cfg,
            tune=tune,
            no_tune=no_tune,
            ensemble=ensemble,
            max_seeds=max_seeds,
            n_trials=n_trials,
            top_pct=top_pct,
            profile=profile,
            max_runtime_hours=max_runtime_hours,
        ):
            return False
    
    # Step 4: Generate reports (includes conformal coverage)
    if reports:
        if not generate_reports(cfg):
            print("⚠️  Reports generation failed, continuing...")

    publish_cfg = _load_publish_audit_cfg()
    target_metrics_path = Path(target_metrics_file) if target_metrics_file else None
    evaluation = _evaluate_against_baseline(
        dataset_name=cfg.name,
        reports_dir=Path(cfg.reports_dir),
        target_metrics_file=target_metrics_path,
        publish_cfg=publish_cfg,
        profile=profile,
    )
    _persist_selection_artifacts(
        cfg=cfg,
        evaluation=evaluation,
        output_dir=Path("reports/publish"),
    )
    if not bool(evaluation.get("accepted", False)):
        print(f"⚠️  Acceptance gates not met for {cfg.display_name}. See reports/publish.")

    print(f"\n✅ Pipeline completed for {cfg.display_name}")
    return True


def train_all_datasets(**kwargs) -> bool:
    """Train all registered datasets with the same settings."""
    print(f"\n{'='*60}")
    print(f"  TRAINING ALL DATASETS: {list(DATASET_REGISTRY.keys())}")
    print(f"{'='*60}")
    
    results = {}
    for name in DATASET_REGISTRY:
        results[name] = train_dataset(name, **kwargs)
    
    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
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
        print(f"            Config: {cfg.config_file}")
        print(f"            Features: {cfg.features_path}")
        print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified training script for GridPulse datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_dataset.py --dataset DE        # Train German data
  python scripts/train_dataset.py --dataset US        # Train US data
  python scripts/train_dataset.py --all               # Train all datasets
  python scripts/train_dataset.py --dataset DE --tune # With hyperparameter tuning
  python scripts/train_dataset.py --list              # Show available datasets
        """,
    )
    
    parser.add_argument(
        "--dataset", "-d",
        choices=list(DATASET_REGISTRY.keys()) + ["ALL"],
        help="Dataset to train (DE, US, etc.)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Train all registered datasets",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets",
    )
    parser.add_argument(
        "--tune", "-t",
        action="store_true",
        help="Enable Optuna hyperparameter tuning",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Disable tuning even if YAML has tuning.enabled=true",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Train multi-seed GBM ensembles using config.seeds",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Optional cap for ensemble seed count",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override Optuna trial count for this run",
    )
    parser.add_argument(
        "--top-pct",
        type=float,
        default=None,
        help="Use top percent of trials for param aggregation (e.g. 0.30 or 0.001)",
    )
    parser.add_argument(
        "--profile",
        choices=["standard", "aggressive"],
        default="standard",
        help="Training profile. aggressive forces tune+ensemble and higher trial budgets.",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=float,
        default=None,
        help="Optional timeout for each training invocation.",
    )
    parser.add_argument(
        "--target-metrics-file",
        default=None,
        help="Optional baseline metrics JSON for acceptance comparison.",
    )
    parser.add_argument(
        "--reports", "-r",
        action="store_true",
        default=True,
        help="Generate evaluation reports (default: True)",
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip report generation",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild features even if they exist",
    )
    parser.add_argument(
        "--reports-only",
        action="store_true",
        help="Only generate reports (skip training)",
    )
    
    args = parser.parse_args()
    
    # Handle --list
    if args.list:
        list_datasets()
        return 0
    
    # Handle --no-reports
    reports = args.reports and not args.no_reports
    
    # Require either --dataset or --all
    if not args.dataset and not args.all:
        parser.print_help()
        print("\n❌ Error: Specify --dataset <NAME> or --all")
        return 1
    
    # Run training
    run_all = bool(args.all or args.dataset == "ALL")
    if run_all:
        success = train_all_datasets(
            tune=args.tune,
            no_tune=args.no_tune,
            ensemble=args.ensemble,
            max_seeds=args.max_seeds,
            n_trials=args.n_trials,
            top_pct=args.top_pct,
            profile=args.profile,
            max_runtime_hours=args.max_runtime_hours,
            target_metrics_file=args.target_metrics_file,
            reports=reports,
            rebuild_features=args.rebuild,
            skip_training=args.reports_only,
        )
    else:
        success = train_dataset(
            args.dataset,
            tune=args.tune,
            no_tune=args.no_tune,
            ensemble=args.ensemble,
            max_seeds=args.max_seeds,
            n_trials=args.n_trials,
            top_pct=args.top_pct,
            profile=args.profile,
            max_runtime_hours=args.max_runtime_hours,
            target_metrics_file=args.target_metrics_file,
            reports=reports,
            rebuild_features=args.rebuild,
            skip_training=args.reports_only,
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
