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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
        reports_dir="reports/eia930",
        raw_data_path="data/raw/us_eia930",
        feature_module="gridpulse.data_pipeline.build_features_eia930",
        ba_code="MISO",
    ),
}


# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def run_command(cmd: list[str], description: str) -> bool:
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
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} - completed successfully")
        return True
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
        "python", "-m", cfg.feature_module,
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
        "python", "-m", "gridpulse.data_pipeline.split_time_series",
        "--in", cfg.features_path,
        "--out", cfg.splits_path,
    ]
    
    return run_command(cmd, f"Creating time series splits for {cfg.display_name}")


def train_models(cfg: DatasetConfig, tune: bool = False) -> bool:
    """
    Train forecasting models.
    
    Args:
        cfg: Dataset configuration
        tune: Enable hyperparameter tuning with Optuna
        
    Returns:
        True if successful
    """
    cmd = [
        "python", "-m", "gridpulse.forecasting.train",
        "--config", cfg.config_file,
    ]
    
    if tune:
        cmd.append("--tune")
    
    return run_command(cmd, f"Training models for {cfg.display_name}")


def generate_reports(cfg: DatasetConfig) -> bool:
    """
    Generate evaluation reports including conformal coverage.
    
    Args:
        cfg: Dataset configuration
        
    Returns:
        True if successful
    """
    cmd = [
        "python", "scripts/build_reports.py",
        "--config", cfg.config_file,
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
        "python", str(conformal_script),
        "--config", cfg.config_file,
    ]
    
    return run_command(cmd, f"Computing conformal intervals for {cfg.display_name}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def train_dataset(
    dataset_name: str,
    tune: bool = False,
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
        if not train_models(cfg, tune=tune):
            return False
    
    # Step 4: Generate reports (includes conformal coverage)
    if reports:
        if not generate_reports(cfg):
            print("⚠️  Reports generation failed, continuing...")
    
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
        choices=list(DATASET_REGISTRY.keys()),
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
    if args.all:
        success = train_all_datasets(
            tune=args.tune,
            reports=reports,
            rebuild_features=args.rebuild,
            skip_training=args.reports_only,
        )
    else:
        success = train_dataset(
            args.dataset,
            tune=args.tune,
            reports=reports,
            rebuild_features=args.rebuild,
            skip_training=args.reports_only,
        )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
