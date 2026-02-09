"""
Training pipeline script for multiple geographic datasets.

This script orchestrates the full training workflow for both German (OPSD)
and US (EIA-930) energy datasets. It handles data validation, feature
engineering, train/val/test splitting, and model training.

Usage:
    python scripts/train_multi_dataset.py
    python scripts/train_multi_dataset.py --ba ERCOT --start 2023-01-01

The script is idempotent - it checks for existing outputs before regenerating.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    """
    Execute a subprocess command with logging and error handling.
    
    Args:
        cmd: Command and arguments as a list of strings
        
    Raises:
        subprocess.CalledProcessError: If command exits with non-zero status
    """
    # Print the command for traceability in CI/CD logs
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    """
    Main entry point for multi-dataset training pipeline.
    
    Workflow:
    1. Validate and build features for German OPSD data (if not already done)
    2. Build features for US EIA-930 data for specified balancing authority
    3. Create temporal train/val/test splits for both datasets
    4. Train LightGBM models using respective config files
    """
    # Parse command-line arguments for flexible dataset configuration
    parser = argparse.ArgumentParser(
        description="Train GridPulse models on German and US energy datasets"
    )
    parser.add_argument(
        "--ba", 
        default="MISO", 
        help="EIA-930 Balancing Authority code (e.g., MISO, ERCOT, PJM)"
    )
    parser.add_argument(
        "--start", 
        default=None, 
        help="Start date for EIA-930 data filtering (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", 
        default=None, 
        help="End date for EIA-930 data filtering (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # STEP 1: German OPSD Dataset (skip if already processed)
    # -------------------------------------------------------------------------
    # Check if features already exist to avoid redundant computation
    if not Path("data/processed/features.parquet").exists():
        # Validate raw data schema and generate quality report
        run([
            "python", "-m", "gridpulse.data_pipeline.validate_schema",
            "--in", "data/raw",
            "--report", "reports/data_quality_report.md"
        ])
        
        # Build engineered features (lag, rolling, calendar, etc.)
        run([
            "python", "-m", "gridpulse.data_pipeline.build_features",
            "--in", "data/raw",
            "--out", "data/processed"
        ])
        
        # Create temporal splits respecting time-series ordering
        run([
            "python", "-m", "gridpulse.data_pipeline.split_time_series",
            "--in", "data/processed/features.parquet",
            "--out", "data/processed/splits"
        ])

    # -------------------------------------------------------------------------
    # STEP 2: US EIA-930 Dataset
    # -------------------------------------------------------------------------
    # Build features for the specified balancing authority region
    eia_cmd = [
        "python", "-m", "gridpulse.data_pipeline.build_features_eia930",
        "--in", "data/raw/us_eia930",
        "--out", "data/processed/us_eia930",
        "--ba", args.ba
    ]
    # Add optional date filtering if specified
    if args.start:
        eia_cmd.extend(["--start", args.start])
    if args.end:
        eia_cmd.extend(["--end", args.end])
    run(eia_cmd)
    
    # Create temporal splits for US data
    run([
        "python", "-m", "gridpulse.data_pipeline.split_time_series",
        "--in", "data/processed/us_eia930/features.parquet",
        "--out", "data/processed/us_eia930/splits"
    ])

    # -------------------------------------------------------------------------
    # STEP 3: Model Training
    # -------------------------------------------------------------------------
    # Train on German dataset (load, solar, wind, price targets)
    run([
        "python", "-m", "gridpulse.forecasting.train",
        "--config", "configs/train_forecast.yaml"
    ])
    
    # Train on US dataset (load, solar, wind targets for the BA)
    run([
        "python", "-m", "gridpulse.forecasting.train",
        "--config", "configs/train_forecast_eia930.yaml"
    ])


if __name__ == "__main__":
    main()
