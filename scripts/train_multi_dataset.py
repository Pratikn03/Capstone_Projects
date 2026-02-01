from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ba", default="MISO", help="EIA930 Balancing Authority code")
    p.add_argument("--start", default=None, help="Start date for EIA930 (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date for EIA930 (YYYY-MM-DD)")
    args = p.parse_args()

    # OPSD
    if not Path("data/processed/features.parquet").exists():
        run(["python", "-m", "gridpulse.data_pipeline.validate_schema", "--in", "data/raw", "--report", "reports/data_quality_report.md"])
        run(["python", "-m", "gridpulse.data_pipeline.build_features", "--in", "data/raw", "--out", "data/processed"])
        run(["python", "-m", "gridpulse.data_pipeline.split_time_series", "--in", "data/processed/features.parquet", "--out", "data/processed/splits"])

    # EIA930
    run(["python", "-m", "gridpulse.data_pipeline.build_features_eia930", "--in", "data/raw/us_eia930", "--out", "data/processed/us_eia930", "--ba", args.ba] +
        (["--start", args.start] if args.start else []) +
        (["--end", args.end] if args.end else []))
    run(["python", "-m", "gridpulse.data_pipeline.split_time_series", "--in", "data/processed/us_eia930/features.parquet", "--out", "data/processed/us_eia930/splits"])

    # Train both
    run(["python", "-m", "gridpulse.forecasting.train", "--config", "configs/train_forecast.yaml"])
    run(["python", "-m", "gridpulse.forecasting.train", "--config", "configs/train_forecast_eia930.yaml"])


if __name__ == "__main__":
    main()
