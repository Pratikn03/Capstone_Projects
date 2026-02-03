"""Run a lightweight production readiness gate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], label: str) -> None:
    print(f"[release_check] {label}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _require(path: Path, msg: str) -> None:
    if not path.exists():
        raise SystemExit(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Run data pipeline and training before checks")
    args = ap.parse_args()

    if args.full:
        _run([sys.executable, "-m", "gridpulse.pipeline.run", "--all"], "data pipeline")
        _run([sys.executable, "-m", "gridpulse.forecasting.train", "--config", "configs/train_forecast.yaml"], "train")

    # Hard requirements: processed data + splits exist.
    _require(Path("data/processed/features.parquet"), "Missing features.parquet. Run `make data`.")
    _require(Path("data/processed/splits/train.parquet"), "Missing train split. Run `make data`.")
    _require(Path("data/processed/splits/test.parquet"), "Missing test split. Run `make data`.")

    _run([sys.executable, "-m", "pytest", "-q"], "tests")
    _run([sys.executable, "scripts/check_api_health.py"], "api health")
    _run([sys.executable, "scripts/validate_dispatch.py"], "dispatch validation")
    _run([sys.executable, "scripts/run_monitoring.py"], "monitoring report")
    _run([sys.executable, "scripts/build_reports.py"], "reports")

    print("[release_check] OK")


if __name__ == "__main__":
    main()
