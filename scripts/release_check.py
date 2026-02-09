"""
Pre-release quality gate to ensure production readiness.

This script runs a comprehensive checklist before any release or deployment:
1. Configuration validation
2. Required data artifacts exist
3. Unit and integration tests pass
4. API health check
5. Dispatch plan validation
6. Monitoring reports generation
7. Model registry update

Usage:
    # Quick check (assumes data already processed)
    python scripts/release_check.py
    
    # Full check including data pipeline and training
    python scripts/release_check.py --full

Exit codes:
    0 - All checks passed, safe to release
    1 - One or more checks failed - DO NOT RELEASE
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str], label: str, env: dict | None = None) -> None:
    """
    Execute a subprocess command as part of the release checklist.
    
    Args:
        cmd: Command and arguments to execute
        label: Human-readable description for logging
        env: Optional environment variables override
        
    Raises:
        subprocess.CalledProcessError: If command fails (blocks release)
    """
    print(f"[release_check] {label}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def _require(path: Path, msg: str) -> None:
    """
    Assert that a required file or directory exists.
    
    This is a hard requirement - if the artifact is missing,
    we cannot proceed with the release.
    
    Args:
        path: Path to the required artifact
        msg: Error message explaining how to fix the issue
        
    Raises:
        SystemExit: If the required path does not exist
    """
    if not path.exists():
        raise SystemExit(msg)


def main() -> None:
    """
    Main release check workflow.
    
    Runs through all quality gates in sequence. Early failure on any
    gate prevents wasted time on downstream checks.
    """
    # Parse command-line arguments
    ap = argparse.ArgumentParser(
        description="Run production readiness checks before release"
    )
    ap.add_argument(
        "--full", 
        action="store_true", 
        help="Run complete pipeline including data processing and training"
    )
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Optional: Full pipeline execution (for CI/CD or fresh environments)
    # -------------------------------------------------------------------------
    if args.full:
        # Run the entire data pipeline from raw data to processed features
        _run(
            [sys.executable, "-m", "gridpulse.pipeline.run", "--all"], 
            "data pipeline"
        )
        # Train models on the processed data
        _run(
            [sys.executable, "-m", "gridpulse.forecasting.train", 
             "--config", "configs/train_forecast.yaml"], 
            "train"
        )

    # -------------------------------------------------------------------------
    # GATE 1: Configuration files are valid
    # -------------------------------------------------------------------------
    _run([sys.executable, "scripts/validate_configs.py"], "config validation")

    # -------------------------------------------------------------------------
    # GATE 2: Required data artifacts exist
    # These are hard requirements - cannot proceed without them
    # -------------------------------------------------------------------------
    _require(
        Path("data/processed/features.parquet"), 
        "Missing features.parquet. Run `make data` to generate."
    )
    _require(
        Path("data/processed/splits/train.parquet"), 
        "Missing train split. Run `make data` to generate."
    )
    _require(
        Path("data/processed/splits/test.parquet"), 
        "Missing test split. Run `make data` to generate."
    )

    # -------------------------------------------------------------------------
    # GATE 3: All tests pass
    # -------------------------------------------------------------------------
    _run([sys.executable, "-m", "pytest", "-q"], "tests")

    # -------------------------------------------------------------------------
    # GATE 4: API service is healthy
    # -------------------------------------------------------------------------
    _run([sys.executable, "scripts/check_api_health.py"], "api health")

    # -------------------------------------------------------------------------
    # GATE 5: Dispatch plans validate correctly
    # -------------------------------------------------------------------------
    _run([sys.executable, "scripts/validate_dispatch.py"], "dispatch validation")

    # -------------------------------------------------------------------------
    # GATE 6: Monitoring and drift reports
    # -------------------------------------------------------------------------
    _run([sys.executable, "scripts/run_monitoring.py"], "monitoring report")

    # -------------------------------------------------------------------------
    # GATE 7: Models registered in artifact store
    # -------------------------------------------------------------------------
    _run([sys.executable, "scripts/register_models.py"], "model registry")

    # -------------------------------------------------------------------------
    # GATE 8: Generate documentation and reports
    # Set up matplotlib to work in headless environments (CI servers)
    # -------------------------------------------------------------------------
    tmp_dir = Path(tempfile.gettempdir())
    mpl_dir = tmp_dir / "mplconfig"
    xdg_dir = tmp_dir / "xdg_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure environment for headless matplotlib rendering
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")  # Non-interactive backend
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))  # Avoid ~/.matplotlib issues
    env.setdefault("XDG_CACHE_HOME", str(xdg_dir))  # Cache directory
    
    _run([sys.executable, "scripts/build_reports.py"], "reports", env=env)

    # -------------------------------------------------------------------------
    # All checks passed - safe to release!
    # -------------------------------------------------------------------------
    print("[release_check] OK - All quality gates passed!")


if __name__ == "__main__":
    main()
