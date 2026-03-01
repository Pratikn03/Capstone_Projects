from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_sensitivity_sweeps_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "scripts/run_sensitivity_sweeps.py",
            "--out-dir",
            str(tmp_path),
            "--horizon",
            "24",
            "--scenarios",
            "drift_combo",
            "--seeds",
            "11",
            "--alpha0",
            "0.10",
            "--ph-lambda",
            "5.0",
            "--kappa-drift-penalty",
            "0.5",
            "--n-bootstrap",
            "100",
            "--confidence",
            "0.95",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    grid_csv = tmp_path / "sensitivity_grid.csv"
    summary_csv = tmp_path / "sensitivity_summary_ci.csv"
    summary_md = tmp_path / "sensitivity_summary_ci.md"

    for path in (grid_csv, summary_csv, summary_md):
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip()

    grid_df = pd.read_csv(grid_csv)
    summary_df = pd.read_csv(summary_csv)

    assert not grid_df.empty
    assert not summary_df.empty

    assert {
        "scenario",
        "seed",
        "controller",
        "alpha0",
        "ph_lambda",
        "kappa_drift_penalty",
        "horizon",
        "seeds",
        "run_dir",
    }.issubset(grid_df.columns)
    assert {
        "scenario",
        "controller",
        "alpha0",
        "ph_lambda",
        "kappa_drift_penalty",
        "picp_90_mean",
        "picp_90_ci_low",
        "picp_90_ci_high",
        "mean_interval_width_mean",
        "expected_cost_usd_mean",
        "intervention_rate_mean",
    }.issubset(summary_df.columns)
