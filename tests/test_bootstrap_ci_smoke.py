from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


def test_bootstrap_ci_script_smoke(tmp_path: Path) -> None:
    rows: list[dict[str, float | int | str]] = []
    for scenario, scenario_offset in [("nominal", 0.0), ("drift_combo", 0.1)]:
        for controller, controller_offset in [("deterministic_lp", 0.0), ("dc3s_wrapped", 0.2)]:
            for seed in range(5):
                rows.append(
                    {
                        "scenario": scenario,
                        "controller": controller,
                        "seed": seed,
                        "picp_90": 0.70 + scenario_offset + controller_offset + 0.01 * seed,
                        "mean_interval_width": 100.0 + 10.0 * seed + 50.0 * controller_offset,
                        "expected_cost_usd": 1000.0 + 100.0 * seed + 500.0 * scenario_offset,
                        "intervention_rate": 0.02 + 0.001 * seed + controller_offset / 10.0,
                        "violation_rate": 0.10 + 0.002 * seed + scenario_offset / 10.0,
                    }
                )

    input_csv = tmp_path / "dc3s_main_table.csv"
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap_ci.py",
            "--in",
            str(input_csv),
            "--out-dir",
            str(tmp_path),
            "--group-cols",
            "scenario,controller",
            "--metrics",
            "picp_90,mean_interval_width,expected_cost_usd,intervention_rate,violation_rate",
            "--n-bootstrap",
            "100",
            "--confidence",
            "0.95",
            "--seed",
            "42",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    csv_path = tmp_path / "dc3s_main_table_ci.csv"
    md_path = tmp_path / "dc3s_main_table_ci.md"
    tex_path = tmp_path / "dc3s_main_table_ci.tex"

    for path in (csv_path, md_path, tex_path):
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip()

    summary = pd.read_csv(csv_path)
    assert len(summary) == 4
    assert {
        "scenario",
        "controller",
        "picp_90_mean",
        "picp_90_ci_low",
        "picp_90_ci_high",
        "mean_interval_width_mean",
        "expected_cost_usd_mean",
        "intervention_rate_mean",
        "violation_rate_mean",
    }.issubset(summary.columns)

    nominal_det = summary[
        (summary["scenario"] == "nominal") & (summary["controller"] == "deterministic_lp")
    ].iloc[0]
    expected_picp_mean = sum(0.70 + 0.01 * seed for seed in range(5)) / 5.0
    assert nominal_det["picp_90_mean"] == pytest.approx(expected_picp_mean)
