from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "run_battery_reliability_baselines.py"
    spec = importlib.util.spec_from_file_location("battery_reliability_baselines", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "controller": "dc3s_ftit",
                "true_soc_violation_rate": 0.0,
                "violation_rate": 0.0,
                "picp_90": 0.97,
                "mean_interval_width": 3900.0,
                "intervention_rate": 0.03,
                "expected_cost_usd": 394_000_000.0,
            },
            {
                "controller": "dc3s_wrapped",
                "true_soc_violation_rate": 0.0,
                "violation_rate": 0.0,
                "picp_90": 0.89,
                "mean_interval_width": 560.0,
                "intervention_rate": 0.0,
                "expected_cost_usd": 393_900_000.0,
            },
            {
                "controller": "aci_conformal",
                "true_soc_violation_rate": 0.0,
                "violation_rate": 0.0,
                "picp_90": 0.96,
                "mean_interval_width": 2820.0,
                "intervention_rate": 0.014,
                "expected_cost_usd": 394_100_000.0,
            },
            {
                "controller": "deterministic_lp",
                "true_soc_violation_rate": 0.04,
                "violation_rate": 0.02,
                "picp_90": 0.88,
                "mean_interval_width": 503.0,
                "intervention_rate": 0.0,
                "expected_cost_usd": 394_050_000.0,
            },
            {
                "controller": "robust_fixed_interval",
                "true_soc_violation_rate": 0.25,
                "violation_rate": 0.25,
                "picp_90": 0.88,
                "mean_interval_width": 503.0,
                "intervention_rate": 0.0,
                "expected_cost_usd": 393_980_000.0,
            },
        ]
    )


def test_summarize_locked_battery_controller_table_reports_real_metrics() -> None:
    mod = _load_module()
    summary = mod.summarize_locked_battery_controller_table(_fixture_frame())

    orius = summary["metrics_by_method"]["orius_repair"]
    deterministic = summary["metrics_by_method"]["deterministic_no_repair"]

    assert orius["source_controller"] == "dc3s_ftit"
    assert orius["true_state_violation_rate"] == 0.0
    assert orius["coverage_90"] == 0.97
    assert deterministic["observed_state_violation_rate"] == 0.02
    assert deterministic["hidden_gap_rate"] == 0.02
    assert "weighted or Mondrian dispatch controllers are not synthesized" in " ".join(summary["notes"])


def test_battery_reliability_baseline_cli_writes_deterministic_artifacts(tmp_path: Path) -> None:
    mod = _load_module()
    input_path = tmp_path / "dc3s_main_table.csv"
    out_dir = tmp_path / "out"
    paper_table = tmp_path / "paper" / "tbl.tex"
    paper_figure = tmp_path / "paper" / "fig.png"
    _fixture_frame().to_csv(input_path, index=False)

    args = [
        "--main-table",
        str(input_path),
        "--output-dir",
        str(out_dir),
        "--paper-table",
        str(paper_table),
        "--paper-figure",
        str(paper_figure),
    ]
    assert mod.main(args) == 0
    first_json = (out_dir / "battery_reliability_baselines_summary.json").read_text()
    first_csv = (out_dir / "battery_reliability_baselines_summary.csv").read_text()

    assert mod.main(args) == 0
    second_json = (out_dir / "battery_reliability_baselines_summary.json").read_text()
    second_csv = (out_dir / "battery_reliability_baselines_summary.csv").read_text()

    assert first_json == second_json
    assert first_csv == second_csv

    summary = json.loads(first_json)
    assert summary["study"] == "battery_reliability_baselines"
    assert summary["outputs"]["paper_table_tex"].endswith("tbl.tex")
    assert paper_table.exists()
    assert paper_figure.exists()
    assert (out_dir / "tbl_battery_reliability_baselines.tex").exists()
    assert (out_dir / "fig_battery_reliability_baselines.png").exists()
