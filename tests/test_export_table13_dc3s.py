from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.export_table13_dc3s import _print_drift_combo_summary, build_table13


def _write_input(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_table13_aggregates_required_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "dc3s_main_table.csv"
    rows = [
        {
            "scenario": "nominal",
            "seed": 0,
            "controller": "dc3s_wrapped",
            "picp_90": 0.91,
            "mean_interval_width": 100.0,
            "expected_cost_usd": 1000.0,
            "intervention_rate": 0.10,
            "violation_rate": 0.02,
        },
        {
            "scenario": "nominal",
            "seed": 1,
            "controller": "dc3s_wrapped",
            "picp_90": 0.93,
            "mean_interval_width": 120.0,
            "expected_cost_usd": 1100.0,
            "intervention_rate": 0.20,
            "violation_rate": 0.04,
        },
    ]
    _write_input(input_path, rows)

    payload = build_table13(input_path=input_path, out_dir=tmp_path)
    summary = pd.read_csv(tmp_path / "table13_dc3s_summary.csv")

    assert payload["rows"] == 1
    assert (tmp_path / "table13_dc3s_summary.md").exists()
    assert (tmp_path / "table13_dc3s_summary.tex").exists()
    assert summary.loc[0, "n_seeds"] == 2
    assert summary.loc[0, "picp_90"] == pytest.approx(0.92)
    assert summary.loc[0, "mean_interval_width"] == pytest.approx(110.0)
    assert summary.loc[0, "expected_cost_usd"] == pytest.approx(1050.0)


def test_build_table13_includes_optional_columns_when_present(tmp_path: Path) -> None:
    input_path = tmp_path / "dc3s_main_table.csv"
    rows = [
        {
            "scenario": "nominal",
            "seed": 0,
            "controller": "deterministic_lp",
            "picp_90": 0.90,
            "picp_95": 0.95,
            "mean_interval_width": 100.0,
            "expected_cost_usd": 1000.0,
            "intervention_rate": 0.10,
            "violation_rate": 0.02,
            "mae": 11.0,
            "rmse": 12.0,
        }
    ]
    _write_input(input_path, rows)

    payload = build_table13(input_path=input_path, out_dir=tmp_path)
    summary = pd.read_csv(tmp_path / "table13_dc3s_summary.csv")
    md = (tmp_path / "table13_dc3s_summary.md").read_text(encoding="utf-8")
    tex = (tmp_path / "table13_dc3s_summary.tex").read_text(encoding="utf-8")

    assert payload["optional_columns"] == ["mae", "rmse", "picp_95"]
    assert {"mae", "rmse", "picp_95"} <= set(summary.columns)
    assert "picp_95" in md
    assert "rmse" in tex


def test_build_table13_omits_missing_optional_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "dc3s_main_table.csv"
    rows = [
        {
            "scenario": "nominal",
            "seed": 0,
            "controller": "deterministic_lp",
            "picp_90": 0.90,
            "mean_interval_width": 100.0,
            "expected_cost_usd": 1000.0,
            "intervention_rate": 0.10,
            "violation_rate": 0.02,
        }
    ]
    _write_input(input_path, rows)

    payload = build_table13(input_path=input_path, out_dir=tmp_path)
    summary = pd.read_csv(tmp_path / "table13_dc3s_summary.csv")

    assert payload["omitted_optional_columns"] == ["mae", "rmse", "picp_95"]
    assert "mae" not in summary.columns
    assert "rmse" not in summary.columns
    assert "picp_95" not in summary.columns


def test_build_table13_requires_required_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "dc3s_main_table.csv"
    pd.DataFrame(
        [{"scenario": "nominal", "seed": 0, "controller": "deterministic_lp"}]
    ).to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        build_table13(input_path=input_path, out_dir=tmp_path)


def test_print_drift_combo_summary_outputs_expected_lines(capsys: pytest.CaptureFixture[str]) -> None:
    summary = pd.DataFrame(
        [
            {
                "scenario": "drift_combo",
                "controller": "dc3s_wrapped",
                "n_seeds": 2,
                "picp_90": 0.930,
                "mean_interval_width": 100.0,
                "expected_cost_usd": 990.0,
                "intervention_rate": 0.120,
                "violation_rate": 0.010,
            },
            {
                "scenario": "drift_combo",
                "controller": "dc3s_ftit",
                "n_seeds": 2,
                "picp_90": 0.950,
                "mean_interval_width": 108.0,
                "expected_cost_usd": 992.0,
                "intervention_rate": 0.080,
                "violation_rate": 0.005,
            },
            {
                "scenario": "drift_combo",
                "controller": "deterministic_lp",
                "n_seeds": 2,
                "picp_90": 0.900,
                "mean_interval_width": 90.0,
                "expected_cost_usd": 1010.0,
                "intervention_rate": 0.000,
                "violation_rate": 0.020,
            },
            {
                "scenario": "drift_combo",
                "controller": "robust_fixed_interval",
                "n_seeds": 2,
                "picp_90": 0.910,
                "mean_interval_width": 95.0,
                "expected_cost_usd": 995.0,
                "intervention_rate": 0.030,
                "violation_rate": 0.015,
            },
        ]
    )

    _print_drift_combo_summary(summary)
    out = capsys.readouterr().out
    assert "[Table13] drift_combo summary" in out
    assert "dc3s_wrapped picp_90: 0.930" in out
    assert "dc3s_ftit picp_90: 0.950" in out
    assert "deterministic_lp picp_90: 0.900" in out
    assert "best_dc3s_controller: dc3s_ftit" in out
    assert "cost delta vs robust_fixed_interval (USD): -3.00" in out
    assert "dc3s_ftit intervention_rate: 0.080" in out


def test_build_table13_sorts_rows_deterministically(tmp_path: Path) -> None:
    input_path = tmp_path / "dc3s_main_table.csv"
    rows = [
        {
            "scenario": "drift_combo",
            "seed": 0,
            "controller": "dc3s_wrapped",
            "picp_90": 0.92,
            "mean_interval_width": 110.0,
            "expected_cost_usd": 1000.0,
            "intervention_rate": 0.10,
            "violation_rate": 0.00,
        },
        {
            "scenario": "drift_combo",
            "seed": 0,
            "controller": "dc3s_ftit",
            "picp_90": 0.94,
            "mean_interval_width": 111.0,
            "expected_cost_usd": 999.0,
            "intervention_rate": 0.08,
            "violation_rate": 0.00,
        },
        {
            "scenario": "nominal",
            "seed": 0,
            "controller": "robust_fixed_interval",
            "picp_90": 0.90,
            "mean_interval_width": 100.0,
            "expected_cost_usd": 1005.0,
            "intervention_rate": 0.05,
            "violation_rate": 0.02,
        },
        {
            "scenario": "nominal",
            "seed": 0,
            "controller": "deterministic_lp",
            "picp_90": 0.89,
            "mean_interval_width": 95.0,
            "expected_cost_usd": 1010.0,
            "intervention_rate": 0.00,
            "violation_rate": 0.03,
        },
    ]
    _write_input(input_path, rows)

    build_table13(input_path=input_path, out_dir=tmp_path)
    summary = pd.read_csv(tmp_path / "table13_dc3s_summary.csv")

    assert summary[["scenario", "controller"]].to_dict(orient="records") == [
        {"scenario": "nominal", "controller": "deterministic_lp"},
        {"scenario": "nominal", "controller": "robust_fixed_interval"},
        {"scenario": "drift_combo", "controller": "dc3s_wrapped"},
        {"scenario": "drift_combo", "controller": "dc3s_ftit"},
    ]
