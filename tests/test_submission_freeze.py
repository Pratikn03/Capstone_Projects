from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

import scripts.build_baseline_comparison_table as baseline
import scripts.build_paper_table_tex as paper_tables
import scripts.run_r1_release as r1


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_tables_formats_missing_values_as_dashes(tmp_path: Path) -> None:
    de_metrics = tmp_path / "de" / "week2_metrics.json"
    us_metrics = tmp_path / "us" / "week2_metrics.json"
    de_uncertainty = tmp_path / "de_uncertainty"
    us_uncertainty = tmp_path / "us_uncertainty"
    out_dir = tmp_path / "publication"

    _write_json(
        de_metrics,
        {
            "targets": {
                "load_mw": {
                    "gbm": {"rmse": 10.0, "mae": 8.0, "smape": 0.12, "r2": 0.9},
                }
            }
        },
    )
    _write_json(
        us_metrics,
        {
            "targets": {
                "load_mw": {
                    "gbm": {"rmse": 11.0, "mae": 9.0, "smape": 0.10, "r2": 0.91},
                }
            }
        },
    )
    _write_json(de_uncertainty / "load_mw_conformal.json", {"meta": {"global_coverage": 0.88, "global_mean_width": 100.0}})
    _write_json(us_uncertainty / "load_mw_conformal.json", {"meta": {"global_coverage": 0.89, "global_mean_width": 120.0}})

    status = baseline.build_tables(
        out_dir=out_dir,
        de_metrics_json=de_metrics,
        us_metrics_json=us_metrics,
        de_uncertainty_dir=de_uncertainty,
        us_uncertainty_dir=us_uncertainty,
    )

    csv_text = (out_dir / "baseline_comparison_all.csv").read_text(encoding="utf-8")
    assert "---" in csv_text
    assert "NaN" not in csv_text
    assert status["thesis_headline_point_metrics_complete"] is False


def test_render_special_tables_are_compact(tmp_path: Path) -> None:
    tbl08_csv = tmp_path / "tbl08.csv"
    pd.DataFrame(
        [
            {
                "Region": "DE",
                "Target": "Load",
                "Model": "GBM",
                "RMSE": "305.12",
                "MAE": "187.87",
                "sMAPE (%)": "0.39",
                "R2": "0.9988",
                "PICP@90 (%)": "88.2",
                "Interval Width (MW)": "1070.2",
            },
            {
                "Region": "DE",
                "Target": "Load",
                "Model": "LSTM",
                "RMSE": "---",
                "MAE": "---",
                "sMAPE (%)": "---",
                "R2": "---",
                "PICP@90 (%)": "---",
                "Interval Width (MW)": "---",
            },
        ]
    ).to_csv(tbl08_csv, index=False)
    tbl08_tex = paper_tables.render_table_tex("TBL08_FORECAST_BASELINES", tbl08_csv, "Forecast")
    assert r"\resizebox{\linewidth}{!}" in tbl08_tex
    assert "NaN" not in tbl08_tex

    tbl02_csv = tmp_path / "tbl02.csv"
    pd.DataFrame(
        [
            {
                "analysis_scope": "primary_aggregate_fault_sweep",
                "fault_dimension": "dropout",
                "n_pairs": 40,
                "true_soc_violation_rate_baseline_mean": 0.25,
                "true_soc_violation_rate_dc3s_mean": 0.0,
                "true_soc_violation_rate_rel_reduction": 1.0,
                "true_soc_violation_rate_wilcoxon_p": 0.001,
                "true_soc_violation_severity_p95_baseline_mean": 333.0,
                "true_soc_violation_severity_p95_dc3s_mean": 0.0,
                "true_soc_violation_severity_p95_rel_reduction": 1.0,
                "true_soc_violation_severity_p95_wilcoxon_p": 0.001,
                "passes_all_thresholds": True,
            }
        ]
    ).to_csv(tbl02_csv, index=False)
    tbl02_tex = paper_tables.render_table_tex("TBL02_ABLATIONS", tbl02_csv, "Ablations")
    assert r"\resizebox{\linewidth}{!}" in tbl02_tex
    assert "Viol. base" in tbl02_tex
    assert "yes" in tbl02_tex


def test_stage_verify_uses_run_manifest_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(r1, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(r1, "_run", lambda cmd, description: True)

    release_id = "FINAL_TEST"
    for dataset in r1.R1_DATASETS:
        dataset_lower = dataset.lower()
        run_root = tmp_path / "artifacts" / "runs" / dataset_lower / release_id
        reports_dir = tmp_path / "reports" / "runs" / dataset_lower / release_id
        uncertainty_dir = run_root / "uncertainty"
        backtests_dir = run_root / "backtests"
        models_dir = run_root / "models"
        registry_dir = run_root / "registry"
        summary_path = registry_dir / f"tuning_summary_{dataset_lower}.json"
        preflight_path = reports_dir / "preflight_dataset_analysis.json"

        for path in (reports_dir, uncertainty_dir, backtests_dir, models_dir, registry_dir):
            path.mkdir(parents=True, exist_ok=True)

        _write_json(
            preflight_path,
            {
                "expected_targets": ["load_mw", "wind_mw", "solar_mw"],
                "expected_model_types": ["gbm_lightgbm", "lstm", "tcn", "nbeats", "tft", "patchtst"],
            },
        )
        _write_json(
            summary_path,
            {
                "accepted": True,
                "targets": [
                    {"target": "load_mw", "accepted": True},
                    {"target": "wind_mw", "accepted": True},
                    {"target": "solar_mw", "accepted": True},
                ],
            },
        )
        _write_json(
            registry_dir / "run_manifest.json",
            {
                "release_id": release_id,
                "run_id": release_id,
                "dataset": dataset,
                "accepted": False,
                "preflight_path": str(preflight_path),
                "selection_summary_path": str(summary_path),
                "artifacts": {
                    "reports_dir": str(reports_dir),
                    "models_dir": str(models_dir),
                    "uncertainty_dir": str(uncertainty_dir),
                    "backtests_dir": str(backtests_dir),
                },
            },
        )

    verification = r1.stage_verify(release_id)

    assert all(detail["passed"] for detail in verification.values())
    report = json.loads((tmp_path / "reports" / "runs" / f"{release_id}_verification.json").read_text(encoding="utf-8"))
    assert report["all_pass"] is True
    manifest = json.loads(
        (tmp_path / "artifacts" / "runs" / "de" / release_id / "registry" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["accepted"] is True
