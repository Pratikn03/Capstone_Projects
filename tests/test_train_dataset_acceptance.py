from __future__ import annotations

import json
from pathlib import Path

import scripts.train_dataset as td


def _write_metrics(path: Path, target_payloads: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"targets": target_payloads}, indent=2), encoding="utf-8")


def test_acceptance_uses_target_specific_metric(tmp_path: Path) -> None:
    reports_dir = tmp_path / "candidate_reports"
    baseline_path = tmp_path / "baseline" / "week2_metrics.json"

    _write_metrics(
        reports_dir / "week2_metrics.json",
        {
            "price_eur_mwh": {
                "gbm": {
                    "mape": 1.37,
                    "smape": 0.099,
                }
            }
        },
    )
    _write_metrics(
        baseline_path,
        {
            "price_eur_mwh": {
                "gbm": {
                    "mape": 1.34,
                    "smape": 0.101,
                }
            }
        },
    )

    result = td._evaluate_against_baseline(
        dataset_name="DE",
        reports_dir=reports_dir,
        target_metrics_file=baseline_path,
        publish_cfg={
            "retraining_acceptance": {
                "metric": "mape",
                "metric_by_target": {"price_eur_mwh": "smape"},
                "require_non_regression": True,
            }
        },
        profile="production-max-fast",
    )

    row = result["targets"][0]
    assert result["accepted"] is True
    assert row["target"] == "price_eur_mwh"
    assert row["metric"] == "smape"
    assert row["current_metric"] == 0.099


def test_incumbent_retention_replaces_regressed_target_artifacts(tmp_path: Path) -> None:
    cfg = td.DATASET_REGISTRY["HEALTHCARE"]
    run_layout = td.RunLayout(
        mode="candidate",
        run_id="TEST_RETAIN",
        dataset="HEALTHCARE",
        artifacts_root=tmp_path / "run",
        models_dir=tmp_path / "run" / "models",
        uncertainty_dir=tmp_path / "run" / "uncertainty",
        backtests_dir=tmp_path / "run" / "backtests",
        registry_dir=tmp_path / "run" / "registry",
        reports_dir=tmp_path / "reports",
        publication_dir=tmp_path / "reports" / "publication",
        validation_report=tmp_path / "reports" / "data_quality.md",
        data_manifest_output=tmp_path / "run" / "registry" / "data_manifest.json",
        walk_forward_report=tmp_path / "reports" / "walk_forward.json",
        selection_output_dir=tmp_path / "run" / "registry",
    )
    for path in (
        run_layout.models_dir,
        run_layout.uncertainty_dir,
        run_layout.backtests_dir,
        run_layout.registry_dir,
        run_layout.reports_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    baseline_root = tmp_path / "canonical"
    baseline_models = baseline_root / "models"
    baseline_uncertainty = baseline_root / "uncertainty"
    baseline_backtests = baseline_root / "backtests"
    baseline_reports = baseline_root / "reports"
    for path in (baseline_models, baseline_uncertainty, baseline_backtests, baseline_reports):
        path.mkdir(parents=True, exist_ok=True)

    (run_layout.models_dir / "gbm_lightgbm_spo2_pct.pkl").write_text("candidate", encoding="utf-8")
    (baseline_models / "gbm_lightgbm_spo2_pct.pkl").write_text("incumbent", encoding="utf-8")
    (baseline_uncertainty / "gbm_spo2_pct_conformal.json").write_text("{}", encoding="utf-8")
    (baseline_backtests / "gbm_spo2_pct_test.npz").write_text("npz", encoding="utf-8")

    _write_metrics(
        run_layout.reports_dir / "week2_metrics.json",
        {"spo2_pct": {"gbm": {"mape": 0.0020, "mae": 0.20}}},
    )
    baseline_metrics = baseline_reports / "week2_metrics.json"
    _write_metrics(
        baseline_metrics,
        {"spo2_pct": {"gbm": {"mape": 0.0015, "mae": 0.14}}},
    )

    evaluation = {
        "targets": [
            {
                "target": "spo2_pct",
                "accepted": False,
                "non_regression_pass": False,
            }
        ]
    }

    retained = td._apply_incumbent_retention(
        cfg=cfg,
        run_layout=run_layout,
        evaluation=evaluation,
        publish_cfg={
            "retraining_acceptance": {
                "retain_incumbent_on_regression_targets": ["spo2_pct"],
            }
        },
        baseline_metrics_file=baseline_metrics,
        canonical_models_dir=baseline_models,
        canonical_uncertainty_dir=baseline_uncertainty,
        canonical_backtests_dir=baseline_backtests,
        canonical_reports_dir=baseline_reports,
    )

    metrics = json.loads((run_layout.reports_dir / "week2_metrics.json").read_text(encoding="utf-8"))
    assert retained == ["spo2_pct"]
    assert (run_layout.models_dir / "gbm_lightgbm_spo2_pct.pkl").read_text(encoding="utf-8") == "incumbent"
    assert (run_layout.uncertainty_dir / "gbm_spo2_pct_conformal.json").exists()
    assert (run_layout.backtests_dir / "gbm_spo2_pct_test.npz").exists()
    assert metrics["targets"]["spo2_pct"]["gbm"]["mape"] == 0.0015
    assert metrics["targets"]["spo2_pct"]["retention_decision"] == "retained_incumbent"


def test_incumbent_retention_replaces_target_that_misses_improvement_gate(tmp_path: Path) -> None:
    cfg = td.DATASET_REGISTRY["DE"]
    run_layout = td.RunLayout(
        mode="candidate",
        run_id="TEST_RETAIN_IMPROVEMENT",
        dataset="DE",
        artifacts_root=tmp_path / "run",
        models_dir=tmp_path / "run" / "models",
        uncertainty_dir=tmp_path / "run" / "uncertainty",
        backtests_dir=tmp_path / "run" / "backtests",
        registry_dir=tmp_path / "run" / "registry",
        reports_dir=tmp_path / "reports",
        publication_dir=tmp_path / "reports" / "publication",
        validation_report=tmp_path / "reports" / "data_quality.md",
        data_manifest_output=tmp_path / "run" / "registry" / "data_manifest.json",
        walk_forward_report=tmp_path / "reports" / "walk_forward.json",
        selection_output_dir=tmp_path / "run" / "registry",
    )
    for path in (
        run_layout.models_dir,
        run_layout.uncertainty_dir,
        run_layout.backtests_dir,
        run_layout.registry_dir,
        run_layout.reports_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    baseline_root = tmp_path / "canonical"
    baseline_models = baseline_root / "models"
    baseline_uncertainty = baseline_root / "uncertainty"
    baseline_backtests = baseline_root / "backtests"
    baseline_reports = baseline_root / "reports"
    for path in (baseline_models, baseline_uncertainty, baseline_backtests, baseline_reports):
        path.mkdir(parents=True, exist_ok=True)

    (run_layout.models_dir / "gbm_lightgbm_load_mw.pkl").write_text("candidate", encoding="utf-8")
    (baseline_models / "gbm_lightgbm_load_mw.pkl").write_text("incumbent", encoding="utf-8")
    (baseline_uncertainty / "gbm_load_mw_conformal.json").write_text("{}", encoding="utf-8")
    (baseline_backtests / "gbm_load_mw_test.npz").write_text("npz", encoding="utf-8")

    _write_metrics(
        run_layout.reports_dir / "week2_metrics.json",
        {"load_mw": {"gbm": {"mape": 0.00999}}},
    )
    baseline_metrics = baseline_reports / "week2_metrics.json"
    _write_metrics(
        baseline_metrics,
        {"load_mw": {"gbm": {"mape": 0.01000}}},
    )

    evaluation = {
        "targets": [
            {
                "target": "load_mw",
                "accepted": False,
                "non_regression_pass": True,
                "improvement_pass": False,
                "regression_delta": -0.00001,
            }
        ]
    }

    retained = td._apply_incumbent_retention(
        cfg=cfg,
        run_layout=run_layout,
        evaluation=evaluation,
        publish_cfg={
            "retraining_acceptance": {
                "retain_incumbent_on_regression_targets": ["load_mw"],
            }
        },
        baseline_metrics_file=baseline_metrics,
        canonical_models_dir=baseline_models,
        canonical_uncertainty_dir=baseline_uncertainty,
        canonical_backtests_dir=baseline_backtests,
        canonical_reports_dir=baseline_reports,
    )

    metrics = json.loads((run_layout.reports_dir / "week2_metrics.json").read_text(encoding="utf-8"))
    assert retained == ["load_mw"]
    assert (run_layout.models_dir / "gbm_lightgbm_load_mw.pkl").read_text(encoding="utf-8") == "incumbent"
    assert metrics["targets"]["load_mw"]["retention_decision"] == "retained_incumbent"
    assert metrics["targets"]["load_mw"]["retention_reason"] == "challenger_missed_acceptance_gate"


def test_retained_incumbent_satisfies_positive_improvement_gate(tmp_path: Path) -> None:
    reports_dir = tmp_path / "candidate_reports"
    baseline_path = tmp_path / "baseline" / "week2_metrics.json"

    _write_metrics(
        reports_dir / "week2_metrics.json",
        {
            "load_mw": {
                "retention_decision": "retained_incumbent",
                "gbm": {
                    "mape": 0.01,
                },
            }
        },
    )
    _write_metrics(
        baseline_path,
        {
            "load_mw": {
                "gbm": {
                    "mape": 0.01,
                }
            }
        },
    )

    result = td._evaluate_against_baseline(
        dataset_name="DE",
        reports_dir=reports_dir,
        target_metrics_file=baseline_path,
        publish_cfg={
            "retraining_acceptance": {
                "metric": "mape",
                "require_non_regression": True,
                "min_improvement_by_target": {"load_mw": 0.005},
            }
        },
        profile="production-max-fast",
    )

    row = result["targets"][0]
    assert result["accepted"] is True
    assert row["target"] == "load_mw"
    assert row["decision"] == "retained_incumbent"
    assert row["improvement"] == 0.0
    assert row["improvement_pass"] is True
