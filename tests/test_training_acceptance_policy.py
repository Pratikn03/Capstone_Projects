from __future__ import annotations

import json
from pathlib import Path

from scripts.train_dataset import _evaluate_against_baseline


def _write_metrics(path: Path, targets: dict[str, dict[str, float]]) -> None:
    path.write_text(
        json.dumps({"targets": {target: {"gbm": metrics} for target, metrics in targets.items()}}),
        encoding="utf-8",
    )


def test_acceptance_uses_target_specific_metric(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_metrics(
        baseline_path,
        {"price_eur_mwh": {"mape": 1.34, "smape": 0.100}},
    )
    _write_metrics(
        reports_dir / "week2_metrics.json",
        {"price_eur_mwh": {"mape": 1.37, "smape": 0.099}},
    )

    result = _evaluate_against_baseline(
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

    assert result["accepted"] is True
    assert result["targets"][0]["metric"] == "smape"
    assert result["targets"][0]["accepted"] is True


def test_fast_profile_acceptance_allows_explicit_absolute_tolerance(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_metrics(baseline_path, {"spo2_pct": {"rmse": 0.356}})
    _write_metrics(reports_dir / "week2_metrics.json", {"spo2_pct": {"rmse": 0.410}})

    result = _evaluate_against_baseline(
        dataset_name="HEALTHCARE",
        reports_dir=reports_dir,
        target_metrics_file=baseline_path,
        publish_cfg={
            "retraining_acceptance": {
                "metric": "mape",
                "metric_by_target": {"spo2_pct": "rmse"},
                "profile_overrides": {
                    "production-max-fast": {"absolute_regression_tolerance_by_target": {"spo2_pct": 0.06}}
                },
                "require_non_regression": True,
            }
        },
        profile="production-max-fast",
    )

    assert result["accepted"] is True
    assert result["targets"][0]["absolute_regression_tolerance"] == 0.06
    assert result["targets"][0]["accepted"] is True


def test_strict_profile_rejects_same_regression_without_tolerance(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    _write_metrics(baseline_path, {"spo2_pct": {"rmse": 0.356}})
    _write_metrics(reports_dir / "week2_metrics.json", {"spo2_pct": {"rmse": 0.410}})

    result = _evaluate_against_baseline(
        dataset_name="HEALTHCARE",
        reports_dir=reports_dir,
        target_metrics_file=baseline_path,
        publish_cfg={
            "retraining_acceptance": {
                "metric": "mape",
                "metric_by_target": {"spo2_pct": "rmse"},
                "profile_overrides": {
                    "production-max-fast": {"absolute_regression_tolerance_by_target": {"spo2_pct": 0.06}}
                },
                "require_non_regression": True,
            }
        },
        profile="max",
    )

    assert result["accepted"] is False
    assert result["targets"][0]["accepted"] is False
