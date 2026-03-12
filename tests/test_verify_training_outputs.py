from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_training_outputs import ArtifactChecker


def test_check_week2_metrics_fails_when_expected_target_missing(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "targets": {
                    "load_mw": {
                        "gbm": {"rmse": 1.0},
                        "lstm": {"rmse": 2.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    checker = ArtifactChecker(verbose=False)
    ok = checker.check_week2_metrics(
        metrics_path,
        targets=["load_mw", "wind_mw"],
        model_types=["gbm_lightgbm", "lstm"],
    )

    assert ok is False
    assert any("wind_mw" in err for err in checker.errors)


def test_check_week2_metrics_fails_when_expected_model_entry_missing(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "targets": {
                    "load_mw": {
                        "gbm": {"rmse": 1.0},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    checker = ArtifactChecker(verbose=False)
    ok = checker.check_week2_metrics(
        metrics_path,
        targets=["load_mw"],
        model_types=["gbm_lightgbm", "lstm"],
    )

    assert ok is False
    assert any("lstm" in err for err in checker.errors)


def test_check_week2_metrics_accepts_conference_baseline_entries(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "targets": {
                    "load_mw": {
                        "gbm": {"rmse": 1.0},
                        "nbeats": {"rmse": 1.1},
                        "tft": {"rmse": 1.2},
                        "patchtst": {"rmse": 1.3},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    checker = ArtifactChecker(verbose=False)
    ok = checker.check_week2_metrics(
        metrics_path,
        targets=["load_mw"],
        model_types=["gbm_lightgbm", "nbeats", "tft", "patchtst"],
    )

    assert ok is True
    assert checker.errors == []
