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


def test_check_conformal_artifacts_warns_for_missing_dl_model(tmp_path: Path) -> None:
    """DL-model conformal JSON absence should produce a warning, not a hard failure."""
    uncertainty_dir = tmp_path / "uncertainty"
    uncertainty_dir.mkdir()
    backtests_dir = tmp_path / "backtests"
    backtests_dir.mkdir()

    # Write the required GBM legacy conformal JSON.
    import json

    gbm_conf = uncertainty_dir / "load_mw_conformal.json"
    gbm_conf.write_text(
        json.dumps({"config": {"alpha": 0.1}, "meta": {"global_coverage": 0.91}}),
        encoding="utf-8",
    )

    import numpy as np

    # Write the required GBM NPZ files.
    for suffix in ("calibration", "test"):
        np.savez(backtests_dir / f"load_mw_{suffix}.npz", y_true=np.zeros(10), y_pred=np.zeros(10))

    # No LSTM conformal JSON written → should warn, not fail.
    checker = ArtifactChecker(verbose=False)
    checker.check_conformal_artifacts(
        uncertainty_dir=uncertainty_dir,
        backtests_dir=backtests_dir,
        targets=["load_mw"],
        model_types=["gbm_lightgbm", "lstm"],
    )

    assert checker.checks_failed == 0, f"Expected no failures; got: {checker.errors}"
    assert any("lstm" in w.lower() for w in checker.warnings), "Expected warning about missing LSTM conformal artifact"


def test_check_conformal_artifacts_passes_for_dl_model_present(tmp_path: Path) -> None:
    """DL-model conformal JSON presence should be validated and counted as a passing check."""
    uncertainty_dir = tmp_path / "uncertainty"
    uncertainty_dir.mkdir()
    backtests_dir = tmp_path / "backtests"
    backtests_dir.mkdir()

    import json

    import numpy as np

    # Write GBM legacy path.
    (uncertainty_dir / "load_mw_conformal.json").write_text(
        json.dumps({"config": {"alpha": 0.1}, "meta": {"global_coverage": 0.91}}),
        encoding="utf-8",
    )
    for suffix in ("calibration", "test"):
        np.savez(backtests_dir / f"load_mw_{suffix}.npz", y_true=np.zeros(10), y_pred=np.zeros(10))

    # Write LSTM conformal JSON + NPZ pairs.
    (uncertainty_dir / "lstm_load_mw_conformal.json").write_text(
        json.dumps({"config": {"alpha": 0.1}, "meta": {"global_coverage": 0.89}}),
        encoding="utf-8",
    )
    for suffix in ("calibration", "test"):
        np.savez(
            backtests_dir / f"lstm_load_mw_{suffix}.npz",
            y_true=np.zeros(10),
            q_lo=np.zeros(10),
            q_hi=np.ones(10),
        )

    checker = ArtifactChecker(verbose=False)
    checker.check_conformal_artifacts(
        uncertainty_dir=uncertainty_dir,
        backtests_dir=backtests_dir,
        targets=["load_mw"],
        model_types=["gbm_lightgbm", "lstm"],
    )

    assert checker.checks_failed == 0, f"Expected no failures; got: {checker.errors}"
    # GBM (legacy npz x2) + LSTM conformal JSON + LSTM (npz x2) = 5 passing checks minimum.
    assert checker.checks_passed >= 5

