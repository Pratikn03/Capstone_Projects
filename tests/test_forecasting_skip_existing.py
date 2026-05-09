import json

import pytest

from orius.forecasting.train import (
    _initial_training_report,
    _load_json_dict,
    _reuse_existing_model_metrics,
)


def test_reuse_existing_model_metrics_returns_copy(tmp_path):
    artifact = tmp_path / "lstm_load_mw.pt"
    artifact.write_bytes(b"model")
    existing_report = {
        "targets": {
            "load_mw": {
                "lstm": {
                    "rmse": 1.2,
                    "mae": 0.8,
                    "uncertainty": {"picp_90": 0.91},
                }
            }
        }
    }

    metrics = _reuse_existing_model_metrics(
        existing_report=existing_report,
        target="load_mw",
        model_key="lstm",
        artifact_path=artifact,
    )

    assert metrics == existing_report["targets"]["load_mw"]["lstm"]
    metrics["rmse"] = 99.0
    assert existing_report["targets"]["load_mw"]["lstm"]["rmse"] == 1.2


def test_reuse_existing_model_metrics_fails_without_matching_metrics(tmp_path):
    artifact = tmp_path / "lstm_load_mw.pt"
    artifact.write_bytes(b"model")

    with pytest.raises(RuntimeError, match="Cannot use --skip-existing"):
        _reuse_existing_model_metrics(
            existing_report={"targets": {"load_mw": {}}},
            target="load_mw",
            model_key="lstm",
            artifact_path=artifact,
        )


def test_load_json_dict_handles_missing_and_invalid_files(tmp_path):
    assert _load_json_dict(tmp_path / "missing.json") == {}

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert _load_json_dict(invalid) == {}

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"ok": True}), encoding="utf-8")
    assert _load_json_dict(valid) == {"ok": True}


def test_initial_training_report_preserves_existing_targets_for_target_split():
    existing = {
        "device": "mps",
        "quantiles": [0.1, 0.9],
        "manifest_id": "old",
        "targets": {
            "target_a": {"gbm": {"rmse": 1.0}},
            "target_b": {"gbm": {"rmse": 2.0}},
        },
    }

    report = _initial_training_report(
        existing_report=existing,
        preserve_existing_targets=True,
        device="cpu",
        quantiles=[0.1, 0.5, 0.9],
        manifest_id="new",
    )

    assert report["device"] == "cpu"
    assert report["manifest_id"] == "new"
    assert report["targets"]["target_a"]["gbm"]["rmse"] == 1.0
    report["targets"]["target_a"]["gbm"]["rmse"] = 99.0
    assert existing["targets"]["target_a"]["gbm"]["rmse"] == 1.0


def test_initial_training_report_resets_targets_for_full_run():
    existing = {"targets": {"stale": {"gbm": {"rmse": 1.0}}}}

    report = _initial_training_report(
        existing_report=existing,
        preserve_existing_targets=False,
        device="cpu",
        quantiles=[0.1, 0.5, 0.9],
        manifest_id="fresh",
    )

    assert report["targets"] == {}
    assert report["manifest_id"] == "fresh"
