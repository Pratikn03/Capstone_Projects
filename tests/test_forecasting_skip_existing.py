import json

import pytest

from orius.forecasting.train import _load_json_dict, _reuse_existing_model_metrics


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
