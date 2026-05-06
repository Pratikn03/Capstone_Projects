from __future__ import annotations

import json
from pathlib import Path

import scripts.build_model_quality_gate as builder
import scripts.validate_model_quality_gate as validator


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _strong_metrics() -> dict:
    return {
        "device": "cpu",
        "manifest_id": "unit-test",
        "targets": {
            "load_mw": {
                "n_features": 12,
                "gbm": {
                    "rmse": 11.0,
                    "r2": 0.91,
                    "split_metrics": {
                        "train": {"rmse": 10.0},
                        "validation": {"rmse": 10.5},
                        "test": {"rmse": 11.0},
                    },
                    "latency": {"p95_per_sample_ms": 0.15},
                    "uncertainty": {"picp_90": 0.91, "mean_interval_width": 3.0},
                    "tuning_meta": {
                        "enabled": True,
                        "n_trials": 100,
                        "n_complete_trials": 100,
                        "best_objective": 10.3,
                        "selected_params": {"learning_rate": 0.03, "num_leaves": 63},
                    },
                },
            }
        },
    }


def _config() -> dict:
    return {
        "tuning": {
            "enabled": True,
            "n_trials": 100,
            "min_top_trials": 5,
            "select_top_pct": 0.1,
            "params": {
                "baseline_gbm": {
                    "learning_rate": {"type": "float", "low": 0.005, "high": 0.1, "log": True},
                    "num_leaves": {"type": "int", "low": 31, "high": 127},
                }
            },
        }
    }


def test_model_quality_gate_passes_strong_metrics(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    config_path = tmp_path / "train.yaml"
    out_path = tmp_path / "model_quality_gate.json"
    _write_json(metrics_path, _strong_metrics())
    _write_json(config_path, _config())

    result = builder.build_model_quality_gate(
        metrics_paths=[metrics_path],
        config_paths=[config_path],
        out_path=out_path,
    )

    assert result["pass"] is True
    assert result["summary"]["model_count"] == 1
    assert result["summary"]["blocking_model_count"] == 0
    assert result["models"][0]["gates"]["generalization"]["status"] == "pass"
    assert result["models"][0]["gates"]["architecture"]["status"] == "pass"
    assert result["models"][0]["gates"]["calibration"]["status"] == "pass"
    assert result["models"][0]["gates"]["latency"]["status"] == "pass"

    validation = validator.validate_model_quality_gate(out_path)
    assert validation["pass"] is True


def test_model_quality_gate_blocks_missing_train_latency_and_tuning(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    out_path = tmp_path / "model_quality_gate.json"
    payload = _strong_metrics()
    model = payload["targets"]["load_mw"]["gbm"]
    model.pop("split_metrics")
    model.pop("latency")
    model["tuning_meta"] = None
    _write_json(metrics_path, payload)

    result = builder.build_model_quality_gate(metrics_paths=[metrics_path], out_path=out_path)

    assert result["pass"] is False
    blockers = "\n".join(result["blockers"])
    assert "train split metrics" in blockers
    assert "latency" in blockers
    assert "hyperparameter tuning" in blockers


def test_model_quality_gate_blocks_missing_architecture_and_calibration(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    out_path = tmp_path / "model_quality_gate.json"
    payload = _strong_metrics()
    payload["targets"]["load_mw"].pop("n_features")
    payload["targets"]["load_mw"]["gbm"].pop("uncertainty")
    _write_json(metrics_path, payload)

    result = builder.build_model_quality_gate(metrics_paths=[metrics_path], out_path=out_path)

    assert result["pass"] is False
    blockers = "\n".join(result["blockers"])
    assert "architecture" in blockers
    assert "calibration" in blockers


def test_model_quality_gate_detects_overfit_underfit_and_gradient_instability(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    out_path = tmp_path / "model_quality_gate.json"
    payload = _strong_metrics()
    payload["targets"]["load_mw"]["gbm"]["split_metrics"]["validation"]["rmse"] = 22.0
    payload["targets"]["load_mw"]["gbm"]["r2"] = -0.2
    payload["targets"]["load_mw"]["lstm"] = {
        "rmse": 12.0,
        "r2": 0.7,
        "split_metrics": {
            "train": {"rmse": 10.0},
            "validation": {"rmse": 10.4},
            "test": {"rmse": 12.0},
        },
        "latency": {"p95_per_sample_ms": 0.4},
        "uncertainty": {"picp_90": 0.92, "mean_interval_width": 2.5},
        "model_architecture": {
            "lookback": 24,
            "horizon": 6,
            "dropout": 0.1,
            "gradient_clip": 1.0,
            "early_stopping_patience": 4,
        },
        "training_summary": {
            "epochs_ran": 4,
            "best_val_loss": 0.8,
            "last_train_loss": 0.7,
            "last_val_loss": 0.8,
            "non_finite_loss": True,
            "gradient_clipped_fraction": 0.9,
            "max_grad_norm": 14.0,
        },
        "tuning_meta": {
            "enabled": True,
            "n_trials": 100,
            "n_complete_trials": 100,
            "selected_params": {"learning_rate": 0.001},
        },
    }
    _write_json(metrics_path, payload)

    result = builder.build_model_quality_gate(metrics_paths=[metrics_path], out_path=out_path)

    assert result["pass"] is False
    blockers = "\n".join(result["blockers"])
    assert "overfit" in blockers
    assert "underfit" in blockers
    assert "gradient stability" in blockers


def test_model_quality_validator_rejects_hand_edited_pass(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    out_path = tmp_path / "model_quality_gate.json"
    payload = _strong_metrics()
    payload["targets"]["load_mw"]["gbm"].pop("latency")
    _write_json(metrics_path, payload)
    builder.build_model_quality_gate(metrics_paths=[metrics_path], out_path=out_path)

    edited = json.loads(out_path.read_text(encoding="utf-8"))
    edited["pass"] = True
    edited["blockers"] = []
    edited["models"][0]["gates"]["latency"]["status"] = "pass"
    edited["models"][0]["gates"]["latency"]["detail"] = "fake pass"
    out_path.write_text(json.dumps(edited, indent=2, sort_keys=True), encoding="utf-8")

    result = validator.validate_model_quality_gate(out_path, metrics_paths=[metrics_path])

    assert result["pass"] is False
    assert any("does not match recomputed gate" in finding for finding in result["findings"])
