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


def _strong_deep_model() -> dict:
    return {
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
            "non_finite_loss": False,
            "gradient_clipped_fraction": 0.1,
            "max_grad_norm": 14.0,
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


def test_model_quality_gate_skips_non_model_metadata_rows(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    out_path = tmp_path / "model_quality_gate.json"
    payload = _strong_metrics()
    payload["targets"]["load_mw"]["challenger_metrics"] = {"rmse": 999.0}
    payload["targets"]["load_mw"]["retention_decision"] = "keep_incumbent"
    payload["targets"]["load_mw"]["retention_reason"] = "unit-test metadata"
    _write_json(metrics_path, payload)

    result = builder.build_model_quality_gate(metrics_paths=[metrics_path], out_path=out_path)

    assert result["pass"] is True
    assert result["summary"]["model_count"] == 1
    assert [row["model"] for row in result["models"]] == ["gbm"]


def test_model_quality_gate_accepts_fixed_deep_architecture_without_tuning(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    out_path = tmp_path / "model_quality_gate.json"
    payload = _strong_metrics()
    payload["targets"]["load_mw"]["lstm"] = _strong_deep_model()
    _write_json(metrics_path, payload)

    result = builder.build_model_quality_gate(metrics_paths=[metrics_path], out_path=out_path)

    assert result["pass"] is True
    rows = {row["model"]: row for row in result["models"]}
    assert rows["gbm"]["release_model"] is True
    assert rows["lstm"]["release_model"] is False
    assert rows["lstm"]["gates"]["hyperparameter_tuning"]["status"] == "pass"
    assert "fixed deep architecture" in rows["lstm"]["gates"]["hyperparameter_tuning"]["detail"]


def test_model_quality_gate_can_require_deep_tuning_when_policy_demands_it(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    out_path = tmp_path / "model_quality_gate.json"
    payload = _strong_metrics()
    payload["targets"]["load_mw"]["lstm"] = _strong_deep_model()
    _write_json(metrics_path, payload)

    result = builder.build_model_quality_gate(
        metrics_paths=[metrics_path],
        out_path=out_path,
        policy={"require_deep_hyperparameter_tuning": True, "block_candidate_models": True},
    )

    assert result["pass"] is False
    assert "missing hyperparameter tuning metadata" in "\n".join(result["blockers"])


def test_model_quality_gate_records_candidate_failures_without_blocking_release(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    out_path = tmp_path / "model_quality_gate.json"
    payload = _strong_metrics()
    payload["targets"]["load_mw"]["tft"] = _strong_deep_model()
    payload["targets"]["load_mw"]["tft"]["r2"] = -0.5
    payload["targets"]["load_mw"]["tft"]["training_summary"]["non_finite_loss"] = True
    _write_json(metrics_path, payload)

    result = builder.build_model_quality_gate(metrics_paths=[metrics_path], out_path=out_path)

    assert result["pass"] is True
    assert result["blockers"] == []
    assert result["summary"]["blocking_release_model_count"] == 0
    assert result["summary"]["candidate_blocking_model_count"] == 1
    assert "gradient stability" in "\n".join(result["candidate_findings"])


def test_model_quality_gate_uses_matching_domain_config_for_boundary_checks(tmp_path: Path) -> None:
    metrics_path = tmp_path / "reports" / "runs" / "de" / "R" / "week2_metrics.json"
    de_config = tmp_path / "train_forecast.yaml"
    av_config = tmp_path / "train_forecast_av.yaml"
    out_path = tmp_path / "model_quality_gate.json"
    payload = _strong_metrics()
    payload["targets"]["load_mw"]["gbm"]["tuning_meta"]["selected_params"] = {"n_estimators": 900}
    _write_json(metrics_path, payload)
    _write_json(
        de_config,
        {
            "dataset": {"key": "DE"},
            "tuning": {"params": {"baseline_gbm": {"n_estimators": {"type": "int", "low": 100, "high": 1000}}}},
        },
    )
    _write_json(
        av_config,
        {
            "dataset": {"key": "AV"},
            "tuning": {"params": {"baseline_gbm": {"n_estimators": {"type": "int", "low": 100, "high": 600}}}},
        },
    )

    result = builder.build_model_quality_gate(
        metrics_paths=[metrics_path],
        config_paths=[de_config, av_config],
        out_path=out_path,
    )

    tuning_gate = result["models"][0]["gates"]["hyperparameter_tuning"]
    assert result["pass"] is True
    assert tuning_gate["status"] == "pass"
    assert tuning_gate["metrics"]["boundary_hits"] == []


def test_model_quality_gate_can_make_candidate_failures_blocking_by_policy(tmp_path: Path) -> None:
    metrics_path = tmp_path / "week2_metrics.json"
    out_path = tmp_path / "model_quality_gate.json"
    payload = _strong_metrics()
    payload["targets"]["load_mw"]["tft"] = _strong_deep_model()
    payload["targets"]["load_mw"]["tft"]["r2"] = -0.5
    _write_json(metrics_path, payload)

    result = builder.build_model_quality_gate(
        metrics_paths=[metrics_path],
        out_path=out_path,
        policy={"block_candidate_models": True},
    )

    assert result["pass"] is False
    assert "underfit risk" in "\n".join(result["blockers"])


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
    payload["targets"]["load_mw"]["lstm"] = _strong_deep_model()
    payload["targets"]["load_mw"]["lstm"]["training_summary"]["non_finite_loss"] = True
    payload["targets"]["load_mw"]["lstm"]["training_summary"]["gradient_clipped_fraction"] = 0.9
    _write_json(metrics_path, payload)

    result = builder.build_model_quality_gate(metrics_paths=[metrics_path], out_path=out_path)

    assert result["pass"] is False
    blockers = "\n".join(result["blockers"])
    candidate_findings = "\n".join(result["candidate_findings"])
    assert "overfit" in blockers
    assert "underfit" in blockers
    assert "gradient stability" in candidate_findings


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
