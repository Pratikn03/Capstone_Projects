"""Tests for training profile command wiring."""

from __future__ import annotations

import pandas as pd

import scripts.train_dataset as td


def _flag_value(cmd: list[str], flag: str) -> str:
    assert flag in cmd
    return cmd[cmd.index(flag) + 1]


def test_report_subprocesses_use_bounded_thread_environment(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(cmd, check, capture_output, timeout, cwd, env):
        captured["cmd"] = cmd
        captured["check"] = check
        captured["env"] = env

    monkeypatch.setattr(td.subprocess, "run", fake_run)

    ok = td.run_command(["python", "scripts/build_reports.py"], "Generating reports for AV")

    assert ok is True
    assert captured["cmd"] == ["python", "scripts/build_reports.py"]
    assert captured["check"] is True
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        assert captured["env"][key] == "1"


def test_train_models_aggressive_profile_adds_expected_flags(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_command(cmd: list[str], description: str, timeout_seconds=None) -> bool:
        captured["cmd"] = cmd
        captured["description"] = description
        captured["timeout_seconds"] = timeout_seconds
        return True

    monkeypatch.setattr(td, "run_command", fake_run_command)
    ok = td.train_models(
        td.DATASET_REGISTRY["DE"],
        profile="aggressive",
        tune=False,
        no_tune=False,
        ensemble=False,
        max_seeds=None,
        n_trials=None,
        top_pct=None,
        max_runtime_hours=1.5,
    )
    assert ok is True
    cmd = captured["cmd"]
    assert "--tune" in cmd
    assert "--ensemble" in cmd
    assert "--n-trials" in cmd and "220" in cmd
    assert "--max-seeds" in cmd and "8" in cmd
    assert "--top-pct" in cmd and "0.2" in cmd
    assert captured["timeout_seconds"] == 5400.0


def test_train_models_max_profile_uses_stronger_defaults(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_command(cmd: list[str], description: str, timeout_seconds=None) -> bool:
        captured["cmd"] = cmd
        captured["description"] = description
        captured["timeout_seconds"] = timeout_seconds
        return True

    monkeypatch.setattr(td, "run_command", fake_run_command)
    ok = td.train_models(
        td.DATASET_REGISTRY["AV"],
        profile="max",
        tune=False,
        no_tune=False,
        ensemble=False,
        max_seeds=None,
        n_trials=None,
        top_pct=None,
        max_runtime_hours=None,
    )
    assert ok is True
    cmd = captured["cmd"]
    assert "--tune" in cmd
    assert "--ensemble" in cmd
    assert "--n-trials" in cmd and "140" in cmd
    assert "--max-seeds" in cmd and "4" in cmd
    assert "--top-pct" in cmd and "0.15" in cmd


def test_train_models_production_max_fast_profile_wires_fast_parallel_defaults(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_command(cmd: list[str], description: str, timeout_seconds=None) -> bool:
        captured["cmd"] = cmd
        captured["description"] = description
        captured["timeout_seconds"] = timeout_seconds
        return True

    monkeypatch.setattr(td, "run_command", fake_run_command)
    ok = td.train_models(
        td.DATASET_REGISTRY["DE"],
        profile="production-max-fast",
        tune=False,
        no_tune=False,
        ensemble=False,
        max_seeds=None,
        n_trials=None,
        top_pct=None,
        max_runtime_hours=None,
        target_metrics_file="reports/week2_metrics.json",
    )

    assert ok is True
    cmd = captured["cmd"]
    assert "--tune" in cmd
    assert "--ensemble" in cmd
    assert "--n-trials" in cmd and "64" in cmd
    assert cmd.count("--n-trials") == 1
    assert "--max-seeds" in cmd and "4" in cmd
    assert "--top-pct" in cmd and "0.2" in cmd
    assert "--tuning-n-jobs" in cmd and "3" in cmd
    assert "--gbm-threads" in cmd and "2" in cmd
    assert _flag_value(cmd, "--max-deep-epochs") == "16"
    assert _flag_value(cmd, "--deep-patience") == "4"
    assert _flag_value(cmd, "--deep-warmup-epochs") == "2"
    assert "--reuse-best-gbm-from" in cmd and "reports/week2_metrics.json" in cmd


def test_train_models_production_max_fast_qmax10_caps_deep_models(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_command(cmd: list[str], description: str, timeout_seconds=None) -> bool:
        captured["cmd"] = cmd
        captured["description"] = description
        captured["timeout_seconds"] = timeout_seconds
        return True

    monkeypatch.setattr(td, "run_command", fake_run_command)
    ok = td.train_models(
        td.DATASET_REGISTRY["DE"],
        profile="production-max-fast",
        tune=False,
        no_tune=False,
        ensemble=False,
        max_seeds=None,
        n_trials=10,
        top_pct=None,
        max_runtime_hours=None,
    )

    assert ok is True
    cmd = captured["cmd"]
    assert _flag_value(cmd, "--n-trials") == "10"
    assert _flag_value(cmd, "--max-deep-epochs") == "2"
    assert _flag_value(cmd, "--deep-patience") == "1"
    assert _flag_value(cmd, "--deep-warmup-epochs") == "1"


def test_av_training_config_matches_nuplan_registry_surface() -> None:
    cfg = td.DATASET_REGISTRY["AV"]
    train_cfg = td._load_training_cfg(cfg)
    target_cols = train_cfg["task"]["targets"]

    assert train_cfg["dataset"]["key"] == "AV"
    assert "nuPlan" in train_cfg["dataset"]["label"]
    assert train_cfg["data"]["processed_path"] == cfg.features_path
    assert cfg.features_path.endswith("processed_nuplan_allzip_grouped/anchor_features.parquet")
    assert train_cfg["data"]["order_cols"] == ["scenario_id", "step_index"]
    assert target_cols == [
        "target_ego_speed_mps__1s",
        "target_relative_gap_m__1s",
        "target_ego_speed_mps__2s",
        "target_relative_gap_m__2s",
        "target_ego_speed_mps__4s",
        "target_relative_gap_m__4s",
    ]


def test_train_models_production_max_fast_profile_wires_av_training_defaults(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_command(cmd: list[str], description: str, timeout_seconds=None) -> bool:
        captured["cmd"] = cmd
        captured["description"] = description
        captured["timeout_seconds"] = timeout_seconds
        return True

    monkeypatch.setattr(td, "run_command", fake_run_command)
    ok = td.train_models(
        td.DATASET_REGISTRY["AV"],
        profile="production-max-fast",
        tune=False,
        no_tune=False,
        ensemble=False,
        max_seeds=None,
        n_trials=None,
        top_pct=None,
        max_runtime_hours=None,
    )

    assert ok is True
    cmd = captured["cmd"]
    assert "configs/train_forecast_av.yaml" in cmd
    assert "--tune" in cmd
    assert "--ensemble" in cmd
    assert "--n-trials" in cmd and "48" in cmd
    assert "--max-seeds" in cmd and "3" in cmd
    assert "--top-pct" in cmd and "0.2" in cmd
    assert "--tuning-n-jobs" in cmd and "2" in cmd
    assert "--gbm-threads" in cmd and "2" in cmd
    assert _flag_value(cmd, "--max-deep-epochs") == "12"
    assert _flag_value(cmd, "--deep-patience") == "3"
    assert _flag_value(cmd, "--deep-warmup-epochs") == "1"
    assert "--reuse-best-gbm-from" not in cmd


def test_av_schema_validation_uses_order_columns_not_timestamp(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    def fake_run_command(cmd: list[str], description: str, timeout_seconds=None) -> bool:
        captured["cmd"] = cmd
        captured["description"] = description
        captured["timeout_seconds"] = timeout_seconds
        return True

    monkeypatch.setattr(td, "run_command", fake_run_command)
    ok = td.validate_features_schema(td.DATASET_REGISTRY["AV"], report_path=tmp_path / "av_schema.md")

    assert ok is True
    cmd = captured["cmd"]
    assert "--required-cols" in cmd
    assert (
        "target_ego_speed_mps__1s,target_relative_gap_m__1s,target_ego_speed_mps__2s,target_relative_gap_m__2s,target_ego_speed_mps__4s,target_relative_gap_m__4s"
        in cmd
    )
    assert "--order-cols" in cmd
    assert "scenario_id,step_index" in cmd
    assert "--timestamp-col" not in cmd


def test_preflight_sort_uses_order_columns_when_timestamp_is_absent() -> None:
    train_cfg = {"data": {"timestamp_col": None, "order_cols": ["scenario_id", "step_index"]}}
    frame = pd.DataFrame(
        {
            "scenario_id": ["b", "a", "a"],
            "step_index": [1, 2, 1],
            "target_ego_speed_mps__1s": [30.0, 20.0, 10.0],
        }
    )

    sorted_frame = td._sort_preflight_frame(train_cfg, frame)

    assert sorted_frame["target_ego_speed_mps__1s"].tolist() == [10.0, 20.0, 30.0]
