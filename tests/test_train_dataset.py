"""Tests for scripts/train_dataset.py command wiring."""
from __future__ import annotations

from pathlib import Path

import scripts.train_dataset as td


def test_generate_reports_uses_build_reports_expected_args_de(monkeypatch) -> None:
    captured = {}

    def fake_run_command(cmd: list[str], description: str) -> bool:
        captured["cmd"] = cmd
        captured["description"] = description
        return True

    monkeypatch.setattr(td, "run_command", fake_run_command)
    ok = td.generate_reports(td.DATASET_REGISTRY["DE"])
    assert ok is True

    cmd = captured["cmd"]
    assert Path(cmd[0]).name.startswith("python")
    assert cmd[1] == "scripts/build_reports.py"
    assert "--features" in cmd and "data/processed/features.parquet" in cmd
    assert "--splits" in cmd and "data/processed/splits" in cmd
    assert "--models-dir" in cmd and "artifacts/models" in cmd
    assert "--reports-dir" in cmd and "reports" in cmd
    assert "--config" not in cmd


def test_generate_reports_uses_build_reports_expected_args_us(monkeypatch) -> None:
    captured = {}

    def fake_run_command(cmd: list[str], description: str) -> bool:
        captured["cmd"] = cmd
        captured["description"] = description
        return True

    monkeypatch.setattr(td, "run_command", fake_run_command)
    ok = td.generate_reports(td.DATASET_REGISTRY["US"])
    assert ok is True

    cmd = captured["cmd"]
    assert Path(cmd[0]).name.startswith("python")
    assert cmd[1] == "scripts/build_reports.py"
    assert "--features" in cmd and "data/processed/us_eia930/features.parquet" in cmd
    assert "--splits" in cmd and "data/processed/us_eia930/splits" in cmd
    assert "--models-dir" in cmd and "artifacts/models_eia930" in cmd
    assert "--reports-dir" in cmd and "reports/eia930" in cmd
    assert "--current-dataset" not in cmd
    assert "--config" not in cmd


def test_train_models_candidate_run_passes_isolated_output_overrides(monkeypatch) -> None:
    captured = {}

    def fake_run_command(cmd: list[str], description: str, timeout_seconds=None) -> bool:  # noqa: ANN001
        captured["cmd"] = cmd
        captured["description"] = description
        captured["timeout_seconds"] = timeout_seconds
        return True

    monkeypatch.setattr(td, "run_command", fake_run_command)
    layout = td._resolve_run_layout(td.DATASET_REGISTRY["DE"], candidate_run=True, run_id="r1-stack")
    ok = td.train_models(td.DATASET_REGISTRY["DE"], run_layout=layout)

    assert ok is True
    cmd = captured["cmd"]
    assert "--artifacts-dir" in cmd and str(layout.models_dir) in cmd
    assert "--reports-dir" in cmd and str(layout.reports_dir) in cmd
    assert "--uncertainty-artifacts-dir" in cmd and str(layout.uncertainty_dir) in cmd
    assert "--backtests-dir" in cmd and str(layout.backtests_dir) in cmd
    assert "--walk-forward-report" in cmd and str(layout.walk_forward_report) in cmd
    assert "--validation-report" in cmd and str(layout.validation_report) in cmd
    assert "--data-manifest-output" in cmd and str(layout.data_manifest_output) in cmd


def test_generate_reports_candidate_run_uses_local_publication_paths(monkeypatch) -> None:
    captured = {}

    def fake_run_command(cmd: list[str], description: str) -> bool:
        captured["cmd"] = cmd
        captured["description"] = description
        return True

    monkeypatch.setattr(td, "run_command", fake_run_command)
    layout = td._resolve_run_layout(td.DATASET_REGISTRY["US"], candidate_run=True, run_id="stage1")
    ok = td.generate_reports(td.DATASET_REGISTRY["US"], run_layout=layout)

    assert ok is True
    cmd = captured["cmd"]
    assert "--publication-dir" in cmd and str(layout.publication_dir) in cmd
    assert "--uncertainty-artifacts-dir" in cmd and str(layout.uncertainty_dir) in cmd
    assert "--backtests-dir" in cmd and str(layout.backtests_dir) in cmd
    assert "--current-dataset" in cmd and "US_MISO" in cmd


def test_dataset_registry_keeps_us_alias_backwards_compatible() -> None:
    alias_cfg = td.DATASET_REGISTRY["US"]
    canonical_cfg = td.DATASET_REGISTRY["US_MISO"]

    assert alias_cfg.alias_of == "US_MISO"
    assert alias_cfg.name == canonical_cfg.name == "US_MISO"
    assert alias_cfg.features_path == canonical_cfg.features_path
    assert alias_cfg.reports_dir == canonical_cfg.reports_dir


def test_ercot_dataset_uses_eia930_authority_code() -> None:
    assert td.DATASET_REGISTRY["US_ERCOT"].ba_code == "ERCO"


def test_dataset_registry_includes_navigation_trainable_row() -> None:
    cfg = td.DATASET_REGISTRY["NAVIGATION"]
    assert cfg.name == "NAVIGATION"
    assert cfg.features_path == "data/navigation/processed/features.parquet"
    assert cfg.reports_dir == "reports/navigation"


def test_configured_model_types_include_new_conference_baselines() -> None:
    cfg = {
        "models": {
            "gbm_lightgbm": {"enabled": True},
            "dl_lstm": {"enabled": False},
            "dl_tcn": {"enabled": False},
            "dl_nbeats": {"enabled": True},
            "dl_tft": {"enabled": True},
            "dl_patchtst": {"enabled": True},
        }
    }

    assert td._configured_model_types(cfg) == ["gbm_lightgbm", "nbeats", "tft", "patchtst"]
