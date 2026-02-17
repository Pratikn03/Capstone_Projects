"""Tests for scripts/train_dataset.py command wiring."""
from __future__ import annotations

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
    assert cmd[0:2] == ["python", "scripts/build_reports.py"]
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
    assert cmd[0:2] == ["python", "scripts/build_reports.py"]
    assert "--features" in cmd and "data/processed/us_eia930/features.parquet" in cmd
    assert "--splits" in cmd and "data/processed/us_eia930/splits" in cmd
    assert "--models-dir" in cmd and "artifacts/models_eia930" in cmd
    assert "--reports-dir" in cmd and "reports/eia930" in cmd
    assert "--config" not in cmd
