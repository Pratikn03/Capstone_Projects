"""Tests for aggressive training profile command wiring."""
from __future__ import annotations

import scripts.train_dataset as td


def test_train_models_aggressive_profile_adds_expected_flags(monkeypatch) -> None:
    captured: dict = {}

    def fake_run_command(cmd: list[str], description: str, timeout_seconds=None) -> bool:  # noqa: ANN001
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
