"""Tests retrain_if_needed activation when DC3S trigger reasons are present."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import scripts.retrain_if_needed as retrain_if_needed


def test_retrain_if_needed_runs_pipeline_on_dc3s_trigger(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "monitoring_summary.json").write_text(
        json.dumps(
            {
                "retraining": {
                    "retrain": True,
                    "reasons": ["dc3s_intervention_spike"],
                    "last_trained_days_ago": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_run(cmd, check):  # noqa: ANN001
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(retrain_if_needed.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["retrain_if_needed.py"])
    retrain_if_needed.main()

    assert calls, "Expected retraining pipeline command to run."
    assert calls[-1][0] == sys.executable
    assert calls[-1][1:5] == ["-m", "gridpulse.pipeline.run", "--steps", "train,reports"]


def test_retrain_if_needed_dry_run_reports_dc3s_decision(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "monitoring_summary.json").write_text(
        json.dumps({"retraining": {"retrain": True, "reasons": ["dc3s_reliability_degradation"]}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["retrain_if_needed.py", "--dry-run"])
    retrain_if_needed.main()

    out = capsys.readouterr().out
    assert "decision=True" in out
    assert "dc3s_reliability_degradation" in out
