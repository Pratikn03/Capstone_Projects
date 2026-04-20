from __future__ import annotations

from pathlib import Path

import scripts.train_dataset as train_dataset
from scripts._dataset_registry import DATASET_REGISTRY


def test_healthcare_bridge_build_uses_max_input(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_command(cmd: list[str], description: str, timeout_seconds: int = 3600) -> bool:
        captured["cmd"] = cmd
        captured["description"] = description
        captured["timeout_seconds"] = timeout_seconds
        return True

    monkeypatch.setattr(train_dataset, "run_command", fake_run_command)

    cfg = DATASET_REGISTRY["HEALTHCARE"]
    features_path = tmp_path / "features.parquet"
    monkeypatch.setattr(cfg, "features_path", str(features_path))

    bidmc_bridge = tmp_path / "healthcare_bidmc_orius.csv"
    mimic_bridge = tmp_path / "mimic3_healthcare_orius.csv"
    bidmc_bridge.write_text("timestamp,target,hr,resp,patient_id\np1_t0,96,70,14,p1\n", encoding="utf-8")
    mimic_bridge.write_text("timestamp,target,hr,resp,patient_id\np2_t0,95,75,16,p2\n", encoding="utf-8")

    def _fake_path(value: str) -> Path:
        if value.endswith("healthcare_bidmc_orius.csv"):
            return bidmc_bridge
        if value.endswith("mimic3_healthcare_orius.csv"):
            return mimic_bridge
        return Path(value)

    monkeypatch.setattr(train_dataset, "Path", _fake_path)

    assert train_dataset.build_features(cfg, force=False) is True
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert "--max-input" in cmd
    assert str(bidmc_bridge) in cmd
    assert str(mimic_bridge) in cmd
