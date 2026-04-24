"""Tests for prepared-but-not-promoted next-tier validation manifests."""
from __future__ import annotations

from pathlib import Path

from scripts.build_next_tier_validation_preparation import build_next_tier_validation_preparation


def test_next_tier_validation_manifests_are_non_claims(tmp_path: Path) -> None:
    summary = build_next_tier_validation_preparation(out_dir=tmp_path)

    assert summary["status"] == "prepared_not_completed"
    assert summary["completed"]["av_nuplan_carla"] is False
    assert summary["completed"]["healthcare_heldout_runtime"] is False

    av_text = (tmp_path / "nuplan_carla_preparation_manifest.json").read_text(encoding="utf-8").lower()
    hc_text = (tmp_path / "healthcare_heldout_runtime_preparation_manifest.json").read_text(encoding="utf-8").lower()

    assert "prepared_not_completed" in av_text
    assert "not claim completed nuplan" in av_text or "does not claim completed nuplan" in av_text
    assert "not claim completed carla" in av_text or "does not claim completed carla" in av_text
    assert "prepared_not_completed" in hc_text
    assert "not claim live clinical" in hc_text or "does not claim live clinical" in hc_text
    assert "prospective trial evidence" in hc_text
