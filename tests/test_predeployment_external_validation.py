from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.run_predeployment_external_validation import build_predeployment_external_validation


def test_predeployment_external_validation_requires_three_external_surfaces(tmp_path: Path) -> None:
    report = build_predeployment_external_validation(out_dir=tmp_path)

    assert report["all_passed"] is True
    rows = {row["domain"]: row for row in report["domains"]}
    assert set(rows) == {
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    }

    assert rows["Battery Energy Storage"]["validation_surface"] == "battery_hil_or_simulator"
    assert rows["Battery Energy Storage"]["safety_violations"] == 0
    assert rows["Battery Energy Storage"]["certificate_valid_rate"] == 1.0

    av = rows["Autonomous Vehicles"]
    assert av["validation_surface"] == "nuplan_allzip_grouped_runtime_replay_surrogate"
    assert av["orius_tsvr"] <= 1e-3
    assert av["fallback_or_intervention_rate"] <= 0.50
    assert av["certificate_valid_rate"] >= 0.999
    assert "not completed CARLA simulation" in av["claim_boundary"]
    assert "road deployment" in av["claim_boundary"]

    healthcare = rows["Medical and Healthcare Monitoring"]
    assert healthcare["validation_surface"] == "healthcare_retrospective_time_forward_source_holdout"
    assert healthcare["orius_tsvr"] == 0.0
    assert healthcare["fallback_or_intervention_rate"] <= 0.50
    assert healthcare["patient_disjoint"] is True
    assert healthcare["time_forward"] is True
    assert healthcare["site_holdout"] is True
    assert healthcare["development_source"] != healthcare["holdout_source"]
    assert "not live clinical deployment" in healthcare["claim_boundary"]


def test_predeployment_healthcare_site_split_manifest_is_source_holdout(tmp_path: Path) -> None:
    report = build_predeployment_external_validation(out_dir=tmp_path)
    manifest_path = tmp_path / "healthcare_site_splits" / "manifest.json"
    details_path = tmp_path / "healthcare_retrospective_holdout_split_details.csv"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    details = pd.read_csv(details_path)
    split_sources = {
        row["split"]: set(str(row["sources"]).split("|"))
        for _, row in details.iterrows()
        if not str(row["split"]).startswith("source:")
    }

    assert manifest["split_strategy"] == "development_site_patient_blocks_plus_later_source_holdout"
    assert set(manifest["source_datasets"]) >= {"bidmc", "mimic3"}
    assert split_sources["test"].isdisjoint(
        split_sources["train"] | split_sources["calibration"] | split_sources["val"]
    )
    assert all(Path(report["detail_artifacts"][name]).exists() for name in ("battery", "av", "healthcare"))
