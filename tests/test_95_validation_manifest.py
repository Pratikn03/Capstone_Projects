from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.run_95_validation_training import build_95_validation_manifest
from scripts.validate_95_validation_manifest import validate_manifest


def test_95_manifest_allows_bounded_package_but_not_full_95_without_carla_eicu_physical_hil(
    tmp_path: Path,
) -> None:
    nuplan = tmp_path / "nuplan"
    carla = tmp_path / "carla"
    healthcare = tmp_path / "healthcare"
    battery = tmp_path / "battery"
    for path in (nuplan, carla, healthcare, battery):
        path.mkdir()
    pd.DataFrame([{"controller": "orius", "tsvr": 0.0}]).to_csv(nuplan / "runtime_summary.csv", index=False)
    (carla / "carla_closed_loop_manifest.json").write_text(
        json.dumps(
            {
                "status": "blocked_by_local_platform",
                "carla_completed": False,
                "claim_boundary": "not completed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"controller": "orius", "tsvr": 0.0}]).to_csv(
        healthcare / "heldout_runtime_summary.csv", index=False
    )
    (healthcare / "heldout_runtime_manifest.json").write_text(
        json.dumps(
            {
                "status": "completed_retrospective_heldout_replay",
                "eicu_status": "not_staged",
                "source_holdout": True,
                "time_forward": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (battery / "hil_summary.json").write_text(json.dumps({"total_violations": 0}) + "\n", encoding="utf-8")

    out = tmp_path / "manifest.json"
    manifest = build_95_validation_manifest(
        nuplan_runtime=nuplan,
        carla_dir=carla,
        healthcare_dir=healthcare,
        battery_hil_dir=battery,
        out=out,
    )

    assert manifest["bounded_package_passed"] is True
    assert manifest["full_95_external_validation_complete"] is False
    assert validate_manifest(out) == []


def test_95_manifest_validator_rejects_physical_hil_claim(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "full_95_external_validation_complete": True,
                "domains": {
                    "av_carla": {"completed": False, "status": "blocked_by_local_platform"},
                    "healthcare": {"eicu_status": "not_staged"},
                    "battery": {"evidence_type": "software_hil", "physical_hil_completed": True},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    findings = validate_manifest(path)
    assert any("Physical HIL" in finding for finding in findings)
    assert any("full_95_external_validation_complete" in finding for finding in findings)


def test_95_manifest_validator_rejects_carla_completion_without_trace_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    missing_manifest = tmp_path / "missing_carla_manifest.json"
    path.write_text(
        json.dumps(
            {
                "full_95_external_validation_complete": False,
                "domains": {
                    "av_carla": {
                        "completed": True,
                        "status": "completed_closed_loop",
                        "manifest": str(missing_manifest),
                    },
                    "healthcare": {"eicu_status": "not_staged"},
                    "battery": {"evidence_type": "software_hil", "physical_hil_completed": False},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    findings = validate_manifest(path)
    assert any("CARLA completed claim points to missing artifact" in finding for finding in findings)
    assert any(
        "CARLA completed claim is missing closed-loop summary/traces" in finding for finding in findings
    )
