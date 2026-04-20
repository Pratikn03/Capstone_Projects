from __future__ import annotations

import csv
import json
from pathlib import Path
import tomllib

from scripts._dataset_registry import DATASET_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_healthcare_registry_uses_promoted_mimic_paths() -> None:
    healthcare = DATASET_REGISTRY["HEALTHCARE"]
    assert healthcare.raw_data_path == "data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv"
    assert healthcare.canonical_runtime_path == "data/healthcare/mimic3/processed/mimic3_healthcare_orius.csv"
    assert healthcare.runtime_provenance_path == "data/healthcare/mimic3/processed/mimic3_manifest.json"


def test_three_domain_scorecard_row_is_green() -> None:
    rows = {
        row["target_tier"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "publication" / "orius_submission_scorecard.csv").open())
    }
    row = rows["three_domain_93_candidate"]
    assert row["meets_93_gate"] == "True"
    assert row["critical_gap_count"] == "0"
    assert row["high_gap_count"] == "0"


def test_promoted_lane_artifacts_are_sanitized_and_healthcare_promoted() -> None:
    release_summary = (REPO_ROOT / "reports" / "battery_av_healthcare" / "overall" / "release_summary.json").read_text(
        encoding="utf-8"
    )
    override = json.loads(
        (REPO_ROOT / "reports" / "battery_av_healthcare" / "overall" / "publication_closure_override.json").read_text(
            encoding="utf-8"
        )
    )
    closure_rows = {
        row["domain"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv").open())
    }

    assert "/Users/" not in release_summary
    assert "battery_av_only" not in release_summary
    assert override["healthcare"]["resulting_tier"] == "proof_validated"
    assert "mimic3_manifest.json" in release_summary
    assert closure_rows["Medical and Healthcare Monitoring"]["tier"] == "proof_validated"
    assert "promoted source=MIMIC" in closure_rows["Medical and Healthcare Monitoring"]["current_status"]
    assert "Industrial Process Control" not in closure_rows


def test_validation_harness_proxy_surfaces_do_not_present_runtime_denominators() -> None:
    closure_rows = list(csv.DictReader((REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv").open()))
    evidence_rows = {
        row["domain"]: row
        for row in csv.DictReader((REPO_ROOT / "reports" / "publication" / "chapters40_44_domain_evidence_register.csv").open())
    }

    by_domain = {row["domain"]: row for row in closure_rows}
    for domain in ("Autonomous Vehicles", "Medical and Healthcare Monitoring"):
        status = by_domain[domain]["current_status"]
        assert "validation_harness proxy" in status
        assert "not a runtime-row denominator" in status

        evidence = evidence_rows[domain]
        assert evidence["metric_basis"].endswith("validation_harness proxy")
        assert "not" in evidence["evidence_volume_note"].lower()
        assert "denominator" in evidence["evidence_volume_note"].lower()


def test_readme_reflects_three_domain_submission_lane() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "battery_av_healthcare" in readme
    assert "mimic3_healthcare_orius.csv" in readme
    assert "equal_domain_93" not in readme


def test_full_check_testclient_dependency_is_locked() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(pyproject["project"]["dependencies"])
    lock_lines = set((REPO_ROOT / "requirements.lock.txt").read_text(encoding="utf-8").splitlines())

    assert "httpx==0.28.1" in dependencies
    assert "httpx==0.28.1" in lock_lines
    assert "httpcore==1.0.9" in lock_lines
