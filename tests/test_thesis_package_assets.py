from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_core_three_domain_assets_exist() -> None:
    required = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "ORIUS_REPRODUCIBILITY.md",
        REPO_ROOT / "paper" / "README.md",
        REPO_ROOT / "paper" / "manifest.yaml",
        REPO_ROOT / "paper" / "orius_program_manifest.json",
        REPO_ROOT / "paper" / "review" / "orius_review_dossier.tex",
        REPO_ROOT / "paper" / "monograph" / "ch09_universal_orius_across_domains.tex",
        REPO_ROOT / "paper" / "monograph" / "ch13_governance_reproducibility_limitations_and_conclusion.tex",
        REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_submission_scorecard.csv",
        REPO_ROOT / "reports" / "publication" / "defended_theorem_core.json",
        REPO_ROOT / "reports" / "publication" / "defended_theorem_core.csv",
        REPO_ROOT / "reports" / "publication" / "defended_theorem_core.md",
        REPO_ROOT / "reports" / "publication" / "defended_assumption_map.csv",
        REPO_ROOT / "reports" / "publication" / "defended_assumption_map.md",
        REPO_ROOT / "reports" / "publication" / "active_theorem_audit.json",
        REPO_ROOT / "reports" / "publication" / "external_proof_audit_packet.md",
        REPO_ROOT / "reports" / "publication" / "external_proof_audit_findings.csv",
        REPO_ROOT / "reports" / "battery_av_healthcare" / "overall" / "release_summary.json",
        REPO_ROOT / "reports" / "battery_av_healthcare" / "overall" / "publication_closure_override.json",
        REPO_ROOT / "scripts" / "build_orius_monograph_assets.py",
        REPO_ROOT / "scripts" / "build_active_theorem_audit.py",
        REPO_ROOT / "scripts" / "build_missing_tables.py",
        REPO_ROOT / "scripts" / "verify_phase_346_closure.py",
        REPO_ROOT / "scripts" / "refresh_real_data_manifests.py",
        REPO_ROOT / "scripts" / "verify_real_data_preflight.py",
        REPO_ROOT / "scripts" / "run_orius_canonical_closure_refresh.py",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    assert missing == []


def test_removed_domains_and_old_gate_absent_from_core_truth_surfaces() -> None:
    forbidden = ("industrial", "navigation", "aerospace", "equal_domain_93", "orius_equal_domain_parity_matrix")
    surfaces = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "ORIUS_REPRODUCIBILITY.md",
        REPO_ROOT / "paper" / "README.md",
        REPO_ROOT / "paper" / "manifest.yaml",
        REPO_ROOT / "paper" / "orius_program_manifest.json",
        REPO_ROOT / "paper" / "review" / "orius_review_dossier.tex",
        REPO_ROOT / "paper" / "monograph" / "ch09_universal_orius_across_domains.tex",
        REPO_ROOT / "paper" / "monograph" / "ch13_governance_reproducibility_limitations_and_conclusion.tex",
        REPO_ROOT / "reports" / "publication" / "README.md",
        REPO_ROOT / "reports" / "publication" / "tbl_orius_calibration_diagnostics.tex",
        REPO_ROOT / "reports" / "publication" / "tbl_orius_governance_lifecycle_matrix.tex",
        REPO_ROOT / "reports" / "publication" / "orius_review_global_gap_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_reviewer_scorecards.csv",
        REPO_ROOT / "reports" / "publication" / "orius_deployment_validation_scope.csv",
        REPO_ROOT / "reports" / "publication" / "tbl_orius_deployment_validation_scope.tex",
        REPO_ROOT / "reports" / "publication" / "orius_transfer_obligation_table.csv",
        REPO_ROOT / "reports" / "publication" / "github_issue_specs.csv",
        REPO_ROOT / "scripts" / "hf_jobs" / "README.md",
        REPO_ROOT / "Makefile",
    ]
    for path in surfaces:
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, f"{token} leaked in {path.relative_to(REPO_ROOT)}"


def test_scorecard_and_closure_matrix_are_literal_three_domain_surfaces() -> None:
    scorecard_rows = list(csv.DictReader((REPO_ROOT / "reports" / "publication" / "orius_submission_scorecard.csv").open()))
    closure_rows = list(csv.DictReader((REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv").open()))

    assert [row["target_tier"] for row in scorecard_rows] == ["three_domain_93_candidate"]
    assert {row["domain"] for row in closure_rows} == {
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    }


def test_defended_theorem_core_is_strict_and_bounded() -> None:
    defended_core = json.loads((REPO_ROOT / "reports" / "publication" / "defended_theorem_core.json").read_text(encoding="utf-8"))
    summary = defended_core["summary"]
    rows = defended_core["rows"]

    assert summary["flagship_defended_ids"] == ["T1", "T2", "T3a", "T4", "T6", "T7", "T11", "T_trajectory_PAC"]
    assert summary["supporting_defended_ids"] == [
        "T3b",
        "T8",
        "T10_T11_ObservationAmbiguitySandwich",
        "T11_AV_BrakeHold",
        "T11_HC_FailSafeRelease",
        "T6_AV_FallbackValidity",
        "T6_HC_FallbackValidity",
        "T_EQ_Battery_RuntimeArtifactPackage",
        "T_EQ_AV_RuntimeArtifactPackage",
        "T_EQ_HC_RuntimeArtifactPackage",
        "T11_Byzantine",
        "T_stale_decay",
    ]
    assert summary["flagship_gate_ready"] is True
    assert all(row["scope_note"] for row in rows)
    assert all(
        row["rigor_rating"] not in {"broken", "has-a-hole"}
        for row in rows
        if row["defense_tier"] == "flagship_defended"
    )


def test_canonical_defense_surfaces_use_current_paths_and_strict_core_language() -> None:
    surfaces = [
        REPO_ROOT / "appendices" / "app_s_claim_evidence_registers.tex",
        REPO_ROOT / "appendices" / "app_n_defense_lock_templates.tex",
        REPO_ROOT / "appendices" / "app_z_theorem_and_paper_sync_registers.tex",
        REPO_ROOT / "chapters" / "ch01_introduction.tex",
        REPO_ROOT / "chapters" / "ch15_assumptions_notation_proof_discipline.tex",
        REPO_ROOT / "reports" / "publication" / "claim_evidence_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "battery_claim_evidence_register.csv",
        REPO_ROOT / "reports" / "publication" / "chapter_theorem_traceability.csv",
        REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "tbl_claim_evidence.tex",
    ]
    for path in surfaces:
        text = path.read_text(encoding="utf-8")
        assert "src/gridpulse/" not in text, f"legacy path leaked in {path.relative_to(REPO_ROOT)}"

    introduction = (REPO_ROOT / "chapters" / "ch01_introduction.tex").read_text(encoding="utf-8")
    assert "Flagship defended." in introduction
    assert "Draft / non-defended extensions." in introduction

    appendix_s = (REPO_ROOT / "appendices" / "app_s_claim_evidence_registers.tex").read_text(encoding="utf-8")
    assert "registry-canonical defended core" in appendix_s
    assert "flagship defended" in appendix_s


def test_three_domain_release_summary_is_sanitized_and_mimic_backed() -> None:
    release_summary = json.loads(
        (REPO_ROOT / "reports" / "battery_av_healthcare" / "overall" / "release_summary.json").read_text(encoding="utf-8")
    )
    override = json.loads(
        (REPO_ROOT / "reports" / "battery_av_healthcare" / "overall" / "publication_closure_override.json").read_text(
            encoding="utf-8"
        )
    )

    text = json.dumps(release_summary)
    assert "/Users/" not in text
    assert "battery_av_only" not in text
    assert release_summary["submission_scope"] == "battery_av_healthcare"
    assert "mimic3_manifest.json" in text
    assert override["vehicle"]["resulting_tier"] == "runtime_contract_closed"
    assert override["healthcare"]["resulting_tier"] == "runtime_contract_closed"


def test_promoted_healthcare_surfaces_are_mimic_only() -> None:
    surfaces = [
        REPO_ROOT / "data" / "DATASET_DOWNLOAD_GUIDE.md",
        REPO_ROOT / "reports" / "publication" / "orius_deployment_validation_scope.csv",
        REPO_ROOT / "paper" / "review" / "generated" / "review_gap_analysis.tex",
    ]
    for path in surfaces:
        text = path.read_text(encoding="utf-8").lower()
        assert "mimic" in text
        assert "bidmc-canonical" not in text
        assert "bidmc canonical" not in text


def test_makefile_no_longer_exposes_removed_domain_targets() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "equal-domain-gate" not in makefile
    assert "navigation-datasets" not in makefile
    assert "industrial-datasets" not in makefile
    assert "aerospace-datasets" not in makefile


def test_canonical_refresh_help_is_three_domain_only() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_orius_canonical_closure_refresh.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout
    assert "three_domain_lane" in stdout
    assert "equal_domain_gate" not in stdout
