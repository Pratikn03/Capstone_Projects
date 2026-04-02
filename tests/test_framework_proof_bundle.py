"""Tests for the ORIUS framework proof bundle builder."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_orius_framework_proof.py"


def test_framework_proof_bundle_builds_expected_outputs(tmp_path: Path) -> None:
    for name in ("training_audit", "sil_validation", "universal_validation"):
        shutil.copytree(REPO_ROOT / "reports" / "orius_framework_proof" / name, tmp_path / name)

    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--reuse-existing-artifacts",
            "--seeds",
            "1",
            "--horizon",
            "24",
            "--out",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "ORIUS Framework Proof Bundle" in run.stdout

    manifest = json.loads((tmp_path / "framework_proof_manifest.json").read_text())
    assert manifest["reference_domain"] == "battery"
    proof_domains = manifest["proof_validated_domains"]
    assert "industrial" in proof_domains
    assert "healthcare" in proof_domains
    assert "vehicle" in proof_domains
    assert manifest["proof_candidate_domains"] == []
    assert manifest["harness_pass"] is True
    assert manifest["evidence_pass"] is True
    assert manifest["integrated_theorem_gate_pass"] is True
    assert manifest["training_audit_pass"] is False
    assert manifest["sil_audit_pass"] is True

    summary_md = (tmp_path / "framework_proof_summary.md").read_text()
    assert "Proof-validated domains" in summary_md
    assert "industrial" in summary_md
    assert "vehicle" in summary_md
    assert "Training audit" in summary_md
    assert "SIL audit" in summary_md

    artifact_register = (tmp_path / "artifact_register.csv").read_text()
    assert "battery" in artifact_register
    assert "industrial" in artifact_register

    controller_summary = (tmp_path / "domain_controller_summary.csv").read_text()
    assert "battery" in controller_summary
    assert "industrial" in controller_summary
