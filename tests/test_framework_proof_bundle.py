"""Tests for the ORIUS framework proof bundle builder."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_orius_framework_proof.py"


def test_framework_proof_bundle_builds_expected_outputs(tmp_path: Path) -> None:
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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
    assert manifest["proof_validated_domains"] == ["industrial", "healthcare"]
    assert manifest["shadow_synthetic_domains"] == ["navigation"]
    assert manifest["harness_pass"] is True
    assert manifest["evidence_pass"] is True
    assert manifest["integrated_theorem_gate_pass"] is True
    assert manifest["training_audit_pass"] is True
    assert manifest["sil_audit_pass"] is True

    summary_md = (tmp_path / "framework_proof_summary.md").read_text()
    assert "Proof-validated domains" in summary_md
    assert "`industrial, healthcare`" in summary_md
    assert "Training audit" in summary_md
    assert "SIL audit" in summary_md

    artifact_register = (tmp_path / "artifact_register.csv").read_text()
    assert "battery" in artifact_register
    assert "industrial" in artifact_register
    assert "navigation" in artifact_register

    controller_summary = (tmp_path / "domain_controller_summary.csv").read_text()
    assert "battery,locked_artifact,reference" in controller_summary
    assert "industrial,locked_csv,proof_validated" in controller_summary
