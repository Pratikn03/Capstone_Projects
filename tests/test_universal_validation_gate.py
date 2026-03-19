"""Regression tests for the canonical universal validation gate."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_universal_orius_validation.py"


def _load_validation_script():
    spec = importlib.util.spec_from_file_location("run_universal_orius_validation", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validation_script = _load_validation_script()


def test_evaluate_proof_candidate_rejects_trivial_baseline() -> None:
    summary = {
        "baseline_tsvr_mean": 0.3,
        "baseline_tsvr_std": 0.1,
        "orius_tsvr_mean": 0.0,
        "orius_tsvr_std": 0.0,
        "data_source": "locked_csv",
    }
    report = validation_script._evaluate_proof_candidate(summary)
    assert report["pass_gate"] is False
    assert "baseline_gap_too_small" in report["failure_reasons"]


def test_evaluate_proof_candidate_rejects_non_improving_orius() -> None:
    summary = {
        "baseline_tsvr_mean": 4.0,
        "baseline_tsvr_std": 0.5,
        "orius_tsvr_mean": 4.0,
        "orius_tsvr_std": 0.5,
        "data_source": "locked_csv",
    }
    report = validation_script._evaluate_proof_candidate(summary)
    assert report["pass_gate"] is False
    assert "orius_did_not_improve" in report["failure_reasons"]


def test_evaluate_proof_candidate_rejects_unstable_results() -> None:
    summary = {
        "baseline_tsvr_mean": 8.0,
        "baseline_tsvr_std": 9.5,
        "orius_tsvr_mean": 0.0,
        "orius_tsvr_std": 0.0,
        "data_source": "locked_csv",
    }
    report = validation_script._evaluate_proof_candidate(summary)
    assert report["pass_gate"] is False
    assert "seed_instability" in report["failure_reasons"]


def test_evaluate_proof_candidate_accepts_stable_improvement() -> None:
    summary = {
        "baseline_tsvr_mean": 12.0,
        "baseline_tsvr_std": 2.0,
        "orius_tsvr_mean": 0.0,
        "orius_tsvr_std": 0.0,
        "data_source": "locked_csv",
    }
    report = validation_script._evaluate_proof_candidate(summary)
    assert report["pass_gate"] is True
    assert report["failure_reasons"] == []


def test_validation_cli_reports_reference_and_promoted_domains(tmp_path: Path) -> None:
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--seeds",
            "3",
            "--horizon",
            "48",
            "--out",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Harness pass:  True" in run.stdout
    assert "Evidence pass: True" in run.stdout

    report = json.loads((tmp_path / "validation_report.json").read_text())
    proof_report = json.loads((tmp_path / "proof_domain_report.json").read_text())
    summary_tex = (tmp_path / "tbl_domain_validation_summary.tex").read_text()
    proof_status_tex = (tmp_path / "tbl_domain_proof_status.tex").read_text()

    assert report["reference_domain"] == "battery"
    assert report["validated_domains"] == ["battery", "industrial", "healthcare"]
    assert report["proof_validated_domains"] == ["industrial", "healthcare"]
    assert report["shadow_synthetic_domains"] == ["navigation"]
    assert report["experimental_domains"] == ["aerospace"]
    assert report["harness_pass"] is True
    assert report["evidence_pass"] is True

    domain_rows = {row["domain"]: row for row in report["domain_results"]}
    assert domain_rows["battery"]["validation_status"] == "reference_validated"
    assert domain_rows["industrial"]["validation_status"] == "proof_validated"
    assert domain_rows["healthcare"]["validation_status"] == "proof_validated"
    assert domain_rows["av"]["validation_status"] == "proof_candidate"
    assert domain_rows["navigation"]["validation_status"] == "shadow_synthetic"
    assert domain_rows["aerospace"]["validation_status"] == "experimental"
    assert domain_rows["av"]["proof_gate_reasons"] == "orius_did_not_improve"
    assert domain_rows["industrial"]["training_verified"] is True
    assert domain_rows["healthcare"]["training_verified"] is True
    assert domain_rows["av"]["training_verified"] is True
    assert domain_rows["industrial"]["sil_pass"] is True
    assert domain_rows["healthcare"]["sil_pass"] is True
    assert domain_rows["av"]["sil_pass"] is True

    assert proof_report["proof_validated_domains"] == ["industrial", "healthcare"]
    assert proof_report["proof_downgraded_domains"][0]["domain"] == "av"

    assert r"\label{tab:domain-validation-summary}" in summary_tex
    assert "proof\\_validated" in summary_tex
    assert "proof\\_candidate" in summary_tex
    assert "shadow\\_synthetic" in summary_tex
    assert r"\label{tab:domain-proof-status}" in proof_status_tex
