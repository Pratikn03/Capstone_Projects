from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_SCRIPT = REPO_ROOT / "scripts" / "run_universal_orius_validation.py"
TRAINING_SCRIPT = REPO_ROOT / "scripts" / "run_universal_training_audit.py"
CLOSURE_SCRIPT = REPO_ROOT / "scripts" / "build_domain_closure_matrix.py"


def test_domain_closure_matrix_builds_with_bounded_p5_p6_surfaces(tmp_path: Path) -> None:
    validation_out = tmp_path / "validation"
    training_out = tmp_path / "training"
    validation_run = subprocess.run(
        [
            sys.executable,
            str(VALIDATION_SCRIPT),
            "--seeds",
            "1",
            "--horizon",
            "24",
            "--out",
            str(validation_out),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert validation_run.returncode in {0, 1}
    assert (validation_out / "validation_report.json").exists()
    training_run = subprocess.run(
        [
            sys.executable,
            str(TRAINING_SCRIPT),
            "--out",
            str(training_out),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert training_run.returncode == 0
    assert (training_out / "training_audit_report.json").exists()
    training_report = json.loads((training_out / "training_audit_report.json").read_text(encoding="utf-8"))
    assert training_report["failed_domains"] == []
    assert "battery" in training_report["training_surface_closed_domains"]
    subprocess.run(
        [
            sys.executable,
            str(CLOSURE_SCRIPT),
            "--validation-report",
            str(validation_out / "validation_report.json"),
            "--training-report",
            str(training_out / "training_audit_report.json"),
            "--out",
            str(validation_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    closure_rows = list(csv.DictReader((validation_out / "domain_closure_matrix.csv").open()))
    p5_rows = list(csv.DictReader((validation_out / "paper5_cross_domain_matrix.csv").open()))
    p6_rows = list(csv.DictReader((validation_out / "paper6_cross_domain_matrix.csv").open()))
    payload = json.loads((validation_out / "domain_closure_matrix.json").read_text())
    assert (validation_out / "tbl_domain_closure_matrix.tex").exists()
    assert (validation_out / "tbl_paper5_cross_domain_matrix.tex").exists()
    assert (validation_out / "tbl_paper6_cross_domain_matrix.tex").exists()

    closure_by_domain = {row["domain"]: row for row in closure_rows}
    assert set(closure_by_domain) == {"battery", "healthcare", "vehicle"}
    assert closure_by_domain["battery"]["resulting_tier"] == "reference"
    assert closure_by_domain["healthcare"]["resulting_tier"] == "proof_validated"
    assert closure_by_domain["vehicle"]["resulting_tier"] == "proof_validated"
    assert closure_by_domain["vehicle"]["safe_action_soundness_status"] == "pass"
    assert closure_by_domain["vehicle"]["training_surface_status"] == "closed"

    p5_by_domain = {row["domain"]: row for row in p5_rows}
    assert p5_by_domain["battery"]["status"] == "evaluated"
    assert p5_by_domain["healthcare"]["status"] == "evaluated"
    assert p5_by_domain["vehicle"]["status"] == "evaluated"

    p6_by_domain = {row["domain"]: row for row in p6_rows}
    assert p6_by_domain["battery"]["status"] == "evaluated"
    assert p6_by_domain["healthcare"]["status"] == "evaluated"
    assert p6_by_domain["vehicle"]["status"] == "evaluated"

    closure_tex = (validation_out / "tbl_domain_closure_matrix.tex").read_text(encoding="utf-8")
    assert r"\textbf{training surface status}" in closure_tex
    assert "proof_validated" not in closure_tex

    assert payload["vehicle_soundness_rows"]


def test_domain_closure_matrix_does_not_paint_failed_domains_green(tmp_path: Path) -> None:
    validation_out = tmp_path / "validation"
    validation_out.mkdir(parents=True, exist_ok=True)
    training_out = tmp_path / "training"
    training_out.mkdir(parents=True, exist_ok=True)

    (validation_out / "validation_report.json").write_text(
        json.dumps(
            {
                "all_passed": False,
                "validated_domains": ["battery", "healthcare"],
                "failed_domains": ["vehicle"],
                "domain_proof_reports": {
                    "healthcare": {"evidence_pass": True, "failure_reasons": []},
                    "vehicle": {
                        "evidence_pass": False,
                        "failure_reasons": ["dc3s_regression_on_tsvr"],
                    },
                },
                "domain_results": {
                    "battery": {
                        "baseline_tsvr_mean": 0.0083,
                        "orius_tsvr_mean": 0.0,
                        "orius_reduction_pct": 100.0,
                    },
                    "healthcare": {
                        "baseline_tsvr_mean": 0.2917,
                        "orius_tsvr_mean": 0.0417,
                        "orius_reduction_pct": 85.7,
                    },
                    "vehicle": {
                        "baseline_tsvr_mean": 0.1250,
                        "orius_tsvr_mean": 0.1200,
                        "orius_reduction_pct": 4.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (training_out / "training_audit_report.json").write_text(
        json.dumps({"training_surface_closed_domains": ["healthcare", "av"]}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(CLOSURE_SCRIPT),
            "--validation-report",
            str(validation_out / "validation_report.json"),
            "--training-report",
            str(training_out / "training_audit_report.json"),
            "--out",
            str(validation_out),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    closure_rows = {
        row["domain"]: row for row in csv.DictReader((validation_out / "domain_closure_matrix.csv").open())
    }
    p5_rows = {
        row["domain"]: row
        for row in csv.DictReader((validation_out / "paper5_cross_domain_matrix.csv").open())
    }
    p6_rows = {
        row["domain"]: row
        for row in csv.DictReader((validation_out / "paper6_cross_domain_matrix.csv").open())
    }

    assert closure_rows["healthcare"]["resulting_tier"] == "proof_validated"
    assert closure_rows["vehicle"]["resulting_tier"] == "proof_candidate"
    assert closure_rows["vehicle"]["safe_action_soundness_status"] == "fail"
    assert closure_rows["vehicle"]["exact_blocker"] == "dc3s_regression_on_tsvr"
    assert p5_rows["vehicle"]["status"] == "fail"
    assert p6_rows["vehicle"]["status"] == "fail"
