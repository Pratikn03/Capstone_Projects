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
    subprocess.run(
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
        check=True,
        capture_output=True,
        text=True,
    )
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
    assert training_run.returncode in {0, 1}
    assert (training_out / "training_audit_report.json").exists()
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
    assert closure_by_domain["battery"]["resulting_tier"] == "reference"
    assert closure_by_domain["industrial"]["resulting_tier"] == "proof_validated"
    assert closure_by_domain["healthcare"]["resulting_tier"] == "proof_validated"
    assert closure_by_domain["navigation"]["exact_blocker"] == "navigation_real_data_gap"
    assert closure_by_domain["aerospace"]["exact_blocker"] == "aerospace_experimental_placeholder"
    assert closure_by_domain["vehicle"]["safe_action_soundness_status"] in {"pass", "fail"}

    p5_by_domain = {row["domain"]: row for row in p5_rows}
    assert p5_by_domain["battery"]["status"] == "evaluated"
    assert p5_by_domain["industrial"]["status"] == "evaluated"
    assert p5_by_domain["vehicle"]["status"] == "gated"

    p6_by_domain = {row["domain"]: row for row in p6_rows}
    assert p6_by_domain["battery"]["status"] == "evaluated"
    assert p6_by_domain["industrial"]["status"] == "evaluated"
    assert p6_by_domain["healthcare"]["status"] == "evaluated"
    assert p6_by_domain["vehicle"]["status"] == "evaluated"
    assert p6_by_domain["navigation"]["status"] == "gated"

    assert payload["vehicle_soundness_rows"]
