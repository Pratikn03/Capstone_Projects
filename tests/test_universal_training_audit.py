"""Regression tests for the non-battery training audit."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_universal_training_audit.py"


def test_universal_training_audit_reports_verified_domains(tmp_path: Path) -> None:
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--out",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Universal Training Audit" in run.stdout

    report = json.loads((tmp_path / "training_audit_report.json").read_text())
    assert report["all_passed"] is True
    assert report["training_verified_domains"] == ["av", "industrial", "healthcare", "aerospace"]

    summary_csv = (tmp_path / "domain_training_summary.csv").read_text()
    assert "industrial" in summary_csv
    assert "healthcare" in summary_csv
    assert "aerospace" in summary_csv

