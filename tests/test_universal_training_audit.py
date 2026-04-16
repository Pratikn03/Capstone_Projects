"""Regression tests for the canonical-domain training audit."""
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
        check=False,
        capture_output=True,
        text=True,
    )
    assert "Universal Training Audit" in run.stdout
    assert run.returncode == 1

    report = json.loads((tmp_path / "training_audit_report.json").read_text())
    assert report["domains"] == ["battery", "av", "industrial", "healthcare", "navigation", "aerospace"]
    assert report["all_passed"] is False
    assert "battery" in report["failed_domains"]
    assert "navigation" in report["failed_domains"]
    assert "navigation" in report["real_data_gap_domains"]
    assert "real_data_backed_domains" in report
    assert "training_surface_closed_domains" in report

    summary_csv = (tmp_path / "domain_training_summary.csv").read_text()
    assert "battery" in summary_csv
    assert "navigation" in summary_csv
    assert "industrial" in summary_csv
    assert "healthcare" in summary_csv
    assert "aerospace" in summary_csv
