"""Regression tests for the universal software-in-loop validator."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_universal_sil_validation.py"


def test_universal_sil_validation_writes_summary_and_traces(tmp_path: Path) -> None:
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--seeds",
            "1",
            "--rows",
            "24",
            "--out",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Universal SIL Validation" in run.stdout

    report = json.loads((tmp_path / "sil_validation_report.json").read_text())
    assert report["all_passed"] is True
    assert "av" in report["sil_pass_domains"]
    assert "industrial" in report["sil_pass_domains"]

    summary_csv = (tmp_path / "domain_sil_summary.csv").read_text()
    assert "healthcare" in summary_csv
    assert "navigation" in summary_csv
    assert (tmp_path / "traces" / "av_seed0.csv").exists()

