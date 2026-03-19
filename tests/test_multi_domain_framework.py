"""Regression tests for the multi-domain framework demo runner."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_multi_domain_framework.py"


def test_multi_domain_framework_cli_includes_navigation(tmp_path: Path) -> None:
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
    assert "Navigation" in run.stdout

    report = json.loads((tmp_path / "multi_domain_report.json").read_text())
    assert report["domains_total"] == 6
    assert report["domains_run"] == 6

    rows = {row["domain_id"]: row for row in report["results"]}
    assert rows["navigation"]["status"] == "ok"
    assert "ax" in rows["navigation"]["safe_action"]
    assert "ay" in rows["navigation"]["safe_action"]
