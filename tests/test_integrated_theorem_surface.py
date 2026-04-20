"""Regression test for the integrated 18-theorem release gate."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_integrated_theorem_surface.py"


def test_integrated_theorem_surface_gate_passes() -> None:
    run = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "PASS: 18/18 theorem rows traceability-locked" in run.stdout

    summary = json.loads((REPO_ROOT / "reports" / "publication" / "integrated_theorem_gate.json").read_text())
    assert summary["gate_kind"] == "traceability_release_gate"
    assert summary["total"] == 18
    assert summary["failed"] == 0
    assert summary["passed"] == 18
