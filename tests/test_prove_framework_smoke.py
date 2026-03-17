"""Smoke test for scripts/prove_framework.py checks."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "prove_framework.py"


@pytest.mark.skipif(not SCRIPT.exists(), reason="prove_framework.py not found")
def test_prove_framework_runs():
    """Run prove_framework.py and verify it exits cleanly."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(SCRIPT.parent.parent),
    )
    assert result.returncode == 0, (
        f"prove_framework.py failed with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "PASS" in result.stdout
