from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_phase_346_closure.py"
STATUS_JSON = REPO_ROOT / "reports" / "publication" / "phase_3_4_6_closure_status.json"


def test_phase_346_closure_status_is_emitted_and_fail_closed() -> None:
    run = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "[verify_phase_346_closure] INCOMPLETE" in run.stdout

    status = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
    assert status["overall_ready"] is False
    assert status["completion_claim_allowed"] is False
    assert status["phase_6_ready"] is False
    assert status["phase_6_blockers"]

    gated = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--require-complete"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert gated.returncode == 1
    assert "[verify_phase_346_closure] FAIL" in gated.stdout
