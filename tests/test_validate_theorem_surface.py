from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_validate_theorem_surface_script_passes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "validate_theorem_surface.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
