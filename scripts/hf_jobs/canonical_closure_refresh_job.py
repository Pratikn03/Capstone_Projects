# /// script
# dependencies = []
# ///
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(os.environ.get("GRIDPULSE_REPO_ROOT", ".")).resolve()
PYTHON = sys.executable


def main() -> None:
    cmd = [
        PYTHON,
        "scripts/run_orius_canonical_closure_refresh.py",
        "--mode",
        "canonical_plus_hf_support",
    ]
    external_root = os.environ.get("ORIUS_EXTERNAL_DATA_ROOT")
    if external_root:
        cmd.extend(["--external-root", str(Path(external_root).resolve())])
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
