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
    subprocess.run(
        [PYTHON, "scripts/build_orius_monograph_assets.py"],
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
