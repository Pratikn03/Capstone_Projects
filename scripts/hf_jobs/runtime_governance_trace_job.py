# /// script
# dependencies = []
# ///
from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(os.environ.get("GRIDPULSE_REPO_ROOT", ".")).resolve()


def main() -> None:
    subprocess.run(
        ["python", "scripts/run_universal_orius_validation.py", "--seeds", "1", "--horizon", "24"],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        ["python", "scripts/build_orius_monograph_assets.py"],
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
