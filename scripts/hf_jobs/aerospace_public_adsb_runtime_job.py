# /// script
# dependencies = ["pandas", "numpy", "huggingface_hub"]
# ///
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(os.environ.get("GRIDPULSE_REPO_ROOT", ".")).resolve()
EXTERNAL_ROOT = os.environ.get("ORIUS_EXTERNAL_DATA_ROOT")
PYTHON = sys.executable


def run(*args: str) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def main() -> None:
    build_args = [
        PYTHON,
        "scripts/build_aerospace_public_adsb_runtime.py",
        "--download",
    ]
    if EXTERNAL_ROOT:
        build_args.extend(["--external-root", str(Path(EXTERNAL_ROOT).resolve())])
    run(*build_args)
    run(PYTHON, "scripts/build_orius_monograph_assets.py")


if __name__ == "__main__":
    main()
