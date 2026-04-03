# /// script
# dependencies = ["pandas", "pyarrow"]
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
    run(PYTHON, "scripts/verify_real_data_preflight.py", "--domain", "aerospace")
    build_args = [
        PYTHON,
        "scripts/download_aerospace_datasets.py",
        "--out",
        "data/aerospace/processed/aerospace_orius.csv",
    ]
    if EXTERNAL_ROOT:
        build_args.extend(["--external-root", str(Path(EXTERNAL_ROOT).resolve())])
    run(*build_args)
    run(PYTHON, "scripts/build_data_manifest.py", "--dataset", "AEROSPACE")
    run(PYTHON, "scripts/train_dataset.py", "--dataset", "AEROSPACE", "--candidate-run", "--run-id", "hf_aerospace_realflight")
    run(PYTHON, "scripts/run_universal_orius_validation.py", "--seeds", "1", "--horizon", "24")


if __name__ == "__main__":
    main()
