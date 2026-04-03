# /// script
# dependencies = ["pandas", "pyarrow", "torch", "matplotlib", "scikit-learn"]
# ///
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(os.environ.get("GRIDPULSE_REPO_ROOT", ".")).resolve()
PYTHON = sys.executable


def run(*args: str) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def main() -> None:
    cmd = [
        PYTHON,
        "scripts/run_battery_deep_novelty.py",
        "--deep-oqe-epochs",
        os.environ.get("ORIUS_DEEP_OQE_EPOCHS", "12"),
        "--forecast-epochs",
        os.environ.get("ORIUS_DEEP_FORECAST_EPOCHS", "8"),
        "--batch-size",
        os.environ.get("ORIUS_DEEP_BATCH_SIZE", "128"),
    ]
    run(*cmd)
    run(PYTHON, "scripts/verify_paper_manifest.py")
    run(PYTHON, "scripts/validate_paper_claims.py")


if __name__ == "__main__":
    main()
