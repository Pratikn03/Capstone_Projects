#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _battery_wrappers_common import REPO_ROOT, copy_outputs, ensure_dir, run_script


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep hyperparameters wrapper")
    p.add_argument("--out-dir", default="reports/calibration")
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    if not (REPO_ROOT / "reports/publication/ablation_table.csv").exists():
        raise SystemExit("Expected reports/publication/ablation_table.csv. Run ablation pipeline first.")
    run_script("generate_priority2_artifacts.py")
    copy_outputs(
        [(REPO_ROOT / "reports/publication/hyperparameter_surfaces.csv", out / "hyperparameter_surfaces.csv")]
    )


if __name__ == "__main__":
    main()
