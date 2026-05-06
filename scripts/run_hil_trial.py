#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _battery_wrappers_common import REPO_ROOT, ensure_dir, run_script


def main() -> None:
    p = argparse.ArgumentParser(description="Run HIL trial wrapper")
    p.add_argument("--steps", type=int, default=72)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="reports/hil")
    args = p.parse_args()
    ensure_dir(REPO_ROOT / args.out_dir)
    run_script(
        "generate_hil_evidence.py",
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--out-dir",
        args.out_dir,
    )


if __name__ == "__main__":
    main()
