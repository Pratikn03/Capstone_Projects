#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _battery_wrappers_common import REPO_ROOT, copy_outputs, ensure_dir, run_script


def main() -> None:
    p = argparse.ArgumentParser(description="Run graceful degradation wrapper")
    p.add_argument("--out-dir", default="reports/blackout")
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    run_script("generate_priority2_artifacts.py")
    copy_outputs(
        [
            (
                REPO_ROOT / "reports/publication/graceful_degradation_trace.csv",
                out / "graceful_degradation_trace.csv",
            ),
            (
                REPO_ROOT / "reports/publication/graceful_degradation_trace.png",
                out / "graceful_degradation_trace.png",
            ),
        ]
    )


if __name__ == "__main__":
    main()
