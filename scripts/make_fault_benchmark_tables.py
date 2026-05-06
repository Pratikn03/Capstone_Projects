#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _battery_wrappers_common import REPO_ROOT, copy_outputs, ensure_dir, run_script


def main() -> None:
    p = argparse.ArgumentParser(description="Build battery fault benchmark tables")
    p.add_argument("--out-dir", default="reports/fault_benchmark")
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    run_script(
        "build_fault_performance_table.py",
        "--in-csv",
        "reports/publication/cpsbench_merged_sweep.csv",
        "--out-dir",
        "reports/publication",
    )
    copy_outputs(
        [
            (
                REPO_ROOT / "reports/publication/fault_performance_table.csv",
                out / "fault_performance_table.csv",
            ),
            (
                REPO_ROOT / "reports/publication/fault_performance_pivot.csv",
                out / "fault_performance_pivot.csv",
            ),
        ]
    )


if __name__ == "__main__":
    main()
