#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _battery_wrappers_common import REPO_ROOT, copy_outputs, ensure_dir, run_script, write_manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Run battery fault benchmark wrapper")
    p.add_argument("--region", default="DE")
    p.add_argument(
        "--controllers", nargs="*", default=["det_lp", "robust_fixed", "cvar", "dc3s", "dc3s_ftit"]
    )
    p.add_argument("--out-dir", default="reports/fault_benchmark")
    args = p.parse_args()
    out = ensure_dir(REPO_ROOT / args.out_dir)
    run_script("run_cpsbench.py", "--out-dir", "reports/publication")
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
            (
                REPO_ROOT / "reports/publication/fault_performance_summary.json",
                out / "fault_performance_summary.json",
            ),
        ]
    )
    write_manifest(
        out,
        "run_fault_benchmark_manifest.json",
        {
            "region_hint": args.region,
            "controllers_hint": args.controllers,
            "source_script": "run_cpsbench.py",
        },
    )


if __name__ == "__main__":
    main()
