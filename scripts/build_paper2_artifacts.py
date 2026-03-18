#!/usr/bin/env python3
"""Build Paper 2 artifacts into reports/paper2/.

Step 2.2 (blackout benchmark): run_paper2_blackout_benchmark.py is canonical.
Produces:
- expiration_horizon.csv
- blackout_policy_compare.csv (4 policies)
- horizon_error.json
- fig_certificate_shrinkage.png

Step 2.1 (runtime trace): run_paper2_runtime_horizon_trace.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    out = REPO / "reports" / "paper2"
    out.mkdir(parents=True, exist_ok=True)

    # Step 2.2: canonical blackout benchmark (4 policies, all sweep dimensions)
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_paper2_blackout_benchmark.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"ERROR: run_paper2_blackout_benchmark failed: {r.stderr[:500]}")
        return 1

    # Step 2.1: runtime horizon trace
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_paper2_runtime_horizon_trace.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        print(f"  runtime_horizon_trace.csv -> {out}")
    else:
        print(f"  WARN: run_paper2_runtime_horizon_trace failed: {r.stderr[:100]}")

    print(f"Paper 2 artifacts -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
