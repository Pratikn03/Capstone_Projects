#!/usr/bin/env python3
"""Build Paper 3 artifacts into reports/paper3/.

Step 3.2 (four-policy comparison): run_paper3_four_policy_benchmark.py is canonical.
Produces:
- policy_compare.csv (4 policies, blackout_duration breakdown)
- gdq_results.csv (GDQ, useful_work, violation_rate, severity by policy)
- fig_degradation_trajectory.png
- intervention_trace.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Paper 3 artifacts")
    parser.add_argument("--out", default="reports/paper3", help="Output directory")
    args = parser.parse_args()

    out = _resolve_repo_path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Step 3.2 + 3.4: canonical four-policy benchmark (produces policy_compare, gdq_results, fig, intervention_trace)
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_paper3_four_policy_benchmark.py"), "--out", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"ERROR: run_paper3_four_policy_benchmark failed: {r.stderr[:500]}")
        return 1

    print(f"Paper 3 artifacts -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
