#!/usr/bin/env python3
"""Paper 5: Run multi-agent transformer-capacity scenario.

Non-composition counterexample: two batteries, shared feeder limit.
Outputs CSV and summary to reports/publication/.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.multi_agent.scenarios import run_transformer_capacity_scenario


def main() -> None:
    out_dir = REPO_ROOT / "reports" / "publication"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_transformer_capacity_scenario(
        feeder_capacity_mw=80.0,
        n_steps=24,
        seed=42,
        out_dir=out_dir,
    )

    print("Multi-agent transformer-capacity scenario (Paper 5)")
    print("-" * 50)
    for name, r in results.items():
        print(f"  {name:12s}: joint_viol={r['joint_violations']:3d} "
              f"local_viol={r['local_violations']:3d} "
              f"work={r['useful_work_mwh']:.0f} "
              f"margin_q={r['margin_quality']:.3f}")
    print(f"\nCSV -> {out_dir / 'multi_agent_transformer_scenario.csv'}")


if __name__ == "__main__":
    main()
