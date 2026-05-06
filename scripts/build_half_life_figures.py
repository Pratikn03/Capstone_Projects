#!/usr/bin/env python3
"""Build Paper 2 half-life figures from blackout benchmark outputs.

Usage:
    python scripts/build_half_life_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    in_path = REPO_ROOT / "reports/publication/certificate_half_life_blackout.csv"
    if not in_path.exists():
        in_path = REPO_ROOT / "reports/publication/blackout_study.csv"
    if not in_path.exists():
        print("Run scripts/run_certificate_half_life_blackout.py first.")
        sys.exit(1)

    out_dir = REPO_ROOT / "reports/publication"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    agg = df.groupby("blackout_hours", as_index=False).agg(
        {
            "tsvr_pct": "mean",
            "coverage_pct": "mean",
        }
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = agg["blackout_hours"]
    ax1.plot(x, agg["tsvr_pct"], marker="o", color="#d62728", linewidth=1.5)
    ax1.set_xlabel("Blackout Duration (hours)")
    ax1.set_ylabel("TSVR (%)")
    ax1.set_title("Violation Rate vs Blackout Duration")
    ax1.grid(alpha=0.3)

    ax2.plot(x, agg["coverage_pct"], marker="s", color="#1f77b4", linewidth=1.5)
    ax2.axhline(50, color="gray", linestyle="--", linewidth=1, label="Half-life threshold")
    ax2.set_xlabel("Blackout Duration (hours)")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_title("Certificate Coverage Decay")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "fig_half_life_blackout.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
