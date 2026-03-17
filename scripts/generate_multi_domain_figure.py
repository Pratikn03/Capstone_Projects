#!/usr/bin/env python3
"""Generate multi-domain ORIUS validation figure for the paper."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OASG_CSV = REPO / "reports" / "universal_orius_validation" / "cross_domain_oasg_table.csv"
OUT = REPO / "paper" / "assets" / "figures" / "fig_multi_domain_validation.png"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def main() -> int:
    if not OASG_CSV.exists():
        print(f"Run scripts/run_universal_orius_validation.py first")
        return 1

    df = pd.read_csv(OASG_CSV)
    domains = df["domain"].tolist()
    baseline = pd.to_numeric(df["oasg_rate_baseline"], errors="coerce").fillna(0)
    orius = pd.to_numeric(df["oasg_rate_orius"], errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(domains))
    w = 0.35
    ax.bar([i - w/2 for i in x], baseline, w, label="Baseline (nominal)", color="#d62728", alpha=0.8)
    ax.bar([i + w/2 for i in x], orius, w, label="ORIUS (DC3S)", color="#2ca02c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15, ha="right")
    ax.set_ylabel("TSVR (true-state violation rate)")
    ax.set_title("Multi-Domain ORIUS Validation: Baseline vs. ORIUS Under Fault Episodes")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(1.0, baseline.max() * 1.1, orius.max() * 1.1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(f"Saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
