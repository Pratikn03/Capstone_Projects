#!/usr/bin/env python3
"""Generate ORIUS Universal Framework architecture figure."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def main() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colors
    stage_color = "#4A90D9"
    domain_color = "#E8A838"
    core_color = "#5CB85C"

    # Pipeline stages (left to right)
    stages = [
        ("1. Detect\n(OQE)", 1.5, 3),
        ("2. Calibrate\n(U_t)", 3.0, 3),
        ("3. Constrain\n(A_t)", 4.5, 3),
        ("4. Shield\n(repair)", 6.0, 3),
        ("5. Certify", 7.5, 3),
    ]
    for label, x, y in stages:
        rect = FancyBboxPatch((x - 0.5, y - 0.4), 1.0, 0.8, boxstyle="round,pad=0.02",
                              facecolor=stage_color, edgecolor="black", linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=8, fontweight="bold")

    # Arrows between stages
    for i in range(len(stages) - 1):
        ax.annotate("", xy=(stages[i + 1][1] - 0.55, 3), xytext=(stages[i][1] + 0.55, 3),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Domains (bottom)
    domains = ["Energy", "AV", "Industrial", "Healthcare"]
    for i, d in enumerate(domains):
        x = 1.5 + i * 2.2
        rect = FancyBboxPatch((x - 0.6, 1.2), 1.2, 0.6, boxstyle="round,pad=0.02",
                              facecolor=domain_color, edgecolor="black", linewidth=1)
        ax.add_patch(rect)
        ax.text(x, 1.5, d, ha="center", va="center", fontsize=7)

    # Core (top)
    core_rect = FancyBboxPatch((2, 4.5), 6, 0.8, boxstyle="round,pad=0.02",
                               facecolor=core_color, edgecolor="black", linewidth=1)
    ax.add_patch(core_rect)
    ax.text(5, 4.9, "ORIUS Universal Physical Safety Framework", ha="center", va="center", fontsize=10, fontweight="bold")

    # Domain adapter arrows to pipeline
    for i in range(4):
        x_d = 1.5 + i * 2.2
        ax.annotate("", xy=(x_d, 2.4), xytext=(x_d, 1.9),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1, linestyle="--"))
    ax.text(0.5, 2.15, "DomainAdapter", fontsize=7, color="gray")

    ax.set_title("Universal Domain Pipeline", fontsize=12, fontweight="bold")
    plt.tight_layout()
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    fig_dir = repo / "paper" / "assets" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "fig_universal_framework.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved -> {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()
