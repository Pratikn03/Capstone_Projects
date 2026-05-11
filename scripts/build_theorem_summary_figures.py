#!/usr/bin/env python3
"""Build three-domain theorem summary tables and figures."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("reports/publication")
MATRIX = OUT / "three_domain_theorem_promotion_matrix.csv"

THEOREMS = ["T5", "T8", "T9", "T10", "T11Byz", "Tstale", "Tsensor"]
DOMAINS = ["battery", "av", "healthcare"]
PASS_MATRIX = {
    "T5": [1, 1, 1],
    "T8": [1, 1, 1],
    "T9": [1, 1, 1],
    "T10": [1, 1, 1],
    "T11Byz": [1, 0, 0],
    "Tstale": [1, 1, 1],
    "Tsensor": [1, 1, 1],
}


def write_matrix() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with MATRIX.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["theorem", *DOMAINS])
        for theorem in THEOREMS:
            writer.writerow([theorem, *PASS_MATRIX[theorem]])


def write_heatmap() -> None:
    data = np.array([PASS_MATRIX[t] for t in THEOREMS], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.imshow(data, vmin=0, vmax=1, cmap="Greens")
    ax.set_xticks(range(len(DOMAINS)), DOMAINS)
    ax.set_yticks(range(len(THEOREMS)), THEOREMS)
    ax.set_title("Theorem Evidence Pass Matrix")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, "pass" if data[i, j] else "n/a", ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_theorem_pass_matrix.png", dpi=160)
    plt.close(fig)


def write_domain_bar() -> None:
    data = np.array([PASS_MATRIX[t] for t in THEOREMS], dtype=float)
    counts = data.sum(axis=0)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(DOMAINS, counts, color="#1f77b4")
    ax.set_ylim(0, len(THEOREMS))
    ax.set_ylabel("promoted theorem evidence rows")
    ax.set_title("Three-Domain Theorem Evidence")
    fig.tight_layout()
    fig.savefig(OUT / "fig_three_domain_theorem_evidence.png", dpi=160)
    plt.close(fig)


def main() -> None:
    write_matrix()
    write_heatmap()
    write_domain_bar()
    print(MATRIX)


if __name__ == "__main__":
    main()
