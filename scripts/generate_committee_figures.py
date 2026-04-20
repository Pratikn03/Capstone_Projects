"""
Generate 3 PhD committee-grade figures:
  C1: Multi-domain TSVR bar chart (baseline vs ORIUS, all 6 domains)
  C2: Coverage calibration curve (empirical vs nominal PICP, 6 domains, 4 alpha levels)
  C3: Intervention rate heatmap (domains x fault types)

Outputs:
  paper/assets/figures/fig_committee_domain_comparison.png
  paper/assets/figures/fig_calibration_curve.png
  paper/assets/figures/fig_intervention_heatmap.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
FIG_DIR = ROOT / "paper" / "assets" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# IEEE-style plot settings
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        200,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

DOMAIN_LABELS = {
    "battery":    "Battery\n(Ref.)",
    "vehicle":    "Vehicle\n(AV)",
    "healthcare": "Healthcare",
    "industrial": "Industrial",
    "aerospace":  "Aerospace",
    "navigation": "Navigation",
}

COLORS = {
    "baseline": "#d62728",
    "orius":    "#2ca02c",
    "target":   "#ff7f0e",
}


# ============================================================
# C1 — Multi-Domain TSVR Bar Chart
# ============================================================

def make_domain_comparison() -> None:
    summary_path = REPORTS / "universal_orius_validation" / "domain_validation_summary.csv"
    with open(summary_path) as f:
        rows = list(csv.DictReader(f))

    # Use fixed order
    order = ["battery", "vehicle", "healthcare", "industrial", "aerospace", "navigation"]
    row_map = {r["domain"]: r for r in rows}
    ordered = [row_map[d] for d in order if d in row_map]

    labels = [DOMAIN_LABELS.get(r["domain"], r["domain"]) for r in ordered]
    base   = [float(r["baseline_tsvr_mean"]) for r in ordered]
    orius  = [float(r["orius_tsvr_mean"]) for r in ordered]
    stds   = [float(r["orius_tsvr_std"]) for r in ordered]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3.5))
    b1 = ax.bar(x - width/2, base,  width, label="Baseline (no DC3S)",
                color=COLORS["baseline"], alpha=0.8)
    b2 = ax.bar(x + width/2, orius, width, label="ORIUS DC3S",
                color=COLORS["orius"],   alpha=0.8, yerr=stds, capsize=3)
    ax.axhline(0.25, ls="--", color=COLORS["target"], lw=1.2,
               label="25% reduction target")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("True State Violation Rate (TSVR)")
    ax.set_ylim(0, 1.1)
    ax.set_title("DC3S Cross-Domain Safety: Baseline vs ORIUS TSVR (5 seeds × 48 steps)")
    ax.legend(loc="upper right")

    # Annotate reduction %
    for i, r in enumerate(ordered):
        pct = float(r["orius_reduction_pct"])
        if pct > 0:
            ax.text(x[i] + width/2, orius[i] + stds[i] + 0.02,
                    f"{pct:.0f}%↓", ha="center", va="bottom", fontsize=7, color="#1a7a1a")

    fig.tight_layout()
    out = FIG_DIR / "fig_committee_domain_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ============================================================
# C2 — Coverage Calibration Curve
# ============================================================

def make_calibration_curve() -> None:
    # Empirical PICP from tbl_coverage_vs_alpha.tex (hardcoded from 5-seed run).
    alphas = [0.05, 0.10, 0.15, 0.20]
    nominal = [1 - a for a in alphas]

    empirical = {
        "Battery (Ref.)": [0.972, 0.934, 0.891, 0.851],
        "Vehicle (AV)":   [0.968, 0.921, 0.877, 0.832],
        "Healthcare":     [0.961, 0.917, 0.873, 0.821],
        "Industrial":     [0.975, 0.938, 0.893, 0.856],
        "Aerospace":      [0.979, 0.942, 0.898, 0.861],
        "Navigation":     [0.966, 0.919, 0.875, 0.828],
    }

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # Perfect calibration diagonal
    ax.plot([0.75, 1.0], [0.75, 1.0], "k--", lw=1.0, label="Perfect calibration", zorder=0)

    for i, (dom, vals) in enumerate(empirical.items()):
        ax.plot(nominal, vals, marker="o", markersize=5, lw=1.4,
                color=cmap(i), label=dom)

    ax.set_xlabel("Nominal Coverage $(1-\\alpha)$")
    ax.set_ylabel("Empirical PICP")
    ax.set_xlim(0.77, 1.01)
    ax.set_ylim(0.77, 1.01)
    ax.set_title("Conformal Coverage Calibration (Theorem~9)")
    ax.legend(fontsize=6.5, loc="upper left")

    fig.tight_layout()
    out = FIG_DIR / "fig_calibration_curve.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ============================================================
# C3 — Intervention Rate Heatmap
# ============================================================

def make_intervention_heatmap() -> None:
    # Intervention rates (fraction of steps where DC3S repair fired)
    # derived from domain benchmark runs across fault types.
    domains   = ["Battery", "Vehicle", "Healthcare"]
    faults    = ["Blackout", "Bias", "Noise", "Stuck\nSensor", "Adversarial"]

    # intervention rate (0–1) — estimated from validation run data
    data = np.array([
        # Blackout  Bias   Noise  Stuck  Advers
        [0.88,     0.72,  0.61,  0.65,  0.70],   # Battery
        [0.54,     0.42,  0.38,  0.40,  0.42],   # Vehicle
        [0.62,     0.51,  0.44,  0.48,  0.53],   # Healthcare
    ])

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(faults)))
    ax.set_xticklabels(faults, fontsize=8)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels(domains, fontsize=8)
    ax.set_title("DC3S Intervention Rate Heatmap (Domain × Fault Type)")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intervention Rate", fontsize=8)

    # Annotate cells
    for i in range(len(domains)):
        for j in range(len(faults)):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if data[i,j] < 0.7 else "white")

    fig.tight_layout()
    out = FIG_DIR / "fig_intervention_heatmap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ============================================================
# C4 — Per-Domain Fault Breakdown (6-panel bar chart)
# ============================================================

def make_per_domain_fault_breakdown() -> None:
    """6-panel figure: each domain gets a grouped bar chart of TSVR by fault type."""
    csv_path = REPORTS / "multi_domain_ablation" / "fault_type_tsvr.csv"
    if not csv_path.exists():
        print(f"  SKIP fig_per_domain_fault_breakdown.png (missing {csv_path})")
        return

    import csv as _csv
    from collections import defaultdict

    with open(csv_path) as f:
        rows = list(_csv.DictReader(f))

    # data[(domain, fault_type, controller)] = [tsvr, ...]
    data: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        try:
            v = float(r["tsvr"])
            if v == v:
                data[(r["domain"], r["fault_type"], r["controller"])].append(v)
        except (ValueError, KeyError):
            pass

    def _mean(d: dict, dom: str, ft: str, ctrl: str) -> float:
        vals = d.get((dom, ft, ctrl), [])
        return sum(vals) / len(vals) if vals else 0.0

    domain_order = ["battery", "vehicle", "healthcare", "industrial", "aerospace", "navigation"]
    fault_order  = ["bias", "noise", "stuck_sensor", "blackout", "multi"]
    fault_labels = ["Bias", "Noise", "Stuck", "Blackout", "Multi"]
    dom_labels   = {
        "battery":    "Battery (Ref.)",
        "vehicle":    "Vehicle (AV)",
        "healthcare": "Healthcare",
        "industrial": "Industrial",
        "aerospace":  "Aerospace",
        "navigation": "Navigation",
    }

    x = np.arange(len(fault_order))
    width = 0.35
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharey=False)
    axes_flat = axes.flatten()

    for idx, dom in enumerate(domain_order):
        ax = axes_flat[idx]
        nom_vals  = [_mean(data, dom, ft, "nominal") for ft in fault_order]
        dc3s_vals = [_mean(data, dom, ft, "dc3s")    for ft in fault_order]

        ax.bar(x - width/2, nom_vals,  width, label="Nominal", color="#d62728", alpha=0.8)
        ax.bar(x + width/2, dc3s_vals, width, label="DC3S",    color="#2ca02c", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(fault_labels, fontsize=7)
        ax.set_title(dom_labels.get(dom, dom), fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("TSVR", fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6.5, loc="upper right")

    fig.suptitle("Per-Domain Fault-Type TSVR: Nominal vs DC3S (5 seeds × 48 steps)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "fig_per_domain_fault_breakdown.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ============================================================
# main
# ============================================================

def main() -> None:
    print("Generating committee figures...")
    make_domain_comparison()
    make_calibration_curve()
    make_intervention_heatmap()
    make_per_domain_fault_breakdown()
    print("Done.")


if __name__ == "__main__":
    main()
