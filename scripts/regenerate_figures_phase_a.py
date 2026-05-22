"""Phase A: Regenerate the two P0 critical figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

OUT = "reports/publication"
DPI = 300
FONT = "DejaVu Sans"

# ── colour palette ─────────────────────────────────────────────────
C_BASE  = "#B22222"   # firebrick – baseline
C_ORIUS = "#1A6B3C"   # forest green – ORIUS
C_FAIL  = "#4A90D9"   # blue – fail-safe
C_BG    = "#F8F9FA"
C_GRID  = "#E0E0E0"

plt.rcParams.update({
    "font.family": FONT,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
})

# ══════════════════════════════════════════════════════════════════════
# FIGURE A1 — Universal Framework Conceptual Diagram (MISSING fig)
# ══════════════════════════════════════════════════════════════════════
def make_universal_framework():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.5)
    ax.axis("off")

    def box(ax, x, y, w, h, label, sub, fc, ec, fontsize=10):
        r = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle="round,pad=0.12", fc=fc, ec=ec, lw=2, zorder=3)
        ax.add_patch(r)
        ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white", zorder=4)
        ax.text(x + w/2, y + h*0.25, sub, ha="center", va="center",
                fontsize=8, color="white", alpha=0.92, zorder=4)

    def arrow(ax, x1, x2, y):
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
            arrowprops=dict(arrowstyle="-|>", color="#444", lw=2,
                            mutation_scale=16), zorder=5)

    # Physical plant / true state (left)
    box(ax, 0.2, 1.8, 2.2, 1.9, "Physical Plant", "True state  $z_t$",
        "#2C3E50", "#1A252F", fontsize=10)

    # Degraded observation channel
    box(ax, 2.9, 2.55, 2.5, 1.4, "Degraded\nObservation",
        "dropout · delay · drift", "#8E44AD", "#6C3483", fontsize=9)

    # Nominal controller
    box(ax, 2.9, 0.8, 2.5, 1.4, "Nominal\nController",
        "planner / optimizer", "#2980B9", "#1F618D", fontsize=9)

    # ORIUS runtime layer (centre, prominent)
    box(ax, 6.0, 1.0, 3.2, 2.5,
        "ORIUS Runtime Layer",
        "Detect · Calibrate · Constrain\nShield · Certify",
        "#1A6B3C", "#145A32", fontsize=10)

    # Physical actuation (right)
    box(ax, 9.8, 1.8, 2.6, 1.9, "Physical\nActuation",
        "safe action  $a_t^{\\mathrm{safe}}$",
        "#2C3E50", "#1A252F", fontsize=10)

    # Arrows
    # true state → observation channel
    arrow(ax, 2.42, 2.9, 3.25)
    # true state → ORIUS (dashed – hidden from nominal controller)
    ax.annotate("", xy=(6.0, 2.0), xytext=(2.42, 2.0),
        arrowprops=dict(arrowstyle="-|>", color="#B22222", lw=1.8,
                        linestyle="dashed", mutation_scale=14), zorder=5)
    # observation → nominal controller
    arrow(ax, 3.15, 2.9, 1.5); ax.annotate("", xy=(2.9, 1.5), xytext=(3.14, 2.55),
        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.8, mutation_scale=14))
    # nominal controller → ORIUS
    arrow(ax, 5.4, 6.0, 1.5)
    # ORIUS → actuation
    arrow(ax, 9.2, 9.8, 2.75)
    # actuation → plant (feedback)
    ax.annotate("", xy=(2.42, 1.8), xytext=(9.8, 1.8),
        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.5,
                        connectionstyle="arc3,rad=-0.35", mutation_scale=14))

    # OASG annotation
    ax.annotate("OASG event:\n$a_t\\in\\mathcal{A}(\\hat{z}_t)$\nbut\n$a_t\\notin\\mathcal{A}(z_t)$",
        xy=(5.4, 1.5), xytext=(5.6, 0.1),
        fontsize=8.5, color="#B22222",
        arrowprops=dict(arrowstyle="->", color="#B22222", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec="#B22222", lw=1))

    # Certificate annotation
    ax.text(9.5, 4.1, "Certificate\n$\\tau_t \\geq 1$", fontsize=8.5,
        color="#1A6B3C", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="#EAFAF1", ec="#1A6B3C", lw=1))

    # Three-domain badge
    ax.text(6.6, 4.1,
        "Battery  ·  AV  ·  Healthcare",
        fontsize=9, ha="center", color="#555",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F0F0F0", ec="#AAA", lw=1))

    ax.set_title(
        "ORIUS: Universal Runtime Safety Layer for Physical AI under Degraded Observation",
        fontsize=13, fontweight="bold", pad=14)

    path = f"{OUT}/fig_universal_framework.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE A2 — Runtime TSVR (dual-panel: overview + zoom)
# ══════════════════════════════════════════════════════════════════════
def make_final_runtime_tsvr():
    domains  = ["Battery\n(BESS)", "AV\n(nuPlan)", "Healthcare\n(MIMIC)"]
    baseline = [0.8333, 28.925, 19.449]   # percent
    orius    = [0.0000,  0.016,   0.000]
    x = np.arange(len(domains))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [3, 2]})
    fig.subplots_adjust(wspace=0.38)

    # ── Left panel: full scale with log Y ──────────────────────────
    ax1.set_facecolor(C_BG)
    b1 = ax1.bar(x - w/2, baseline, w, label="Baseline TSVR",
                 color=C_BASE, zorder=3, alpha=0.88, edgecolor="white", lw=0.5)
    b2 = ax1.bar(x + w/2, orius,    w, label="ORIUS TSVR",
                 color=C_ORIUS, zorder=3, alpha=0.88, edgecolor="white", lw=0.5)

    ax1.set_yscale("log")
    ax1.set_ylim(1e-4, 100)
    ax1.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.3g}%"))
    ax1.set_xticks(x); ax1.set_xticklabels(domains, fontsize=11)
    ax1.set_ylabel("True-State Violation Rate (log scale, %)", fontsize=11)
    ax1.set_title("TSVR: Baseline vs ORIUS (log scale)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)

    # value labels
    for bar, val in zip(b1, baseline):
        ax1.text(bar.get_x()+bar.get_width()/2, val*1.6,
                 f"{val:.3f}%", ha="center", va="bottom", fontsize=9,
                 color=C_BASE, fontweight="bold")
    for bar, val in zip(b2, orius):
        ypos = max(val, 2e-4)*1.6
        label = "0.000%" if val == 0 else f"{val:.4f}%"
        ax1.text(bar.get_x()+bar.get_width()/2, ypos,
                 label, ha="center", va="bottom", fontsize=9,
                 color=C_ORIUS, fontweight="bold")

    # ── Right panel: ORIUS-only zoom ──────────────────────────────
    ax2.set_facecolor(C_BG)
    colors = [C_ORIUS]*3
    bars = ax2.bar(x, orius, 0.55, color=colors, zorder=3,
                   edgecolor="white", lw=0.5, alpha=0.9)
    ax2.set_ylim(0, 0.025)
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.3f}%"))
    ax2.set_xticks(x); ax2.set_xticklabels(domains, fontsize=11)
    ax2.set_ylabel("ORIUS TSVR (%)", fontsize=11)
    ax2.set_title("ORIUS TSVR — Zoomed", fontsize=12, fontweight="bold")

    for bar, val in zip(bars, orius):
        label = "0.000%" if val == 0 else f"{val:.4f}%"
        ypos = val + 0.0005
        ax2.text(bar.get_x()+bar.get_width()/2, ypos,
                 label, ha="center", va="bottom", fontsize=11,
                 color="#0A3D1F", fontweight="bold")

    # Annotate step counts
    steps = ["n=288", "n=1.53M", "n=137K"]
    for i, s in enumerate(steps):
        ax2.text(i, -0.003, s, ha="center", va="top",
                 fontsize=8.5, color="#555", style="italic")

    ax2.text(0.5, 0.97,
        "All ORIUS bars scored on true physical state,\nnot observed state",
        transform=ax2.transAxes, ha="center", va="top",
        fontsize=8, color="#555",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F0F0F0", ec="#CCC", lw=0.8))

    fig.suptitle(
        "Claim-Governing Runtime TSVR Across Three Domains",
        fontsize=14, fontweight="bold", y=1.01)

    path = f"{OUT}/fig_final_runtime_tsvr.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


if __name__ == "__main__":
    print("Phase A — generating P0 figures …")
    make_universal_framework()
    make_final_runtime_tsvr()
    print("Phase A complete.")
