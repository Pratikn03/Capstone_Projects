"""Phase B: Regenerate P1 figures — architecture diagram and theorem figures."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUT = "reports/publication"
DPI = 300
FONT = "DejaVu Sans"

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
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
})

C_BASE  = "#B22222"
C_ORIUS = "#1A6B3C"


# ══════════════════════════════════════════════════════════════════════
# FIGURE B1 — Theory-to-Runtime-to-Domain Flow (replaces low-quality PNG)
# ══════════════════════════════════════════════════════════════════════
def make_theory_runtime_domain_flow():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    ax.axis("off")

    stages = [
        ("Physical AI\nHazard",        "Degraded observation\ncauses OASG event",         "#2C3E50", "#1A252F"),
        ("Theory\nBridge",             "OASG · repair · temporal\ncertificate validity",   "#6C3483", "#4A235A"),
        ("Runtime Kernel\n(ORIUS)",    "Detect · Calibrate\nConstrain · Shield · Certify","#1A6B3C", "#145A32"),
        ("Domain\nAdapters",           "Battery · AV\nHealthcare",                        "#1F618D", "#154360"),
        ("Parity Gate\n& Evidence",    "Claim promoted only\nby artifact evidence",        "#784212", "#6E2C00"),
    ]

    xs = [0.4, 3.1, 5.8, 8.5, 11.2]
    w, h = 2.5, 2.8
    yc = 1.0

    for i, ((title, sub, fc, ec), x) in enumerate(zip(stages, xs)):
        r = mpatches.FancyBboxPatch((x, yc), w, h,
            boxstyle="round,pad=0.15", fc=fc, ec=ec, lw=2.5, zorder=3)
        ax.add_patch(r)
        ax.text(x+w/2, yc+h*0.67, title, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white", zorder=4)
        ax.text(x+w/2, yc+h*0.28, sub, ha="center", va="center",
                fontsize=8.5, color="white", alpha=0.9, zorder=4,
                linespacing=1.4)
        if i < len(stages)-1:
            ax.annotate("", xy=(xs[i+1], yc+h/2),
                        xytext=(x+w, yc+h/2),
                arrowprops=dict(arrowstyle="-|>", color="#555",
                                lw=2, mutation_scale=18), zorder=5)

    # Domain tier badges below
    tier_data = [
        ("Battery",    0.9, "#1A6B3C", "Primary"),
        ("AV",         3.5, "#1F618D", "Bounded"),
        ("Healthcare", 6.1, "#1F618D", "Bounded"),
    ]
    for name, bx, bc, tier in tier_data:
        r = mpatches.FancyBboxPatch((bx, 0.0), 2.2, 0.75,
            boxstyle="round,pad=0.1", fc=bc, ec=bc, lw=1.5, alpha=0.85, zorder=3)
        ax.add_patch(r)
        ax.text(bx+1.1, 0.38, f"{name}  [{tier}]", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=4)

    ax.text(7.0, 0.38, "Three-domain evidence boundary",
            ha="center", va="center", fontsize=9, color="#555", style="italic")

    ax.set_title(
        "ORIUS: Theory-to-Runtime-to-Domain Flow\n"
        "One universal safety contract · Three defended domain instantiations",
        fontsize=13, fontweight="bold", pad=10)

    path = f"{OUT}/fig_orius_theory_runtime_domain_flow.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE B2 — T5 Certificate Horizon vs Reliability
# ══════════════════════════════════════════════════════════════════════
def make_t5_horizon():
    w = np.linspace(0.05, 0.99, 200)
    # T5: tau*(w) = floor(log(alpha) / log(1-w))  illustrative with alpha=0.05
    # Simplified: horizon ~ -log(0.05)/(-log(1-w)) clipped
    horizon = np.floor(-np.log(0.05) / (-np.log(1 - w + 1e-9)))
    horizon = np.clip(horizon, 0, 20)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(w, 0, horizon, alpha=0.12, color=C_ORIUS)
    ax.plot(w, horizon, color=C_ORIUS, lw=2.5, zorder=3,
            label=r"$\tau^*(w_t)$ — max valid horizon")

    # Illustrative data points (from fig_T5_horizon_vs_reliability.png)
    wp = [0.2, 0.5, 0.8, 0.95]
    hp = [0,   3,   8,   12]
    ax.scatter(wp, hp, s=80, color=C_ORIUS, zorder=5, edgecolors="white", lw=1.5)

    ax.axhline(1, color="#AAA", lw=1.2, ls="--", label="Minimum valid horizon ($\\tau=1$)")
    ax.axvline(0.5, color="#DDD", lw=1, ls=":")

    ax.set_xlabel("Observation Reliability  $w_t \\in [0,1]$", fontsize=12)
    ax.set_ylabel("Certificate Validity Horizon  $\\tau^*(w_t)$ (steps)", fontsize=12)
    ax.set_title(
        "Theorem T5: Certificate Validity Horizon vs Reliability\n"
        r"$\tau^*(w_t) \geq 1 \Rightarrow$ certificate governs release",
        fontsize=12, fontweight="bold")

    ax.set_xlim(0.05, 1.0); ax.set_ylim(-0.5, 16)
    ax.legend(fontsize=10, loc="upper left")

    # Annotation
    ax.annotate("Below threshold:\ncertificate expires\n(safe-hold triggered)",
        xy=(0.2, 0), xytext=(0.25, 6),
        fontsize=9, color=C_BASE,
        arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="#FDEDEC", ec=C_BASE, lw=1))

    path = f"{OUT}/fig_T5_horizon_vs_reliability.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# ══════════════════════════════════════════════════════════════════════
# FIGURE B3 — T8 Safety-Useful Work Frontier
# ══════════════════════════════════════════════════════════════════════
def make_t8_frontier():
    # Four policies from graceful_four_policy_metrics.csv
    policies = ["Blind\nPersistence", "Immediate\nShutdown",
                "Simple\nRamp-Down", "Optimized\nGraceful\n(ORIUS)"]
    work  = [19.2, 0.0,  5.05, 10.1]   # MWh
    tsvr  = [1.333, 0.0, 0.0,  0.0]    # %
    colors = [C_BASE, "#888", "#E67E22", C_ORIUS]
    markers = ["X", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for i, (w, t, col, mk, pol) in enumerate(
            zip(work, tsvr, colors, markers, policies)):
        ax.scatter(w, t, s=160, color=col, marker=mk, zorder=5,
                   edgecolors="white", lw=1.5, label=pol)
        offset_x = 0.3 if i != 0 else -0.3
        offset_y = 0.08 if i != 0 else 0.08
        ax.annotate(pol.replace("\n", " "),
            xy=(w, t), xytext=(w+offset_x, t+offset_y),
            fontsize=9, color=col, fontweight="bold",
            ha="left" if i != 0 else "right")

    # Pareto frontier line (zero-violation policies only)
    safe_w = sorted([w for w, t in zip(work, tsvr) if t == 0.0])
    ax.plot(safe_w, [0]*len(safe_w), color=C_ORIUS, lw=1.5,
            ls="--", alpha=0.5, label="Zero-violation frontier")

    # Shaded safe region
    ax.axhline(0, color="#CCC", lw=1.2, ls="-")
    ax.fill_between([0, 22], 0, -0.15, alpha=0.08, color=C_ORIUS)
    ax.text(11, -0.1, "Zero true-state violation region (T8 requirement)",
            ha="center", fontsize=9, color=C_ORIUS, style="italic")

    ax.set_xlabel("Useful Work Retained (MWh)", fontsize=12)
    ax.set_ylabel("True-State Violation Rate (%)", fontsize=12)
    ax.set_title(
        "Theorem T8: Safety–Useful Work Frontier\n"
        "ORIUS Optimized Graceful policy dominates immediate shutdown",
        fontsize=12, fontweight="bold")
    ax.set_xlim(-1, 22); ax.set_ylim(-0.2, 1.8)

    ax.annotate("+10.1 MWh over\nimmediate shutdown",
        xy=(10.1, 0), xytext=(12, 0.6),
        fontsize=9.5, color=C_ORIUS, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_ORIUS, lw=1.3),
        bbox=dict(boxstyle="round,pad=0.3", fc="#EAFAF1", ec=C_ORIUS, lw=1))

    path = f"{OUT}/fig_T8_safety_useful_work_frontier.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


if __name__ == "__main__":
    print("Phase B — generating P1 figures …")
    make_theory_runtime_domain_flow()
    make_t5_horizon()
    make_t8_frontier()
    print("Phase B complete.")
