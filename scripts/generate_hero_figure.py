#!/usr/bin/env python3
"""Generate the OASG hero figure for the thesis Introduction chapter.

Two-panel matplotlib figure (12 x 5 inches):

Panel A (left):  OASG timeline — observed vs true SOC trajectories showing
                  how observed SOC appears above the safety floor while the
                  true SOC has already breached it.  A vertical marker shows
                  the moment DC3S activates the shield.

Panel B (right): DC3S five-stage pipeline block diagram:
                  Detect → Calibrate → Constrain → Shield → Certify

Output: paper/assets/figures/fig_oasg_hero.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    print("matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def _make_panel_a(ax: "plt.Axes") -> None:
    """OASG timeline: observed vs true SOC with shield activation."""
    np.random.seed(42)
    T = 60
    t = np.arange(T)

    # True SOC: starts healthy, drifts below 20% floor around t=30
    true_soc = 0.55 - 0.008 * t + 0.015 * np.random.randn(T).cumsum() * 0.05
    true_soc = np.clip(true_soc, 0.05, 0.95)
    # Inject gradual drift below floor at t=28-40
    for i in range(28, 48):
        true_soc[i] -= 0.012 * (i - 27)
    true_soc = np.clip(true_soc, 0.03, 0.95)

    # Observed SOC: optimistic (stale / spike faults hide true drop)
    obs_soc = true_soc.copy()
    obs_soc[25:45] = obs_soc[25] - 0.002 * np.arange(20) + 0.01 * np.random.randn(20)
    obs_soc = np.clip(obs_soc, 0.15, 0.95)

    floor = 0.20
    shield_t = 37  # DC3S detects and shields at this step

    # Observed SOC line (blue)
    ax.plot(t, obs_soc * 100, color="#1565C0", lw=2.0, label="Observed SOC", zorder=3)
    # True SOC line (red)
    ax.plot(t, true_soc * 100, color="#C62828", lw=2.0, linestyle="--",
            label="True SOC", zorder=3)
    # Safety floor (orange dashed)
    ax.axhline(y=floor * 100, color="#E65100", lw=1.5, linestyle=":", label="Safety floor (20 %)", zorder=2)
    # Shade the OASG region: where obs >= floor but true < floor
    oasg_mask = (obs_soc >= floor) & (true_soc < floor)
    if oasg_mask.any():
        oasg_t = t[oasg_mask]
        ax.fill_between(oasg_t,
                        floor * 100, obs_soc[oasg_mask] * 100,
                        alpha=0.18, color="#FF8F00", label="OASG region")

    # Shield activation marker
    ax.axvline(x=shield_t, color="#2E7D32", lw=2.0, linestyle="-.", zorder=4)
    ax.annotate(
        "Shield\nactivates",
        xy=(shield_t, 35),
        xytext=(shield_t + 5, 50),
        fontsize=8.5,
        color="#2E7D32",
        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2),
        ha="left",
    )

    # OASG label in the gap region
    ax.text(31, 23, "OASG\nzone", fontsize=8, color="#E65100", ha="center",
            style="italic")

    ax.set_xlabel("Time step", fontsize=10)
    ax.set_ylabel("State of Charge (%)", fontsize=10)
    ax.set_title("(A) OASG: Observed vs True SOC", fontsize=11, fontweight="bold")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _make_panel_b(ax: "plt.Axes") -> None:
    """DC3S five-stage pipeline block diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    stages = [
        ("Detect\n(OQE)", "#1565C0", "Scores telemetry\nreliability  $w_t$"),
        ("Calibrate\n(RUI)", "#6A1B9A", "Inflates uncertainty\nset by $1/w_t$"),
        ("Constrain\n(SAF)", "#B71C1C", "Tightens safe\naction set"),
        ("Shield\n(Repair)", "#E65100", "Projects action\ninto safe set"),
        ("Certify\n(CERTos)", "#2E7D32", "Emits per-step\nsafety certificate"),
    ]

    box_w = 1.55
    box_h = 1.4
    gap = 0.3
    x_start = 0.3
    y_center = 6.8
    arrow_len = gap - 0.05

    for i, (label, color, sub) in enumerate(stages):
        x0 = x_start + i * (box_w + gap)
        y0 = y_center - box_h / 2

        rect = mpatches.FancyBboxPatch(
            (x0, y0), box_w, box_h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="white", alpha=0.88, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x0 + box_w / 2, y0 + box_h * 0.62, label,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white", zorder=4)
        ax.text(x0 + box_w / 2, y0 + box_h * 0.22, sub,
                ha="center", va="center", fontsize=6.8, color="white",
                alpha=0.92, zorder=4)

        # Arrow to next stage
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x0 + box_w + arrow_len, y_center),
                xytext=(x0 + box_w + 0.02, y_center),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2),
                zorder=5,
            )

    # Telemetry input arrow
    in_x = x_start + 0.75
    ax.annotate(
        "",
        xy=(in_x, y_center + box_h / 2),
        xytext=(in_x, y_center + box_h / 2 + 1.2),
        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2),
        zorder=5,
    )
    ax.text(in_x, y_center + box_h / 2 + 1.4, "Raw\ntelemetry $z_t$",
            ha="center", va="bottom", fontsize=8, color="#333")

    # Safe action output arrow
    last_x = x_start + (len(stages) - 1) * (box_w + gap)
    out_x = last_x + box_w * 0.5
    ax.annotate(
        "",
        xy=(out_x, y_center - box_h / 2 - 1.1),
        xytext=(out_x, y_center - box_h / 2 - 0.02),
        arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=1.5),
        zorder=5,
    )
    ax.text(out_x, y_center - box_h / 2 - 1.25, "Safe action $a_t^*$\n+ Certificate",
            ha="center", va="top", fontsize=8, color="#2E7D32", fontweight="bold")

    # OQE feedback label
    ax.annotate(
        "$w_t$ flows right →",
        xy=(x_start + box_w + gap / 2, y_center + 0.15),
        xytext=(x_start + box_w + 1.1, y_center + 1.6),
        fontsize=7.5, color="#555",
        arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8, linestyle="dashed"),
    )

    ax.set_title("(B) DC3S Five-Stage Pipeline", fontsize=11, fontweight="bold", pad=14)


def main() -> None:
    out_dir = Path("paper/assets/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig_oasg_hero.png"

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.2),
                                      gridspec_kw={"width_ratios": [1.1, 1.0]})
    _make_panel_a(ax_a)
    _make_panel_b(ax_b)

    fig.suptitle(
        "The Observation–Action Safety Gap (OASG) and the DC3S Remedy",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Hero figure → {out_path}")


if __name__ == "__main__":
    main()
