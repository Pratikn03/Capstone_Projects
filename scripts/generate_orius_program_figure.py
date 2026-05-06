#!/usr/bin/env python3
"""Generate the ORIUS universal monograph spine figure.

The figure is intentionally conceptual rather than data-driven. It is used as a
navigation surface for the book: battery remains the deepest witness row, while
temporal validity, graceful fallback, benchmark discipline, bounded composition,
and runtime governance extend the same universal safety-layer argument.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BG = "#F7F8FA"
TEXT = "#14202B"
MUTED = "#4B5563"
LINE = "#475569"
SHADOW = "#CBD5E1"

CORE_FILL = "#EFF6FF"
CORE_EDGE = "#1D4ED8"
TEMP_FILL = "#FEF3C7"
TEMP_EDGE = "#B45309"
GRACE_FILL = "#ECFCCB"
GRACE_EDGE = "#3F6212"
BENCH_FILL = "#F3E8FF"
BENCH_EDGE = "#7E22CE"
FLEET_FILL = "#E0F2FE"
FLEET_EDGE = "#0F766E"
RUNTIME_FILL = "#FCE7F3"
RUNTIME_EDGE = "#BE185D"
BRIDGE_FILL = "#F8FAFC"
BRIDGE_EDGE = "#64748B"


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    lines: list[str],
    *,
    fc: str,
    ec: str,
    fs: float = 10.5,
) -> tuple[float, float, float, float]:
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.006, y - 0.006),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            lw=0,
            fc=SHADOW,
            alpha=0.22,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            lw=1.8,
            ec=ec,
            fc=fc,
            zorder=2,
        )
    )
    line_gap = min(0.035, h / max(3, len(lines) + 1))
    for idx, line in enumerate(lines):
        ty = y + h / 2 + (len(lines) - 1) * line_gap / 2 - idx * line_gap
        weight = "bold" if idx == 0 else "normal"
        ax.text(
            x + w / 2,
            ty,
            line,
            ha="center",
            va="center",
            fontsize=fs if idx == 0 else fs - 0.8,
            fontweight=weight,
            color=TEXT,
            zorder=3,
        )
    return (x, y, w, h)


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2] / 2, box[1] + box[3] / 2)


def _right(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2], box[1] + box[3] / 2)


def _left(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0], box[1] + box[3] / 2)


def _arrow(
    ax: plt.Axes,
    src: tuple[float, float],
    dst: tuple[float, float],
    *,
    label: str | None = None,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            src,
            dst,
            arrowstyle="-|>",
            mutation_scale=18,
            lw=1.8,
            color=LINE,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
            zorder=4,
        )
    )
    if label:
        mx = (src[0] + dst[0]) / 2
        my = (src[1] + dst[1]) / 2
        ax.text(
            mx,
            my + 0.022,
            label,
            ha="center",
            va="center",
            fontsize=9,
            color=MUTED,
            zorder=5,
            bbox={"fc": "white", "ec": "none", "pad": 1.0, "alpha": 0.9},
        )


def build_figure(paper_out: Path, report_out: Path) -> None:
    for target in (paper_out, report_out):
        target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.5, 9.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "ORIUS Universal Safety Monograph Spine",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        0.5,
        0.935,
        "Battery is the deepest witness row; temporal validity, graceful fallback, benchmark discipline, bounded composition, and runtime governance extend the same universal runtime layer.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=MUTED,
    )

    core = _box(
        ax,
        0.06,
        0.33,
        0.27,
        0.34,
        [
            "Battery Witness Row",
            "Observation–action safety gap",
            "DC3S kernel and T1–T8 ladder",
            "Locked empirical and theorem anchor",
        ],
        fc=CORE_FILL,
        ec=CORE_EDGE,
        fs=12.0,
    )

    p2 = _box(
        ax,
        0.44,
        0.73,
        0.22,
        0.14,
        [
            "Temporal validity",
            "Certificate validity horizon",
            "Expiration bound and blackout safe-hold",
            "Ch. 20 + Ch. 28",
        ],
        fc=TEMP_FILL,
        ec=TEMP_EDGE,
    )
    p3 = _box(
        ax,
        0.72,
        0.73,
        0.22,
        0.14,
        [
            "Graceful fallback",
            "Graceful degradation",
            "Safe landing under prolonged blindness",
            "Ch. 20 + Ch. 29",
        ],
        fc=GRACE_FILL,
        ec=GRACE_EDGE,
    )
    p4 = _box(
        ax,
        0.44,
        0.50,
        0.22,
        0.14,
        [
            "Universal benchmark",
            "ORIUS-Bench",
            "Truth-vs-observation benchmark contract",
            "Ch. 10 + Ch. 30",
        ],
        fc=BENCH_FILL,
        ec=BENCH_EDGE,
    )
    p5 = _box(
        ax,
        0.72,
        0.50,
        0.22,
        0.14,
        [
            "Bounded composition",
            "Compositional safety",
            "Shared-constraint battery fleet coordination",
            "Ch. 31",
        ],
        fc=FLEET_FILL,
        ec=FLEET_EDGE,
    )
    p6 = _box(
        ax,
        0.58,
        0.24,
        0.28,
        0.16,
        [
            "Runtime governance",
            "CertOS runtime governance",
            "Certificate lifecycle, audit chain, explicit fallback",
            "Governance block + Ch. 27 + Ch. 32",
        ],
        fc=RUNTIME_FILL,
        ec=RUNTIME_EDGE,
    )

    bridge = _box(
        ax,
        0.43,
        0.06,
        0.47,
        0.11,
        [
            "Deployment bridge",
            "HIL rehearsal  •  locked release artifacts  •  runtime evidence  •  conservative claim discipline",
        ],
        fc=BRIDGE_FILL,
        ec=BRIDGE_EDGE,
        fs=10.5,
    )

    ax.text(
        0.19,
        0.275,
        "Reference proof and measurement surface",
        ha="center",
        va="center",
        fontsize=10.2,
        color=CORE_EDGE,
        fontweight="bold",
    )

    _arrow(ax, _right(core), _left(p2), label="temporal extension", rad=0.07)
    _arrow(ax, _right(core), _left(p4), label="evaluation discipline", rad=0.02)
    _arrow(ax, _right(core), _left(p6), label="runtime and governance", rad=-0.03)
    _arrow(ax, _right(p2), _left(p3), label="fallback timing")
    _arrow(ax, _right(p4), _left(p5), label="single asset → fleet")
    _arrow(ax, _center(p6), _center(bridge), label="bounded path toward deployment", rad=0.0)

    ax.text(
        0.78,
        0.425,
        "Program-first thesis block",
        ha="center",
        va="center",
        fontsize=10.2,
        color=RUNTIME_EDGE,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(report_out, dpi=300, bbox_inches="tight")
    fig.savefig(paper_out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-out",
        default="paper/assets/figures/fig46_orius_program_spine.png",
    )
    parser.add_argument(
        "--report-out",
        default="reports/publication/fig46_orius_program_spine.png",
    )
    args = parser.parse_args()
    build_figure(Path(args.paper_out), Path(args.report_out))


if __name__ == "__main__":
    main()
