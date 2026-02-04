"""Generate a static architecture diagram (PNG/SVG) for reports."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _add_box(ax, xy, text, width=0.18, height=0.12, fc="#F3F6FB", ec="#2C3E50"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, src, dst):
    arrow = FancyArrowPatch(
        src,
        dst,
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.0,
        color="#2C3E50",
    )
    ax.add_patch(arrow)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/figures/architecture.png")
    parser.add_argument("--svg", default="reports/figures/architecture.svg")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Node positions
    nodes = {
        "sources": (0.05, 0.72),
        "pipeline": (0.27, 0.72),
        "features": (0.49, 0.72),
        "forecast": (0.71, 0.72),
        "optimization": (0.71, 0.46),
        "anomaly": (0.49, 0.46),
        "api": (0.88, 0.63),
        "dashboard": (0.88, 0.35),
        "monitoring": (0.27, 0.35),
        "retrain": (0.49, 0.20),
        "artifacts": (0.71, 0.20),
    }

    _add_box(ax, nodes["sources"], "Data Sources\nOPSD, Weather\nPrice/Carbon, Holidays")
    _add_box(ax, nodes["pipeline"], "Data Pipeline\nvalidate → build\nsplit_time_series")
    _add_box(ax, nodes["features"], "Feature Store\nfeatures.parquet\nsplits")
    _add_box(ax, nodes["forecast"], "Forecasting\nGBM + Quantile\nLSTM/TCN")
    _add_box(ax, nodes["optimization"], "Optimization\nLP dispatch\nbaselines")
    _add_box(ax, nodes["anomaly"], "Anomaly Detection\nresidual + isolation")
    _add_box(ax, nodes["api"], "API\nFastAPI")
    _add_box(ax, nodes["dashboard"], "Dashboard\nStreamlit")
    _add_box(ax, nodes["monitoring"], "Monitoring\ndata/model drift\nalerts")
    _add_box(ax, nodes["retrain"], "Retraining\nretrain_if_needed")
    _add_box(ax, nodes["artifacts"], "Reports & Artifacts\nreports/\nartifacts/")

    # Arrows
    _arrow(ax, (0.23, 0.78), (0.27, 0.78))  # sources -> pipeline
    _arrow(ax, (0.45, 0.78), (0.49, 0.78))  # pipeline -> features
    _arrow(ax, (0.67, 0.78), (0.71, 0.78))  # features -> forecast
    _arrow(ax, (0.80, 0.72), (0.88, 0.66))  # forecast -> api
    _arrow(ax, (0.80, 0.50), (0.88, 0.66))  # optimization -> api
    _arrow(ax, (0.58, 0.50), (0.65, 0.50))  # anomaly -> optimization? (visual link)
    _arrow(ax, (0.58, 0.50), (0.88, 0.66))  # anomaly -> api
    _arrow(ax, (0.88, 0.59), (0.88, 0.47))  # api -> dashboard
    _arrow(ax, (0.45, 0.72), (0.33, 0.43))  # features -> monitoring
    _arrow(ax, (0.36, 0.35), (0.49, 0.26))  # monitoring -> retrain
    _arrow(ax, (0.58, 0.26), (0.71, 0.72))  # retrain -> forecast (loop)
    _arrow(ax, (0.80, 0.20), (0.71, 0.30))  # artifacts connection
    _arrow(ax, (0.80, 0.46), (0.80, 0.26))  # optimization -> artifacts
    _arrow(ax, (0.80, 0.72), (0.80, 0.26))  # forecast -> artifacts

    ax.text(0.5, 0.95, "GridPulse Architecture", ha="center", va="center", fontsize=14, weight="bold")

    out_png = Path(args.out)
    out_svg = Path(args.svg)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_svg, dpi=200, bbox_inches="tight")
    print(f"Wrote {out_png} and {out_svg}")


if __name__ == "__main__":
    main()

