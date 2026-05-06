#!/usr/bin/env python3
"""Paper 3 publication figure builder backed by the canonical benchmark surface."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.run_paper3_four_policy_benchmark import POLICY_ORDER, run_benchmark


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_publication_figure(
    *,
    paper3_dir: Path,
    publication_dir: Path,
) -> dict[str, Path]:
    """Build the publication-facing figure and keep the promoted CSV in sync."""
    summary_path = paper3_dir / "graceful_four_policy_metrics.csv"
    detail_path = paper3_dir / "policy_compare.csv"
    if not summary_path.exists() or not detail_path.exists():
        run_benchmark(paper3_dir=paper3_dir, publication_dir=publication_dir)

    rows = _load_csv(detail_path)
    publication_dir.mkdir(parents=True, exist_ok=True)
    paper3_dir.mkdir(parents=True, exist_ok=True)

    by_policy: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    colors = {
        "blind_persistence": "#b02a37",
        "immediate_shutdown": "#4b5563",
        "simple_ramp_down": "#d97706",
        "optimized_graceful": "#047857",
    }
    labels = {
        "blind_persistence": "Blind persistence",
        "immediate_shutdown": "Immediate shutdown",
        "simple_ramp_down": "Simple ramp-down",
        "optimized_graceful": "Optimized graceful",
    }
    metrics = [
        ("gdq", "GDQ"),
        ("useful_work_mwh", "Useful work (MWh)"),
        ("violation_rate", "Violation rate"),
        ("severity_mwh", "Severity (MWh)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4))
    for ax, (metric, ylabel) in zip(axes.flat, metrics, strict=False):
        for policy in POLICY_ORDER:
            series = by_policy.get(policy, [])
            if not series:
                continue
            ax.plot(
                [int(row["blackout_duration"]) for row in series],
                [float(row[metric]) for row in series],
                "o-",
                color=colors[policy],
                label=labels[policy],
                linewidth=1.8,
                markersize=4.5,
            )
        ax.set_xlabel("Blackout duration (h)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Graceful fallback benchmark over prolonged blindness")
    fig.tight_layout()

    publication_figure_path = publication_dir / "fig_graceful_four_policies.png"
    paper3_figure_path = paper3_dir / "fig_graceful_four_policies.png"
    fig.savefig(publication_figure_path, dpi=200)
    fig.savefig(paper3_figure_path, dpi=200)
    plt.close(fig)

    publication_summary_path = publication_dir / "graceful_four_policy_metrics.csv"
    shutil.copyfile(summary_path, publication_summary_path)

    return {
        "publication_figure": publication_figure_path,
        "paper3_figure": paper3_figure_path,
        "publication_summary": publication_summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper 3 publication figure")
    parser.add_argument("--paper3-dir", default="reports/paper3")
    parser.add_argument("--publication-dir", default="reports/publication")
    args = parser.parse_args()

    outputs = build_publication_figure(
        paper3_dir=(REPO_ROOT / args.paper3_dir)
        if not Path(args.paper3_dir).is_absolute()
        else Path(args.paper3_dir),
        publication_dir=(REPO_ROOT / args.publication_dir)
        if not Path(args.publication_dir).is_absolute()
        else Path(args.publication_dir),
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
