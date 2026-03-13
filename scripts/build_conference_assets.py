#!/usr/bin/env python3
"""Build lightweight conference-paper figures and dataset cards from publication artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "publication"
DATASET_SUMMARY_DEFAULT = DEFAULT_OUT_DIR / "tables" / "table1_dataset_summary.csv"

DATASET_META = {
    "DE": {
        "stats_path": REPO_ROOT / "data" / "dashboard" / "de_stats.json",
        "provenance_path": REPO_ROOT / "data" / "processed" / "dataset_provenance.json",
        "source": "OPSD + SMARD",
        "weather_source": "Open-Meteo archive API",
        "carbon_source": "SMARD generation mix",
    },
    "US_MISO": {
        "stats_path": REPO_ROOT / "data" / "dashboard" / "us_stats.json",
        "provenance_path": REPO_ROOT / "data" / "processed" / "us_eia930" / "dataset_provenance.json",
        "source": "EIA-930 MISO",
        "weather_source": "Open-Meteo archive API",
        "carbon_source": "proxy_generation_mix",
    },
    "US_PJM": {
        "stats_path": REPO_ROOT / "data" / "dashboard" / "us_pjm_stats.json",
        "provenance_path": REPO_ROOT / "data" / "processed" / "us_eia930_pjm" / "dataset_provenance.json",
        "source": "EIA-930 PJM",
        "weather_source": "Open-Meteo archive API",
        "carbon_source": "proxy_generation_mix",
    },
    "US_ERCOT": {
        "stats_path": REPO_ROOT / "data" / "dashboard" / "us_ercot_stats.json",
        "provenance_path": REPO_ROOT / "data" / "processed" / "us_eia930_ercot" / "dataset_provenance.json",
        "source": "EIA-930 ERCOT",
        "weather_source": "Open-Meteo archive API",
        "carbon_source": "proxy_generation_mix",
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conference-style dataset cards and paper figures")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dataset-summary", default=str(DATASET_SUMMARY_DEFAULT))
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_dataset_cards(dataset_summary_path: Path, out_dir: Path) -> dict[str, Any]:
    summary_df = pd.read_csv(dataset_summary_path)
    rows: list[dict[str, Any]] = []
    for dataset_key, sub in summary_df.groupby("DatasetKey", sort=True):
        first = sub.iloc[0]
        meta = DATASET_META.get(str(dataset_key), {})
        stats = _read_json(Path(str(meta.get("stats_path", "")))) if meta.get("stats_path") else {}
        provenance = _read_json(Path(str(meta.get("provenance_path", "")))) if meta.get("provenance_path") else {}
        rows.append(
            {
                "dataset_key": str(dataset_key),
                "dataset_label": str(first["Dataset"]),
                "country": str(first["Country"]),
                "rows": int(first["Rows"]),
                "date_start": str(first["Start"]),
                "date_end": str(first["End"]),
                "targets": ",".join(sorted(sub["Signal"].astype(str).unique().tolist())),
                "min_coverage_pct": float(pd.to_numeric(sub["Coverage%"], errors="coerce").min()),
                "feature_count": stats.get("total_features") or stats.get("feature_count") or provenance.get("columns"),
                "columns": stats.get("columns") or provenance.get("columns"),
                "source": meta.get("source"),
                "weather_source": provenance.get("weather_source") or meta.get("weather_source") or "n/a",
                "carbon_source": provenance.get("carbon_source") or meta.get("carbon_source") or "n/a",
            }
        )

    if rows:
        cards_df = pd.DataFrame(rows).sort_values(["country", "dataset_key"]).reset_index(drop=True)
    else:
        cards_df = pd.DataFrame(
            columns=[
                "dataset_key",
                "dataset_label",
                "country",
                "rows",
                "date_start",
                "date_end",
                "targets",
                "min_coverage_pct",
                "feature_count",
                "columns",
                "source",
                "weather_source",
                "carbon_source",
            ]
        )
    cards_path = out_dir / "dataset_cards.csv"
    cards_df.to_csv(cards_path, index=False)

    fig_path = out_dir / "fig_region_dataset_cards.png"
    n_cards = len(cards_df)
    ncols = 2 if n_cards > 1 else 1
    nrows = max(1, (n_cards + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 3.8 * nrows))
    if not isinstance(axes, (list, tuple)):
        axes_iter = [axes]
    else:
        axes_iter = list(axes)
    flat_axes = []
    for ax in axes_iter:
        if isinstance(ax, (list, tuple)):
            flat_axes.extend(ax)
        else:
            flat_axes.append(ax)
    if hasattr(axes, "flat"):
        flat_axes = list(axes.flat)

    for ax, row in zip(flat_axes, cards_df.to_dict(orient="records")):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.90, facecolor="#f7f7f4", edgecolor="#2e3d30", linewidth=1.5))
        ax.text(0.05, 0.88, row["dataset_label"], fontsize=14, fontweight="bold", color="#1f2d1f")
        ax.text(0.05, 0.79, f"Key: {row['dataset_key']}  |  Country: {row['country']}", fontsize=10, color="#334")
        ax.text(0.05, 0.67, f"Rows: {row['rows']:,}", fontsize=11)
        ax.text(0.05, 0.58, f"Range: {row['date_start']} to {row['date_end']}", fontsize=11)
        ax.text(0.05, 0.49, f"Targets: {row['targets']}", fontsize=11)
        ax.text(0.05, 0.40, f"Min coverage: {row['min_coverage_pct']:.2f}%", fontsize=11)
        ax.text(0.05, 0.31, f"Features: {row.get('feature_count') or 'n/a'}  |  Columns: {row.get('columns') or 'n/a'}", fontsize=11)
        ax.text(0.05, 0.22, f"Source: {row.get('source') or 'n/a'}", fontsize=11)
        if row.get("weather_source"):
            ax.text(0.05, 0.13, f"Weather: {row['weather_source']}", fontsize=10, color="#455")
        if row.get("carbon_source"):
            ax.text(0.05, 0.07, f"Carbon: {row['carbon_source']}", fontsize=10, color="#455")

    for ax in flat_axes[n_cards:]:
        ax.axis("off")

    fig.suptitle("GridPulse Conference Dataset Cards", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return {"rows": int(len(cards_df)), "csv": str(cards_path), "figure": str(fig_path)}


def build_calibration_tradeoff(out_dir: Path) -> dict[str, Any]:
    cov_path = out_dir / "cqr_group_coverage.csv"
    cov_df = pd.read_csv(cov_path)
    fig_path = out_dir / "fig_calibration_tradeoff.png"

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    markers = {"low": "o", "mid": "s", "high": "^", "med": "s"}
    colors = {"load_mw": "#1f77b4", "wind_mw": "#2ca02c", "solar_mw": "#ff7f0e"}
    for _, row in cov_df.iterrows():
        target = str(row.get("target", "unknown"))
        group = str(row.get("group", "group"))
        ax.scatter(
            float(row["mean_width"]),
            float(row["picp_90"]),
            s=90,
            marker=markers.get(group, "o"),
            color=colors.get(target, "#555555"),
            alpha=0.9,
        )
        ax.text(float(row["mean_width"]), float(row["picp_90"]), f" {target}:{group}", fontsize=8)
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mean Interval Width")
    ax.set_ylabel("PICP@90")
    ax.set_title("Calibration Tradeoff by Target and Volatility Group")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return {"rows": int(len(cov_df)), "figure": str(fig_path)}


def build_transfer_generalization(out_dir: Path) -> dict[str, Any]:
    summary_path = out_dir / "cross_region_transfer_summary.csv"
    transfer_path = out_dir / "transfer_stress.csv"
    fig_path = out_dir / "fig_transfer_generalization.png"

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        plot_df = df[df["scenario"].isin(["nominal", "dropout", "drift_combo"])].copy()
        if plot_df.empty:
            plot_df = df.copy()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for region, sub in plot_df.groupby("region", sort=True):
            axes[0].plot(
                sub["scenario"].astype(str),
                pd.to_numeric(sub["picp_90_mean"], errors="coerce"),
                marker="o",
                linewidth=1.8,
                label=str(region),
            )
            axes[1].plot(
                sub["scenario"].astype(str),
                pd.to_numeric(sub["true_soc_violation_rate_mean"], errors="coerce"),
                marker="o",
                linewidth=1.8,
                label=str(region),
            )
        axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1.0)
        axes[0].set_title("Transfer Calibration")
        axes[0].set_ylabel("PICP@90")
        axes[1].set_title("Transfer Safety")
        axes[1].set_ylabel("True-SOC Violation Rate")
        for ax in axes:
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=20)
        axes[1].legend(fontsize=8)
        fig.suptitle("Cross-Region Transfer / Generalization")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=220)
        plt.close(fig)
        return {"rows": int(len(plot_df)), "figure": str(fig_path), "source": str(summary_path)}

    transfer_df = pd.read_csv(transfer_path)
    if "transfer_case" in transfer_df.columns and transfer_df["transfer_case"].astype(str).str.contains("pending_transfer_artifacts", na=False).any():
        raise RuntimeError("transfer_stress.csv contains placeholder rows; cannot build transfer generalization figure")

    agg = transfer_df.groupby(["transfer_case"], as_index=False).agg(
        picp_90=("picp_90", "mean"),
        true_soc_violation_rate=("true_soc_violation_rate", "mean"),
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(agg["transfer_case"].astype(str), agg["picp_90"], marker="o", label="PICP@90")
    ax.plot(agg["transfer_case"].astype(str), agg["true_soc_violation_rate"], marker="s", label="True-SOC Violation Rate")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Transfer / Generalization Summary")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    return {"rows": int(len(agg)), "figure": str(fig_path), "source": str(transfer_path)}


def build_figure_inventory(out_dir: Path) -> dict[str, Any]:
    figures = {
        "architecture": [
            REPO_ROOT / "reports" / "publication" / "figures" / "fig01_geographic_scope.png",
            REPO_ROOT / "reports" / "figures" / "architecture.png",
        ],
        "dc3s_runtime_flow": [
            REPO_ROOT / "reports" / "publication" / "figures" / "fig11_dispatch_comparison.png",
            REPO_ROOT / "reports" / "figures" / "dispatch_compare.png",
        ],
        "region_dataset_cards": [out_dir / "fig_region_dataset_cards.png"],
        "calibration_tradeoff": [out_dir / "fig_calibration_tradeoff.png"],
        "fault_violation_rate_sweep": [out_dir / "fig_true_soc_violation_vs_dropout.png"],
        "fault_severity_sweep": [out_dir / "fig_true_soc_severity_p95_vs_dropout.png"],
        "cost_safety_frontier": [out_dir / "fig_cost_safety_pareto.png"],
        "transfer_generalization": [
            out_dir / "fig_transfer_generalization.png",
            out_dir / "fig_transfer_coverage.png",
        ],
    }
    payload = {"figures": []}
    for name, candidates in figures.items():
        selected = next((path for path in candidates if path.exists()), None)
        payload["figures"].append(
            {
                "name": name,
                "selected": str(selected.relative_to(REPO_ROOT)) if selected else None,
                "ready": selected is not None,
                "candidates": [str(path.relative_to(REPO_ROOT)) for path in candidates],
            }
        )
    payload["summary"] = {
        "required_total": len(figures),
        "ready_total": sum(1 for row in payload["figures"] if row["ready"]),
        "ready": all(row["ready"] for row in payload["figures"]),
    }
    out_path = out_dir / "conference_figure_inventory.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary_path = Path(args.dataset_summary)
    if not dataset_summary_path.is_absolute():
        dataset_summary_path = REPO_ROOT / dataset_summary_path

    dataset_cards = build_dataset_cards(dataset_summary_path=dataset_summary_path, out_dir=out_dir)
    calibration = build_calibration_tradeoff(out_dir=out_dir)
    transfer = build_transfer_generalization(out_dir=out_dir)
    inventory = build_figure_inventory(out_dir=out_dir)

    payload = {
        "dataset_cards": dataset_cards,
        "calibration_tradeoff": calibration,
        "transfer_generalization": transfer,
        "figure_inventory": inventory.get("summary", {}),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
