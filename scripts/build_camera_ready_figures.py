#!/usr/bin/env python3
"""Regenerate all publication figures at camera-ready quality (300 DPI, serif fonts)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PUB = REPO / "reports" / "publication"
PAPER_FIG = REPO / "paper" / "assets" / "figures"
STYLE_TOKEN_PATH = PUB / "camera_ready_figure_style_tokens.json"

# ── Publication-quality style ──────────────────────────────
RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "grid.alpha": 0.3,
    "axes.grid": True,
}

CONTROLLER_STYLE = {
    "cvar_interval": {"color": "#1f77b4", "marker": "o"},
    "dc3s_ftit": {"color": "#2ca02c", "marker": "D"},
    "deterministic_lp": {"color": "#ff7f0e", "marker": "s"},
    "robust_fixed_interval": {"color": "#d62728", "marker": "^"},
    "threshold_rule": {"color": "#9467bd", "marker": "v"},
}

TARGET_COLORS = {"load_mw": "#1f77b4", "wind_mw": "#2ca02c", "solar_mw": "#ff7f0e"}
GROUP_MARKERS = {"low": "o", "mid": "s", "med": "s", "high": "^"}

CONTROLLER_LABELS = {
    "cvar_interval": "CVaR",
    "dc3s_ftit": "DC³S (ours)",
    "deterministic_lp": "Det. LP",
    "robust_fixed_interval": "Robust Fixed",
    "threshold_rule": "Threshold Rule",
}


def _style(controller: str) -> dict:
    s = CONTROLLER_STYLE.get(str(controller), {"color": "#555", "marker": "o"})
    return {**s, "label": CONTROLLER_LABELS.get(str(controller), str(controller))}


def _write_style_tokens() -> None:
    tokens = {
        "schema_version": 1,
        "font_family": RCPARAMS["font.family"],
        "font_serif": RCPARAMS["font.serif"],
        "figure_dpi": RCPARAMS["figure.dpi"],
        "savefig_dpi": RCPARAMS["savefig.dpi"],
        "controller_style": CONTROLLER_STYLE,
        "controller_labels": CONTROLLER_LABELS,
        "target_colors": TARGET_COLORS,
        "group_markers": GROUP_MARKERS,
        "artifact_policy": {
            "data_plots": "Generated from tracked CSV/JSON/runtime artifacts.",
            "static_diagrams": "Exported assets are committed; editable source is registered in the Figma design manifest.",
        },
    }
    STYLE_TOKEN_PATH.write_text(json.dumps(tokens, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_publication_figure(fig: plt.Figure, pub_stem: str, paper_stem: str | None = None) -> None:
    pub_stem = Path(pub_stem).with_suffix("").name
    outputs = [PUB / f"{pub_stem}.png", PUB / f"{pub_stem}.pdf"]
    for out in outputs:
        fig.savefig(out)
    if paper_stem:
        paper_stem = Path(paper_stem).with_suffix("").name
        for out in outputs:
            dst = PAPER_FIG / f"{paper_stem}{out.suffix}"
            shutil.copy2(out, dst)
    plt.close(fig)
    print(f"  {pub_stem}: png+pdf")


def _run_static_export_generators() -> None:
    """Rebuild static/conceptual exports before writing lineage."""
    for script in [
        REPO / "scripts" / "generate_architecture_diagram.py",
        REPO / "scripts" / "generate_hero_figure.py",
    ]:
        if script.exists():
            subprocess.run([sys.executable, str(script)], cwd=REPO, check=True)


def _write_lineage_outputs() -> None:
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_camera_ready_figure_lineage.py"), "--write"],
        cwd=REPO,
        check=True,
    )


# ═══════════════════════════════════════════════════════════
# FIG03 & FIG04 — True-SOC violation/severity vs fault seed
# ═══════════════════════════════════════════════════════════
def build_fig03_fig04():
    df = pd.read_csv(PUB / "dc3s_main_table.csv")
    subset = df[df["scenario"].isin(["dropout", "drift_combo"])].copy()
    if subset.empty:
        print("  SKIP fig03/fig04: no dropout/drift_combo rows")
        return

    sev_col = (
        "true_soc_violation_severity_p95_mwh"
        if "true_soc_violation_severity_p95_mwh" in subset.columns
        else "true_soc_violation_severity_p95"
    )

    # --- FIG03: violation rate ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in subset.groupby("controller", sort=True):
        s = _style(controller)
        ax.plot(
            sub["seed"],
            sub["true_soc_violation_rate"],
            marker=s["marker"],
            color=s["color"],
            label=s["label"],
        )
    ax.set_xlabel("Simulation Seed")
    ax.set_ylabel("True-SOC Violation Rate")
    ax.set_title("True-SOC Violation Rate Under Faulted Telemetry")
    ax.legend(loc="best")
    fig.tight_layout()
    _save_publication_figure(
        fig,
        "fig_true_soc_violation_vs_dropout",
        "fig03_true_soc_violation_vs_dropout",
    )
    print(f"  fig03: violation rate ({len(subset)} points)")

    # --- FIG04: severity P95 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in subset.groupby("controller", sort=True):
        s = _style(controller)
        ax.plot(sub["seed"], sub[sev_col], marker=s["marker"], color=s["color"], label=s["label"])
    ax.set_xlabel("Simulation Seed")
    ax.set_ylabel("True-SOC Severity P95 (MWh)")
    ax.set_title("True-SOC Violation Severity (P95) Under Faulted Telemetry")
    ax.legend(loc="best")
    fig.tight_layout()
    _save_publication_figure(
        fig,
        "fig_true_soc_severity_p95_vs_dropout",
        "fig04_true_soc_severity_p95_vs_dropout",
    )
    print(f"  fig04: severity P95 ({len(subset)} points)")


# ═══════════════════════════════════════════════════════════
# FIG05 — CQR group coverage
# ═══════════════════════════════════════════════════════════
def build_fig05():
    cov = pd.read_csv(PUB / "cqr_group_coverage.csv")
    cov["group"] = cov["group"].astype(str).replace({"mid": "med"})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, row in cov.iterrows():
        w = float(row["mean_width"])
        p = float(row["picp_90"])
        t = str(row.get("target", "load_mw"))
        g = str(row["group"])
        ax.scatter(
            w,
            p,
            s=100,
            marker=GROUP_MARKERS.get(g, "o"),
            color=TARGET_COLORS.get(t, "#555"),
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        ax.annotate(f" {g}", (w, p), fontsize=8, va="center")
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0, label="90% target")
    ax.set_xlabel("Mean Interval Width (MW)")
    ax.set_ylabel("PICP@90")
    ax.set_title("Coverage vs Width by Volatility Group")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save_publication_figure(fig, "fig_cqr_group_coverage", "fig05_cqr_group_coverage")
    print(f"  fig05: {len(cov)} groups")


# ═══════════════════════════════════════════════════════════
# FIG06 — Transfer coverage
# ═══════════════════════════════════════════════════════════
def build_fig06():
    summary_path = PUB / "cross_region_transfer_summary.csv"
    if not summary_path.exists():
        print("  SKIP fig06: no transfer summary CSV")
        return
    df = pd.read_csv(summary_path)
    plot_df = df[df["scenario"].isin(["nominal", "dropout", "drift_combo"])].copy()
    if plot_df.empty:
        plot_df = df.copy()

    region_colors = {
        "DE": "#1f77b4",
        "US": "#ff7f0e",
        "US_MISO": "#ff7f0e",
        "US_PJM": "#2ca02c",
        "US_ERCOT": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    for region, sub in plot_df.groupby("region", sort=True):
        picp = pd.to_numeric(sub["picp_90_mean"], errors="coerce")
        c = region_colors.get(str(region), "#555")
        ax.bar(
            sub["scenario"].astype(str) + f"\n({region})",
            picp,
            color=c,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
            label=str(region),
        )
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    ax.set_ylabel("PICP@90")
    ax.set_title("Transfer Coverage Across Regions")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    fig.tight_layout()
    _save_publication_figure(fig, "fig_transfer_coverage", "fig06_transfer_coverage")
    print(f"  fig06: {len(plot_df)} bars")


# ═══════════════════════════════════════════════════════════
# FIG07 — Cost-safety Pareto frontier
# ═══════════════════════════════════════════════════════════
def build_fig07():
    df = pd.read_csv(PUB / "dc3s_main_table.csv")
    sev_col = (
        "true_soc_violation_severity_p95_mwh"
        if "true_soc_violation_severity_p95_mwh" in df.columns
        else "true_soc_violation_severity_p95"
    )

    baseline = df[df["controller"] == "robust_fixed_interval"][
        ["scenario", "seed", "true_soc_violation_rate", sev_col, "expected_cost_usd"]
    ].rename(
        columns={
            "true_soc_violation_rate": "baseline_violation_rate",
            sev_col: "baseline_severity_p95",
            "expected_cost_usd": "baseline_cost_usd",
        }
    )
    merged = df.merge(baseline, on=["scenario", "seed"], how="left")
    merged["violation_reduction_pct"] = 100.0 * (
        (merged["baseline_violation_rate"] - merged["true_soc_violation_rate"])
        / np.maximum(merged["baseline_violation_rate"], 1e-9)
    )
    merged["cost_delta_pct"] = 100.0 * (
        (merged["expected_cost_usd"] - merged["baseline_cost_usd"])
        / np.maximum(merged["baseline_cost_usd"], 1e-9)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in merged.groupby("controller", sort=True):
        s = _style(controller)
        ax.scatter(
            sub["cost_delta_pct"].to_numpy(dtype=float),
            sub["violation_reduction_pct"].to_numpy(dtype=float),
            alpha=0.8,
            marker=s["marker"],
            color=s["color"],
            edgecolors="black",
            linewidths=0.3,
            label=s["label"],
            s=50,
        )
    ax.axhline(10.0, color="black", linestyle="--", linewidth=1.0, label="10% threshold")
    ax.set_xlabel("Cost Delta vs Robust Baseline (%)")
    ax.set_ylabel("Violation Reduction (%)")
    ax.set_title("Cost\u2013Safety Pareto Frontier")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save_publication_figure(fig, "fig_cost_safety_pareto", "fig07_cost_safety_frontier")
    print(f"  fig07: {len(merged)} points")


# ═══════════════════════════════════════════════════════════
# FIG08 — RAC sensitivity vs width
# ═══════════════════════════════════════════════════════════
def build_fig08():
    df = pd.read_csv(PUB / "dc3s_main_table.csv")
    required = {"rac_sensitivity_mean", "adaptive_width_mean", "controller"}
    if not required.issubset(df.columns):
        print(f"  SKIP fig08: missing columns {required - set(df.columns)}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in df.groupby("controller", sort=True):
        s = _style(controller)
        x = pd.to_numeric(sub["rac_sensitivity_mean"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(sub["adaptive_width_mean"], errors="coerce").to_numpy(float)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.any():
            ax.scatter(
                x[mask],
                y[mask],
                alpha=0.8,
                marker=s["marker"],
                color=s["color"],
                edgecolors="black",
                linewidths=0.3,
                label=s["label"],
                s=50,
            )
    ax.set_xlabel("RAC Sensitivity (mean)")
    ax.set_ylabel("Adaptive Width (mean, MW)")
    ax.set_title("RAC Sensitivity vs Interval Width Expansion")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _save_publication_figure(fig, "fig_rac_sensitivity_vs_width", "fig08_rac_sensitivity_vs_width")
    print(f"  fig08: {len(df)} points")


# ═══════════════════════════════════════════════════════════
# FIG09 — Dataset cards (re-export from build_conference_assets)
# ═══════════════════════════════════════════════════════════
def build_fig09():
    """Dataset cards — inlined to respect 300 DPI rcParams."""
    summary_csv = PUB / "tables" / "table1_dataset_summary.csv"
    if not summary_csv.exists():
        print("  SKIP fig09: no dataset summary CSV")
        return
    summary_df = pd.read_csv(summary_csv)

    DATASET_META_LOCAL = {
        "DE": {"source": "OPSD + SMARD"},
        "US_MISO": {"source": "EIA-930 MISO"},
        "US_PJM": {"source": "EIA-930 PJM"},
        "US_ERCOT": {"source": "EIA-930 ERCOT"},
    }

    rows = []
    for dkey, sub in summary_df.groupby("DatasetKey", sort=True):
        first = sub.iloc[0]
        meta = DATASET_META_LOCAL.get(str(dkey), {})
        rows.append(
            {
                "dataset_key": str(dkey),
                "dataset_label": str(first["Dataset"]),
                "country": str(first["Country"]),
                "rows": int(first["Rows"]),
                "date_start": str(first["Start"]),
                "date_end": str(first["End"]),
                "targets": ", ".join(sorted(sub["Signal"].astype(str).unique().tolist())),
                "min_coverage_pct": float(pd.to_numeric(sub["Coverage%"], errors="coerce").min()),
                "source": meta.get("source", "n/a"),
            }
        )

    ncols = 2 if len(rows) > 1 else 1
    nrows = max(1, (len(rows) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 3.8 * nrows))
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, r in zip(flat_axes, rows, strict=False):
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(
            plt.Rectangle((0.02, 0.05), 0.96, 0.90, facecolor="#f7f7f4", edgecolor="#2e3d30", linewidth=1.5)
        )
        ax.text(0.05, 0.88, r["dataset_label"], fontsize=14, fontweight="bold", color="#1f2d1f")
        ax.text(0.05, 0.76, f"Key: {r['dataset_key']}  |  Country: {r['country']}", fontsize=10, color="#334")
        ax.text(
            0.05, 0.64, f"Rows: {r['rows']:,}  |  Range: {r['date_start']} to {r['date_end']}", fontsize=11
        )
        ax.text(0.05, 0.52, f"Targets: {r['targets']}", fontsize=11)
        ax.text(0.05, 0.40, f"Min coverage: {r['min_coverage_pct']:.1f}%", fontsize=11)
        ax.text(0.05, 0.28, f"Source: {r['source']}", fontsize=11)
    for ax in flat_axes[len(rows) :]:
        ax.axis("off")

    fig.suptitle("ORIUS Dataset Cards", fontsize=16, fontweight="bold")
    fig.tight_layout()
    _save_publication_figure(fig, "fig_region_dataset_cards", "fig09_region_dataset_cards")
    print(f"  fig09: {len(rows)} cards")


# ═══════════════════════════════════════════════════════════
# FIG10 — Calibration tradeoff
# ═══════════════════════════════════════════════════════════
def build_fig10():
    cov = pd.read_csv(PUB / "cqr_group_coverage.csv")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for _, row in cov.iterrows():
        t = str(row.get("target", "load_mw"))
        g = str(row.get("group", "group"))
        ax.scatter(
            float(row["mean_width"]),
            float(row["picp_90"]),
            s=100,
            marker=GROUP_MARKERS.get(g, "o"),
            color=TARGET_COLORS.get(t, "#555"),
            edgecolors="black",
            linewidths=0.5,
            zorder=5,
        )
        ax.annotate(
            f" {t.replace('_mw', '')}:{g}",
            (float(row["mean_width"]), float(row["picp_90"])),
            fontsize=7,
            va="center",
        )
    ax.axhline(0.90, color="black", linestyle="--", linewidth=1.0, label="90% target")
    ax.set_xlabel("Mean Interval Width (MW)")
    ax.set_ylabel("PICP@90")
    ax.set_title("Calibration Tradeoff by Target and Volatility Group")
    # custom legend for targets and groups
    from matplotlib.lines import Line2D

    legend_elements = []
    for t, c in TARGET_COLORS.items():
        legend_elements.append(
            Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=t.replace("_mw", "")
            )
        )
    for g, m in [("low", "o"), ("med", "s"), ("high", "^")]:
        legend_elements.append(
            Line2D([0], [0], marker=m, color="w", markerfacecolor="#888", markersize=8, label=g)
        )
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    _save_publication_figure(fig, "fig_calibration_tradeoff", "fig10_calibration_tradeoff")
    print(f"  fig10: {len(cov)} points")


# ═══════════════════════════════════════════════════════════
# FIG11 — Transfer generalization (2-panel)
# ═══════════════════════════════════════════════════════════
def build_fig11():
    summary_path = PUB / "cross_region_transfer_summary.csv"
    if not summary_path.exists():
        print("  SKIP fig11: no transfer summary CSV")
        return
    df = pd.read_csv(summary_path)
    plot_df = df[df["scenario"].isin(["nominal", "dropout", "drift_combo"])].copy()
    if plot_df.empty:
        plot_df = df.copy()

    region_colors = {
        "DE": "#1f77b4",
        "US": "#ff7f0e",
        "US_MISO": "#ff7f0e",
        "US_PJM": "#2ca02c",
        "US_ERCOT": "#d62728",
    }
    region_markers = {"DE": "o", "US": "s", "US_MISO": "s", "US_PJM": "^", "US_ERCOT": "v"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for region, sub in plot_df.groupby("region", sort=True):
        c = region_colors.get(str(region), "#555")
        m = region_markers.get(str(region), "o")
        axes[0].plot(
            sub["scenario"].astype(str),
            pd.to_numeric(sub["picp_90_mean"], errors="coerce"),
            marker=m,
            color=c,
            linewidth=1.8,
            label=str(region),
        )
        axes[1].plot(
            sub["scenario"].astype(str),
            pd.to_numeric(sub["true_soc_violation_rate_mean"], errors="coerce"),
            marker=m,
            color=c,
            linewidth=1.8,
            label=str(region),
        )

    axes[0].axhline(0.90, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Transfer Calibration")
    axes[0].set_ylabel("PICP@90")
    axes[0].legend(fontsize=8)
    axes[1].set_title("Transfer Safety")
    axes[1].set_ylabel("True-SOC Violation Rate")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Cross-Region Transfer and Generalization", fontweight="bold")
    fig.tight_layout()
    _save_publication_figure(fig, "fig_transfer_generalization", "fig11_transfer_generalization")
    print(f"  fig11: {len(plot_df)} rows × 2 panels")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    plt.rcParams.update(RCPARAMS)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Camera-Ready Figure Generation (300 DPI, serif)")
    print("=" * 60)

    _write_style_tokens()
    _run_static_export_generators()
    print("fig01/fig02: static/conceptual exports refreshed")

    build_fig03_fig04()
    build_fig05()
    build_fig06()
    build_fig07()
    build_fig08()

    # FIG09 uses build_conference_assets — generate with rcParams already set
    try:
        build_fig09()
    except Exception as e:
        print(f"  WARN fig09: {e}")

    build_fig10()
    build_fig11()

    print()
    _write_lineage_outputs()
    print("All camera-ready figures saved to:")
    print(f"  {PUB}")
    print(f"  {PAPER_FIG}")
    print("Done!")


if __name__ == "__main__":
    main()
