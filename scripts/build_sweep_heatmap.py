#!/usr/bin/env python3
"""Build hyperparameter sweep heatmaps from sensitivity_grid.csv for thesis ch23."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    out_dir = Path("reports/publication")
    grid_csv = out_dir / "sensitivity_grid.csv"
    if not grid_csv.exists():
        raise SystemExit(f"Missing {grid_csv} — run scripts/run_sensitivity_sweeps.py first")

    df = pd.read_csv(grid_csv)
    dc3s = df[df["controller"] == "dc3s_wrapped"].copy()
    if dc3s.empty:
        dc3s = df[df["controller"] == "dc3s_ftit"].copy()

    agg = (
        dc3s.groupby(["alpha0", "ph_lambda", "kappa_drift_penalty"], as_index=False)
        .agg(
            tsvr=("true_soc_violation_rate", "mean"),
            ir=("intervention_rate", "mean"),
            picp=("picp_90", "mean"),
            width=("mean_interval_width", "mean"),
        )
    )

    kpen_values = sorted(agg["kappa_drift_penalty"].unique())

    fig, axes = plt.subplots(1, len(kpen_values), figsize=(6 * len(kpen_values), 5), squeeze=False)
    for col_idx, kpen in enumerate(kpen_values):
        ax = axes[0, col_idx]
        sub = agg[agg["kappa_drift_penalty"] == kpen]
        pivot = sub.pivot_table(index="ph_lambda", columns="alpha0", values="tsvr")
        pivot = pivot.sort_index(ascending=False)
        im = ax.imshow(pivot.values * 100, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=max(5.0, (agg["tsvr"].max() * 100) + 1))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
        ax.set_xlabel(r"$\alpha_0$")
        ax.set_ylabel(r"$\lambda_{\mathrm{PH}}$")
        ax.set_title(f"$\\kappa_d$ = {kpen}")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j] * 100
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color="white" if val > 2.5 else "black", fontsize=9)
    fig.suptitle("True SOC Violation Rate (%) by Hyperparameter Setting — DC3S", fontsize=13)
    fig.colorbar(im, ax=axes[0, :], label="TSVR (%)", shrink=0.7)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    heatmap_path = out_dir / "fig_sweep_heatmap_tsvr.png"
    fig.savefig(heatmap_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap -> {heatmap_path}")

    fig, axes = plt.subplots(1, len(kpen_values), figsize=(6 * len(kpen_values), 5), squeeze=False)
    for col_idx, kpen in enumerate(kpen_values):
        ax = axes[0, col_idx]
        sub = agg[agg["kappa_drift_penalty"] == kpen]
        pivot = sub.pivot_table(index="ph_lambda", columns="alpha0", values="ir")
        pivot = pivot.sort_index(ascending=False)
        im = ax.imshow(pivot.values * 100, aspect="auto", cmap="Blues",
                       vmin=0, vmax=max(20.0, (agg["ir"].max() * 100) + 1))
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
        ax.set_xlabel(r"$\alpha_0$")
        ax.set_ylabel(r"$\lambda_{\mathrm{PH}}$")
        ax.set_title(f"$\\kappa_d$ = {kpen}")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j] * 100
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color="white" if val > 10 else "black", fontsize=9)
    fig.suptitle("Intervention Rate (%) by Hyperparameter Setting — DC3S", fontsize=13)
    fig.colorbar(im, ax=axes[0, :], label="IR (%)", shrink=0.7)
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    heatmap_ir_path = out_dir / "fig_sweep_heatmap_ir.png"
    fig.savefig(heatmap_ir_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap IR -> {heatmap_ir_path}")


if __name__ == "__main__":
    main()
