#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_cost_safety_pareto(main_table_path: Path, out_dir: Path) -> dict[str, str | int]:
    if not main_table_path.exists():
        raise FileNotFoundError(main_table_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(main_table_path)
    sev_col = "true_soc_violation_severity_p95_mwh"
    if sev_col not in df.columns:
        sev_col = "true_soc_violation_severity_p95"

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
    merged["severity_reduction_pct"] = 100.0 * (
        (merged["baseline_severity_p95"] - merged[sev_col])
        / np.maximum(merged["baseline_severity_p95"], 1e-9)
    )
    merged["cost_delta_pct"] = 100.0 * (
        (merged["expected_cost_usd"] - merged["baseline_cost_usd"])
        / np.maximum(merged["baseline_cost_usd"], 1e-9)
    )

    keep_cols = [
        "scenario",
        "seed",
        "controller",
        "cost_delta_pct",
        "violation_reduction_pct",
        "severity_reduction_pct",
    ]
    pareto = merged[keep_cols].copy()
    csv_path = out_dir / "cost_safety_pareto.csv"
    pareto.to_csv(csv_path, index=False, float_format="%.6f")

    fig_path = out_dir / "fig_cost_safety_pareto.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in pareto.groupby("controller", sort=True):
        ax.scatter(
            sub["cost_delta_pct"].to_numpy(dtype=float),
            sub["violation_reduction_pct"].to_numpy(dtype=float),
            alpha=0.85,
            label=controller,
        )
    ax.axhline(10.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Cost Delta vs Robust Baseline (%)")
    ax.set_ylabel("Violation Reduction (%)")
    ax.set_title("Cost-Safety Pareto Frontier")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    payload = {
        "rows": int(len(pareto)),
        "csv": str(csv_path),
        "figure": str(fig_path),
    }
    return payload


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build cost-safety Pareto artifact")
    p.add_argument("--main-table", default="reports/publication/dc3s_main_table.csv")
    p.add_argument("--out-dir", default="reports/publication")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    payload = build_cost_safety_pareto(
        main_table_path=Path(args.main_table),
        out_dir=Path(args.out_dir),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
