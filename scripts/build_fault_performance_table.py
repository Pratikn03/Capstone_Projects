#!/usr/bin/env python3
"""Build fault_performance_table.csv from cpsbench_merged_sweep.csv for thesis ch21.

Reads the raw sweep CSV and produces a compact, thesis-ready table showing
violation rate, violation severity (p95), intervention rate, and cost delta
per fault dimension × severity × controller.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fault performance table")
    parser.add_argument("--in-csv", default="reports/publication/cpsbench_merged_sweep.csv")
    parser.add_argument("--out-dir", default="reports/publication")
    return parser.parse_args()


def build_table(in_csv: Path, out_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(in_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    key_controllers = [
        "deterministic_lp",
        "robust_fixed_interval",
        "dc3s_wrapped",
        "dc3s_ftit",
    ]
    df = df[df["controller"].isin(key_controllers)].copy()

    sev_col = (
        "true_soc_violation_severity_p95_mwh"
        if "true_soc_violation_severity_p95_mwh" in df.columns
        else "true_soc_violation_severity_p95"
    )

    agg = (
        df.groupby(["fault_dimension", "severity", "controller"], as_index=False)
        .agg(
            tsvr_mean=("true_soc_violation_rate", "mean"),
            tsvr_std=("true_soc_violation_rate", "std"),
            sev_p95_mean=(sev_col, "mean"),
            ir_mean=("intervention_rate", "mean"),
            cost_delta_pct_mean=("cost_delta_pct", lambda x: np.nanmean(x)),
            n_episodes=("seed", "count"),
        )
        .sort_values(["fault_dimension", "severity", "controller"])
        .reset_index(drop=True)
    )
    agg["tsvr_pct"] = (agg["tsvr_mean"] * 100).round(2)
    agg["ir_pct"] = (agg["ir_mean"] * 100).round(2)
    agg["sev_p95_mwh"] = agg["sev_p95_mean"].round(3)
    agg["cost_delta"] = agg["cost_delta_pct_mean"].round(2)

    table = agg[
        [
            "fault_dimension",
            "severity",
            "controller",
            "tsvr_pct",
            "sev_p95_mwh",
            "ir_pct",
            "cost_delta",
            "n_episodes",
        ]
    ].copy()
    table.columns = [
        "Fault Dimension",
        "Severity",
        "Controller",
        "TSVR (%)",
        "Sev P95 (MWh)",
        "IR (%)",
        "Cost Δ (%)",
        "N",
    ]

    csv_path = out_dir / "fault_performance_table.csv"
    table.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Table -> {csv_path} ({len(table)} rows)")

    pivot = agg.pivot_table(
        index=["fault_dimension", "severity"],
        columns="controller",
        values="tsvr_pct",
        aggfunc="first",
    ).reset_index()
    pivot_path = out_dir / "fault_performance_pivot.csv"
    pivot.to_csv(pivot_path, index=False, float_format="%.2f")
    print(f"  Pivot -> {pivot_path}")

    summary = {
        "total_rows": int(len(table)),
        "fault_dimensions": sorted(table["Fault Dimension"].unique().tolist()),
        "controllers": sorted(table["Controller"].unique().tolist()),
        "dc3s_wrapped_max_tsvr": float(table.loc[table["Controller"] == "dc3s_wrapped", "TSVR (%)"].max())
        if "dc3s_wrapped" in table["Controller"].values
        else None,
        "dc3s_ftit_max_tsvr": float(table.loc[table["Controller"] == "dc3s_ftit", "TSVR (%)"].max())
        if "dc3s_ftit" in table["Controller"].values
        else None,
        "det_lp_max_tsvr": float(table.loc[table["Controller"] == "deterministic_lp", "TSVR (%)"].max())
        if "deterministic_lp" in table["Controller"].values
        else None,
    }
    summary_path = out_dir / "fault_performance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Summary -> {summary_path}")

    return table


def main() -> None:
    args = _parse_args()
    build_table(Path(args.in_csv), Path(args.out_dir))


if __name__ == "__main__":
    main()
