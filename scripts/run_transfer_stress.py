#!/usr/bin/env python3
"""Run the transfer stress experiment and generate tbl04 data."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from orius.cpsbench_iot.runner import run_single


def run_transfer_stress(
    seeds: list[int] = [11, 22, 33],
    horizon: int = 96,
) -> pd.DataFrame:
    transfer_cases: dict[str, dict[str, float]] = {
        "DE_to_US_no_retrain": {
            "load_scale": 1.18,
            "renewables_scale": 0.82,
            "seasonal_shift_hours": 6,
            "load_bias_mw": 600.0,
        },
        "US_to_DE_no_retrain": {
            "load_scale": 0.86,
            "renewables_scale": 1.14,
            "seasonal_shift_hours": -6,
            "load_bias_mw": -450.0,
        },
        "DE_season_shift": {
            "seasonal_shift_hours": 12,
            "load_scale": 1.04,
            "renewables_scale": 0.94,
        },
        "US_season_shift": {
            "seasonal_shift_hours": -12,
            "load_scale": 0.98,
            "renewables_scale": 1.05,
        },
    }

    sweep_specs: list[tuple[str, str, float | str, dict[str, Any]]] = []
    sweep_specs.append(("nominal", "nominal", 0.0, {}))
    for p in (0.0, 0.10, 0.20, 0.30):
        sweep_specs.append(("dropout", "dropout", p, {"dropout_rate": float(p), "soc_dropout_prob": float(p)}))
    for p in (0.0, 0.10, 0.20):
        sweep_specs.append(("stale", "nominal", p, {"soc_stale_prob": float(p)}))
    sweep_specs.append(("delay", "delay_jitter", 0, {"delay_seconds": 0.0, "delay_rate": 0.0, "soc_stale_prob": 0.0}))
    sweep_specs.append(("delay", "delay_jitter", "high", {"delay_seconds": 15.0, "delay_rate": 0.50, "soc_stale_prob": 0.35}))

    total = len(transfer_cases) * len(sweep_specs) * len(seeds)
    print(f"Running {total} transfer stress simulations ({len(transfer_cases)} cases × {len(sweep_specs)} sweeps × {len(seeds)} seeds)")

    rows: list[dict[str, Any]] = []
    done = 0
    t0 = time.time()

    for case_name, case_overrides in transfer_cases.items():
        for sweep_type, scenario, level, sweep_overrides in sweep_specs:
            overrides = {**case_overrides, **sweep_overrides}
            for seed in seeds:
                done += 1
                elapsed = time.time() - t0
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                print(f"  [{done}/{total}] case={case_name} sweep={sweep_type}({level}) seed={seed} (ETA {eta:.0f}s)", flush=True)

                payload = run_single(
                    scenario=scenario,
                    seed=int(seed),
                    horizon=int(horizon),
                    fault_overrides=overrides,
                )
                for row in payload["main_rows"]:
                    rows.append({
                        "transfer_case": case_name,
                        "sweep_type": sweep_type,
                        "sweep_value": level,
                        "scenario": scenario,
                        "seed": int(seed),
                        "controller": row.get("controller"),
                        "picp_90": row.get("picp_90"),
                        "mean_width": row.get("mean_interval_width"),
                        "true_soc_violation_rate": row.get("true_soc_violation_rate"),
                        "true_soc_violation_severity_p95_mwh": row.get(
                            "true_soc_violation_severity_p95_mwh",
                            row.get("true_soc_violation_severity_p95"),
                        ),
                        "cost_delta_pct": row.get("cost_delta_pct"),
                    })

    return pd.DataFrame(rows)


def build_tbl04(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transfer stress results into the tbl04 format."""
    agg = (
        df.groupby(["transfer_case", "controller"], as_index=False)
        .agg(
            picp_90=("picp_90", "mean"),
            mean_width=("mean_width", "mean"),
            true_soc_violation_rate=("true_soc_violation_rate", "mean"),
            true_soc_violation_severity_p95_mwh=("true_soc_violation_severity_p95_mwh", "mean"),
            cost_delta_pct=("cost_delta_pct", "mean"),
        )
        .rename(columns={"controller": "source_artifact"})
        .sort_values(["transfer_case", "source_artifact"])
    )
    return agg


def build_tex(tbl04: pd.DataFrame) -> str:
    """Generate LaTeX for tbl04."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Transfer stress outcomes across DE/US and seasonal shifts.}",
        r"\label{tab:TBL04_TRANSFER_STRESS}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Transfer Case & Controller & PICP@90 & Mean Width & Viol.\ Rate & Severity P95 (MWh) & Cost $\Delta$\% \\",
        r"\midrule",
    ]
    for _, row in tbl04.iterrows():
        case = str(row["transfer_case"]).replace("_", r"\_")
        ctrl = str(row["source_artifact"]).replace("_", r"\_")
        picp = f'{float(row["picp_90"]):.3f}'
        width = f'{float(row["mean_width"]):.1f}'
        viol = f'{float(row["true_soc_violation_rate"]):.4f}'
        sev = f'{float(row["true_soc_violation_severity_p95_mwh"]):.3f}'
        cost = f'{float(row["cost_delta_pct"]):.3f}'
        lines.append(f"{case} & {ctrl} & {picp} & {width} & {viol} & {sev} & {cost} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    out_dir = REPO_ROOT / "reports" / "publication"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Transfer Stress Experiment")
    print("=" * 60)

    df = run_transfer_stress()
    print(f"\nTotal rows: {len(df)}")

    # Save full results
    full_path = out_dir / "transfer_stress.csv"
    df.to_csv(full_path, index=False, float_format="%.6f")
    print(f"Full results → {full_path}")

    alias_path = out_dir / "table5_transfer.csv"
    df.to_csv(alias_path, index=False, float_format="%.6f")

    # Build aggregated tbl04
    tbl04 = build_tbl04(df)
    print(f"\nAggregated tbl04 ({len(tbl04)} rows):")
    print(tbl04.to_string(index=False))

    # Save CSV
    csv_path = REPO_ROOT / "paper" / "assets" / "tables" / "tbl04_transfer_stress.csv"
    tbl04.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\nCSV → {csv_path}")

    # Save LaTeX
    tex_path = REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "tbl04_transfer_stress.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_content = build_tex(tbl04)
    tex_path.write_text(tex_content, encoding="utf-8")
    print(f"LaTeX → {tex_path}")

    # Save JSON summary
    summary = {
        "total_simulations": len(df),
        "transfer_cases": sorted(df["transfer_case"].unique().tolist()),
        "controllers": sorted(df["controller"].unique().tolist()),
        "seeds": sorted(df["seed"].unique().tolist()),
    }
    summary_path = out_dir / "transfer_stress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary → {summary_path}")

    print("\n✓ Transfer stress experiment complete")


if __name__ == "__main__":
    main()
