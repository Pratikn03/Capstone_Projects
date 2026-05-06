#!/usr/bin/env python3
"""Build tbl04 from existing cross_region_transfer.csv data."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd

df = pd.read_csv(REPO / "reports/publication/cross_region_transfer.csv")

# Build aggregated tbl04 from drift_combo + nominal + dropout scenarios
rows = []
for scenario in ["drift_combo", "nominal", "dropout"]:
    sub = df[df["scenario"] == scenario]
    for region in ["DE", "US"]:
        rsub = sub[sub["region"] == region]
        if rsub.empty:
            continue
        if scenario == "drift_combo":
            case = f"{region}_drift_combo"
        elif scenario == "nominal":
            case = f"{region}_nominal_baseline"
        else:
            case = f"{region}_{scenario}"

        agg = rsub.groupby("controller", as_index=False).agg(
            picp_90=("picp_90", "mean"),
            mean_width=("mean_interval_width", "mean"),
            true_soc_violation_rate=("true_soc_violation_rate", "mean"),
            true_soc_violation_severity_p95_mwh=("true_soc_violation_severity_p95_mwh", "mean"),
            cost_delta_pct=("cost_delta_pct", "mean"),
        )
        for _, r in agg.iterrows():
            rows.append(
                {
                    "transfer_case": case,
                    "source_artifact": r["controller"],
                    "picp_90": r["picp_90"],
                    "mean_width": r["mean_width"],
                    "true_soc_violation_rate": r["true_soc_violation_rate"],
                    "true_soc_violation_severity_p95_mwh": r["true_soc_violation_severity_p95_mwh"],
                    "cost_delta_pct": r["cost_delta_pct"],
                }
            )

tbl = pd.DataFrame(rows)
print(tbl.to_string(index=False))
print(f"\nTotal rows: {len(tbl)}")

# Save CSV for paper
csv_path = REPO / "paper/assets/tables/tbl04_transfer_stress.csv"
tbl.to_csv(csv_path, index=False, float_format="%.6f")
print(f"CSV -> {csv_path}")

# Update reports/publication files
full = df[df["scenario"].isin(["drift_combo", "nominal", "dropout"])].copy()
cols = [
    "scenario",
    "seed",
    "controller",
    "picp_90",
    "mean_interval_width",
    "true_soc_violation_rate",
    "true_soc_violation_severity_p95_mwh",
    "cost_delta_pct",
    "region",
]
full_out = full[cols].copy()
full_out.rename(
    columns={
        "scenario": "transfer_case",
        "controller": "source_artifact",
        "mean_interval_width": "mean_width",
    },
    inplace=True,
)
full_out["transfer_case"] = full_out["region"] + "_" + full_out["transfer_case"]
full_out.drop(columns=["region"], inplace=True)
full_out.to_csv(REPO / "reports/publication/transfer_stress.csv", index=False, float_format="%.6f")
full_out.to_csv(REPO / "reports/publication/table5_transfer.csv", index=False, float_format="%.6f")
print("Updated reports/publication/transfer_stress.csv + table5_transfer.csv")

# Build LaTeX
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
for _, row in tbl.iterrows():
    case = str(row["transfer_case"]).replace("_", r"\_")
    ctrl = str(row["source_artifact"]).replace("_", r"\_")
    picp = f"{float(row['picp_90']):.3f}"
    width = f"{float(row['mean_width']):.1f}"
    viol = f"{float(row['true_soc_violation_rate']):.4f}"
    sev = f"{float(row['true_soc_violation_severity_p95_mwh']):.3f}"
    cost = f"{float(row['cost_delta_pct']):.3f}"
    lines.append(f"{case} & {ctrl} & {picp} & {width} & {viol} & {sev} & {cost} \\\\")
lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
    ]
)

tex_path = REPO / "paper/assets/tables/generated/tbl04_transfer_stress.tex"
tex_path.write_text("\n".join(lines), encoding="utf-8")
print(f"LaTeX -> {tex_path}")
print("\nDone!")
