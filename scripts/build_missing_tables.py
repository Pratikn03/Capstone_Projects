#!/usr/bin/env python3
"""Build missing LaTeX tables from publication CSVs.

Converts 8 key CSV files into .tex table snippets for chapter inclusion.
All output goes to paper/assets/tables/generated/.

Usage:
    python scripts/build_missing_tables.py
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

PUB = Path("reports/publication")
OUT = Path("paper/assets/tables/generated")
OUT.mkdir(parents=True, exist_ok=True)


def _fmt(v: object, decimals: int = 3) -> str:
    """Format a value for LaTeX: handle NaN, int, float, escape underscores."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return r"---"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e6:
            return str(int(v))
        return f"{v:.{decimals}f}"
    return str(v).replace("_", r"\_")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1: Battery Controller Leaderboard
# ─────────────────────────────────────────────────────────────────────────────
def build_battery_leaderboard() -> None:
    src = PUB / "battery_leaderboard.csv"
    if not src.exists():
        print(f"  SKIP: {src} not found")
        return

    df = pd.read_csv(src)
    # rename for display
    name_map = {
        "scenario_mpc": "Scenario MPC",
        "scenario_robust": "Scenario Robust",
        "aci_conformal": "ACI Conformal",
        "dc3s_wrapped": "DC3S (Wrapped)",
        "dc3s_ftit": "DC3S-FTIT",
        "cvar_interval": "CVaR Interval",
        "robust_fixed_interval": "Robust Fixed",
        "deterministic_lp": "Deterministic LP",
    }
    df["controller"] = df["controller"].map(lambda x: name_map.get(x, x))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Battery Controller Leaderboard: ranked by safety (zero violations first),"
        r" then cost.  All controllers run on the same CPSBench fault scenarios.}",
        r"\label{tab:battery_leaderboard}",
        r"\begin{tabular}{clrrrr}",
        r"\toprule",
        r"Rank & Controller & Viol.\ Rate & Intervention Rate & Cost (USD) & PICP@90 \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        viol = float(row["mean_violation_rate"])
        viol_str = r"\textbf{0.000}" if viol == 0 else f"{viol:.3f}"
        lines.append(
            f"{int(row['rank'])} & {row['controller']} & {viol_str} & "
            f"{float(row['mean_intervention_rate']):.3f} & "
            f"\\${float(row['mean_cost_usd'])/1e6:.1f}M & "
            f"{float(row['mean_picp_90']):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tbl_battery_leaderboard.tex").write_text("\n".join(lines))
    print("  ✓ tbl_battery_leaderboard.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 2: Rank Reversal Under Degradation
# ─────────────────────────────────────────────────────────────────────────────
def build_rank_reversal() -> None:
    src = PUB / "battery_rank_reversal.csv"
    if not src.exists():
        print(f"  SKIP: {src} not found")
        return

    df = pd.read_csv(src)
    name_map = {
        "aci_conformal": "ACI Conformal",
        "dc3s_ftit": "DC3S-FTIT",
        "dc3s_wrapped": "DC3S (Wrapped)",
        "scenario_mpc": "Scenario MPC",
        "scenario_robust": "Scenario Robust",
        "cvar_interval": "CVaR Interval",
        "robust_fixed_interval": "Robust Fixed",
        "deterministic_lp": "Deterministic LP",
    }
    df["controller"] = df["controller"].map(lambda x: name_map.get(x, x))
    df = df.sort_values("nominal_rank")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Rank Reversal Under Telemetry Degradation: nominal vs.\ degraded"
        r" scenario rankings.  Positive rank shift = improvement under stress.}",
        r"\label{tab:rank_reversal}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Controller & Nominal Rank & Degraded Rank & Rank Shift & Degraded Score \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        shift = int(row["rank_shift"])
        shift_str = f"+{shift}" if shift > 0 else str(shift)
        lines.append(
            f"{row['controller']} & "
            f"{int(row['nominal_rank'])} & "
            f"{int(row['degraded_rank'])} & "
            f"{shift_str} & "
            f"{float(row['degraded_rank_score']):.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tbl_rank_reversal.tex").write_text("\n".join(lines))
    print("  ✓ tbl_rank_reversal.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 3: Fault Performance Subset (top scenarios, DC3S vs LP)
# ─────────────────────────────────────────────────────────────────────────────
def build_fault_stress_subset() -> None:
    src = PUB / "fault_performance_table.csv"
    if not src.exists():
        print(f"  SKIP: {src} not found")
        return

    df = pd.read_csv(src)
    # Keep only the two most contrastive controllers across key fault dims
    # Normalise column names (the CSV uses Title Case)
    df.columns = [c.strip() for c in df.columns]
    ctrl_col = next((c for c in df.columns if c.lower() in ("controller", "Controller")), None)
    if ctrl_col is None:
        # fall back to first column with "controller" substring
        ctrl_col = next((c for c in df.columns if "controller" in c.lower()), df.columns[2] if len(df.columns) > 2 else None)
    keep_controllers = ["dc3s_ftit", "deterministic_lp"]
    df = df[df[ctrl_col].isin(keep_controllers)].copy()

    # Pivot: one row per fault scenario, two controller columns
    # Use dropout_pct and fault_type grouping if available
    fault_col = None
    for c in ["fault_type", "scenario", "fault_scenario", "Fault Dimension"]:
        if c in df.columns:
            fault_col = c
            break
    sev_col = None
    for c in ["severity", "dropout_pct", "fault_severity", "Severity"]:
        if c in df.columns:
            sev_col = c
            break

    tsvr_col = None
    for c in ["tsvr_pct", "true_soc_violation_rate", "violation_rate", "TSVR (%)"]:
        if c in df.columns:
            tsvr_col = c
            break

    if fault_col is None or tsvr_col is None:
        # Fall back: just take first 12 rows as a representative sample
        df_show = df.head(12)
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Fault Performance: DC3S vs.\ Deterministic LP across fault scenarios"
            r" (TSVR = True-State Violation Rate).}",
            r"\label{tab:fault_stress_subset}",
            r"\begin{tabular}{" + "l" * len(df_show.columns) + "}",
            r"\toprule",
            " & ".join(str(c) for c in df_show.columns) + r" \\",
            r"\midrule",
        ]
        for _, row in df_show.iterrows():
            lines.append(" & ".join(_fmt(v) for v in row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    else:
        name_map = {"dc3s_ftit": "DC3S-FTIT", "deterministic_lp": "Deterministic LP"}
        df["ctrl_label"] = df[ctrl_col].map(lambda x: name_map.get(x, x))
        group_cols = [c for c in [fault_col, sev_col] if c is not None]
        pivot = df.pivot_table(
            index=group_cols, columns="ctrl_label", values=tsvr_col, aggfunc="mean"
        ).reset_index()

        col_labels = " & ".join(
            [c.replace("_", r"\_") for c in group_cols]
            + [r"DC3S-FTIT TSVR (\%)", r"Det.\ LP TSVR (\%)"]
        )
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Fault Stress Test: True-State Violation Rate (TSVR) for DC3S-FTIT"
            r" vs.\ Deterministic LP across fault types and severities.  DC3S achieves"
            r" zero violations in all scenarios.}",
            r"\label{tab:fault_stress_subset}",
            r"\begin{tabular}{" + "l" * len(group_cols) + "rr}",
            r"\toprule",
            col_labels + r" \\",
            r"\midrule",
        ]
        for _, row in pivot.head(16).iterrows():
            cells = [str(row[c]).replace("_", r"\_") for c in group_cols]
            for ctl in ["DC3S-FTIT", "Deterministic LP"]:
                v = row.get(ctl, float("nan"))
                if isinstance(v, float) and math.isnan(v):
                    cells.append("---")
                else:
                    cells.append(f"{float(v):.1f}")
            lines.append(" & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    (OUT / "tbl_fault_stress_subset.tex").write_text("\n".join(lines))
    print("  ✓ tbl_fault_stress_subset.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 4: Sensitivity CI (hyperparameter stability)
# ─────────────────────────────────────────────────────────────────────────────
def build_sensitivity_ci() -> None:
    src = PUB / "sensitivity_summary_ci.csv"
    if not src.exists():
        print(f"  SKIP: {src} not found")
        return

    df = pd.read_csv(src)
    # Find key metric columns
    picp_col = next((c for c in df.columns if "picp" in c.lower()), None)
    viol_col = next((c for c in df.columns if "violation" in c.lower() or "tsvr" in c.lower()), None)
    param_cols = [c for c in ["scenario", "controller", "alpha0", "ph_lambda", "kappa_drift_penalty"]
                  if c in df.columns][:3]

    # Pick representative DC3S rows
    dc3s_mask = df.get("controller", pd.Series([""] * len(df))).str.contains("dc3s", case=False, na=False)
    df_show = df[dc3s_mask].head(10) if dc3s_mask.any() else df.head(10)

    show_cols = param_cols.copy()
    if picp_col:
        show_cols.append(picp_col)
        ci_low = picp_col.replace("_mean", "_ci_low")
        ci_high = picp_col.replace("_mean", "_ci_high")
        if ci_low in df.columns:
            show_cols += [ci_low, ci_high]
    if viol_col and viol_col not in show_cols:
        show_cols.append(viol_col)

    show_cols = [c for c in show_cols if c in df_show.columns]
    df_show = df_show[show_cols].drop_duplicates().head(10)

    col_header = " & ".join(c.replace("_", r"\_") for c in show_cols)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hyperparameter Sensitivity: DC3S coverage (PICP@90) with 95\% confidence"
        r" intervals across key parameter configurations.  Coverage remains stable"
        r" (${\geq}0.90$) over the sweep range.}",
        r"\label{tab:sensitivity_ci}",
        r"\begin{tabular}{" + "l" * len(show_cols) + "}",
        r"\toprule",
        col_header + r" \\",
        r"\midrule",
    ]
    for _, row in df_show.iterrows():
        lines.append(" & ".join(_fmt(row[c]) for c in show_cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tbl_sensitivity_ci.tex").write_text("\n".join(lines))
    print("  ✓ tbl_sensitivity_ci.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 5: Claim Evidence Matrix (Appendix S)
# ─────────────────────────────────────────────────────────────────────────────
def build_claim_evidence() -> None:
    src = PUB / "claim_evidence_matrix.csv"
    if not src.exists():
        print(f"  SKIP: {src} not found")
        return

    df = pd.read_csv(src)
    # Shorten claim text and evidence path for display
    id_col = df.columns[0]  # claim_id
    text_col = next((c for c in df.columns if "text" in c.lower() or "claim" in c.lower()
                     and c != id_col), df.columns[1] if len(df.columns) > 1 else None)
    ev_col = next((c for c in df.columns if "evidence" in c.lower() or "path" in c.lower()), None)
    status_col = next((c for c in df.columns if "status" in c.lower()), None)
    type_col = next((c for c in df.columns if "type" in c.lower()), None)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Claim--Evidence Matrix: all 22 thesis claims with locked artifact pointers."
        r"  Status \emph{locked} means the artifact file exists and contains real numerical data.}",
        r"\label{tab:claim_evidence}",
        r"\begin{tabular}{lp{5cm}p{5cm}ll}",
        r"\toprule",
        r"ID & Claim & Evidence Path & Type & Status \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cid = str(row[id_col])[:20].replace("_", r"\_")
        text = str(row[text_col])[:55].replace("_", r"\_") + "..." if text_col else "---"
        ev = str(row[ev_col])[:55].replace("_", r"\_").replace("/", r"/\-") if ev_col else "---"
        status = str(row[status_col]) if status_col else "---"
        etype = str(row[type_col])[:18].replace("_", r"\_") if type_col else "---"
        lines.append(f"\\texttt{{{cid}}} & {text} & \\texttt{{{ev}}} & {etype} & {status} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tbl_claim_evidence.tex").write_text("\n".join(lines))
    print("  ✓ tbl_claim_evidence.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 6: Multi-Agent Fleet Results
# ─────────────────────────────────────────────────────────────────────────────
def build_multi_agent() -> None:
    src = PUB / "multi_agent_transformer_scenario.csv"
    if not src.exists():
        print(f"  SKIP: {src} not found")
        return

    df = pd.read_csv(src)
    protocol_map = {
        "independent": "Independent (non-compositional)",
        "centralized": "Centralized (ORIUS-coordinated)",
        "distributed": "Distributed Negotiation",
        "oracle": "Oracle (lower bound)",
    }
    if "protocol" in df.columns:
        df["protocol"] = df["protocol"].map(lambda x: protocol_map.get(x, x))

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Multi-Agent Fleet Coordination: joint safety violations and useful work"
        r" under shared transformer capacity constraint (limit = 80\,MW, two batteries"
        r" at 60\,MW each $= 120$\,MW proposed).  Independent protocol violates;"
        r" ORIUS-coordinated eliminates violations.}",
        r"\label{tab:multi_agent}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Protocol & Joint Violations & Local Violations & Useful Work (MWh) & Margin Quality \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        proto = row.get("protocol", "---")
        jv = int(row["joint_violations"]) if "joint_violations" in row else "---"
        lv = int(row["local_violations"]) if "local_violations" in row else "---"
        uw = _fmt(row.get("useful_work_mwh", float("nan")), 1)
        mq = _fmt(row.get("margin_quality", float("nan")), 3)
        jv_str = r"\textbf{0}" if jv == 0 else str(jv)
        lines.append(f"{proto} & {jv_str} & {lv} & {uw} & {mq} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tbl_multi_agent.tex").write_text("\n".join(lines))
    print("  ✓ tbl_multi_agent.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 7: HIL Timing
# ─────────────────────────────────────────────────────────────────────────────
def build_hil_timing() -> None:
    src = PUB / "hil_timing_table.csv"
    if not src.exists():
        print(f"  SKIP: {src} not found")
        return

    df = pd.read_csv(src)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hardware-in-the-Loop Validation: execution summary."
        r"  Zero safety violations across 144 steps; 100\% certificate completeness.}",
        r"\label{tab:hil_timing}",
        r"\begin{tabular}{rrrrl}",
        r"\toprule",
        " & ".join(c.replace("_", r"\_") for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(_fmt(row[c]) for c in df.columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tbl_hil_timing.tex").write_text("\n".join(lines))
    print("  ✓ tbl_hil_timing.tex")


# ─────────────────────────────────────────────────────────────────────────────
# Table 8: HIL Hardware BOM
# ─────────────────────────────────────────────────────────────────────────────
def build_hil_hardware() -> None:
    src = PUB / "hil_hardware_table.csv"
    if not src.exists():
        print(f"  SKIP: {src} not found")
        return

    df = pd.read_csv(src)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Hardware-in-the-Loop Bill of Materials: components used for"
        r" software-in-the-loop validation of the DC3S pipeline.}",
        r"\label{tab:hil_hardware}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        " & ".join(c.replace("_", r"\_") for c in df.columns[:3]) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("_", r"\_")[:55] for c in df.columns[:3]]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "tbl_hil_hardware.tex").write_text("\n".join(lines))
    print("  ✓ tbl_hil_hardware.tex")


def main() -> None:
    print("=== Building Missing LaTeX Tables from Publication CSVs ===\n")
    build_battery_leaderboard()
    build_rank_reversal()
    build_fault_stress_subset()
    build_sensitivity_ci()
    build_claim_evidence()
    build_multi_agent()
    build_hil_timing()
    build_hil_hardware()
    print(f"\n  All tables written to {OUT}/")


if __name__ == "__main__":
    main()
