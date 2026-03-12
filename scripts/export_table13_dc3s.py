#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SCENARIO_ORDER = [
    "nominal",
    "dropout",
    "delay_jitter",
    "out_of_order",
    "spikes",
    "drift_combo",
]

CONTROLLER_ORDER = [
    "deterministic_lp",
    "robust_fixed_interval",
    "cvar_interval",
    "dc3s_wrapped",
    "dc3s_ftit",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export publication-ready Table 13 from the DC3S main table")
    parser.add_argument("--input", default="reports/publication/dc3s_main_table.csv")
    parser.add_argument("--out-dir", default="reports/publication")
    return parser.parse_args()


def _required_columns() -> list[str]:
    return [
        "scenario",
        "seed",
        "controller",
        "picp_90",
        "mean_interval_width",
        "expected_cost_usd",
        "intervention_rate",
        "violation_rate",
    ]


def _optional_columns(df: pd.DataFrame) -> list[str]:
    return [name for name in ["mae", "rmse", "picp_95"] if name in df.columns]


def _scenario_sort_key(name: str) -> tuple[int, str]:
    try:
        return (SCENARIO_ORDER.index(str(name)), str(name))
    except ValueError:
        return (len(SCENARIO_ORDER), str(name))


def _controller_sort_key(name: str) -> tuple[int, str]:
    try:
        return (CONTROLLER_ORDER.index(str(name)), str(name))
    except ValueError:
        return (len(CONTROLLER_ORDER), str(name))


def _validate_input(df: pd.DataFrame) -> None:
    missing = [col for col in _required_columns() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _format_value(column: str, value: Any, *, latex: bool = False) -> str:
    if pd.isna(value):
        return ""
    if column == "n_seeds":
        return str(int(round(float(value))))
    if column in {"picp_90", "picp_95", "intervention_rate", "violation_rate"}:
        return f"{float(value):.3f}"
    if column == "expected_cost_usd":
        return f"{float(value):.2f}" if latex else f"{float(value):,.2f}"
    if column in {"mean_interval_width", "mae", "rmse"}:
        return f"{float(value):.2f}"
    return str(value)


def _markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "# Table 13: DC3S Controller Summary",
        "",
        "Derived from `reports/publication/dc3s_main_table.csv` by averaging metrics across seeds for each `(scenario, controller)` pair.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.to_dict(orient="records"):
        values = [_format_value(col, row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _escape_latex(text: Any) -> str:
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def _latex_alignment(columns: list[str]) -> str:
    parts = []
    for col in columns:
        if col in {"scenario", "controller"}:
            parts.append("l")
        else:
            parts.append("r")
    return "".join(parts)


def _latex_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        f"\\begin{{tabular}}{{{_latex_alignment(columns)}}}",
        "\\toprule",
        " & ".join(_escape_latex(col) for col in columns) + " \\\\",
        "\\midrule",
    ]
    for row in df.to_dict(orient="records"):
        rendered: list[str] = []
        for col in columns:
            if col in {"scenario", "controller"}:
                rendered.append(_escape_latex(row[col]))
            else:
                rendered.append(_format_value(col, row[col], latex=True))
        lines.append(" & ".join(rendered) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _write_markdown(df: pd.DataFrame, path: Path) -> None:
    path.write_text(_markdown_table(df), encoding="utf-8")


def _write_latex(df: pd.DataFrame, path: Path) -> None:
    path.write_text(_latex_table(df), encoding="utf-8")


def _print_drift_combo_summary(df_summary: pd.DataFrame) -> None:
    subset = df_summary[df_summary["scenario"] == "drift_combo"]
    if subset.empty:
        print("[Table13] drift_combo summary")
        print("  warning: drift_combo rows are missing from the aggregated summary")
        return

    dc3s_rows = subset[subset["controller"].isin(["dc3s_wrapped", "dc3s_ftit"])]
    deterministic = subset[subset["controller"] == "deterministic_lp"]
    robust = subset[subset["controller"] == "robust_fixed_interval"]
    if dc3s_rows.empty or deterministic.empty or robust.empty:
        print("[Table13] drift_combo summary")
        print("  warning: required drift_combo controller rows are missing from the aggregated summary")
        return

    best_dc3s = (
        dc3s_rows.sort_values(["picp_90", "expected_cost_usd"], ascending=[False, True], kind="stable")
        .iloc[0]
    )
    det_picp = float(deterministic.iloc[0]["picp_90"])
    cost_delta = float(best_dc3s["expected_cost_usd"]) - float(robust.iloc[0]["expected_cost_usd"])
    intervention_rate = float(best_dc3s["intervention_rate"])

    print("[Table13] drift_combo summary")
    for _, row in dc3s_rows.sort_values("controller", kind="stable").iterrows():
        print(f"  {row['controller']} picp_90: {float(row['picp_90']):.3f}")
    print(f"  deterministic_lp picp_90: {det_picp:.3f}")
    print(f"  best_dc3s_controller: {best_dc3s['controller']}")
    print(f"  picp_90 delta (best_dc3s - deterministic_lp): {float(best_dc3s['picp_90']) - det_picp:.3f}")
    print(f"  cost delta vs robust_fixed_interval (USD): {cost_delta:.2f}")
    print(f"  {best_dc3s['controller']} intervention_rate: {intervention_rate:.3f}")


def build_table13(input_path: Path, out_dir: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    _validate_input(df)

    optional = _optional_columns(df)
    numeric_cols = ["picp_90", "mean_interval_width", "expected_cost_usd", "intervention_rate", "violation_rate", *optional]
    group_cols = ["scenario", "controller"]

    summary = (
        df.groupby(group_cols, dropna=False)[numeric_cols]
        .mean()
        .reset_index()
    )
    seed_counts = (
        df.groupby(group_cols, dropna=False)["seed"]
        .nunique()
        .rename("n_seeds")
        .reset_index()
    )
    summary = summary.merge(seed_counts, on=group_cols, how="left")

    ordered_cols = [
        "scenario",
        "controller",
        "n_seeds",
        "picp_90",
        *([col for col in ["picp_95"] if col in optional]),
        "mean_interval_width",
        "expected_cost_usd",
        "intervention_rate",
        "violation_rate",
        *([col for col in ["mae", "rmse"] if col in optional]),
    ]
    summary = summary[ordered_cols]
    summary = (
        summary.assign(
            _scenario_order=summary["scenario"].map(_scenario_sort_key),
            _controller_order=summary["controller"].map(_controller_sort_key),
        )
        .sort_values(["_scenario_order", "_controller_order", "scenario", "controller"], kind="stable")
        .drop(columns=["_scenario_order", "_controller_order"])
        .reset_index(drop=True)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "table13_dc3s_summary.csv"
    md_path = out_dir / "table13_dc3s_summary.md"
    tex_path = out_dir / "table13_dc3s_summary.tex"

    summary.to_csv(csv_path, index=False, float_format="%.6f")
    _write_markdown(summary, md_path)
    _write_latex(summary, tex_path)

    omitted_optional = [name for name in ["mae", "rmse", "picp_95"] if name not in optional]
    return {
        "rows": int(len(summary)),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "latex": str(tex_path),
        "optional_columns": optional,
        "omitted_optional_columns": omitted_optional,
        "summary_df": summary,
    }


def main() -> None:
    args = _parse_args()
    payload = build_table13(input_path=Path(args.input), out_dir=Path(args.out_dir))
    if payload["omitted_optional_columns"]:
        print(f"[Table13] optional columns omitted: {payload['omitted_optional_columns']}")
    _print_drift_combo_summary(payload["summary_df"])
    printable = {key: value for key, value in payload.items() if key != "summary_df"}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
