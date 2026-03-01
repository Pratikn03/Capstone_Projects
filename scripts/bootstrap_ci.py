#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from gridpulse.evaluation.stats import bootstrap_ci


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
    "aci_conformal",
    "scenario_robust",
]


def _parse_csv_list(raw: str) -> list[str]:
    items = [item.strip() for item in raw.split(",")]
    return [item for item in items if item]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute grouped bootstrap confidence intervals for the CPSBench main table"
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default="reports/publication/dc3s_main_table.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/publication",
        help="Directory for the CI summary outputs",
    )
    parser.add_argument(
        "--group-cols",
        default="scenario,controller",
        help="Comma-separated grouping columns",
    )
    parser.add_argument(
        "--metrics",
        default="picp_90,mean_interval_width,expected_cost_usd,intervention_rate,violation_rate",
        help="Comma-separated metric columns to summarize",
    )
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _scenario_sort_key(name: Any) -> tuple[int, str]:
    value = str(name)
    try:
        return (SCENARIO_ORDER.index(value), value)
    except ValueError:
        return (len(SCENARIO_ORDER), value)


def _controller_sort_key(name: Any) -> tuple[int, str]:
    value = str(name)
    try:
        return (CONTROLLER_ORDER.index(value), value)
    except ValueError:
        return (len(CONTROLLER_ORDER), value)


def _validate_inputs(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    metrics: list[str],
    confidence: float,
    n_bootstrap: int,
) -> None:
    if not group_cols:
        raise ValueError("group-cols must contain at least one column")
    if not metrics:
        raise ValueError("metrics must contain at least one column")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")
    if n_bootstrap < 1:
        raise ValueError("n-bootstrap must be >= 1")

    missing_group = [col for col in group_cols if col not in df.columns]
    if missing_group:
        raise ValueError(f"Missing group columns: {missing_group}")

    missing_metrics = [col for col in metrics if col not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing metric columns: {missing_metrics}")


def _format_value(column: str, value: Any, *, latex: bool = False) -> str:
    if pd.isna(value):
        return ""
    if "_cost_" in column or "cost" in column or "width" in column:
        return f"{float(value):.2f}" if latex else f"{float(value):,.2f}"
    if column.endswith("_mean") or column.endswith("_ci_low") or column.endswith("_ci_high"):
        return f"{float(value):.3f}"
    return str(value)


def _escape_latex(text: Any) -> str:
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def _sort_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    sort_cols: list[str] = []
    ordered = df.copy()

    if "scenario" in group_cols:
        ordered["_scenario_order"] = ordered["scenario"].map(_scenario_sort_key)
        sort_cols.append("_scenario_order")
    if "controller" in group_cols:
        ordered["_controller_order"] = ordered["controller"].map(_controller_sort_key)
        sort_cols.append("_controller_order")

    sort_cols.extend(group_cols)
    ordered = ordered.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    drop_cols = [col for col in ["_scenario_order", "_controller_order"] if col in ordered.columns]
    return ordered.drop(columns=drop_cols)


def _markdown_table(df: pd.DataFrame, group_cols: list[str]) -> str:
    columns = list(df.columns)
    group_desc = ", ".join(group_cols)
    lines = [
        "# CPSBench Bootstrap Confidence Intervals",
        "",
        f"Derived from `reports/publication/dc3s_main_table.csv` by grouping on `({group_desc})` and bootstrapping group means across rows.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.to_dict(orient="records"):
        values = []
        for col in columns:
            if col in group_cols:
                values.append(str(row[col]))
            else:
                values.append(_format_value(col, row[col]))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _latex_alignment(columns: list[str], group_cols: list[str]) -> str:
    return "".join("l" if col in group_cols else "r" for col in columns)


def _latex_table(df: pd.DataFrame, group_cols: list[str]) -> str:
    columns = list(df.columns)
    lines = [
        f"\\begin{{tabular}}{{{_latex_alignment(columns, group_cols)}}}",
        "\\toprule",
        " & ".join(_escape_latex(col) for col in columns) + " \\\\",
        "\\midrule",
    ]
    for row in df.to_dict(orient="records"):
        rendered: list[str] = []
        for col in columns:
            if col in group_cols:
                rendered.append(_escape_latex(row[col]))
            else:
                rendered.append(_format_value(col, row[col], latex=True))
        lines.append(" & ".join(rendered) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def compute_ci_summary_df(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    metrics: list[str],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    _validate_inputs(
        df,
        group_cols=group_cols,
        metrics=metrics,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
    )

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False, sort=False)
    for group_key, group_df in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_cols, group_key))
        for metric in metrics:
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            if values.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_ci_low"] = np.nan
                row[f"{metric}_ci_high"] = np.nan
                continue
            stats = bootstrap_ci(
                values.to_numpy(dtype=float),
                statistic=np.mean,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            row[f"{metric}_mean"] = stats["point_estimate"]
            row[f"{metric}_ci_low"] = stats["ci_lower"]
            row[f"{metric}_ci_high"] = stats["ci_upper"]
        rows.append(row)

    summary = pd.DataFrame(rows)
    ordered_cols = list(group_cols)
    for metric in metrics:
        ordered_cols.extend(
            [f"{metric}_mean", f"{metric}_ci_low", f"{metric}_ci_high"]
        )
    summary = summary[ordered_cols]
    return _sort_summary(summary, group_cols)


def write_ci_outputs(
    summary: pd.DataFrame,
    *,
    out_dir: Path,
    output_stem: str,
    group_cols: list[str],
    title: str,
    provenance: str,
    include_latex: bool = True,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{output_stem}.csv"
    md_path = out_dir / f"{output_stem}.md"

    summary.to_csv(csv_path, index=False, float_format="%.6f")
    columns = list(summary.columns)
    md_lines = [
        title,
        "",
        provenance,
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in summary.to_dict(orient="records"):
        values = []
        for col in columns:
            if col in group_cols:
                values.append(str(row[col]))
            else:
                values.append(_format_value(col, row[col]))
        md_lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    paths = {
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
    if include_latex:
        tex_path = out_dir / f"{output_stem}.tex"
        tex_path.write_text(_latex_table(summary, group_cols), encoding="utf-8")
        paths["latex"] = str(tex_path)
    return paths


def build_ci_summary(
    *,
    input_path: Path,
    out_dir: Path,
    group_cols: list[str],
    metrics: list[str],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    summary = compute_ci_summary_df(
        df,
        group_cols=group_cols,
        metrics=metrics,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    paths = write_ci_outputs(
        summary,
        out_dir=out_dir,
        output_stem="dc3s_main_table_ci",
        group_cols=group_cols,
        title="# CPSBench Bootstrap Confidence Intervals",
        provenance=(
            f"Derived from `reports/publication/dc3s_main_table.csv` by grouping on "
            f"`({', '.join(group_cols)})` and bootstrapping group means across rows."
        ),
        include_latex=True,
    )
    return {
        "rows": int(len(summary)),
        **paths,
        "group_cols": group_cols,
        "metrics": metrics,
        "summary_df": summary,
    }


def main() -> None:
    args = _parse_args()
    payload = build_ci_summary(
        input_path=Path(args.input_path),
        out_dir=Path(args.out_dir),
        group_cols=_parse_csv_list(args.group_cols),
        metrics=_parse_csv_list(args.metrics),
        n_bootstrap=int(args.n_bootstrap),
        confidence=float(args.confidence),
        seed=int(args.seed),
    )
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "summary_df"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
