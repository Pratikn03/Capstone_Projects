"""Summarize the locked battery controller-family comparison for ORIUS.

This script is intentionally conservative. It does not fabricate new replay
rows or rename unsupported controllers into stronger claims. Instead, it
normalizes the real replay-backed battery controllers already present in the
locked publication artifact family and emits:

- machine-readable summary CSV/JSON
- one manuscript-ready LaTeX comparison table
- one manuscript-ready PNG figure

The current lock contains deterministic, fixed-interval robust, adaptive
conformal with repair, and two ORIUS/DC3S variants. Separate weighted or
Mondrian dispatch baselines are not synthesized here because they are not
present as promoted replay-backed controller rows in the locked artifact set.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAIN_TABLE = REPO_ROOT / "reports" / "publication" / "dc3s_main_table.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "publication"
DEFAULT_TABLE_OUT = DEFAULT_OUT_DIR / "tbl_battery_reliability_baselines.tex"
DEFAULT_FIGURE_OUT = DEFAULT_OUT_DIR / "fig_battery_reliability_baselines.png"
DEFAULT_SUMMARY_CSV = DEFAULT_OUT_DIR / "battery_reliability_baselines_summary.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_OUT_DIR / "battery_reliability_baselines_summary.json"
DEFAULT_PAPER_TABLE = REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "tbl_battery_reliability_baselines.tex"
DEFAULT_PAPER_FIGURE = REPO_ROOT / "paper" / "assets" / "figures" / "fig59_battery_reliability_baselines.png"


@dataclass(frozen=True, slots=True)
class ControllerSpec:
    controller: str
    method_id: str
    display_name: str
    family: str
    repair_enabled: bool
    color: str


@dataclass(frozen=True, slots=True)
class BatteryMethodMetrics:
    method_id: str
    display_name: str
    source_controller: str
    family: str
    repair_enabled: bool
    n_runs: int
    true_state_violation_rate: float
    observed_state_violation_rate: float
    hidden_gap_rate: float
    coverage_90: float
    mean_interval_width_mw: float
    repair_rate: float
    expected_cost_usd: float
    expected_cost_musd: float


DEFAULT_CONTROLLER_SPECS: tuple[ControllerSpec, ...] = (
    ControllerSpec(
        controller="dc3s_ftit",
        method_id="orius_repair",
        display_name="ORIUS FTIT",
        family="repair-and-certify",
        repair_enabled=True,
        color="#1b9e77",
    ),
    ControllerSpec(
        controller="dc3s_wrapped",
        method_id="orius_wrapped",
        display_name="ORIUS Wrapped",
        family="repair-and-certify",
        repair_enabled=True,
        color="#66a61e",
    ),
    ControllerSpec(
        controller="aci_conformal",
        method_id="adaptive_conformal_repair",
        display_name="Adaptive Conformal + Repair",
        family="adaptive conformal",
        repair_enabled=True,
        color="#7570b3",
    ),
    ControllerSpec(
        controller="deterministic_lp",
        method_id="deterministic_no_repair",
        display_name="Deterministic LP",
        family="point forecast",
        repair_enabled=False,
        color="#d95f02",
    ),
    ControllerSpec(
        controller="robust_fixed_interval",
        method_id="robust_fixed_interval",
        display_name="Robust Fixed Interval",
        family="static interval",
        repair_enabled=False,
        color="#e7298a",
    ),
)


REQUIRED_COLUMNS = {
    "controller",
    "true_soc_violation_rate",
    "violation_rate",
    "picp_90",
    "mean_interval_width",
    "intervention_rate",
    "expected_cost_usd",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_or_abs(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _ensure_columns(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Locked controller table is missing required columns: {missing}")


def _controller_specs(selected: Iterable[str] | None = None) -> list[ControllerSpec]:
    if selected is None:
        return list(DEFAULT_CONTROLLER_SPECS)
    allowed = {item.strip() for item in selected if item.strip()}
    specs = [spec for spec in DEFAULT_CONTROLLER_SPECS if spec.controller in allowed or spec.method_id in allowed]
    if not specs:
        raise ValueError("No selected controllers matched the supported locked controller-family specs.")
    return specs


def summarize_locked_battery_controller_table(
    frame: pd.DataFrame,
    *,
    controller_specs: Sequence[ControllerSpec] | None = None,
) -> dict[str, Any]:
    """Aggregate the replay-backed battery controller rows into a stable summary."""

    _ensure_columns(frame)
    specs = list(controller_specs or DEFAULT_CONTROLLER_SPECS)
    metrics: list[BatteryMethodMetrics] = []

    for spec in specs:
        subset = frame.loc[frame["controller"] == spec.controller].copy()
        if subset.empty:
            raise ValueError(f"Locked controller table does not contain required controller '{spec.controller}'.")
        true_rate = float(subset["true_soc_violation_rate"].mean())
        observed_rate = float(subset["violation_rate"].mean())
        hidden_gap = float(max(true_rate - observed_rate, 0.0))
        expected_cost = float(subset["expected_cost_usd"].mean())
        metrics.append(
            BatteryMethodMetrics(
                method_id=spec.method_id,
                display_name=spec.display_name,
                source_controller=spec.controller,
                family=spec.family,
                repair_enabled=spec.repair_enabled,
                n_runs=int(len(subset)),
                true_state_violation_rate=true_rate,
                observed_state_violation_rate=observed_rate,
                hidden_gap_rate=hidden_gap,
                coverage_90=float(subset["picp_90"].mean()),
                mean_interval_width_mw=float(subset["mean_interval_width"].mean()),
                repair_rate=float(subset["intervention_rate"].mean()),
                expected_cost_usd=expected_cost,
                expected_cost_musd=float(expected_cost / 1_000_000.0),
            )
        )

    reference = next(item for item in metrics if item.method_id == "orius_repair")
    rows = [asdict(item) for item in metrics]
    deltas_vs_reference = {
        row["method_id"]: {
            "delta_true_state_violation_rate": float(row["true_state_violation_rate"] - reference.true_state_violation_rate),
            "delta_observed_state_violation_rate": float(
                row["observed_state_violation_rate"] - reference.observed_state_violation_rate
            ),
            "delta_coverage_90": float(row["coverage_90"] - reference.coverage_90),
            "delta_mean_interval_width_mw": float(row["mean_interval_width_mw"] - reference.mean_interval_width_mw),
            "delta_repair_rate": float(row["repair_rate"] - reference.repair_rate),
            "delta_expected_cost_usd": float(row["expected_cost_usd"] - reference.expected_cost_usd),
        }
        for row in rows
    }
    return {
        "study": "battery_reliability_baselines",
        "source_family": "locked_battery_controller_table",
        "reference_method": reference.method_id,
        "metrics_by_method": {row["method_id"]: row for row in rows},
        "ordered_methods": [row["method_id"] for row in rows],
        "deltas_vs_reference": deltas_vs_reference,
        "notes": [
            "Summary derived from the real locked battery controller-family artifact reports/publication/dc3s_main_table.csv.",
            "The promoted replay-backed comparison surface currently contains deterministic LP, robust fixed interval, adaptive conformal with repair, and the two ORIUS/DC3S variants.",
            "Separate weighted or Mondrian dispatch controllers are not synthesized here because they are not present as promoted replay-backed rows in the locked artifact family.",
            "observed_state_violation_rate is the controller-side violation_rate emitted by the CPSBench battery replay harness.",
        ],
    }


def _format_pct(value: float, decimals: int = 2) -> str:
    return f"{100.0 * float(value):.{decimals}f}"


def _format_usd_m(value: float) -> str:
    return f"{float(value) / 1_000_000.0:.2f}"


def _write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    rows = [summary["metrics_by_method"][method_id] for method_id in summary["ordered_methods"]]
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")


def _write_summary_json(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_tex_table(summary: dict[str, Any]) -> str:
    rows = [summary["metrics_by_method"][method_id] for method_id in summary["ordered_methods"]]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Locked battery controller-family comparison aggregated from \texttt{reports/publication/dc3s\_main\_table.csv} across six telemetry-fault scenarios and five seeds. The comparison is limited to replay-backed controllers that are actually promoted in the locked artifact family; weighted and Mondrian dispatch rows are not synthesized.}",
        r"\label{tab:battery_reliability_controller_compare}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Controller & True TSVR (\%) & Controller TSVR (\%) & PICP@90 & Width (MW) & Repair (\%) & Cost (\$M) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['display_name']} & "
            f"{_format_pct(row['true_state_violation_rate'])} & "
            f"{_format_pct(row['observed_state_violation_rate'])} & "
            f"{row['coverage_90']:.3f} & "
            f"{row['mean_interval_width_mw']:.1f} & "
            f"{_format_pct(row['repair_rate'])} & "
            f"{_format_usd_m(row['expected_cost_usd'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def _write_tex_table(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_tex_table(summary), encoding="utf-8")


def _render_figure(summary: dict[str, Any], path: Path) -> None:
    rows = [summary["metrics_by_method"][method_id] for method_id in summary["ordered_methods"]]
    specs = {spec.method_id: spec for spec in DEFAULT_CONTROLLER_SPECS}
    labels = [row["display_name"] for row in rows]
    colors = [specs[row["method_id"]].color for row in rows]
    y = list(range(len(rows)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        ("true_state_violation_rate", "True TSVR (%)", lambda x: 100.0 * float(x), "{:.2f}"),
        ("coverage_90", "PICP@90", float, "{:.3f}"),
        ("mean_interval_width_mw", "Mean Interval Width (MW)", float, "{:.1f}"),
        ("repair_rate", "Repair Rate (%)", lambda x: 100.0 * float(x), "{:.2f}"),
    ]
    for ax, (key, title, transform, fmt) in zip(axes.flat, panels, strict=True):
        values = [transform(row[key]) for row in rows]
        ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title)
        ax.set_yticks(y, labels if key in {"true_state_violation_rate", "mean_interval_width_mw"} else [""] * len(y))
        ax.grid(axis="x", alpha=0.3)
        xmax = max(values) if values else 1.0
        if xmax <= 0:
            xmax = 1.0
        ax.set_xlim(0.0, xmax * 1.18)
        for idx, value in enumerate(values):
            ax.text(value + xmax * 0.02, idx, fmt.format(value), va="center", fontsize=9)
    axes[0, 0].invert_yaxis()
    axes[1, 0].invert_yaxis()
    fig.suptitle("Locked Battery Controller-Family Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_battery_reliability_baselines(
    *,
    main_table: Path = DEFAULT_MAIN_TABLE,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    summary_json: Path = DEFAULT_SUMMARY_JSON,
    table_out: Path = DEFAULT_TABLE_OUT,
    figure_out: Path = DEFAULT_FIGURE_OUT,
    paper_table: Path = DEFAULT_PAPER_TABLE,
    paper_figure: Path = DEFAULT_PAPER_FIGURE,
    controller_specs: Sequence[ControllerSpec] | None = None,
) -> dict[str, Any]:
    frame = pd.read_csv(main_table)
    summary = summarize_locked_battery_controller_table(frame, controller_specs=controller_specs)
    summary["source_artifact"] = _repo_relative_or_abs(main_table)
    summary["source_artifact_sha256"] = _sha256(main_table)
    _write_summary_csv(summary, summary_csv)
    _write_summary_json(summary, summary_json)
    _write_tex_table(summary, table_out)
    _render_figure(summary, figure_out)
    _copy(table_out, paper_table)
    _copy(figure_out, paper_figure)
    summary["outputs"] = {
        "summary_csv": _repo_relative_or_abs(summary_csv),
        "summary_json": _repo_relative_or_abs(summary_json),
        "table_tex": _repo_relative_or_abs(table_out),
        "figure_png": _repo_relative_or_abs(figure_out),
        "paper_table_tex": _repo_relative_or_abs(paper_table),
        "paper_figure_png": _repo_relative_or_abs(paper_figure),
    }
    _write_summary_json(summary, summary_json)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-table", type=Path, default=DEFAULT_MAIN_TABLE, help="Locked controller-family CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Publication output directory.")
    parser.add_argument("--summary-csv", type=Path, help="Optional override for summary CSV output.")
    parser.add_argument("--summary-json", type=Path, help="Optional override for summary JSON output.")
    parser.add_argument("--table-out", type=Path, help="Optional override for publication LaTeX table output.")
    parser.add_argument("--figure-out", type=Path, help="Optional override for publication figure output.")
    parser.add_argument("--paper-table", type=Path, default=DEFAULT_PAPER_TABLE, help="Canonical paper table copy.")
    parser.add_argument("--paper-figure", type=Path, default=DEFAULT_PAPER_FIGURE, help="Canonical paper figure copy.")
    parser.add_argument(
        "--controllers",
        nargs="*",
        help="Optional subset of supported controller ids/controller names. Defaults to the promoted locked family.",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the final summary to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir = args.output_dir
    summary_csv = args.summary_csv or (out_dir / "battery_reliability_baselines_summary.csv")
    summary_json = args.summary_json or (out_dir / "battery_reliability_baselines_summary.json")
    table_out = args.table_out or (out_dir / "tbl_battery_reliability_baselines.tex")
    figure_out = args.figure_out or (out_dir / "fig_battery_reliability_baselines.png")
    summary = build_battery_reliability_baselines(
        main_table=args.main_table,
        summary_csv=summary_csv,
        summary_json=summary_json,
        table_out=table_out,
        figure_out=figure_out,
        paper_table=args.paper_table,
        paper_figure=args.paper_figure,
        controller_specs=_controller_specs(args.controllers),
    )
    if args.pretty:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
