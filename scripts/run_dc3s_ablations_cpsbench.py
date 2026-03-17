#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.cpsbench_iot.baselines import dc3s_wrapped_dispatch
from orius.cpsbench_iot.metrics import compute_all_metrics
from orius.cpsbench_iot.scenarios import FAULT_COLUMNS, generate_episode


POLICY_ORDER = [
    "dc3s_no_wt",
    "dc3s_no_drift",
    "dc3s_linear",
    "dc3s_kappa",
]

VARIANT_ORDER = [
    ("no_wt", "dc3s_no_wt"),
    ("no_drift", "dc3s_no_drift"),
    ("linear", "dc3s_linear"),
    ("kappa", "dc3s_kappa"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused DC3S CPSBench ablations for drift_combo")
    parser.add_argument("--scenario", default="drift_combo")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--out-dir", default="reports/publication")
    return parser.parse_args()


def _to_telemetry_events(x_obs: pd.DataFrame, event_log: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx in range(len(x_obs)):
        payload: dict[str, Any] = {
            "ts_utc": pd.to_datetime(event_log.loc[idx, "arrived_timestamp"], utc=True).isoformat(),
            "device_id": "bench-device",
            "zone_id": "DE",
            "load_mw": float(x_obs.loc[idx, "load_mw"]),
            "renewables_mw": float(x_obs.loc[idx, "renewables_mw"]),
        }
        for fault_col in FAULT_COLUMNS:
            payload[fault_col] = bool(event_log.loc[idx, fault_col])
        events.append(payload)
    return events


def _format_value(column: str, value: Any, *, latex: bool = False) -> str:
    if pd.isna(value):
        return ""
    if column == "n_seeds":
        return str(int(round(float(value))))
    if column in {"picp_90", "picp_95", "intervention_rate", "violation_rate", "true_soc_violation_rate"}:
        return f"{float(value):.3f}"
    if column == "expected_cost_usd":
        return f"{float(value):.2f}" if latex else f"{float(value):,.2f}"
    if column in {"mean_interval_width", "mae", "rmse"}:
        return f"{float(value):.2f}"
    return str(value)


def _escape_latex(text: Any) -> str:
    return str(text).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def _latex_alignment(columns: Iterable[str]) -> str:
    parts: list[str] = []
    for col in columns:
        parts.append("l" if col in {"scenario", "policy"} else "r")
    return "".join(parts)


def _markdown_table(df: pd.DataFrame, *, scenario: str) -> str:
    columns = list(df.columns)
    lines = [
        "# DC3S Ablation Table",
        "",
        f"Mean CPSBench-IoT `{scenario}` metrics across seeds for four DC3S ablation variants.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.to_dict(orient="records"):
        values = [_format_value(col, row[col]) for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


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
            if col in {"scenario", "policy"}:
                rendered.append(_escape_latex(row[col]))
            else:
                rendered.append(_format_value(col, row[col], latex=True))
        lines.append(" & ".join(rendered) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def _ordered_summary(summary: pd.DataFrame) -> pd.DataFrame:
    ordered_cols = [
        "scenario",
        "policy",
        "n_seeds",
        "picp_90",
        "picp_95",
        "mean_interval_width",
        "intervention_rate",
        "violation_rate",
        "true_soc_violation_rate",
        "expected_cost_usd",
        "mae",
        "rmse",
    ]
    summary = summary[ordered_cols]
    return (
        summary.assign(_policy_order=summary["policy"].map(lambda name: POLICY_ORDER.index(str(name))))
        .sort_values(["_policy_order", "policy"], kind="stable")
        .drop(columns=["_policy_order"])
        .reset_index(drop=True)
    )


def run_dc3s_ablations(
    *,
    scenario: str = "drift_combo",
    seeds: Iterable[int] = (11, 22, 33, 44, 55),
    horizon: int = 96,
    out_dir: str | Path = "reports/publication",
) -> dict[str, Any]:
    seed_list = [int(seed) for seed in seeds]
    if not seed_list:
        raise ValueError("seeds must be non-empty")

    rows: list[dict[str, Any]] = []
    for seed in seed_list:
        x_obs, x_true, event_log = generate_episode(scenario=scenario, seed=seed, horizon=horizon)
        telemetry_events = _to_telemetry_events(x_obs=x_obs, event_log=event_log)

        load_obs = x_obs["load_mw"].to_numpy(dtype=float)
        renew_obs = x_obs["renewables_mw"].to_numpy(dtype=float)
        load_true = x_true["load_mw"].to_numpy(dtype=float)
        renew_true = x_true["renewables_mw"].to_numpy(dtype=float)
        price = x_obs["price_per_mwh"].to_numpy(dtype=float)

        for variant, expected_policy in VARIANT_ORDER:
            result = dc3s_wrapped_dispatch(
                load_forecast=load_obs,
                renewables_forecast=renew_obs,
                load_true=load_true,
                telemetry_events=telemetry_events,
                price=price,
                command_prefix=f"abl-{scenario}-{seed}",
                variant=variant,
            )
            metrics = compute_all_metrics(
                y_true=load_true,
                y_pred=load_obs,
                lower_90=result["interval_lower"],
                upper_90=result["interval_upper"],
                proposed_charge_mw=result["proposed_charge_mw"],
                proposed_discharge_mw=result["proposed_discharge_mw"],
                safe_charge_mw=result["safe_charge_mw"],
                safe_discharge_mw=result["safe_discharge_mw"],
                soc_mwh=result["soc_mwh"],
                true_soc_mwh=result["soc_mwh"],
                constraints=result["constraints"],
                certificates=result["certificates"],
                event_log=event_log,
                load_true=load_true,
                renewables_true=renew_true,
            )
            rows.append(
                {
                    "scenario": str(scenario),
                    "seed": int(seed),
                    "policy": result["policy"],
                    "picp_90": float(metrics["picp_90"]),
                    "picp_95": float(metrics["picp_95"]),
                    "mean_interval_width": float(metrics["mean_interval_width"]),
                    "intervention_rate": float(metrics["intervention_rate"]),
                    "violation_rate": float(metrics["violation_rate"]),
                    "true_soc_violation_rate": float(metrics["true_soc_violation_rate"]),
                    "expected_cost_usd": float(result["expected_cost_usd"]) if result["expected_cost_usd"] is not None else np.nan,
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                }
            )
            if result["policy"] != expected_policy:
                raise ValueError(f"Unexpected policy mapping for variant {variant}: {result['policy']}")

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["scenario", "policy"], dropna=False)[
            [
                "picp_90",
                "picp_95",
                "mean_interval_width",
                "intervention_rate",
                "violation_rate",
                "true_soc_violation_rate",
                "expected_cost_usd",
                "mae",
                "rmse",
            ]
        ]
        .mean()
        .reset_index()
    )
    seed_counts = raw.groupby(["scenario", "policy"], dropna=False)["seed"].nunique().rename("n_seeds").reset_index()
    summary = _ordered_summary(summary.merge(seed_counts, on=["scenario", "policy"], how="left"))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "dc3s_ablation_table.csv"
    md_path = out_path / "dc3s_ablation_table.md"
    tex_path = out_path / "dc3s_ablation_table.tex"

    summary.to_csv(csv_path, index=False, float_format="%.6f")
    md_path.write_text(_markdown_table(summary, scenario=str(scenario)), encoding="utf-8")
    tex_path.write_text(_latex_table(summary), encoding="utf-8")

    return {
        "scenario": str(scenario),
        "seed_count": len(seed_list),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "latex": str(tex_path),
        "summary_df": summary,
    }


def main() -> None:
    args = _parse_args()
    payload = run_dc3s_ablations(
        scenario=args.scenario,
        seeds=args.seeds,
        horizon=args.horizon,
        out_dir=args.out_dir,
    )
    summary = payload["summary_df"][["policy", "picp_90", "intervention_rate", "expected_cost_usd"]]
    printable = {key: value for key, value in payload.items() if key != "summary_df"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    print(summary.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
