#!/usr/bin/env python3
"""Paper 3 four-policy benchmark and promoted publication summary.

Canonical outputs:
  reports/paper3/policy_compare.csv
  reports/paper3/gdq_results.csv
  reports/paper3/graceful_four_policy_metrics.csv
  reports/paper3/fig_degradation_trajectory.png
  reports/paper3/intervention_trace.json
  reports/publication/graceful_four_policy_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
POLICY_ORDER = [
    "blind_persistence",
    "immediate_shutdown",
    "simple_ramp_down",
    "optimized_graceful",
]
BLACKOUT_DURATIONS = [4, 8, 12, 24, 48]
N_SEEDS = 5
CANONICAL_CONSTRAINTS = {
    "min_soc_mwh": 0.0,
    "max_soc_mwh": 10000.0,
    "capacity_mwh": 10000.0,
    "max_power_mw": 200.0,
    "time_step_hours": 1.0,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
}
CANONICAL_LAST_ACTION = {"charge_mw": 0.0, "discharge_mw": 100.0}
CANONICAL_SOC_MWH = 5000.0
CANONICAL_SIGMA_D = 50.0

spec = importlib.util.spec_from_file_location(
    "graceful", REPO / "src" / "orius" / "dc3s" / "graceful.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
compare_policies = mod.compare_policies


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def collect_policy_surface(
    *,
    blackout_durations: list[int] | None = None,
    n_seeds: int = N_SEEDS,
    constraints: dict[str, float] | None = None,
    last_action: dict[str, float] | None = None,
    soc_mwh: float = CANONICAL_SOC_MWH,
    sigma_d: float = CANONICAL_SIGMA_D,
) -> list[dict[str, Any]]:
    """Collect the detailed per-policy, per-blackout benchmark surface."""
    durations = blackout_durations or list(BLACKOUT_DURATIONS)
    env_constraints = dict(constraints or CANONICAL_CONSTRAINTS)
    env_last_action = dict(last_action or CANONICAL_LAST_ACTION)

    rows_compare: list[dict[str, Any]] = []
    for blackout_duration in durations:
        for seed in range(n_seeds):
            result = compare_policies(
                env_last_action,
                blackout_duration,
                soc_mwh,
                env_constraints,
                sigma_d=sigma_d,
                seed=seed,
            )
            for policy_name, metrics in result.items():
                row = {
                    "policy": policy_name,
                    "blackout_duration": int(blackout_duration),
                    "seed": int(seed),
                    "gdq": float(metrics["gdq"]),
                    "useful_work_mwh": float(metrics["useful_work_mwh"]),
                    "violation_rate": float(metrics["violation_rate"]),
                    "severity_mwh": float(metrics.get("severity_mwh", 0.0)),
                    "violations": float(metrics["violations"]),
                    "retained_cost_frac": float(metrics["retained_cost_frac"]),
                    "descent_stability": float(metrics["descent_stability"]),
                }
                rows_compare.append(row)
    return rows_compare


def summarize_surface(rows_compare: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate the detailed benchmark into duration-level and promoted summaries."""
    duration_buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows_compare:
        key = (str(row["policy"]), int(row["blackout_duration"]))
        duration_buckets.setdefault(key, []).append(row)

    policy_compare_rows: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        for duration in BLACKOUT_DURATIONS:
            runs = duration_buckets.get((policy, duration), [])
            if not runs:
                continue
            n = len(runs)
            policy_compare_rows.append(
                {
                    "policy": policy,
                    "blackout_duration": duration,
                    "gdq": _round(sum(x["gdq"] for x in runs) / n),
                    "useful_work_mwh": _round(sum(x["useful_work_mwh"] for x in runs) / n),
                    "violation_rate": _round(sum(x["violation_rate"] for x in runs) / n),
                    "severity_mwh": _round(sum(x["severity_mwh"] for x in runs) / n),
                    "violations_mean": _round(sum(x["violations"] for x in runs) / n, 2),
                    "retained_cost_frac": _round(sum(x["retained_cost_frac"] for x in runs) / n),
                    "descent_stability": _round(sum(x["descent_stability"] for x in runs) / n),
                }
            )

    by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in policy_compare_rows:
        by_policy.setdefault(str(row["policy"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for policy in POLICY_ORDER:
        runs = by_policy.get(policy, [])
        if not runs:
            continue
        n = len(runs)
        zero_violation_durations = [
            int(row["blackout_duration"]) for row in runs if float(row["violation_rate"]) <= 1e-12
        ]
        summary_rows.append(
            {
                "policy": policy,
                "gdq_mean": _round(sum(float(x["gdq"]) for x in runs) / n),
                "useful_work_mwh_mean": _round(sum(float(x["useful_work_mwh"]) for x in runs) / n),
                "violation_rate_mean": _round(sum(float(x["violation_rate"]) for x in runs) / n),
                "severity_mwh_mean": _round(sum(float(x["severity_mwh"]) for x in runs) / n),
                "violations_mean": _round(sum(float(x["violations_mean"]) for x in runs) / n, 2),
                "retained_cost_frac_mean": _round(
                    sum(float(x["retained_cost_frac"]) for x in runs) / n
                ),
                "descent_stability_mean": _round(
                    sum(float(x["descent_stability"]) for x in runs) / n
                ),
                "max_zero_violation_blackout_h": max(zero_violation_durations, default=0),
            }
        )
    return policy_compare_rows, summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, policy_compare_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    for row in policy_compare_rows:
        rows_by_policy.setdefault(str(row["policy"]), []).append(row)

    colors = {
        "blind_persistence": "#b02a37",
        "immediate_shutdown": "#4b5563",
        "simple_ramp_down": "#d97706",
        "optimized_graceful": "#047857",
    }
    labels = {
        "blind_persistence": "Blind persistence",
        "immediate_shutdown": "Immediate shutdown",
        "simple_ramp_down": "Simple ramp-down",
        "optimized_graceful": "Optimized graceful",
    }
    metrics = [
        ("gdq", "GDQ"),
        ("useful_work_mwh", "Useful work (MWh)"),
        ("violation_rate", "Violation rate"),
        ("severity_mwh", "Severity (MWh)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for policy in POLICY_ORDER:
            series = rows_by_policy.get(policy, [])
            if not series:
                continue
            ax.plot(
                [int(row["blackout_duration"]) for row in series],
                [float(row[metric]) for row in series],
                "o-",
                color=colors[policy],
                label=labels[policy],
                markersize=4,
            )
        ax.set_xlabel("Blackout duration (h)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Paper 3: graceful fallback benchmark over blackout duration")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _build_trace(
    *,
    paper3_dir: Path,
    summary_rows: list[dict[str, Any]],
    policy_compare_rows: list[dict[str, Any]],
) -> None:
    rep_duration = 24
    rep_result = compare_policies(
        CANONICAL_LAST_ACTION,
        rep_duration,
        CANONICAL_SOC_MWH,
        CANONICAL_CONSTRAINTS,
        sigma_d=CANONICAL_SIGMA_D,
        seed=0,
    )
    rep_traj = rep_result["optimized_graceful"]["trajectory"]
    step_level_timing = [
        {
            "step": event["step"],
            "remaining_horizon": rep_duration - event["step"],
            "discharge_mw": _round(event["discharge_mw"], 4),
            "soc_mwh": _round(event["soc_mwh"], 4),
        }
        for event in rep_traj
    ]
    zero_violation_frontier = {
        row["policy"]: int(row["max_zero_violation_blackout_h"]) for row in summary_rows
    }
    trace = {
        "canonical_blackout_durations_h": list(BLACKOUT_DURATIONS),
        "representative_run": {
            "policy": "optimized_graceful",
            "blackout_duration": rep_duration,
            "gdq": rep_result["optimized_graceful"]["gdq"],
            "useful_work_mwh": rep_result["optimized_graceful"]["useful_work_mwh"],
            "violation_rate": rep_result["optimized_graceful"]["violation_rate"],
        },
        "step_level_timing": step_level_timing,
        "zero_violation_frontier_h": zero_violation_frontier,
        "surface_rows": len(policy_compare_rows),
    }
    (paper3_dir / "intervention_trace.json").write_text(
        json.dumps(trace, indent=2), encoding="utf-8"
    )


def run_benchmark(
    *,
    paper3_dir: Path,
    publication_dir: Path,
) -> dict[str, Path]:
    """Run the canonical Paper 3 benchmark and sync the promoted publication files."""
    rows_compare = collect_policy_surface()
    policy_compare_rows, summary_rows = summarize_surface(rows_compare)

    paper3_dir.mkdir(parents=True, exist_ok=True)
    publication_dir.mkdir(parents=True, exist_ok=True)

    policy_compare_path = paper3_dir / "policy_compare.csv"
    summary_path = paper3_dir / "graceful_four_policy_metrics.csv"
    legacy_summary_path = paper3_dir / "gdq_results.csv"
    publication_summary_path = publication_dir / "graceful_four_policy_metrics.csv"
    paper3_figure_path = paper3_dir / "fig_degradation_trajectory.png"

    _write_csv(
        policy_compare_path,
        policy_compare_rows,
        [
            "policy",
            "blackout_duration",
            "gdq",
            "useful_work_mwh",
            "violation_rate",
            "severity_mwh",
            "violations_mean",
            "retained_cost_frac",
            "descent_stability",
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        [
            "policy",
            "gdq_mean",
            "useful_work_mwh_mean",
            "violation_rate_mean",
            "severity_mwh_mean",
            "violations_mean",
            "retained_cost_frac_mean",
            "descent_stability_mean",
            "max_zero_violation_blackout_h",
        ],
    )
    _write_csv(
        legacy_summary_path,
        [
            {
                "policy": row["policy"],
                "gdq_mean": row["gdq_mean"],
                "useful_work_mwh_mean": row["useful_work_mwh_mean"],
                "violation_rate_mean": row["violation_rate_mean"],
                "severity_mwh_mean": row["severity_mwh_mean"],
            }
            for row in summary_rows
        ],
        [
            "policy",
            "gdq_mean",
            "useful_work_mwh_mean",
            "violation_rate_mean",
            "severity_mwh_mean",
        ],
    )
    shutil.copyfile(summary_path, publication_summary_path)
    _write_figure(paper3_figure_path, policy_compare_rows)
    _build_trace(
        paper3_dir=paper3_dir,
        summary_rows=summary_rows,
        policy_compare_rows=policy_compare_rows,
    )

    return {
        "policy_compare": policy_compare_path,
        "summary": summary_path,
        "legacy_summary": legacy_summary_path,
        "publication_summary": publication_summary_path,
        "figure": paper3_figure_path,
        "trace": paper3_dir / "intervention_trace.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper 3 four-policy benchmark")
    parser.add_argument("--out", default="reports/paper3", help="Paper 3 output directory")
    parser.add_argument(
        "--publication-out",
        default="reports/publication",
        help="Publication artifact directory for the promoted summary",
    )
    args = parser.parse_args()

    outputs = run_benchmark(
        paper3_dir=_resolve_repo_path(args.out),
        publication_dir=_resolve_repo_path(args.publication_out),
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
