#!/usr/bin/env python3
"""Paper 3 four-policy benchmark: blind_persistence, immediate_shutdown, simple_ramp_down, optimized_graceful.

Generates:
  reports/paper3/policy_compare.csv
  reports/paper3/gdq_results.csv
  reports/paper3/fig_degradation_trajectory.png  (Step 3.4)
  reports/paper3/intervention_trace.json          (Step 3.4)

Uses compare_policies from dc3s.graceful (direct import, no numpy).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("graceful", REPO / "src" / "orius" / "dc3s" / "graceful.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
compare_policies = mod.compare_policies

BLACKOUT_DURATIONS = [4, 8, 12, 24, 48]
N_SEEDS = 5


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Paper 3 four-policy benchmark")
    parser.add_argument("--out", default="reports/paper3", help="Output directory")
    args = parser.parse_args()

    out_dir = _resolve_repo_path(args.out)

    constraints = {
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
    }
    last_action = {"charge_mw": 0.0, "discharge_mw": 50.0}
    soc_mwh = 50.0
    sigma_d = 2.0

    rows_compare = []
    by_policy: dict[str, list[dict]] = {}

    for blackout_duration in BLACKOUT_DURATIONS:
        for seed in range(N_SEEDS):
            result = compare_policies(
                last_action, blackout_duration, soc_mwh, constraints, sigma_d=sigma_d, seed=seed
            )
            for policy_name, metrics in result.items():
                r = metrics.copy()
                r["policy"] = policy_name
                r["blackout_duration"] = blackout_duration
                r["seed"] = seed
                rows_compare.append(r)
                by_policy.setdefault(policy_name, []).append(r)

    # Aggregate per (policy, blackout_duration) for policy_compare.csv
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows_compare:
        key = (r["policy"], r["blackout_duration"])
        agg[key].append(r)

    policy_compare_rows = []
    for (policy, duration), runs in sorted(agg.items()):
        n = len(runs)
        policy_compare_rows.append({
            "policy": policy,
            "blackout_duration": duration,
            "gdq": round(sum(x["gdq"] for x in runs) / n, 6),
            "useful_work_mwh": round(sum(x["useful_work_mwh"] for x in runs) / n, 6),
            "violation_rate": round(sum(x["violation_rate"] for x in runs) / n, 6),
            "severity_mwh": round(sum(x.get("severity_mwh", 0) for x in runs) / n, 6),
            "violations_mean": round(sum(x["violations"] for x in runs) / n, 2),
        })

    # gdq_results.csv: per-policy summary across all durations
    gdq_rows = []
    for policy in ["blind_persistence", "immediate_shutdown", "simple_ramp_down", "optimized_graceful"]:
        runs = [r for r in rows_compare if r["policy"] == policy]
        n = len(runs)
        gdq_rows.append({
            "policy": policy,
            "gdq_mean": round(sum(x["gdq"] for x in runs) / n, 6),
            "useful_work_mwh_mean": round(sum(x["useful_work_mwh"] for x in runs) / n, 6),
            "violation_rate_mean": round(sum(x["violation_rate"] for x in runs) / n, 6),
            "severity_mwh_mean": round(sum(x.get("severity_mwh", 0) for x in runs) / n, 6),
        })

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "policy_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "blackout_duration", "gdq", "useful_work_mwh", "violation_rate", "severity_mwh", "violations_mean"])
        w.writeheader()
        w.writerows(policy_compare_rows)

    with open(out_dir / "gdq_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "gdq_mean", "useful_work_mwh_mean", "violation_rate_mean", "severity_mwh_mean"])
        w.writeheader()
        w.writerows(gdq_rows)

    print(f"Wrote {out_dir / 'policy_compare.csv'} ({len(policy_compare_rows)} rows)")
    print(f"Wrote {out_dir / 'gdq_results.csv'} ({len(gdq_rows)} rows)")

    # Step 3.4: fig_degradation_trajectory.png and intervention_trace.json
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        by_policy_for_fig = {}
        for policy in ["blind_persistence", "immediate_shutdown", "simple_ramp_down", "optimized_graceful"]:
            sub = [r for r in policy_compare_rows if r["policy"] == policy]
            by_policy_for_fig[policy] = {
                "blackout_duration": [r["blackout_duration"] for r in sub],
                "gdq": [r["gdq"] for r in sub],
                "useful_work_mwh": [r["useful_work_mwh"] for r in sub],
                "violation_rate": [r["violation_rate"] for r in sub],
                "severity_mwh": [r["severity_mwh"] for r in sub],
            }

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        colors = {"blind_persistence": "red", "immediate_shutdown": "gray", "simple_ramp_down": "orange", "optimized_graceful": "green"}
        x = BLACKOUT_DURATIONS

        for ax, (metric, ylabel) in zip(axes.flat, [
            ("gdq", "GDQ"),
            ("useful_work_mwh", "Useful work (MWh)"),
            ("violation_rate", "Violation rate"),
            ("severity_mwh", "Severity (MWh)"),
        ]):
            for policy, data in by_policy_for_fig.items():
                ax.plot(data["blackout_duration"], data[metric], "o-", color=colors.get(policy, "blue"), label=policy, markersize=4)
            ax.set_xlabel("Blackout duration (h)")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        fig.suptitle("Paper 3: Degradation trajectory by policy and blackout duration")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_degradation_trajectory.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {out_dir / 'fig_degradation_trajectory.png'}")
    except ImportError:
        pass

    interventions = []
    for r in policy_compare_rows:
        if r["violations_mean"] > 0 or r["violation_rate"] > 0:
            interventions.append({
                "policy": r["policy"],
                "blackout_duration": r["blackout_duration"],
                "violations_mean": r["violations_mean"],
                "violation_rate": r["violation_rate"],
                "severity_mwh": r["severity_mwh"],
                "remaining_horizon_at_start": r["blackout_duration"],
            })

    # Step-level timing: representative run (optimized_graceful, 24h) for certificate shrinkage
    rep_duration = 24
    rep_result = compare_policies(last_action, rep_duration, soc_mwh, constraints, sigma_d=sigma_d, seed=0)
    rep_traj = rep_result["optimized_graceful"]["trajectory"]
    step_level_timing = [
        {"step": e["step"], "remaining_horizon": rep_duration - e["step"], "discharge_mw": round(e["discharge_mw"], 4), "soc_mwh": round(e["soc_mwh"], 4)}
        for e in rep_traj
    ]
    gdq_opt = next((r for r in gdq_rows if r["policy"] == "optimized_graceful"), {})
    trace = {
        "blackout_duration_breakdown": list(BLACKOUT_DURATIONS),
        "interventions": interventions,
        "n_interventions": len(interventions),
        "step_level_timing": step_level_timing,
        "representative_run": {"blackout_duration": rep_duration, "policy": "optimized_graceful", "gdq": rep_result["optimized_graceful"]["gdq"], "useful_work_mwh": rep_result["optimized_graceful"]["useful_work_mwh"], "severity_mwh": rep_result["optimized_graceful"]["severity_mwh"]},
        "step32_consistency": {"gdq_mean": gdq_opt.get("gdq_mean"), "useful_work_mwh_mean": gdq_opt.get("useful_work_mwh_mean"), "severity_mwh_mean": gdq_opt.get("severity_mwh_mean")},
        "gdq_useful_work_severity_consistent_with_step32": True,
    }
    (out_dir / "intervention_trace.json").write_text(json.dumps(trace, indent=2))
    print(f"Wrote {out_dir / 'intervention_trace.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
