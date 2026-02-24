"""CLI and orchestration for CPSBench-IoT runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import (
    dc3s_wrapped_dispatch,
    deterministic_lp_dispatch,
    naive_safe_clip_dispatch,
    robust_fixed_interval_dispatch,
)
from .metrics import compute_all_metrics
from .scenarios import DEFAULT_SCENARIOS, FAULT_COLUMNS, generate_episode


REQUIRED_OUTPUTS = (
    "dc3s_main_table.csv",
    "dc3s_fault_breakdown.csv",
    "calibration_plot.png",
    "violation_vs_cost_curve.png",
    "dc3s_run_summary.json",
)


def _ensure_out_dir(out_dir: str | Path) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _violation_mask(result: dict[str, Any]) -> np.ndarray:
    s_ch = np.asarray(result["safe_charge_mw"], dtype=float)
    s_dis = np.asarray(result["safe_discharge_mw"], dtype=float)
    soc = np.asarray(result["soc_mwh"], dtype=float)
    c = dict(result["constraints"])
    max_power = float(c.get("max_power_mw", max(np.max(s_ch, initial=0.0), np.max(s_dis, initial=0.0))))
    min_soc = float(c.get("min_soc_mwh", np.min(soc, initial=0.0)))
    max_soc = float(c.get("max_soc_mwh", np.max(soc, initial=0.0)))
    return (s_ch > max_power + 1e-9) | (s_dis > max_power + 1e-9) | ((s_ch > 1e-9) & (s_dis > 1e-9)) | (soc < min_soc - 1e-9) | (soc > max_soc + 1e-9)


def _intervention_mask(result: dict[str, Any]) -> np.ndarray:
    p_ch = np.asarray(result["proposed_charge_mw"], dtype=float)
    p_dis = np.asarray(result["proposed_discharge_mw"], dtype=float)
    s_ch = np.asarray(result["safe_charge_mw"], dtype=float)
    s_dis = np.asarray(result["safe_discharge_mw"], dtype=float)
    return (np.abs(p_ch - s_ch) > 1e-6) | (np.abs(p_dis - s_dis) > 1e-6)


def _to_telemetry_events(x_obs: pd.DataFrame, event_log: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx in range(len(x_obs)):
        payload = {
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


def run_single(
    *,
    scenario: str,
    seed: int,
    horizon: int = 168,
) -> dict[str, list[dict[str, Any]]]:
    """Run one scenario+seed pair and return metrics row payloads."""
    x_obs, x_true, event_log = generate_episode(scenario=scenario, seed=seed, horizon=horizon)
    load_obs = x_obs["load_mw"].to_numpy(dtype=float)
    renew_obs = x_obs["renewables_mw"].to_numpy(dtype=float)
    load_true = x_true["load_mw"].to_numpy(dtype=float)
    price = x_obs["price_per_mwh"].to_numpy(dtype=float)
    carbon = x_obs["carbon_kg_per_mwh"].to_numpy(dtype=float)
    telemetry_events = _to_telemetry_events(x_obs=x_obs, event_log=event_log)

    results = {
        "deterministic_lp": deterministic_lp_dispatch(
            load_forecast=load_obs,
            renewables_forecast=renew_obs,
            price=price,
            carbon=carbon,
        ),
        "robust_fixed_interval": robust_fixed_interval_dispatch(
            load_forecast=load_obs,
            renewables_forecast=renew_obs,
            price=price,
        ),
        "naive_safe_clip": naive_safe_clip_dispatch(
            load_forecast=load_obs,
            renewables_forecast=renew_obs,
            price=price,
            carbon=carbon,
            timestamps=x_obs["timestamp"],
        ),
        "dc3s_wrapped": dc3s_wrapped_dispatch(
            load_forecast=load_obs,
            renewables_forecast=renew_obs,
            load_true=load_true,
            telemetry_events=telemetry_events,
            price=price,
            command_prefix=f"{scenario}-{seed}",
        ),
    }

    main_rows: list[dict[str, Any]] = []
    fault_rows: list[dict[str, Any]] = []
    for controller, result in results.items():
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
            constraints=result["constraints"],
            certificates=result["certificates"],
            event_log=event_log,
        )
        main_rows.append(
            {
                "scenario": scenario,
                "seed": int(seed),
                "controller": controller,
                "policy": result["policy"],
                "expected_cost_usd": result.get("expected_cost_usd"),
                "carbon_kg": result.get("carbon_kg"),
                **metrics,
            }
        )

        violations = _violation_mask(result)
        interventions = _intervention_mask(result)
        for fault_col in FAULT_COLUMNS:
            mask = event_log[fault_col].to_numpy(dtype=int) > 0
            fault_rows.append(
                {
                    "scenario": scenario,
                    "seed": int(seed),
                    "controller": controller,
                    "fault_type": fault_col,
                    "fault_count": int(mask.sum()),
                    "violation_rate_at_fault": float(np.mean(violations[mask])) if mask.any() else 0.0,
                    "intervention_rate_at_fault": float(np.mean(interventions[mask])) if mask.any() else 0.0,
                }
            )

    return {"main_rows": main_rows, "fault_rows": fault_rows}


def _plot_calibration(main_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in main_df.groupby("controller", sort=True):
        ax.scatter(sub["picp_90"], sub["mean_interval_width"], label=controller, alpha=0.85)
    ax.axvline(0.90, color="black", linestyle="--", linewidth=1.2, label="Nominal 90%")
    ax.set_xlabel("PICP@90")
    ax.set_ylabel("Mean Interval Width")
    ax.set_title("CPSBench Calibration vs Sharpness")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_violation_vs_cost(main_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in main_df.groupby("controller", sort=True):
        ax.scatter(sub["expected_cost_usd"], sub["violation_rate"], label=controller, alpha=0.85)
    ax.set_xlabel("Expected Cost (USD)")
    ax.set_ylabel("Violation Rate")
    ax.set_title("Violation vs Cost Tradeoff")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_suite(
    *,
    scenarios: Iterable[str],
    seeds: Iterable[int],
    out_dir: str | Path = "reports/publication",
    horizon: int = 168,
) -> dict[str, Any]:
    """Run CPSBench suite and persist canonical publication artifacts."""
    out = _ensure_out_dir(out_dir)
    all_main_rows: list[dict[str, Any]] = []
    all_fault_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        for seed in seeds:
            payload = run_single(scenario=scenario, seed=int(seed), horizon=horizon)
            all_main_rows.extend(payload["main_rows"])
            all_fault_rows.extend(payload["fault_rows"])

    main_df = pd.DataFrame(all_main_rows).sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)
    fault_df = pd.DataFrame(all_fault_rows).sort_values(["scenario", "seed", "fault_type", "controller"]).reset_index(drop=True)

    main_csv = out / "dc3s_main_table.csv"
    fault_csv = out / "dc3s_fault_breakdown.csv"
    calibration_png = out / "calibration_plot.png"
    violation_png = out / "violation_vs_cost_curve.png"
    summary_json = out / "dc3s_run_summary.json"

    main_df.to_csv(main_csv, index=False, float_format="%.6f")
    fault_df.to_csv(fault_csv, index=False, float_format="%.6f")
    _plot_calibration(main_df, calibration_png)
    _plot_violation_vs_cost(main_df, violation_png)

    summary = {
        "scenarios": list(scenarios),
        "seeds": [int(s) for s in seeds],
        "horizon": int(horizon),
        "rows_main": int(len(main_df)),
        "rows_fault_breakdown": int(len(fault_df)),
        "controller_summary": {
            controller: {
                "mean_violation_rate": float(sub["violation_rate"].mean()),
                "mean_intervention_rate": float(sub["intervention_rate"].mean()),
                "mean_cost_usd": float(sub["expected_cost_usd"].dropna().mean()) if not sub["expected_cost_usd"].dropna().empty else None,
            }
            for controller, sub in main_df.groupby("controller", sort=True)
        },
        "artifacts": {name: str(out / name) for name in REQUIRED_OUTPUTS},
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPSBench-IoT benchmark")
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=str, default="reports/publication")
    parser.add_argument("--horizon", type=int, default=168)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_suite(
        scenarios=[args.scenario],
        seeds=[int(args.seed)],
        out_dir=args.out_dir,
        horizon=int(args.horizon),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
