#!/usr/bin/env python3
"""Generate a 48-hour operational trace from CPSBench for thesis ch13.

Produces:
  - reports/publication/48h_trace.csv  (step-level trace)
  - reports/publication/fig_48h_trace.png (multi-panel figure)

The trace shows observed SOC, true SOC, reliability score, interval width,
candidate vs safe action, and fault-active shading over a 48-hour window.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orius.cpsbench_iot.runner import run_single
from orius.cpsbench_iot.scenarios import DEFAULT_SCENARIOS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 48h operational trace")
    parser.add_argument("--region", default="DE", help="Region label (DE or US)")
    parser.add_argument("--fault", default="stale_sensor", choices=[
        "nominal", "dropout", "delay_jitter", "out_of_order",
        "spikes", "stale_sensor", "drift_combo",
    ])
    parser.add_argument("--window", type=int, default=48, help="Trace window in hours")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="reports/publication")
    return parser.parse_args()


def _extract_trace(
    scenario: str,
    seed: int,
    horizon: int,
) -> dict[str, pd.DataFrame]:
    """Run two controllers (deterministic_lp, dc3s_ftit) and return step data."""
    payload = run_single(scenario=scenario, seed=seed, horizon=horizon)
    main_rows = payload["main_rows"]

    from orius.cpsbench_iot.scenarios import generate_episode
    x_obs, x_true, event_log = generate_episode(
        scenario=scenario, seed=seed, horizon=horizon,
    )

    from orius.cpsbench_iot.runner import (
        _load_optimization_cfg, _load_dc3s_cfg, _battery_constraints,
        _DC3SLoopState, _controller_step_deterministic,
        _controller_step_dc3s, _init_controller_buffers,
    )
    from orius.dc3s.drift import PageHinkleyDetector
    from orius.cpsbench_iot.telemetry_soc import SOCTelemetryChannel
    from orius.cpsbench_iot.plant import BatteryPlant
    from orius.cpsbench_iot.runner import (
        _to_telemetry_events, _soc_fault_config_for_episode,
    )

    optimization_cfg = _load_optimization_cfg()
    dc3s_cfg = _load_dc3s_cfg()
    constraints = _battery_constraints(optimization_cfg)

    load_obs = x_obs["load_mw"].to_numpy(dtype=float)
    renew_obs = x_obs["renewables_mw"].to_numpy(dtype=float)
    load_true = x_true["load_mw"].to_numpy(dtype=float)
    price = x_obs["price_per_mwh"].to_numpy(dtype=float)
    carbon = x_obs["carbon_kg_per_mwh"].to_numpy(dtype=float)
    telemetry_events = _to_telemetry_events(x_obs=x_obs, event_log=event_log)
    n = len(load_obs)

    traces: dict[str, pd.DataFrame] = {}

    for controller_name in ("deterministic_lp", "dc3s_ftit"):
        plant = BatteryPlant(
            soc_mwh=float(constraints["initial_soc_mwh"]),
            min_soc_mwh=float(constraints["min_soc_mwh"]),
            max_soc_mwh=float(constraints["max_soc_mwh"]),
            charge_eff=float(constraints["charge_efficiency"]),
            discharge_eff=float(constraints["discharge_efficiency"]),
            dt_hours=float(constraints["time_step_hours"]),
        )
        soc_channel = SOCTelemetryChannel(
            _soc_fault_config_for_episode(
                scenario=scenario, event_log=event_log,
                seed=seed, fault_overrides=None,
            )
        )
        dc3s_state = _DC3SLoopState(
            detector=PageHinkleyDetector.from_state(None, cfg=dc3s_cfg.get("drift", {})),
            sigma_sq=float(np.var(np.abs(load_true - load_obs))),
        )

        rows = []
        for t in range(n):
            t_end = n
            load_window = load_obs[t:t_end]
            renew_window = renew_obs[t:t_end]
            price_window = price[t:t_end]
            carbon_window = carbon[t:t_end]
            current_true_soc = float(plant.soc_mwh)
            observed_soc, _ = soc_channel.observe(current_true_soc)

            if controller_name == "deterministic_lp":
                step = _controller_step_deterministic(
                    load_window=load_window, renew_window=renew_window,
                    price_window=price_window, carbon_window=carbon_window,
                    optimization_cfg=optimization_cfg, dc3s_cfg=dc3s_cfg,
                    observed_soc_mwh=float(observed_soc),
                )
            else:
                step = _controller_step_dc3s(
                    load_window=load_window, renew_window=renew_window,
                    price_window=price_window, carbon_window=carbon_window,
                    load_true_t=float(load_true[t]),
                    observed_soc_mwh=float(observed_soc),
                    current_true_soc_mwh=current_true_soc,
                    telemetry_event=telemetry_events[t],
                    optimization_cfg=optimization_cfg, dc3s_cfg=dc3s_cfg,
                    state=dc3s_state,
                    command_id=f"trace-{scenario}-{seed}-{controller_name}-{t:04d}",
                    controller_name="dc3s_ftit",
                    law_override="ftit_ro",
                )

            safe_charge = float(step["safe_charge_mw"])
            safe_discharge = float(step["safe_discharge_mw"])
            next_soc = float(plant.step(
                charge_mw=safe_charge, discharge_mw=safe_discharge,
            ))
            viol = plant.violation()

            fault_active = bool(
                event_log.loc[t, "dropout"]
                | event_log.loc[t, "delay_jitter"]
                | event_log.loc[t, "out_of_order"]
                | event_log.loc[t, "spikes"]
                | event_log.loc[t, "stale_sensor"]
                | event_log.loc[t, "covariate_drift"]
            )

            rows.append({
                "step": t + 1,
                "timestamp": str(x_obs.loc[t, "timestamp"]),
                "soc_observed_mwh": float(observed_soc),
                "soc_true_mwh": next_soc,
                "reliability_w": float(step.get("w_t", 1.0)),
                "drift_flag": bool(step.get("drift_flag", False)),
                "inflation": float(step.get("rac_inflation", 1.0)),
                "interval_width_mw": float(step.get("interval_width", 0.0)),
                "proposed_charge_mw": float(step["proposed_charge_mw"]),
                "proposed_discharge_mw": float(step["proposed_discharge_mw"]),
                "safe_charge_mw": safe_charge,
                "safe_discharge_mw": safe_discharge,
                "intervened": bool(
                    abs(safe_charge - float(step["proposed_charge_mw"])) > 1e-6
                    or abs(safe_discharge - float(step["proposed_discharge_mw"])) > 1e-6
                ),
                "fault_active": fault_active,
                "true_soc_violated": bool(viol["violated"]),
                "violation_severity_mwh": float(viol["severity_mwh"]),
                "price_per_mwh": float(price[t]),
            })

        traces[controller_name] = pd.DataFrame(rows)

    return traces


def _plot_48h_trace(
    det_df: pd.DataFrame,
    dc3s_df: pd.DataFrame,
    out_path: Path,
    window: int,
    region: str,
    fault: str,
) -> None:
    det = det_df.iloc[:window].copy()
    dc3s = dc3s_df.iloc[:window].copy()
    x = np.arange(len(det))

    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

    # Panel 1: SOC trajectories
    ax = axes[0]
    ax.plot(x, det["soc_true_mwh"], label="True SOC (det. LP)", color="#d62728", linewidth=1.4)
    ax.plot(x, det["soc_observed_mwh"], label="Observed SOC (det. LP)",
            color="#d62728", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.plot(x, dc3s["soc_true_mwh"], label="True SOC (DC3S FTIT)", color="#1f77b4", linewidth=1.4)
    ax.plot(x, dc3s["soc_observed_mwh"], label="Observed SOC (DC3S FTIT)",
            color="#1f77b4", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
    fault_mask = dc3s["fault_active"].to_numpy(dtype=bool)
    for i in range(len(fault_mask)):
        if fault_mask[i]:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="red")
    ax.set_ylabel("SOC (MWh)")
    ax.set_title(f"48-Hour Operational Trace — {region} / {fault} (seed {det_df['step'].iloc[0] - 1})")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: Reliability score
    ax = axes[1]
    ax.plot(x, dc3s["reliability_w"], color="#2ca02c", linewidth=1.2)
    ax.set_ylabel("Reliability $w_t$")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.grid(alpha=0.3)

    # Panel 3: Interval width
    ax = axes[2]
    ax.fill_between(x, 0, dc3s["interval_width_mw"], alpha=0.4, color="#ff7f0e", label="DC3S FTIT")
    ax.fill_between(x, 0, det["interval_width_mw"], alpha=0.25, color="#d62728", label="Det. LP")
    ax.set_ylabel("Interval Width (MW)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 4: Candidate vs safe action
    ax = axes[3]
    net_proposed = dc3s["proposed_discharge_mw"] - dc3s["proposed_charge_mw"]
    net_safe = dc3s["safe_discharge_mw"] - dc3s["safe_charge_mw"]
    ax.plot(x, net_proposed, label="Candidate action", color="#9467bd", linewidth=1.0)
    ax.plot(x, net_safe, label="Safe action", color="#1f77b4", linewidth=1.2)
    intervened = dc3s["intervened"].to_numpy(dtype=bool)
    if np.any(intervened):
        ax.scatter(x[intervened], net_safe.to_numpy()[intervened],
                   color="red", zorder=5, s=20, label="Intervention")
    ax.set_ylabel("Net Dispatch (MW)\n(+dis / -chg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 5: Price signal
    ax = axes[4]
    ax.plot(x, dc3s["price_per_mwh"], color="#8c564b", linewidth=1.0)
    ax.set_ylabel("Price ($/MWh)")
    ax.set_xlabel("Hour")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon = max(args.window, 48)
    print(f"Generating {args.window}h trace: region={args.region}, fault={args.fault}, seed={args.seed}")

    traces = _extract_trace(
        scenario=args.fault,
        seed=args.seed,
        horizon=horizon,
    )

    det_df = traces["deterministic_lp"]
    dc3s_df = traces["dc3s_ftit"]

    combined = dc3s_df.copy()
    combined["controller"] = "dc3s_ftit"
    det_copy = det_df.copy()
    det_copy["controller"] = "deterministic_lp"
    full_trace = pd.concat([combined, det_copy], ignore_index=True)

    csv_path = out_dir / "48h_trace.csv"
    full_trace.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV -> {csv_path} ({len(full_trace)} rows)")

    fig_path = out_dir / "fig_48h_trace.png"
    _plot_48h_trace(
        det_df=det_df, dc3s_df=dc3s_df,
        out_path=fig_path,
        window=args.window,
        region=args.region, fault=args.fault,
    )
    print(f"  Figure -> {fig_path}")

    summary = {
        "region": args.region,
        "fault": args.fault,
        "window_hours": args.window,
        "seed": args.seed,
        "det_violations": int(det_df["true_soc_violated"].sum()),
        "dc3s_violations": int(dc3s_df["true_soc_violated"].sum()),
        "dc3s_interventions": int(dc3s_df["intervened"].sum()),
        "dc3s_mean_reliability": float(dc3s_df["reliability_w"].mean()),
    }
    summary_path = out_dir / "48h_trace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Summary -> {summary_path}")
    print(f"  Det LP violations: {summary['det_violations']}")
    print(f"  DC3S FTIT violations: {summary['dc3s_violations']}")
    print(f"  DC3S interventions: {summary['dc3s_interventions']}")


if __name__ == "__main__":
    main()
