#!/usr/bin/env python3
"""Fault-performance table under packet drop for runtime-governance stress.

Runs DC3S with varying packet drop rates and records TSVR, intervention rate.
Output: reports/publication/fault_performance_packet_drop.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from orius.cpsbench_iot.scenarios import generate_episode
from orius.cpsbench_iot.runner import (
    _load_optimization_cfg,
    _load_dc3s_cfg,
    _battery_constraints,
    _controller_step_dc3s,
    _to_telemetry_events,
    _soc_fault_config_for_episode,
    _DC3SLoopState,
)
from orius.cpsbench_iot.plant import BatteryPlant
from orius.cpsbench_iot.telemetry_soc import SOCTelemetryChannel


def run_with_packet_drop(
    drop_rate: float,
    seed: int = 42,
    horizon: int = 48,
) -> dict[str, float]:
    """Run DC3S with packet drop; return TSVR, IR, violations."""
    x_obs, x_true, event_log = generate_episode(
        scenario="drift_combo", seed=seed, horizon=horizon
    )
    opt_cfg = _load_optimization_cfg()
    dc3s_cfg = _load_dc3s_cfg()
    constraints = _battery_constraints(opt_cfg)
    load_obs = x_obs["load_mw"].to_numpy(dtype=float)
    renew_obs = x_obs["renewables_mw"].to_numpy(dtype=float)
    price = x_obs["price_per_mwh"].to_numpy(dtype=float)
    carbon = x_obs["carbon_kg_per_mwh"].to_numpy(dtype=float)
    telemetry_events = _to_telemetry_events(x_obs=x_obs, event_log=event_log)
    rng = np.random.default_rng(seed)

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
            scenario="drift_combo", event_log=event_log, seed=seed, fault_overrides=None
        )
    )

    from orius.dc3s.drift import PageHinkleyDetector
    dc3s_state = _DC3SLoopState(
        detector=PageHinkleyDetector.from_state(None, cfg=dc3s_cfg.get("drift", {})),
        sigma_sq=float(np.var(np.abs(x_true["load_mw"].to_numpy() - load_obs))),
    )

    violations = 0
    interventions = 0
    n = len(load_obs)

    for t in range(n):
        true_soc = float(plant.soc_mwh)
        observed_soc, _ = soc_channel.observe(true_soc)

        if rng.random() < drop_rate:
            observed_soc = float("nan")

        step = _controller_step_dc3s(
            load_window=load_obs[t:n],
            renew_window=renew_obs[t:n],
            price_window=price[t:n],
            carbon_window=carbon[t:n],
            load_true_t=float(x_true["load_mw"].iloc[t]),
            observed_soc_mwh=float(observed_soc) if not np.isnan(observed_soc) else 0.5 * constraints["capacity_mwh"],
            current_true_soc_mwh=true_soc,
            telemetry_event=telemetry_events[t],
            optimization_cfg=opt_cfg,
            dc3s_cfg=dc3s_cfg,
            state=dc3s_state,
            command_id=f"drop-{t:04d}",
            controller_name="dc3s_ftit",
            law_override="ftit_ro",
        )
        safe_c = step["safe_charge_mw"]
        safe_d = step["safe_discharge_mw"]
        if step.get("intervention_reason", "none") != "none":
            interventions += 1
        plant.step(charge_mw=safe_c, discharge_mw=safe_d)
        v = plant.violation()
        if v["violated"]:
            violations += 1

    return {
        "drop_rate": drop_rate,
        "tsvr_pct": 100.0 * violations / max(n, 1),
        "ir_pct": 100.0 * interventions / max(n, 1),
        "violations": violations,
        "n_steps": n,
    }


def main() -> None:
    out_dir = REPO_ROOT / "reports" / "publication"
    out_dir.mkdir(parents=True, exist_ok=True)

    drop_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
    rows = []
    for dr in drop_rates:
        r = run_with_packet_drop(drop_rate=dr, seed=42, horizon=48)
        rows.append(r)
        print(f"  drop_rate={dr:.2f} TSVR={r['tsvr_pct']:.2f}% IR={r['ir_pct']:.2f}%")

    csv_path = out_dir / "fault_performance_packet_drop.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["drop_rate", "tsvr_pct", "ir_pct", "violations", "n_steps"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
