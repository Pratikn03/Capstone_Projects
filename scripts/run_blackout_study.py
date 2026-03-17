#!/usr/bin/env python3
"""Certificate half-life blackout study for thesis ch28.

Freezes a DC3S certificate and measures violation rate as the blackout
duration increases: 0, 1, 4, 12, 24, 48 hours.

Outputs:
  - reports/publication/blackout_study.csv
  - reports/publication/fig_blackout_halflife.png
"""
from __future__ import annotations

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

from orius.cpsbench_iot.scenarios import generate_episode
from orius.cpsbench_iot.plant import BatteryPlant
from orius.cpsbench_iot.runner import (
    _load_optimization_cfg, _load_dc3s_cfg, _battery_constraints,
)


def run_blackout_study(
    seed: int = 42,
    scenario: str = "drift_combo",
    horizon: int = 168,
    freeze_step: int = 72,
    blackout_durations: list[int] | None = None,
) -> pd.DataFrame:
    if blackout_durations is None:
        blackout_durations = [0, 1, 4, 12, 24, 48]

    from orius.cpsbench_iot.runner import (
        _load_dc3s_cfg, _DC3SLoopState, _controller_step_dc3s,
        _to_telemetry_events, _soc_fault_config_for_episode,
    )
    from orius.dc3s.drift import PageHinkleyDetector
    from orius.cpsbench_iot.telemetry_soc import SOCTelemetryChannel

    x_obs, x_true, event_log = generate_episode(
        scenario=scenario, seed=seed, horizon=horizon,
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

    # Run DC3S up to freeze_step to get a realistic frozen action
    plant_pre = BatteryPlant(
        soc_mwh=float(constraints["initial_soc_mwh"]),
        min_soc_mwh=float(constraints["min_soc_mwh"]),
        max_soc_mwh=float(constraints["max_soc_mwh"]),
        charge_eff=float(constraints["charge_efficiency"]),
        discharge_eff=float(constraints["discharge_efficiency"]),
        dt_hours=float(constraints["time_step_hours"]),
    )
    soc_channel_pre = SOCTelemetryChannel(
        _soc_fault_config_for_episode(
            scenario=scenario, event_log=event_log, seed=seed, fault_overrides=None,
        )
    )
    dc3s_state = _DC3SLoopState(
        detector=PageHinkleyDetector.from_state(None, cfg=dc3s_cfg.get("drift", {})),
        sigma_sq=float(np.var(np.abs(load_true - load_obs))),
    )

    last_safe_charge = 0.0
    last_safe_discharge = 0.0
    soc_at_freeze = float(constraints["initial_soc_mwh"])

    for t in range(min(freeze_step, n)):
        current_true_soc = float(plant_pre.soc_mwh)
        observed_soc, _ = soc_channel_pre.observe(current_true_soc)
        step = _controller_step_dc3s(
            load_window=load_obs[t:n], renew_window=renew_obs[t:n],
            price_window=price[t:n], carbon_window=carbon[t:n],
            load_true_t=float(load_true[t]),
            observed_soc_mwh=float(observed_soc),
            current_true_soc_mwh=current_true_soc,
            telemetry_event=telemetry_events[t],
            optimization_cfg=optimization_cfg, dc3s_cfg=dc3s_cfg,
            state=dc3s_state,
            command_id=f"blackout-{seed}-{t:04d}",
            controller_name="dc3s_ftit", law_override="ftit_ro",
        )
        last_safe_charge = float(step["safe_charge_mw"])
        last_safe_discharge = float(step["safe_discharge_mw"])
        plant_pre.step(charge_mw=last_safe_charge, discharge_mw=last_safe_discharge)

    soc_at_freeze = float(plant_pre.soc_mwh)

    rows = []
    for duration in blackout_durations:
        plant = BatteryPlant(
            soc_mwh=soc_at_freeze,
            min_soc_mwh=float(constraints["min_soc_mwh"]),
            max_soc_mwh=float(constraints["max_soc_mwh"]),
            charge_eff=float(constraints["charge_efficiency"]),
            discharge_eff=float(constraints["discharge_efficiency"]),
            dt_hours=float(constraints["time_step_hours"]),
        )

        violations = 0
        severities = []
        coverage_count = 0
        total_after = min(duration, horizon - freeze_step)

        for dt_step in range(total_after):
            plant.step(charge_mw=last_safe_charge, discharge_mw=last_safe_discharge)
            viol = plant.violation()
            if viol["violated"]:
                violations += 1
                severities.append(float(viol["severity_mwh"]))
            else:
                coverage_count += 1

        tsvr = violations / max(total_after, 1)
        sev_p95 = float(np.quantile(severities, 0.95)) if severities else 0.0
        coverage = coverage_count / max(total_after, 1)

        rows.append({
            "blackout_hours": duration,
            "frozen_charge_mw": round(last_safe_charge, 2),
            "frozen_discharge_mw": round(last_safe_discharge, 2),
            "soc_at_freeze_mwh": round(soc_at_freeze, 2),
            "tsvr_pct": round(tsvr * 100, 2),
            "sev_p95_mwh": round(sev_p95, 3),
            "coverage_pct": round(coverage * 100, 1),
            "total_steps": total_after,
            "violations": violations,
        })

    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path("reports/publication")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_blackout_study()
    csv_path = out_dir / "blackout_study.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  CSV -> {csv_path}")
    print(df.to_string(index=False))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = df["blackout_hours"]
    ax1.plot(x, df["tsvr_pct"], marker="o", color="#d62728", linewidth=1.5)
    ax1.set_xlabel("Blackout Duration (hours)")
    ax1.set_ylabel("TSVR (%)")
    ax1.set_title("Violation Rate vs Blackout Duration")
    ax1.grid(alpha=0.3)

    ax2.plot(x, df["coverage_pct"], marker="s", color="#1f77b4", linewidth=1.5)
    ax2.axhline(50, color="gray", linestyle="--", linewidth=1, label="Half-life threshold")
    ax2.set_xlabel("Blackout Duration (hours)")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_title("Certificate Coverage Decay")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / "fig_blackout_halflife.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f"  Figure -> {fig_path}")


if __name__ == "__main__":
    main()
