#!/usr/bin/env python3
"""Paper 3: Build four-policy trajectory figures for graceful degradation.

Produces:
  - reports/publication/fig_graceful_four_policies.png
  - reports/publication/graceful_four_policy_metrics.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-orius")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from orius.dc3s.graceful import compare_policies

_CONSTRAINTS = {
    "min_soc_mwh": 0.0,
    "max_soc_mwh": 10000.0,
    "capacity_mwh": 10000.0,
    "max_power_mw": 200.0,
    "time_step_hours": 1.0,
    "charge_efficiency": 0.95,
    "discharge_efficiency": 0.95,
}
_LAST_ACTION = {"charge_mw": 0.0, "discharge_mw": 100.0}


def main() -> None:
    out_dir = REPO_ROOT / "reports" / "publication"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = compare_policies(
        last_action=_LAST_ACTION,
        horizon_steps=20,
        soc_mwh=5000.0,
        constraints=_CONSTRAINTS,
        sigma_d=50.0,
    )

    # Metrics CSV
    rows = []
    for name, data in results.items():
        rows.append({
            "policy": name,
            "gdq": data["gdq"],
            "tsvr": data["tsvr"],
            "useful_work_mwh": data["useful_work_mwh"],
            "retained_cost_frac": data["retained_cost_frac"],
            "descent_stability": data["descent_stability"],
            "violations": data["violations"],
        })
    df = pd.DataFrame(rows)
    csv_path = out_dir / "graceful_four_policy_metrics.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"Wrote {csv_path}")

    # Trajectory figure: SOC and action taper for each policy
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    eta_c = _CONSTRAINTS["charge_efficiency"]
    eta_d = _CONSTRAINTS["discharge_efficiency"]
    dt = _CONSTRAINTS["time_step_hours"]
    soc_min = _CONSTRAINTS["min_soc_mwh"]
    soc_max = _CONSTRAINTS["max_soc_mwh"]

    for idx, (name, data) in enumerate(results.items()):
        ax1 = axes[idx // 2, idx % 2]
        traj = data["trajectory"]
        steps = [e["step"] for e in traj]
        discharge = [e["discharge_mw"] for e in traj]
        charge = [e["charge_mw"] for e in traj]
        soc = 5000.0
        socs = [soc]
        for e in traj:
            soc = soc + eta_c * e["charge_mw"] * dt - (e["discharge_mw"] / eta_d) * dt
            socs.append(soc)

        ax1.plot(steps, discharge, "b-", label="discharge_mw")
        ax1.plot(steps, charge, "g--", label="charge_mw")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Power (MW)")
        ax1.set_title(f"{name}\nGDQ={data['gdq']:.4f} TSVR={data['tsvr']:.4f}")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(alpha=0.3)
        ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()
        ax2.plot([0] + steps, socs, "r:", alpha=0.8, label="SOC")
        ax2.axhline(soc_min, color="gray", linestyle="--", alpha=0.5)
        ax2.axhline(soc_max, color="gray", linestyle="--", alpha=0.5)
        ax2.set_ylabel("SOC (MWh)", color="red", alpha=0.8)
        ax2.tick_params(axis="y", labelcolor="red", labelsize=8)

    plt.tight_layout()
    fig_path = out_dir / "fig_graceful_four_policies.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
