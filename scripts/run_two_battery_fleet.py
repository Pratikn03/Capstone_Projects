#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil

import pandas as pd

from _battery_wrappers_common import REPO_ROOT, ensure_dir, write_manifest
from orius.multi_agent.scenarios import (
    run_transformer_capacity_scenario,
    summarize_transformer_capacity_results,
)


def _sync_step_log_alias(out_dir) -> None:
    step_log = out_dir / "multi_agent_step_log.csv"
    compatibility_alias = out_dir / "fleet_composition_two_battery.csv"
    if step_log.exists():
        shutil.copy2(step_log, compatibility_alias)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the canonical two-battery fleet scenario")
    p.add_argument("--out-dir", default="reports/fleet")
    args = p.parse_args()

    out = ensure_dir(REPO_ROOT / args.out_dir)

    feeder_capacity_mw = 80.0
    proposal_per_agent_mw = 60.0
    n_steps = 24
    results = run_transformer_capacity_scenario(
        feeder_capacity_mw=feeder_capacity_mw,
        n_steps=n_steps,
        out_dir=out,
    )

    metrics = pd.DataFrame(
        [
            summarize_transformer_capacity_results(
                results,
                feeder_capacity_mw=feeder_capacity_mw,
                proposal_per_agent_mw=proposal_per_agent_mw,
                n_steps=n_steps,
            )
        ]
    )
    metrics_path = out / "two_battery_composition_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    _sync_step_log_alias(out)

    manifest = {
        "scenario": "two_battery_transformer_capacity",
        "source_function": "orius.multi_agent.scenarios.run_transformer_capacity_scenario",
        "feeder_capacity_mw": feeder_capacity_mw,
        "proposal_per_agent_mw": proposal_per_agent_mw,
        "n_steps": n_steps,
        "generated_files": [
            "multi_agent_transformer_scenario.csv",
            "protocol_compare.csv",
            "fairness_metrics.csv",
            "multi_agent_step_log.csv",
            "fleet_composition_two_battery.csv",
            "two_battery_composition_metrics.csv",
        ],
        "summary_metrics": metrics.iloc[0].to_dict(),
    }
    write_manifest(out, "run_two_battery_fleet_manifest.json", manifest)


if __name__ == "__main__":
    main()
