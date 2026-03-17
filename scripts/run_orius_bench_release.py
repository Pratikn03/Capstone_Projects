#!/usr/bin/env python3
"""ORIUS-Bench release runner (Paper 4).

Runs the full benchmark suite across domains (battery, navigation) and
controllers (nominal, robust, dc3s, naive, fallback), computes the
seven standardised metrics, and exports a leaderboard CSV + artefact
bundle JSON.

Usage:
    python scripts/run_orius_bench_release.py [--seeds 5] [--horizon 96] [--out reports/orius_bench]
"""
from __future__ import annotations

import argparse
from typing import Any
import json
import math
import tarfile
from pathlib import Path

import numpy as np

from orius.orius_bench.adapter import BenchmarkAdapter
from orius.orius_bench.aerospace_track import AerospaceTrackAdapter
from orius.orius_bench.battery_track import BatteryTrackAdapter
from orius.orius_bench.controller_api import (
    ControllerAPI,
    DC3SController,
    DomainAwareController,
    FallbackController,
    NaiveController,
    NominalController,
    RobustController,
)
from orius.orius_bench.export import (
    leaderboard_row,
    write_bundle_json,
    write_leaderboard_csv,
    write_schemas,
)
from orius.orius_bench.healthcare_track import HealthcareTrackAdapter
from orius.orius_bench.industrial_track import IndustrialTrackAdapter
from orius.orius_bench.fault_engine import (
    active_faults,
    apply_faults,
    generate_fault_schedule,
)
from orius.orius_bench.metrics_engine import (
    StepRecord,
    compute_all_metrics,
)
from orius.orius_bench.navigation_track import NavigationTrackAdapter
from orius.orius_bench.vehicle_track import VehicleTrackAdapter


def _run_episode(
    adapter: BenchmarkAdapter,
    controller: ControllerAPI,
    seed: int,
    horizon: int,
    *,
    use_domain_aware: bool = True,
) -> list[StepRecord]:
    """Run one benchmark episode and return step records."""
    schedule = generate_fault_schedule(seed, horizon)
    adapter.reset(seed)
    records: list[StepRecord] = []
    trajectory: list[dict[str, Any]] = []

    for t in range(horizon):
        ts = adapter.true_state()
        faults = active_faults(schedule, t)

        # Build a single fault dict for observe()
        fault_dict = None
        if faults:
            fault_dict = {
                "kind": faults[0].kind,
                **faults[0].params,
            }

        obs = adapter.observe(ts, fault_dict)
        ctrl = DomainAwareController(controller, adapter.domain_name) if use_domain_aware else controller
        action = ctrl.propose_action(obs, certificate_state=None)

        # Step the environment
        new_state = adapter.step(action)
        violation = adapter.check_violation(new_state)

        # soc_after: battery uses actual SOC; other domains use 0.5 if safe else 0.0 for TSVR
        if adapter.domain_name == "battery":
            soc_after = new_state.get("soc", 0.5)
        else:
            soc_after = 0.5 if not violation["violated"] else 0.0
        is_nan = isinstance(soc_after, float) and math.isnan(soc_after)
        if is_nan:
            soc_after = 0.5

        # Per-step useful work via adapter.compute_useful_work (enriched with action for battery)
        step_rec = {**dict(new_state), **dict(action)}
        trajectory.append(step_rec)
        # Battery: discharge_mw from action; navigation: distance from prev to curr
        if len(trajectory) >= 2:
            useful_work = adapter.compute_useful_work(trajectory[-2:])
        else:
            useful_work = adapter.compute_useful_work([step_rec])

        records.append(
            StepRecord(
                step=t,
                true_state=dict(ts),
                observed_state=dict(obs),
                action=dict(action),
                soc_after=soc_after,
                soc_min=0.1,
                soc_max=0.9,
                certificate_valid=not violation["violated"],
                certificate_predicted_valid=not violation["violated"],
                fallback_active=bool(faults and faults[0].kind == "blackout"),
                useful_work=0.0 if is_nan else useful_work,
                audit_fields_present=1,
                audit_fields_required=1,
            )
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="ORIUS-Bench release runner")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--horizon", type=int, default=96, help="Episode length (steps)")
    parser.add_argument("--out", type=str, default="reports/orius_bench", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Tracks (all five thesis domains + navigation)
    tracks: list[BenchmarkAdapter] = [
        BatteryTrackAdapter(),
        NavigationTrackAdapter(),
        IndustrialTrackAdapter(),
        HealthcareTrackAdapter(),
        AerospaceTrackAdapter(),
        VehicleTrackAdapter(),
    ]

    # Controllers
    controllers: list[ControllerAPI] = [
        NominalController(),
        RobustController(),
        DC3SController(),
        NaiveController(),
        FallbackController(),
    ]

    rows = []
    fault_digests: dict[int, str] = {}

    for track in tracks:
        for ctrl in controllers:
            for s in range(args.seeds):
                seed = 1000 + s
                # Collect fault digest
                if seed not in fault_digests:
                    sched = generate_fault_schedule(seed, args.horizon)
                    fault_digests[seed] = sched.digest

                records = _run_episode(track, ctrl, seed, args.horizon)
                metrics = compute_all_metrics(records)
                row = leaderboard_row(ctrl.name, track.domain_name, seed, metrics)
                rows.append(row)
                print(
                    f"  {track.domain_name:12s} | {ctrl.name:12s} | seed={seed} | "
                    f"TSVR={metrics.tsvr:.4f} GDQ={metrics.gdq:.4f} CVA={metrics.cva:.4f}"
                )

    # Write outputs
    csv_path = write_leaderboard_csv(rows, out / "leaderboard.csv")
    json_path = write_bundle_json(rows, fault_digests, out / "artefact_bundle.json")
    schema_dir = out / "schemas"
    write_schemas(schema_dir)

    # Create benchmark_bundle.tar.gz
    bundle_path = out / "benchmark_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        tf.add(csv_path, arcname="leaderboard.csv")
        tf.add(json_path, arcname="artefact_bundle.json")
        for f in schema_dir.glob("*.json"):
            tf.add(f, arcname=f"schemas/{f.name}")

    print(f"\nLeaderboard → {csv_path}")
    print(f"Bundle      → {json_path}")
    print(f"Schemas     → {schema_dir}")
    print(f"Tar.gz      → {bundle_path}")
    print(f"Total runs  → {len(rows)}")


if __name__ == "__main__":
    main()
