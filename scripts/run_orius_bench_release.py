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
import json
import math
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from orius.adapters.aerospace import AerospaceTrackAdapter
from orius.adapters.battery import BatteryTrackAdapter
from orius.adapters.healthcare import HealthcareTrackAdapter
from orius.adapters.industrial import IndustrialTrackAdapter
from orius.adapters.navigation import NavigationTrackAdapter
from orius.adapters.vehicle import VehicleTrackAdapter
from orius.orius_bench.adapter import BenchmarkAdapter
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
from orius.orius_bench.fault_engine import (
    active_faults,
    apply_faults,
    generate_fault_schedule,
)
from orius.orius_bench.metrics_engine import (
    StepRecord,
    compute_all_metrics,
)


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
        observed_safe = adapter.observed_constraint_satisfied(obs)
        true_margin = adapter.constraint_margin(new_state)
        observed_margin = adapter.constraint_margin(obs)
        fallback_used = bool(faults and faults[0].kind == "blackout")

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
                true_constraint_violated=bool(violation["violated"]),
                observed_constraint_satisfied=observed_safe,
                true_margin=true_margin,
                observed_margin=observed_margin,
                intervened=fallback_used,
                fallback_used=fallback_used,
                soc_after=soc_after,
                soc_min=0.1,
                soc_max=0.9,
                certificate_valid=not violation["violated"],
                certificate_predicted_valid=not violation["violated"],
                fallback_active=fallback_used,
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

    # Tracks (six runtime domains)
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
    fault_schedules: dict[int, list[dict[str, Any]]] = {}

    for track in tracks:
        for ctrl in controllers:
            for s in range(args.seeds):
                seed = 1000 + s
                # Collect fault digest and serialized schedule
                if seed not in fault_digests:
                    sched = generate_fault_schedule(seed, args.horizon)
                    fault_digests[seed] = sched.digest
                    fault_schedules[seed] = [
                        {"step": e.step, "kind": e.kind, "params": e.params, "duration": e.duration}
                        for e in sched.events
                    ]

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
    json_path = write_bundle_json(
        rows, fault_digests, out / "artefact_bundle.json", fault_schedules=fault_schedules
    )
    schema_dir = out / "schemas"
    write_schemas(schema_dir)

    # Write run metadata for replayability audit
    run_metadata = {
        "schema_version": "2.0.0",
        "metric_semantics": {
            "tsvr": "true_state_violation_rate",
            "oasg": "observation_action_safety_gap",
        },
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "command_line": " ".join(sys.argv) if sys.argv else "",
        "seeds": list(range(1000, 1000 + args.seeds)),
        "horizon": args.horizon,
        "n_tracks": len(tracks),
        "n_controllers": len(controllers),
        "n_runs": len(rows),
        "fault_digests": {str(k): v for k, v in fault_digests.items()},
        "artifacts": {
            "leaderboard": str(csv_path.name),
            "artefact_bundle": str(json_path.name),
            "schemas_dir": str(schema_dir.name),
            "bundle_tar": "benchmark_bundle.tar.gz",
        },
    }
    metadata_path = out / "run_metadata.json"
    metadata_path.write_text(json.dumps(run_metadata, indent=2))

    # Create benchmark_bundle.tar.gz
    bundle_path = out / "benchmark_bundle.tar.gz"
    with tarfile.open(bundle_path, "w:gz") as tf:
        tf.add(csv_path, arcname="leaderboard.csv")
        tf.add(json_path, arcname="artefact_bundle.json")
        tf.add(metadata_path, arcname="run_metadata.json")
        for f in schema_dir.glob("*.json"):
            tf.add(f, arcname=f"schemas/{f.name}")

    print(f"\nMetadata   → {metadata_path}")
    print(f"Leaderboard → {csv_path}")
    print(f"Bundle      → {json_path}")
    print(f"Schemas     → {schema_dir}")
    print(f"Tar.gz      → {bundle_path}")
    print(f"Total runs  → {len(rows)}")


if __name__ == "__main__":
    main()
