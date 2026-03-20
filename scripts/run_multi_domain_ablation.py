#!/usr/bin/env python3
"""Multi-domain fault-type ablation benchmark.

Runs all 6 ORIUS-Bench domains under fault-type isolation to produce a
per-domain, per-fault-type TSVR breakdown — the multi-domain analog of
``tbl02_ablations.tex`` (which is battery-only).

Fault types evaluated
---------------------
bias         : sensor bias injection
noise        : Gaussian noise on telemetry
stuck_sensor : frozen telemetry value
blackout     : complete sensor blackout (multi-step)
multi        : all fault types mixed (standard generate_fault_schedule)

For each (domain, fault_type) pair, runs Nominal and DC3S controllers
over ``--seeds`` seeds × ``--horizon`` steps.

Outputs
-------
- reports/multi_domain_ablation/fault_type_tsvr.csv
  Columns: domain, fault_type, controller, seed, tsvr, intervention_rate

Usage
-----
    python scripts/run_multi_domain_ablation.py [--seeds 5] [--horizon 48]
        [--out reports/multi_domain_ablation]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from orius.orius_bench.adapter import BenchmarkAdapter
from orius.orius_bench.battery_track import BatteryTrackAdapter
from orius.orius_bench.vehicle_track import VehicleTrackAdapter
from orius.orius_bench.healthcare_track import HealthcareTrackAdapter
from orius.orius_bench.industrial_track import IndustrialTrackAdapter
from orius.orius_bench.aerospace_track import AerospaceTrackAdapter
from orius.orius_bench.navigation_track import NavigationTrackAdapter
from orius.orius_bench.controller_api import NominalController, DC3SController, DomainAwareController
from orius.orius_bench.fault_engine import FaultEvent, FaultSchedule, active_faults, generate_fault_schedule
from orius.orius_bench.metrics_engine import StepRecord, compute_all_metrics
from orius.universal_framework import run_universal_step
from orius.adapters.vehicle import VehicleDomainAdapter, VehicleTrackAdapter as VehicleAdapter
from orius.adapters.healthcare import HealthcareDomainAdapter, HealthcareTrackAdapter as HCAdapter
from orius.adapters.industrial import IndustrialDomainAdapter, IndustrialTrackAdapter as IndAdapter
from orius.adapters.aerospace import AerospaceDomainAdapter, AerospaceTrackAdapter as AeroAdapter
from orius.adapters.navigation import NavigationDomainAdapter, NavigationTrackAdapter as NavAdapter

# ---------------------------------------------------------------------------
# Domain catalogue (same 6 as universal validation)
# ---------------------------------------------------------------------------

TRACKS: list[BenchmarkAdapter] = [
    BatteryTrackAdapter(),
    NavigationTrackAdapter(),
    IndustrialTrackAdapter(),
    HealthcareTrackAdapter(),
    AerospaceTrackAdapter(),
    VehicleTrackAdapter(),
]

FAULT_TYPES = ["bias", "noise", "stuck_sensor", "blackout", "multi"]

# DC3S domain configs (mirrors run_universal_orius_validation.py)
_DOMAIN_CFGS: dict[str, dict] = {
    "vehicle":    {"expected_cadence_s": 0.25},
    "healthcare": {"expected_cadence_s": 1.0},
    "industrial": {"expected_cadence_s": 3600.0},
    "aerospace":  {"expected_cadence_s": 1.0},
    "navigation": {"expected_cadence_s": 0.25},
    "battery":    {"expected_cadence_s": 3600.0},
}

_DOMAIN_QUANTILES: dict[str, float] = {
    "vehicle":    0.9,
    "healthcare": 5.0,
    "industrial": 30.0,
    "aerospace":  5.0,
    "navigation": 1.0,
    "battery":    10.0,
}

_DOMAIN_HOLD_KEYS: dict[str, tuple[str, ...]] = {
    "vehicle":    ("position_m", "speed_mps", "speed_limit_mps", "lead_position_m"),
    "healthcare": ("hr_bpm", "spo2_pct", "respiratory_rate"),
    "industrial": ("temp_c", "vacuum_cmhg", "pressure_mbar", "humidity_pct", "power_mw"),
    "aerospace":  ("altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct"),
    "navigation": ("x", "y", "vx", "vy"),
    "battery":    ("soc", "load_mw"),
}

PROOF_DOMAINS = {"vehicle", "healthcare", "industrial", "aerospace", "navigation"}


# ---------------------------------------------------------------------------
# Fault schedule generators (single-type isolation)
# ---------------------------------------------------------------------------

def _make_single_fault_schedule(seed: int, horizon: int, fault_type: str) -> FaultSchedule:
    """Generate a schedule with ONLY faults of *fault_type*."""
    if fault_type == "multi":
        return generate_fault_schedule(seed, horizon)

    rng = np.random.default_rng(seed)
    events: list[FaultEvent] = []
    fault_rate = 0.25  # higher rate to ensure meaningful TSVR signal

    for t in range(horizon):
        if rng.random() < fault_rate:
            params: dict = {}
            if fault_type == "bias":
                params["magnitude"] = float(rng.normal(0, 5))
            elif fault_type == "noise":
                params["sigma"] = float(rng.uniform(1, 10))
            elif fault_type == "stuck_sensor":
                params["frozen_value"] = float(rng.uniform(0.1, 0.9))
            elif fault_type == "blackout":
                dur = int(rng.integers(1, 8))
                dur = min(dur, horizon - t)
                events.append(FaultEvent(step=t, kind="blackout", duration=dur, params={}))
                continue
            events.append(FaultEvent(step=t, kind=fault_type, params=params, duration=1))

    return FaultSchedule(seed=seed, events=events, horizon=horizon)


# ---------------------------------------------------------------------------
# Domain adapter factory (mirrors run_universal_orius_validation.py)
# ---------------------------------------------------------------------------

def _make_domain_adapter(domain: str) -> object:
    cfg = _DOMAIN_CFGS.get(domain, {"expected_cadence_s": 1.0})
    if domain == "vehicle":
        return VehicleDomainAdapter(cfg)
    if domain == "healthcare":
        return HealthcareDomainAdapter(cfg)
    if domain == "industrial":
        return IndustrialDomainAdapter(cfg)
    if domain == "aerospace":
        return AerospaceDomainAdapter(cfg)
    if domain == "navigation":
        return NavigationDomainAdapter(cfg)
    return None


def _make_domain_constraints(domain: str, state: dict) -> dict:
    if domain == "vehicle":
        return {
            "speed_limit_mps": float(state.get("speed_limit_mps", 30.0)),
            "accel_min_mps2": -5.0, "accel_max_mps2": 3.0,
            "dt_s": 0.25, "min_headway_m": 5.0, "headway_time_s": 2.0,
        }
    if domain == "healthcare":
        return {"spo2_min_pct": 90.0, "hr_min_bpm": 40.0, "hr_max_bpm": 120.0}
    if domain == "industrial":
        return {"power_max_mw": 500.0, "temp_min_c": 0.0, "temp_max_c": 120.0}
    if domain == "aerospace":
        return {"v_min_kt": 60.0, "v_max_kt": 350.0, "max_bank_deg": 30.0}
    if domain == "navigation":
        return {"arena_size": 10.0, "speed_limit": 1.0}
    return {}


def _iso_step(step: int) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=step)).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _run_baseline_episode(
    track: BenchmarkAdapter, controller, seed: int, horizon: int, fault_type: str
) -> list[StepRecord]:
    """Nominal or Robust controller baseline — no DC3S repair."""
    import math
    schedule = _make_single_fault_schedule(seed, horizon, fault_type)
    track.reset(seed)
    records: list[StepRecord] = []
    trajectory: list[dict] = []

    for t in range(horizon):
        ts = dict(track.true_state())
        faults = active_faults(schedule, t)
        fault_dict = {"kind": faults[0].kind, **faults[0].params} if faults else None
        obs = dict(track.observe(ts, fault_dict))
        action = dict(controller.propose_action(obs, certificate_state=None))
        new_state = track.step(action)
        violation = track.check_violation(new_state)
        trajectory.append({**dict(new_state), **action})
        useful_work = track.compute_useful_work(trajectory[-2:] if len(trajectory) >= 2 else [trajectory[-1]])
        soc_after = 0.5 if not violation["violated"] else 0.0
        records.append(StepRecord(
            step=t, true_state=ts, observed_state=obs, action=action,
            soc_after=soc_after, soc_min=0.1, soc_max=0.9,
            certificate_valid=not violation["violated"],
            certificate_predicted_valid=not violation["violated"],
            fallback_active=bool(faults and faults[0].kind == "blackout"),
            useful_work=0.0 if math.isnan(useful_work) else useful_work,
            audit_fields_present=1, audit_fields_required=1,
        ))
    return records


def _run_dc3s_episode(
    track: BenchmarkAdapter, controller, seed: int, horizon: int, fault_type: str
) -> list[StepRecord]:
    """DC3S controller with universal repair (proof domains) or raw (battery)."""
    import math
    domain = track.domain_name
    schedule = _make_single_fault_schedule(seed, horizon, fault_type)
    track.reset(seed)

    use_universal = domain in PROOF_DOMAINS
    universal_adapter = _make_domain_adapter(domain) if use_universal else None
    cfg = _DOMAIN_CFGS.get(domain, {"expected_cadence_s": 1.0})
    quantile = _DOMAIN_QUANTILES.get(domain, 5.0)
    hold_keys = _DOMAIN_HOLD_KEYS.get(domain, ())

    history: list[dict] = []
    records: list[StepRecord] = []
    trajectory: list[dict] = []
    wrapped = DomainAwareController(controller, domain) if use_universal else controller

    for t in range(horizon):
        ts = dict(track.true_state())
        faults = active_faults(schedule, t)
        fault_dict = {"kind": faults[0].kind, **faults[0].params} if faults else None
        obs = dict(track.observe(ts, fault_dict))
        raw_telemetry = dict(obs)
        raw_telemetry["ts_utc"] = _iso_step(t)

        if history and use_universal:
            prev = history[-1]
            for key in hold_keys:
                raw_telemetry.setdefault(f"_hold_{key}", prev.get(key, 0.0))

        candidate = wrapped.propose_action(obs, certificate_state=None)

        if use_universal and universal_adapter is not None:
            constraints = _make_domain_constraints(domain, ts)
            repaired = run_universal_step(
                domain_adapter=universal_adapter,
                raw_telemetry=raw_telemetry,
                history=history,
                candidate_action=candidate,
                constraints=constraints,
                quantile=quantile,
                cfg=cfg,
                controller=f"orius-ablation-{domain}-{fault_type}",
            )
            action = dict(repaired["safe_action"])
            history.append(dict(repaired["state"]))
        else:
            action = dict(candidate)

        new_state = track.step(action)
        violation = track.check_violation(new_state)
        soc_after = 0.5 if not violation["violated"] else 0.0
        trajectory.append({**dict(new_state), **action})
        useful_work = track.compute_useful_work(trajectory[-2:] if len(trajectory) >= 2 else [trajectory[-1]])
        records.append(StepRecord(
            step=t, true_state=ts, observed_state=obs, action=action,
            soc_after=soc_after, soc_min=0.1, soc_max=0.9,
            certificate_valid=not violation["violated"],
            certificate_predicted_valid=not violation["violated"],
            fallback_active=bool(faults and faults[0].kind == "blackout"),
            useful_work=0.0 if math.isnan(useful_work) else useful_work,
            audit_fields_present=1, audit_fields_required=1,
        ))
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-domain fault-type ablation")
    parser.add_argument("--seeds",   type=int, default=5)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--out",     default="reports/multi_domain_ablation")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nominal_ctrl = NominalController()
    dc3s_ctrl    = DC3SController()
    controllers  = [("nominal", nominal_ctrl), ("dc3s", dc3s_ctrl)]

    all_rows: list[dict] = []

    for fault_type in FAULT_TYPES:
        for track in TRACKS:
            domain = track.domain_name
            print(f"  [{domain}] fault_type={fault_type} ...", flush=True)
            for ctrl_name, ctrl in controllers:
                for s in range(args.seeds):
                    seed = 3000 + s
                    try:
                        if ctrl_name == "nominal":
                            records = _run_baseline_episode(track, ctrl, seed, args.horizon, fault_type)
                        else:
                            records = _run_dc3s_episode(track, ctrl, seed, args.horizon, fault_type)
                        metrics = compute_all_metrics(records)
                        all_rows.append({
                            "domain":            domain,
                            "fault_type":        fault_type,
                            "controller":        ctrl_name,
                            "seed":              seed,
                            "tsvr":              round(metrics.tsvr, 6),
                            "intervention_rate": round(metrics.intervention_rate, 6),
                            "oasg":              round(metrics.oasg, 6),
                        })
                    except Exception as exc:  # noqa: BLE001
                        print(f"    ERROR {domain}/{fault_type}/{ctrl_name}/seed={seed}: {exc}")
                        all_rows.append({
                            "domain": domain, "fault_type": fault_type,
                            "controller": ctrl_name, "seed": seed,
                            "tsvr": float("nan"), "intervention_rate": float("nan"),
                            "oasg": float("nan"),
                        })

    csv_path = out / "fault_type_tsvr.csv"
    with open(csv_path, "w", newline="") as f:
        if all_rows:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    print(f"  Wrote {len(all_rows)} rows → {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
