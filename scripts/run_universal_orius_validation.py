#!/usr/bin/env python3
"""Unified Universal ORIUS validation — runs all domains through ORIUS-Bench.

Closes Gap B (unified harness) and Gap F (pass gate). Outputs:
- reports/universal_orius_validation/validation_report.json
- reports/universal_orius_validation/cross_domain_oasg_table.csv (Gap D)
- reports/universal_orius_validation/domain_validation_summary.csv
- reports/universal_orius_validation/proof_domain_report.json

Exit 0 only if the harness completes and the selected proof domain passes the
evidence gate. This keeps portability evidence separate from proof-domain
validation.

Usage:
    python scripts/run_universal_orius_validation.py [--seeds 3] [--horizon 48] [--out reports/universal_orius_validation]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from orius.adapters.aerospace import AerospaceTrackAdapter
from orius.adapters.battery import BatteryTrackAdapter
from orius.adapters.healthcare import HealthcareTrackAdapter
from orius.adapters.industrial import IndustrialTrackAdapter
from orius.adapters.navigation import NavigationTrackAdapter
from orius.adapters.vehicle import VehicleDomainAdapter, VehicleTrackAdapter
from orius.orius_bench.adapter import BenchmarkAdapter
from orius.orius_bench.controller_api import (
    DC3SController,
    DomainAwareController,
    FallbackController,
    NaiveController,
    NominalController,
    RobustController,
)
from orius.orius_bench.fault_engine import active_faults, generate_fault_schedule
from orius.orius_bench.metrics_engine import StepRecord, compute_all_metrics
from orius.universal_framework import run_universal_step

TRACKS: list[BenchmarkAdapter] = [
    BatteryTrackAdapter(),
    NavigationTrackAdapter(),
    IndustrialTrackAdapter(),
    HealthcareTrackAdapter(),
    AerospaceTrackAdapter(),
    VehicleTrackAdapter(),
]

CONTROLLERS = [
    NominalController(),
    RobustController(),
    DC3SController(),
    NaiveController(),
    FallbackController(),
]

REFERENCE_DOMAIN = "battery"
PROOF_DOMAIN = "vehicle"
DOMAIN_MATURITY = {
    "battery": "reference",
    "vehicle": "proof_domain",
    "navigation": "portability_only",
    "industrial": "portability_only",
    "healthcare": "portability_only",
    "aerospace": "experimental",
}
PROOF_BASELINE_MIN_TSVR = 0.05
PROOF_MIN_REDUCTION_PCT = 25.0
PROOF_MAX_TSVR_STD = 0.05
VEHICLE_PROOF_QUANTILE = 0.9


def _iso_step_timestamp(step: int) -> str:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=step)
    return ts.isoformat().replace("+00:00", "Z")


def _vehicle_constraints(state: dict[str, float]) -> dict[str, float]:
    return {
        "speed_limit_mps": float(state.get("speed_limit_mps", 30.0)),
        "accel_min_mps2": -5.0,
        "accel_max_mps2": 3.0,
        "dt_s": 0.25,
        "min_headway_m": 5.0,
        "headway_time_s": 2.0,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _domain_validation_status(
    domain: str,
    maturity_label: str,
    *,
    proof_evidence_pass: bool,
) -> str:
    if domain == REFERENCE_DOMAIN:
        return "reference_validated"
    if domain == PROOF_DOMAIN:
        return "proof_validated" if proof_evidence_pass else "proof_candidate_only"
    if maturity_label == "experimental":
        return "experimental"
    return "portability_only"


def _evaluate_proof_domain(summary: dict[str, object]) -> dict[str, object]:
    baseline_vals = [float(v) for v in summary.get("tsvr_nominal", [])]
    orius_vals = [float(v) for v in summary.get("tsvr_dc3s", [])]
    baseline_mean, baseline_std = _mean_std(baseline_vals)
    orius_mean, orius_std = _mean_std(orius_vals)
    reduction_pct = (1.0 - orius_mean / baseline_mean) * 100.0 if baseline_mean > 0 else 0.0

    baseline_nontrivial = baseline_mean > PROOF_BASELINE_MIN_TSVR
    orius_improved = reduction_pct >= PROOF_MIN_REDUCTION_PCT and orius_mean < baseline_mean
    stable = max(baseline_std, orius_std) <= PROOF_MAX_TSVR_STD

    reasons: list[str] = []
    if not baseline_nontrivial:
        reasons.append("baseline_gap_too_small")
    if not orius_improved:
        reasons.append("orius_did_not_improve")
    if not stable:
        reasons.append("proof_domain_unstable")

    return {
        "baseline_tsvr_mean": baseline_mean,
        "baseline_tsvr_std": baseline_std,
        "orius_tsvr_mean": orius_mean,
        "orius_tsvr_std": orius_std,
        "orius_reduction_pct": reduction_pct,
        "baseline_nontrivial": baseline_nontrivial,
        "orius_improved": orius_improved,
        "stable": stable,
        "evidence_pass": baseline_nontrivial and orius_improved and stable,
        "failure_reasons": reasons,
    }


def _run_episode(
    adapter: BenchmarkAdapter,
    controller,
    seed: int,
    horizon: int,
) -> list[StepRecord]:
    """Run one episode and return step records."""
    import math
    schedule = generate_fault_schedule(seed, horizon)
    adapter.reset(seed)
    records: list[StepRecord] = []
    trajectory: list[dict] = []

    for t in range(horizon):
        ts = adapter.true_state()
        faults = active_faults(schedule, t)
        fault_dict = None
        if faults:
            fault_dict = {"kind": faults[0].kind, **faults[0].params}

        obs = adapter.observe(ts, fault_dict)
        ctrl = DomainAwareController(controller, adapter.domain_name)
        action = ctrl.propose_action(obs, certificate_state=None)

        new_state = adapter.step(action)
        violation = adapter.check_violation(new_state)

        if adapter.domain_name == "battery":
            soc_after = new_state.get("soc", 0.5)
        else:
            soc_after = 0.5 if not violation["violated"] else 0.0
        if isinstance(soc_after, float) and math.isnan(soc_after):
            soc_after = 0.5

        step_rec = {**dict(new_state), **dict(action)}
        trajectory.append(step_rec)
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
                useful_work=0.0 if math.isnan(useful_work) else useful_work,
                audit_fields_present=1,
                audit_fields_required=1,
            )
        )
    return records


def _run_vehicle_proof_episode(
    controller,
    seed: int,
    horizon: int,
) -> list[StepRecord]:
    """Run the selected proof domain through the universal repair path.

    Vehicle is the designated second-domain proof surface. The ORIUS path must
    therefore exercise the universal adapter and repair semantics rather than an
    aggressive benchmark controller alone.
    """
    schedule = generate_fault_schedule(seed, horizon)
    track = VehicleTrackAdapter()
    universal_adapter = VehicleDomainAdapter({"expected_cadence_s": 1.0})
    track.reset(seed)

    history: list[dict[str, object]] = []
    records: list[StepRecord] = []
    trajectory: list[dict[str, object]] = []
    wrapped = DomainAwareController(controller, track.domain_name)

    for t in range(horizon):
        ts = dict(track.true_state())
        faults = active_faults(schedule, t)
        fault_dict = None
        if faults:
            fault_dict = {"kind": faults[0].kind, **faults[0].params}

        obs = dict(track.observe(ts, fault_dict))
        raw_telemetry = dict(obs)
        raw_telemetry["ts_utc"] = _iso_step_timestamp(t)
        if history:
            prev_state = history[-1]
            for key in ("position_m", "speed_mps", "speed_limit_mps", "lead_position_m"):
                raw_telemetry.setdefault(f"_hold_{key}", prev_state.get(key, 0.0))

        candidate = wrapped.propose_action(obs, certificate_state=None)
        repaired = run_universal_step(
            domain_adapter=universal_adapter,
            raw_telemetry=raw_telemetry,
            history=history,
            candidate_action=candidate,
            constraints=_vehicle_constraints(ts),
            quantile=VEHICLE_PROOF_QUANTILE,
            cfg={"expected_cadence_s": 1.0},
            controller="orius-universal-vehicle-proof",
        )
        action = dict(repaired["safe_action"])

        new_state = track.step(action)
        violation = track.check_violation(new_state)
        soc_after = 0.5 if not violation["violated"] else 0.0

        step_rec = {**dict(new_state), **dict(action)}
        trajectory.append(step_rec)
        if len(trajectory) >= 2:
            useful_work = track.compute_useful_work(trajectory[-2:])
        else:
            useful_work = track.compute_useful_work([step_rec])

        records.append(
            StepRecord(
                step=t,
                true_state=ts,
                observed_state=dict(obs),
                action=action,
                soc_after=soc_after,
                soc_min=0.1,
                soc_max=0.9,
                certificate_valid=not violation["violated"],
                certificate_predicted_valid=not violation["violated"],
                fallback_active=bool(faults and faults[0].kind == "blackout"),
                useful_work=0.0 if math.isnan(useful_work) else useful_work,
                audit_fields_present=1,
                audit_fields_required=1,
            )
        )
        history.append(dict(repaired["state"]))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Universal ORIUS validation")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--out", default="reports/universal_orius_validation")
    parser.add_argument("--no-fail", action="store_true", help="Do not exit 1 on failure")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    domain_summary: dict[str, dict] = {}
    harness_failed_domains: list[dict[str, str]] = []

    for track in TRACKS:
        domain = track.domain_name
        domain_summary[domain] = {
            "tsvr_dc3s": [],
            "tsvr_nominal": [],
            "oasg_dc3s": [],
            "primary_fault": "multi",
            "maturity_label": DOMAIN_MATURITY.get(domain, "portability_only"),
            "harness_status": "pass",
        }
        try:
            for ctrl in CONTROLLERS:
                for s in range(args.seeds):
                    seed = 2000 + s
                    if domain == PROOF_DOMAIN and ctrl.name == "dc3s":
                        records = _run_vehicle_proof_episode(ctrl, seed, args.horizon)
                    else:
                        records = _run_episode(track, ctrl, seed, args.horizon)
                    metrics = compute_all_metrics(records)
                    row = {
                        "domain": domain,
                        "controller": ctrl.name,
                        "seed": seed,
                        "tsvr": metrics.tsvr,
                        "oasg": metrics.oasg,
                        "intervention_rate": metrics.intervention_rate,
                    }
                    results.append(row)
                    if ctrl.name == "dc3s":
                        domain_summary[domain]["tsvr_dc3s"].append(metrics.tsvr)
                        domain_summary[domain]["oasg_dc3s"].append(metrics.oasg)
                    elif ctrl.name == "nominal":
                        domain_summary[domain]["tsvr_nominal"].append(metrics.tsvr)
        except Exception as exc:
            domain_summary[domain]["harness_status"] = "fail"
            domain_summary[domain]["error"] = str(exc)
            harness_failed_domains.append({"domain": domain, "error": str(exc)})

    # Cross-domain OASG table (Gap D)
    oasg_rows = []
    domain_rows = []
    proof_gate = {
        "domain": PROOF_DOMAIN,
        "maturity_label": DOMAIN_MATURITY[PROOF_DOMAIN],
        "evidence_pass": False,
        "failure_reasons": ["proof_domain_not_evaluated"],
    }
    for domain, summary in domain_summary.items():
        tsvr_dc3s = summary["tsvr_dc3s"]
        tsvr_nom = summary["tsvr_nominal"]
        tsvr_dc3s_mean, tsvr_dc3s_std = _mean_std(tsvr_dc3s)
        tsvr_nom_mean, tsvr_nom_std = _mean_std(tsvr_nom)
        reduction = (1.0 - tsvr_dc3s_mean / tsvr_nom_mean) * 100.0 if tsvr_nom_mean > 0 else 0.0
        oasg_rows.append({
            "domain": domain,
            "primary_fault": summary["primary_fault"],
            "oasg_rate_baseline": f"{tsvr_nom_mean:.4f}",
            "oasg_rate_orius": f"{tsvr_dc3s_mean:.4f}",
            "orius_reduction_pct": f"{reduction:.1f}",
        })
        evidence_row = ""
        if domain == PROOF_DOMAIN and summary.get("harness_status") == "pass":
            proof_gate = {
                **proof_gate,
                **_evaluate_proof_domain(summary),
            }
            evidence_row = str(bool(proof_gate["evidence_pass"])).lower()
        validation_status = _domain_validation_status(
            domain,
            str(summary["maturity_label"]),
            proof_evidence_pass=bool(proof_gate.get("evidence_pass", False)),
        )
        domain_rows.append({
            "domain": domain,
            "maturity_label": summary["maturity_label"],
            "validation_status": validation_status,
            "harness_status": summary["harness_status"],
            "baseline_tsvr_mean": f"{tsvr_nom_mean:.4f}",
            "baseline_tsvr_std": f"{tsvr_nom_std:.4f}",
            "orius_tsvr_mean": f"{tsvr_dc3s_mean:.4f}",
            "orius_tsvr_std": f"{tsvr_dc3s_std:.4f}",
            "orius_reduction_pct": f"{reduction:.1f}",
            "evidence_pass": evidence_row,
        })

    csv_path = out / "cross_domain_oasg_table.csv"
    with open(csv_path, "w", newline="") as f:
        if oasg_rows:
            writer = csv.DictWriter(f, fieldnames=list(oasg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(oasg_rows)

    summary_csv_path = out / "domain_validation_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        if domain_rows:
            writer = csv.DictWriter(f, fieldnames=list(domain_rows[0].keys()))
            writer.writeheader()
            writer.writerows(domain_rows)

    proof_report_path = out / "proof_domain_report.json"
    with open(proof_report_path, "w") as f:
        json.dump(
            {
                "reference_domain": REFERENCE_DOMAIN,
                "proof_domain": PROOF_DOMAIN,
                "locked_protocol": {
                    "seeds": args.seeds,
                    "horizon": args.horizon,
                    "candidate_controller": "dc3s",
                    "baseline_controller": "nominal",
                },
                **proof_gate,
            },
            f,
            indent=2,
        )

    harness_pass = len(harness_failed_domains) == 0
    evidence_pass = bool(proof_gate.get("evidence_pass", False))
    validated_domains = [REFERENCE_DOMAIN]
    if evidence_pass:
        validated_domains.append(PROOF_DOMAIN)

    report = {
        "domains_run": len(TRACKS),
        "domains_passed": len(TRACKS) - len(harness_failed_domains),
        "domains_failed": len(harness_failed_domains),
        "failed_domains": harness_failed_domains,
        "harness_pass": harness_pass,
        "evidence_pass": evidence_pass,
        "all_passed": harness_pass and evidence_pass,
        "reference_domain": REFERENCE_DOMAIN,
        "proof_domain": PROOF_DOMAIN,
        "domain_maturity": DOMAIN_MATURITY,
        "validated_domains": validated_domains,
        "experimental_domains": [d for d, label in DOMAIN_MATURITY.items() if label == "experimental"],
        "portability_only_domains": [d for d, label in DOMAIN_MATURITY.items() if label == "portability_only"],
        "domain_results": domain_rows,
        "proof_domain_report": str(proof_report_path),
        "evidence_failure_reasons": proof_gate.get("failure_reasons", []),
        "results_count": len(results),
        "cross_domain_oasg_csv": str(csv_path),
        "domain_summary_csv": str(summary_csv_path),
    }

    report_path = out / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("=== Universal ORIUS Validation ===")
    print(f"  Domains: {len(TRACKS)}")
    print(f"  Harness pass:  {harness_pass}")
    print(f"  Evidence pass: {evidence_pass}")
    print(f"  Harness passed domains:  {report['domains_passed']}")
    print(f"  Harness failed domains:  {report['domains_failed']}")
    if harness_failed_domains:
        for failure in harness_failed_domains:
            print(f"    ✗ {failure['domain']}: {failure['error']}")
    if not evidence_pass:
        reasons = ", ".join(report["evidence_failure_reasons"]) or "unknown"
        print(f"  Proof-domain failure ({PROOF_DOMAIN}): {reasons}")
    print(f"  Report → {report_path}")
    print(f"  OASG table → {csv_path}")
    print(f"  Domain summary → {summary_csv_path}")
    print(f"  Proof-domain report → {proof_report_path}")

    if not args.no_fail and not report["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
