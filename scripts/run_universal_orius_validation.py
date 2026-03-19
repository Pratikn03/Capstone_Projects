#!/usr/bin/env python3
"""Unified Universal ORIUS validation — runs ALL domains through ORIUS-Bench.

Multi-Domain Universal Framework
---------------------------------
All non-battery domains now run through the full universal DC3S adapter
pipeline and are evaluated against the same evidence gate (TSVR reduction
≥ 25 % vs nominal baseline).

Domain tiers
------------
reference     : battery   (full DC3S, locked PhD-thesis metrics)
proof_domain  : vehicle, healthcare, industrial, aerospace, navigation
                (full DC3S + evidence gate: TSVR reduction ≥ 25 %)

Outputs
-------
- reports/universal_orius_validation/validation_report.json
- reports/universal_orius_validation/proof_domain_report.json
- reports/universal_orius_validation/portability_validation_report.json
- reports/universal_orius_validation/cross_domain_oasg_table.csv
- reports/universal_orius_validation/domain_validation_summary.csv

Exit 0 only when harness completes without errors AND every proof domain
passes the evidence gate.

Usage:
    python scripts/run_universal_orius_validation.py [--seeds 3] [--horizon 48] \\
        [--out reports/universal_orius_validation]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from orius.adapters.aerospace import AerospaceDomainAdapter, AerospaceTrackAdapter
from orius.adapters.battery import BatteryTrackAdapter
from orius.adapters.healthcare import HealthcareDomainAdapter, HealthcareTrackAdapter
from orius.adapters.industrial import IndustrialDomainAdapter, IndustrialTrackAdapter
from orius.adapters.navigation import NavigationDomainAdapter, NavigationTrackAdapter
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

# ---------------------------------------------------------------------------
# Domain catalogue
# ---------------------------------------------------------------------------

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

# All non-battery domains now run through the universal DC3S adapter with
# the full evidence gate (TSVR reduction ≥ 25 %).
PROOF_DOMAINS: list[str] = ["vehicle", "healthcare", "industrial", "aerospace", "navigation"]

# Keep for backward-compat with existing test assertions (no domains in soft-gate-only tier)
PORTABILITY_VALIDATED_DOMAINS: list[str] = []

DOMAIN_MATURITY: dict[str, str] = {
    "battery":    "reference",
    "vehicle":    "proof_domain",
    "healthcare": "proof_domain",
    "industrial": "proof_domain",
    "aerospace":  "proof_domain",
    "navigation": "proof_domain",
}

# Evidence-gate thresholds for the proof domain
PROOF_BASELINE_MIN_TSVR = 0.05
PROOF_MIN_REDUCTION_PCT = 25.0
PROOF_MAX_TSVR_STD = 0.05

# Soft-gate threshold for portability_validated domains:
# DC3S is allowed up to this much extra TSVR vs nominal (effectively no-regression).
PORTABILITY_MAX_TSVR_REGRESSION = 0.01

# ---------------------------------------------------------------------------
# Per-domain universal-step parameters
# ---------------------------------------------------------------------------

# Conformal quantile fed into run_universal_step per domain.
_DOMAIN_QUANTILES: dict[str, float] = {
    "vehicle":    0.9,
    "healthcare": 5.0,
    "industrial": 30.0,
    "aerospace":  5.0,
    "navigation": 1.0,
}

# DC3S config passed as `cfg` to run_universal_step.
_DOMAIN_CFGS: dict[str, dict[str, Any]] = {
    "vehicle":    {"expected_cadence_s": 0.25},
    "healthcare": {"expected_cadence_s": 1.0},
    "industrial": {"expected_cadence_s": 3600.0},
    "aerospace":  {"expected_cadence_s": 1.0},
    "navigation": {"expected_cadence_s": 0.25},
}

# Telemetry keys for which zero-order-hold values are injected.
_DOMAIN_HOLD_KEYS: dict[str, tuple[str, ...]] = {
    "vehicle":    ("position_m", "speed_mps", "speed_limit_mps", "lead_position_m"),
    "healthcare": ("hr_bpm", "spo2_pct", "respiratory_rate"),
    "industrial": ("temp_c", "vacuum_cmhg", "pressure_mbar", "humidity_pct", "power_mw"),
    "aerospace":  ("altitude_m", "airspeed_kt", "bank_angle_deg", "fuel_remaining_pct"),
    "navigation": ("x", "y", "vx", "vy"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_step_timestamp(step: int) -> str:
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=step)
    return ts.isoformat().replace("+00:00", "Z")


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def _make_domain_adapter(domain: str, cfg: dict[str, Any]) -> Any:
    """Instantiate the universal domain adapter for *domain*."""
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
    raise ValueError(f"No universal adapter registered for domain '{domain}'")


def _make_domain_constraints(domain: str, state: dict[str, Any]) -> dict[str, Any]:
    """Return the hard-constraint dict for *domain* at the current *state*."""
    if domain == "vehicle":
        return {
            "speed_limit_mps":  float(state.get("speed_limit_mps", 30.0)),
            "accel_min_mps2":   -5.0,
            "accel_max_mps2":    3.0,
            "dt_s":              0.25,
            "min_headway_m":     5.0,
            "headway_time_s":    2.0,
        }
    if domain == "healthcare":
        return {
            "spo2_min_pct":  90.0,
            "hr_min_bpm":    40.0,
            "hr_max_bpm":   120.0,
        }
    if domain == "industrial":
        return {
            "power_max_mw":  500.0,
            "temp_min_c":      0.0,
            "temp_max_c":    120.0,
        }
    if domain == "aerospace":
        return {
            "v_min_kt":      60.0,
            "v_max_kt":     350.0,
            "max_bank_deg":  30.0,
        }
    if domain == "navigation":
        return {
            "arena_size":   10.0,
            "speed_limit":   1.0,
        }
    return {}


def _domain_validation_status(
    domain: str,
    maturity_label: str,
    *,
    proof_evidence_pass: bool,
    portability_pass: bool = False,
) -> str:
    if domain == REFERENCE_DOMAIN:
        return "reference_validated"
    if maturity_label == "proof_domain":
        return "proof_validated" if proof_evidence_pass else "proof_candidate_only"
    if maturity_label == "portability_validated":
        return "portability_validated" if portability_pass else "portability_candidate"
    if maturity_label == "experimental":
        return "experimental"
    return "portability_only"


# ---------------------------------------------------------------------------
# Evidence-gate evaluators
# ---------------------------------------------------------------------------

def _evaluate_proof_domain(summary: dict[str, Any]) -> dict[str, Any]:
    """Full evidence gate for the designated proof domain (vehicle).

    Criteria
    --------
    1. Baseline TSVR is non-trivial  (mean > PROOF_BASELINE_MIN_TSVR).
    2. DC3S achieves ≥ PROOF_MIN_REDUCTION_PCT % TSVR reduction vs baseline.
    3. Both baseline and DC3S TSVR are stable across seeds (std ≤ PROOF_MAX_TSVR_STD).
    """
    baseline_vals = [float(v) for v in summary.get("tsvr_nominal", [])]
    orius_vals    = [float(v) for v in summary.get("tsvr_dc3s", [])]
    baseline_mean, baseline_std = _mean_std(baseline_vals)
    orius_mean,    orius_std    = _mean_std(orius_vals)
    reduction_pct = (1.0 - orius_mean / baseline_mean) * 100.0 if baseline_mean > 0 else 0.0

    baseline_nontrivial = baseline_mean > PROOF_BASELINE_MIN_TSVR
    orius_improved      = reduction_pct >= PROOF_MIN_REDUCTION_PCT and orius_mean < baseline_mean
    stable              = max(baseline_std, orius_std) <= PROOF_MAX_TSVR_STD

    reasons: list[str] = []
    if not baseline_nontrivial:
        reasons.append("baseline_gap_too_small")
    if not orius_improved:
        reasons.append("orius_did_not_improve")
    if not stable:
        reasons.append("proof_domain_unstable")

    return {
        "baseline_tsvr_mean":  baseline_mean,
        "baseline_tsvr_std":   baseline_std,
        "orius_tsvr_mean":     orius_mean,
        "orius_tsvr_std":      orius_std,
        "orius_reduction_pct": reduction_pct,
        "baseline_nontrivial": baseline_nontrivial,
        "orius_improved":      orius_improved,
        "stable":              stable,
        "evidence_pass":       baseline_nontrivial and orius_improved and stable,
        "failure_reasons":     reasons,
    }


def _evaluate_portability_domain(domain: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Soft evidence gate for portability_validated domains.

    Criteria
    --------
    1. Harness ran without errors for all seeds.
    2. DC3S TSVR ≤ nominal TSVR + PORTABILITY_MAX_TSVR_REGRESSION (no regression).
    """
    tsvr_dc3s = [float(v) for v in summary.get("tsvr_dc3s", [])]
    tsvr_nom  = [float(v) for v in summary.get("tsvr_nominal", [])]
    dc3s_mean = float(np.mean(tsvr_dc3s)) if tsvr_dc3s else 0.0
    nom_mean  = float(np.mean(tsvr_nom))  if tsvr_nom  else 0.0

    no_regression = dc3s_mean <= nom_mean + PORTABILITY_MAX_TSVR_REGRESSION
    harness_ok    = summary.get("harness_status") == "pass"

    reasons: list[str] = []
    if not no_regression:
        reasons.append("dc3s_regression_on_tsvr")
    if not harness_ok:
        reasons.append("harness_failed")

    return {
        "domain":                domain,
        "portability_tsvr_nom":  nom_mean,
        "portability_tsvr_dc3s": dc3s_mean,
        "no_regression":         no_regression,
        "harness_ok":            harness_ok,
        "portability_pass":      harness_ok and no_regression,
        "failure_reasons":       reasons,
    }


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _run_episode(
    adapter: BenchmarkAdapter,
    controller: Any,
    seed: int,
    horizon: int,
) -> list[StepRecord]:
    """Baseline episode: controller proposes action, no universal repair."""
    schedule = generate_fault_schedule(seed, horizon)
    adapter.reset(seed)
    records: list[StepRecord] = []
    trajectory: list[dict[str, Any]] = []

    for t in range(horizon):
        ts    = adapter.true_state()
        faults = active_faults(schedule, t)
        fault_dict = {"kind": faults[0].kind, **faults[0].params} if faults else None

        obs    = adapter.observe(ts, fault_dict)
        ctrl   = DomainAwareController(controller, adapter.domain_name)
        action = ctrl.propose_action(obs, certificate_state=None)

        new_state = adapter.step(action)
        violation = adapter.check_violation(new_state)

        if adapter.domain_name == "battery":
            soc_after = new_state.get("soc", 0.5)
        else:
            soc_after = 0.5 if not violation["violated"] else 0.0
        if isinstance(soc_after, float) and math.isnan(soc_after):
            soc_after = 0.5

        step_rec_d = {**dict(new_state), **dict(action)}
        trajectory.append(step_rec_d)
        useful_work = adapter.compute_useful_work(trajectory[-2:] if len(trajectory) >= 2 else [step_rec_d])

        records.append(StepRecord(
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
        ))
    return records


def _run_domain_proof_episode(
    track: BenchmarkAdapter,
    controller: Any,
    seed: int,
    horizon: int,
) -> list[StepRecord]:
    """Universal proof episode: every action passes through run_universal_step().

    Works for any domain that has a registered DomainAdapter (vehicle, healthcare,
    industrial, aerospace).  The DC3S repair is applied every step — this is what
    distinguishes the proof/portability path from the baseline ``_run_episode``.
    """
    domain = track.domain_name
    cfg    = _DOMAIN_CFGS.get(domain, {"expected_cadence_s": 1.0})
    universal_adapter = _make_domain_adapter(domain, cfg)
    quantile   = _DOMAIN_QUANTILES.get(domain, 5.0)
    hold_keys  = _DOMAIN_HOLD_KEYS.get(domain, ())

    schedule = generate_fault_schedule(seed, horizon)
    track.reset(seed)

    history:    list[dict[str, Any]] = []
    records:    list[StepRecord]     = []
    trajectory: list[dict[str, Any]] = []
    wrapped = DomainAwareController(controller, domain)

    for t in range(horizon):
        ts    = dict(track.true_state())
        faults = active_faults(schedule, t)
        fault_dict = {"kind": faults[0].kind, **faults[0].params} if faults else None

        obs          = dict(track.observe(ts, fault_dict))
        raw_telemetry = dict(obs)
        raw_telemetry["ts_utc"] = _iso_step_timestamp(t)

        # Inject zero-order-hold values for any NaN fields using previous state.
        if history:
            prev = history[-1]
            for key in hold_keys:
                raw_telemetry.setdefault(f"_hold_{key}", prev.get(key, 0.0))

        candidate   = wrapped.propose_action(obs, certificate_state=None)
        constraints = _make_domain_constraints(domain, ts)

        repaired = run_universal_step(
            domain_adapter=universal_adapter,
            raw_telemetry=raw_telemetry,
            history=history,
            candidate_action=candidate,
            constraints=constraints,
            quantile=quantile,
            cfg=cfg,
            controller=f"orius-universal-{domain}-proof",
        )
        action = dict(repaired["safe_action"])

        new_state = track.step(action)
        violation = track.check_violation(new_state)
        soc_after = 0.5 if not violation["violated"] else 0.0

        step_rec_d = {**dict(new_state), **dict(action)}
        trajectory.append(step_rec_d)
        useful_work = track.compute_useful_work(trajectory[-2:] if len(trajectory) >= 2 else [step_rec_d])

        records.append(StepRecord(
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
        ))
        history.append(dict(repaired["state"]))
    return records


# Backward-compat alias used by test_universal_validation_gate.py
def _run_vehicle_proof_episode(
    controller: Any,
    seed: int,
    horizon: int,
) -> list[StepRecord]:
    return _run_domain_proof_episode(VehicleTrackAdapter(), controller, seed, horizon)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Universal ORIUS validation")
    parser.add_argument("--seeds",    type=int, default=3)
    parser.add_argument("--horizon",  type=int, default=48)
    parser.add_argument("--out",      default="reports/universal_orius_validation")
    parser.add_argument("--no-fail",  action="store_true", help="Do not exit 1 on failure")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results:               list[dict[str, Any]] = []
    domain_summary:        dict[str, dict[str, Any]] = {}
    harness_failed_domains: list[dict[str, str]] = []

    # All non-battery domains use the universal proof episode for the DC3S run.
    proof_domains_for_dc3s = set(PROOF_DOMAINS)

    for track in TRACKS:
        domain = track.domain_name
        domain_summary[domain] = {
            "tsvr_dc3s":       [],
            "tsvr_nominal":    [],
            "oasg_dc3s":       [],
            "primary_fault":   "multi",
            "maturity_label":  DOMAIN_MATURITY.get(domain, "portability_only"),
            "harness_status":  "pass",
        }
        try:
            for ctrl in CONTROLLERS:
                for s in range(args.seeds):
                    seed = 2000 + s
                    # Route dc3s controller through universal adapter for eligible domains.
                    if ctrl.name == "dc3s" and domain in proof_domains_for_dc3s:
                        records = _run_domain_proof_episode(track, ctrl, seed, args.horizon)
                    else:
                        records = _run_episode(track, ctrl, seed, args.horizon)

                    metrics = compute_all_metrics(records)
                    results.append({
                        "domain":              domain,
                        "controller":          ctrl.name,
                        "seed":                seed,
                        "tsvr":                metrics.tsvr,
                        "oasg":                metrics.oasg,
                        "intervention_rate":   metrics.intervention_rate,
                    })
                    if ctrl.name == "dc3s":
                        domain_summary[domain]["tsvr_dc3s"].append(metrics.tsvr)
                        domain_summary[domain]["oasg_dc3s"].append(metrics.oasg)
                    elif ctrl.name == "nominal":
                        domain_summary[domain]["tsvr_nominal"].append(metrics.tsvr)

        except Exception as exc:  # noqa: BLE001
            domain_summary[domain]["harness_status"] = "fail"
            domain_summary[domain]["error"] = str(exc)
            harness_failed_domains.append({"domain": domain, "error": str(exc)})

    # ------------------------------------------------------------------
    # Multi-domain evidence gate: all proof domains evaluated equally
    # ------------------------------------------------------------------
    # Primary proof-domain gate (vehicle) — kept for backward-compat keys
    proof_gate: dict[str, Any] = {
        "domain":          PROOF_DOMAIN,
        "maturity_label":  DOMAIN_MATURITY[PROOF_DOMAIN],
        "evidence_pass":   False,
        "failure_reasons": ["proof_domain_not_evaluated"],
    }
    if domain_summary.get(PROOF_DOMAIN, {}).get("harness_status") == "pass":
        proof_gate = {**proof_gate, **_evaluate_proof_domain(domain_summary[PROOF_DOMAIN])}

    # Per-domain evidence reports for all proof domains
    domain_proof_reports: dict[str, dict[str, Any]] = {}
    for pd in PROOF_DOMAINS:
        if pd not in domain_summary:
            domain_proof_reports[pd] = {
                "evidence_pass":   False,
                "failure_reasons": ["domain_not_evaluated"],
            }
        elif domain_summary[pd].get("harness_status") != "pass":
            domain_proof_reports[pd] = {
                "evidence_pass":   False,
                "failure_reasons": ["harness_failed"],
            }
        else:
            domain_proof_reports[pd] = _evaluate_proof_domain(domain_summary[pd])

    all_proof_pass = all(
        r.get("evidence_pass", False) for r in domain_proof_reports.values()
    )

    # Portability reports kept for backward-compat (empty — no portability-only domains)
    portability_reports: dict[str, dict[str, Any]] = {}
    portability_all_pass = True

    # ------------------------------------------------------------------
    # Build output tables
    # ------------------------------------------------------------------
    oasg_rows:   list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []

    for domain, summary in domain_summary.items():
        tsvr_dc3s = summary["tsvr_dc3s"]
        tsvr_nom  = summary["tsvr_nominal"]
        dc3s_mean, dc3s_std = _mean_std(tsvr_dc3s)
        nom_mean,  nom_std  = _mean_std(tsvr_nom)
        reduction = (1.0 - dc3s_mean / nom_mean) * 100.0 if nom_mean > 0 else 0.0

        oasg_rows.append({
            "domain":               domain,
            "primary_fault":        summary["primary_fault"],
            "oasg_rate_baseline":   f"{nom_mean:.4f}",
            "oasg_rate_orius":      f"{dc3s_mean:.4f}",
            "orius_reduction_pct":  f"{reduction:.1f}",
        })

        maturity_label = str(summary["maturity_label"])
        pv_pass = portability_reports.get(domain, {}).get("portability_pass", False)
        # Per-domain evidence pass: use domain_proof_reports for all proof domains
        pd_pass = bool(domain_proof_reports.get(domain, {}).get("evidence_pass", False))
        val_status = _domain_validation_status(
            domain, maturity_label,
            proof_evidence_pass=pd_pass,
            portability_pass=pv_pass,
        )

        evidence_row = ""
        if maturity_label == "proof_domain":
            evidence_row = str(pd_pass).lower()
        elif maturity_label == "portability_validated":
            evidence_row = str(pv_pass).lower()

        domain_rows.append({
            "domain":                domain,
            "maturity_label":        maturity_label,
            "validation_status":     val_status,
            "harness_status":        summary["harness_status"],
            "baseline_tsvr_mean":    f"{nom_mean:.4f}",
            "baseline_tsvr_std":     f"{nom_std:.4f}",
            "orius_tsvr_mean":       f"{dc3s_mean:.4f}",
            "orius_tsvr_std":        f"{dc3s_std:.4f}",
            "orius_reduction_pct":   f"{reduction:.1f}",
            "evidence_pass":         evidence_row,
        })

    # Write CSV artefacts
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

    # Proof-domain report
    proof_report_path = out / "proof_domain_report.json"
    with open(proof_report_path, "w") as f:
        json.dump(
            {
                "reference_domain": REFERENCE_DOMAIN,
                "proof_domain":     PROOF_DOMAIN,
                "locked_protocol":  {
                    "seeds":                args.seeds,
                    "horizon":              args.horizon,
                    "candidate_controller": "dc3s",
                    "baseline_controller":  "nominal",
                },
                **proof_gate,
            },
            f, indent=2,
        )

    # Portability validation report (kept for backward-compat; empty in multi-domain framework)
    portability_report_path = out / "portability_validation_report.json"
    with open(portability_report_path, "w") as f:
        json.dump(
            {
                "portability_validated_domains": PORTABILITY_VALIDATED_DOMAINS,
                "portability_all_pass":          portability_all_pass,
                "domain_reports":                portability_reports,
                "locked_protocol": {
                    "seeds":   args.seeds,
                    "horizon": args.horizon,
                    "note":    "All non-battery domains now use the full evidence gate",
                },
            },
            f, indent=2,
        )

    # Master validation report
    harness_pass  = len(harness_failed_domains) == 0
    evidence_pass = bool(proof_gate.get("evidence_pass", False))

    # All proof domains that passed the evidence gate
    validated_domains = [REFERENCE_DOMAIN] + [
        d for d in PROOF_DOMAINS
        if domain_proof_reports.get(d, {}).get("evidence_pass", False)
    ]
    # Backward-compat: portability_validated_domains is now empty
    portability_validated_confirmed: list[str] = []

    report = {
        "domains_run":                    len(TRACKS),
        "domains_passed":                 len(TRACKS) - len(harness_failed_domains),
        "domains_failed":                 len(harness_failed_domains),
        "failed_domains":                 harness_failed_domains,
        "harness_pass":                   harness_pass,
        "evidence_pass":                  evidence_pass,
        "all_proof_domains_pass":         all_proof_pass,
        "portability_all_pass":           portability_all_pass,
        "all_passed":                     harness_pass and all_proof_pass,
        "reference_domain":               REFERENCE_DOMAIN,
        "proof_domain":                   PROOF_DOMAIN,
        "proof_domains":                  PROOF_DOMAINS,
        "domain_maturity":                DOMAIN_MATURITY,
        "validated_domains":              validated_domains,
        "portability_validated_domains":  portability_validated_confirmed,
        "experimental_domains":           [d for d, lbl in DOMAIN_MATURITY.items() if lbl == "experimental"],
        "portability_only_domains":       [d for d, lbl in DOMAIN_MATURITY.items() if lbl == "portability_only"],
        "domain_results":                 domain_rows,
        "domain_proof_reports":           domain_proof_reports,
        "proof_domain_report":            str(proof_report_path),
        "portability_validation_report":  str(portability_report_path),
        "evidence_failure_reasons":       proof_gate.get("failure_reasons", []),
        "proof_domain_failure_reasons": {
            d: r.get("failure_reasons", [])
            for d, r in domain_proof_reports.items()
            if r.get("failure_reasons")
        },
        "results_count":                  len(results),
        "cross_domain_oasg_csv":          str(csv_path),
        "domain_summary_csv":             str(summary_csv_path),
    }

    report_path = out / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("=== Universal ORIUS Multi-Domain Validation ===")
    print(f"  Domains run:             {len(TRACKS)}")
    print(f"  Harness pass:            {harness_pass}")
    print(f"  Evidence pass (vehicle): {evidence_pass}")
    print(f"  All proof domains pass:  {all_proof_pass}  {PROOF_DOMAINS}")
    print(f"  Harness passed domains:  {report['domains_passed']}")
    print(f"  Harness failed domains:  {report['domains_failed']}")
    if harness_failed_domains:
        for failure in harness_failed_domains:
            print(f"    ✗ {failure['domain']}: {failure['error']}")
    for d, dr in domain_proof_reports.items():
        ep = dr.get("evidence_pass", False)
        symbol = "✓" if ep else "✗"
        reasons_str = ""
        if not ep:
            reasons_str = "  ← " + ", ".join(dr.get("failure_reasons", []))
        print(f"  Harness pass:  {symbol} {d}{reasons_str}")
    print(f"  Report                → {report_path}")
    print(f"  OASG table            → {csv_path}")
    print(f"  Domain summary        → {summary_csv_path}")
    print(f"  Proof-domain report   → {proof_report_path}")
    print(f"  Portability report    → {portability_report_path}")

    if not args.no_fail and not report["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
