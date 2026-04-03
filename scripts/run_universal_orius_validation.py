#!/usr/bin/env python3
"""Unified universal ORIUS validation over the canonical bounded-universal ladder.

Tier model
----------
battery        : reference row (locked theorem/economics witness)
industrial     : defended bounded row
healthcare     : defended bounded row
vehicle        : defended bounded row
navigation     : shadow_synthetic
aerospace      : experimental

NOTE: Energy management validation uses the legacy _run_episode() path with
the full forecasting stack (CQR + OQPE). All other domains use the universal
_run_domain_proof_episode() adapter path. This is an intentional design
distinction: energy management exercises the complete DC3S pipeline including
the forecaster; the other five domains exercise the safety-shield layer via
the DomainAdapter interface. Both paths enforce the same degraded-observation
repair contract, but not all rows carry the same evidence claim.

Outputs
-------
- reports/universal_orius_validation/validation_report.json
- reports/universal_orius_validation/proof_domain_report.json
- reports/universal_orius_validation/portability_validation_report.json
- reports/universal_orius_validation/cross_domain_oasg_table.csv
- reports/universal_orius_validation/domain_validation_summary.csv

Exit 0 only when harness completes without errors AND every defended domain
(industrial + healthcare + vehicle) passes the strong evidence gate.

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
DEFENDED_DOMAINS: list[str] = ["industrial", "healthcare", "vehicle"]
PROOF_CANDIDATE_DOMAINS: list[str] = []
SHADOW_SYNTHETIC_DOMAINS: list[str] = ["navigation"]
EXPERIMENTAL_DOMAINS: list[str] = ["aerospace"]
PROOF_DOMAINS: list[str] = DEFENDED_DOMAINS + PROOF_CANDIDATE_DOMAINS
SUPPORT_DOMAINS: list[str] = SHADOW_SYNTHETIC_DOMAINS + EXPERIMENTAL_DOMAINS
PORTABILITY_VALIDATED_DOMAINS: list[str] = []

DOMAIN_MATURITY: dict[str, str] = {
    "battery": "reference",
    "industrial": "proof_validated",
    "healthcare": "proof_validated",
    "vehicle": "proof_validated",
    "navigation": "shadow_synthetic",
    "aerospace": "experimental",
}

CLOSURE_TARGET_TIER: dict[str, str] = {
    "battery": "witness_row",
    "industrial": "defended_bounded_row",
    "healthcare": "defended_bounded_row",
    "vehicle": "defended_bounded_row",
    "navigation": "defended_bounded_row",
    "aerospace": "defended_bounded_row",
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
    if maturity_label == "proof_validated":
        return "proof_validated" if proof_evidence_pass else "proof_candidate_only"
    if maturity_label == "proof_candidate":
        return "proof_candidate"
    if maturity_label == "shadow_synthetic":
        return "proof_validated" if proof_evidence_pass else "shadow_synthetic"
    if maturity_label == "experimental":
        return "proof_validated" if proof_evidence_pass else "experimental"
    return "portability_only"


def _closure_target_ready(
    domain: str,
    validation_status: str,
    portability_pass: bool,
) -> bool:
    if domain == REFERENCE_DOMAIN:
        return True
    if domain in DEFENDED_DOMAINS:
        return validation_status == "proof_validated"
    if domain in SHADOW_SYNTHETIC_DOMAINS or domain in EXPERIMENTAL_DOMAINS:
        return portability_pass and validation_status == "proof_validated"
    return False


def _closure_blocker(
    domain: str,
    validation_status: str,
    proof_report: Mapping[str, Any] | None,
    portability_report: Mapping[str, Any] | None,
) -> str:
    if domain == REFERENCE_DOMAIN:
        return ""
    if validation_status == "proof_validated":
        return ""
    if domain == "navigation":
        return "navigation_real_data_row_missing"
    if domain == "aerospace":
        return "real_multi_flight_safety_task_missing"
    reasons = []
    if proof_report:
        reasons.extend(str(item) for item in proof_report.get("failure_reasons", []))
    if portability_report:
        reasons.extend(str(item) for item in portability_report.get("failure_reasons", []))
    return reasons[0] if reasons else "promotion_gate_open"


# ---------------------------------------------------------------------------
# Evidence-gate evaluators
# ---------------------------------------------------------------------------

def _evaluate_proof_domain(summary: dict[str, Any]) -> dict[str, Any]:
    """Strong evidence gate for defended or promotable non-battery domains.

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
    """Soft support gate for shadow-synthetic or experimental domains.

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
        observed_safe = adapter.observed_constraint_satisfied(obs)
        true_margin = adapter.constraint_margin(new_state)
        observed_margin = adapter.constraint_margin(obs)
        fallback_used = bool(faults and faults[0].kind == "blackout")

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
        observed_safe = track.observed_constraint_satisfied(obs)
        true_margin = track.constraint_margin(new_state)
        observed_margin = track.constraint_margin(obs)
        fallback_used = bool(faults and faults[0].kind == "blackout")
        soc_after = 0.5 if not violation["violated"] else 0.0

        step_rec_d = {**dict(new_state), **dict(action)}
        trajectory.append(step_rec_d)
        useful_work = track.compute_useful_work(trajectory[-2:] if len(trajectory) >= 2 else [step_rec_d])

        records.append(StepRecord(
            step=t,
            true_state=ts,
            observed_state=dict(obs),
            action=action,
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
    parser.add_argument("--seeds",       type=int, default=3)
    parser.add_argument("--horizon",     type=int, default=48)
    parser.add_argument("--out",         default="reports/universal_orius_validation")
    parser.add_argument("--no-fail",     action="store_true", help="Do not exit 1 on failure")
    parser.add_argument(
        "--equal-domain-gate",
        action="store_true",
        help="Require canonical navigation and aerospace real-data rows and promote them only through the same defended-domain gate.",
    )
    parser.add_argument(
        "--real-data", action="store_true",
        help="Use real datasets (data/ccpp/CCPP.csv, data/bidmc/bidmc_vitals.csv) "
             "for industrial and healthcare tracks. Falls back to calibrated synthetic "
             "if files are absent.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Build track list — optionally wire real-data paths
    from orius.orius_bench.real_data_loader import (
        AEROSPACE_RUNTIME_PATH,
        BIDMC_PATH,
        CCPP_PATH,
        NAVIGATION_PATH,
    )
    nav_track = NavigationTrackAdapter(dataset_path=NAVIGATION_PATH if NAVIGATION_PATH.exists() else None)
    aero_track = AerospaceTrackAdapter(dataset_path=AEROSPACE_RUNTIME_PATH if AEROSPACE_RUNTIME_PATH.exists() else None)
    if args.equal_domain_gate and (not nav_track.using_real_data or not aero_track.using_real_data):
        missing = []
        if not nav_track.using_real_data:
            missing.append(str(NAVIGATION_PATH))
        if not aero_track.using_real_data:
            missing.append(str(AEROSPACE_RUNTIME_PATH))
        print("Equal-domain gate requires canonical real-data rows:", ", ".join(missing))
        return 1
    if args.real_data:
        # Pre-load to confirm availability and log source
        ccpp_src  = "real" if CCPP_PATH.exists() else "calibrated-synthetic"
        bidmc_src = "real" if BIDMC_PATH.exists() else "calibrated-synthetic"
        ind_track  = IndustrialTrackAdapter(dataset_path=CCPP_PATH  if CCPP_PATH.exists()  else None)
        hc_track   = HealthcareTrackAdapter(dataset_path=BIDMC_PATH if BIDMC_PATH.exists() else None)
        print(f"  [real-data] industrial → {ccpp_src} ({CCPP_PATH})")
        print(f"  [real-data] healthcare → {bidmc_src} ({BIDMC_PATH})")
        active_tracks: list[BenchmarkAdapter] = [
            BatteryTrackAdapter(),
            nav_track,
            ind_track,
            hc_track,
            aero_track,
            VehicleTrackAdapter(),
        ]
    else:
        active_tracks = [
            BatteryTrackAdapter(),
            nav_track,
            IndustrialTrackAdapter(),
            HealthcareTrackAdapter(),
            aero_track,
            VehicleTrackAdapter(),
        ]

    run_domain_maturity = dict(DOMAIN_MATURITY)
    promotable_support_domains: list[str] = []
    if args.equal_domain_gate and nav_track.using_real_data:
        run_domain_maturity["navigation"] = "proof_validated"
        promotable_support_domains.append("navigation")
    if args.equal_domain_gate and aero_track.using_real_data:
        run_domain_maturity["aerospace"] = "proof_validated"
        promotable_support_domains.append("aerospace")
    active_defended_domains = DEFENDED_DOMAINS + promotable_support_domains
    active_support_domains = [domain for domain in SUPPORT_DOMAINS if domain not in promotable_support_domains]

    results:               list[dict[str, Any]] = []
    domain_summary:        dict[str, dict[str, Any]] = {}
    harness_failed_domains: list[dict[str, str]] = []

    # Every non-battery row uses the universal repair path for its DC3S run.
    proof_domains_for_dc3s = set(PROOF_DOMAINS + SUPPORT_DOMAINS)

    for track in active_tracks:
        domain = track.domain_name
        domain_summary[domain] = {
            "tsvr_dc3s":       [],
            "tsvr_nominal":    [],
            "oasg_dc3s":       [],
            "primary_fault":   "multi",
            "maturity_label":  run_domain_maturity.get(domain, "portability_only"),
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
    # Multi-domain evidence gate: defended peers plus promotable candidates
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

    # Per-domain strong-gate reports for defended domains and proof candidates
    domain_proof_reports: dict[str, dict[str, Any]] = {}
    for pd in active_defended_domains:
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
        domain_proof_reports.get(domain, {}).get("evidence_pass", False)
        for domain in active_defended_domains
    )

    portability_reports: dict[str, dict[str, Any]] = {}
    for domain in active_support_domains:
        if domain not in domain_summary:
            portability_reports[domain] = {
                "domain": domain,
                "portability_pass": False,
                "failure_reasons": ["domain_not_evaluated"],
            }
        else:
            portability_reports[domain] = _evaluate_portability_domain(domain, domain_summary[domain])
    portability_all_pass = all(
        portability_reports.get(domain, {}).get("portability_pass", False)
        for domain in [d for d in SHADOW_SYNTHETIC_DOMAINS if d in active_support_domains]
    )

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
        if maturity_label in {"proof_validated", "proof_candidate"}:
            evidence_row = str(pd_pass).lower()
        elif maturity_label in {"shadow_synthetic", "experimental"}:
            evidence_row = str(pv_pass).lower()

        domain_rows.append({
            "domain":                domain,
            "maturity_label":        maturity_label,
            "validation_status":     val_status,
            "closure_target_tier":   CLOSURE_TARGET_TIER.get(domain, "defended_bounded_row"),
            "closure_target_ready":  _closure_target_ready(domain, val_status, bool(pv_pass)),
            "closure_blocker":       _closure_blocker(
                domain,
                val_status,
                domain_proof_reports.get(domain),
                portability_reports.get(domain),
            ),
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

    # Per-controller breakdown CSV (all domains × all controllers × all seeds)
    per_ctrl_csv_path = out / "per_controller_tsvr.csv"
    with open(per_ctrl_csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    promoted_candidates = [
        d for d in PROOF_CANDIDATE_DOMAINS
        if domain_proof_reports.get(d, {}).get("evidence_pass", False)
    ]
    proof_downgraded_domains = [
        {
            "domain": d,
            "failure_reasons": domain_proof_reports.get(d, {}).get("failure_reasons", []),
        }
        for d in PROOF_CANDIDATE_DOMAINS
        if not domain_proof_reports.get(d, {}).get("evidence_pass", False)
    ]

    # Proof-domain report
    proof_report_path = out / "proof_domain_report.json"
    with open(proof_report_path, "w") as f:
        json.dump(
            {
                "reference_domain": REFERENCE_DOMAIN,
                "proof_domain":     PROOF_DOMAIN,
                "proof_validated_domains": DEFENDED_DOMAINS,
                "evaluated_proof_candidates": PROOF_CANDIDATE_DOMAINS + promotable_support_domains,
                "promoted_proof_candidates": promoted_candidates,
                "proof_downgraded_domains": proof_downgraded_domains,
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

    # Support-tier report (kept on the portability path for backward-compat)
    portability_report_path = out / "portability_validation_report.json"
    with open(portability_report_path, "w") as f:
        json.dump(
            {
                "portability_validated_domains": PORTABILITY_VALIDATED_DOMAINS,
                "shadow_synthetic_domains":      [d for d in SHADOW_SYNTHETIC_DOMAINS if d in active_support_domains],
                "experimental_domains":          [d for d in EXPERIMENTAL_DOMAINS if d in active_support_domains],
                "portability_all_pass":          portability_all_pass,
                "domain_reports":                portability_reports,
                "locked_protocol": {
                    "seeds":   args.seeds,
                    "horizon": args.horizon,
                    "note":    "Shadow/experimental rows retain the universal repair path but not the defended-domain claim tier",
                },
            },
            f, indent=2,
        )

    # Master validation report
    harness_pass  = len(harness_failed_domains) == 0
    evidence_pass = bool(all_proof_pass)

    # All defended domains that passed the strong evidence gate
    validated_domains = [REFERENCE_DOMAIN] + [
        d for d in DEFENDED_DOMAINS
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
        "proof_domains":                  active_defended_domains,
        "defended_domains":               active_defended_domains,
        "proof_candidate_domains":        PROOF_CANDIDATE_DOMAINS + promotable_support_domains,
        "shadow_synthetic_domains":       [d for d in SHADOW_SYNTHETIC_DOMAINS if d in active_support_domains],
        "domain_maturity":                run_domain_maturity,
        "closure_target_tier":            CLOSURE_TARGET_TIER,
        "validated_domains":              validated_domains,
        "portability_validated_domains":  portability_validated_confirmed,
        "experimental_domains":           [d for d in EXPERIMENTAL_DOMAINS if d in active_support_domains],
        "bounded_universal_target_domains": [
            domain for domain, tier in CLOSURE_TARGET_TIER.items() if tier == "defended_bounded_row"
        ],
        "bounded_universal_target_ready": all(
            bool(row["closure_target_ready"]) for row in domain_rows
        ),
        "portability_only_domains":       [d for d, lbl in DOMAIN_MATURITY.items() if lbl == "portability_only"],
        "domain_results":                 domain_rows,
        "domain_proof_reports":           domain_proof_reports,
        "domain_support_reports":         portability_reports,
        "proof_domain_report":            str(proof_report_path),
        "portability_validation_report":  str(portability_report_path),
        "evidence_failure_reasons":       [
            reason
            for domain in active_defended_domains
            for reason in domain_proof_reports.get(domain, {}).get("failure_reasons", [])
        ],
        "proof_domain_failure_reasons": {
            d: r.get("failure_reasons", [])
            for d, r in domain_proof_reports.items()
            if r.get("failure_reasons")
        },
        "results_count":                  len(results),
        "cross_domain_oasg_csv":          str(csv_path),
        "domain_summary_csv":             str(summary_csv_path),
        "per_controller_tsvr_csv":        str(per_ctrl_csv_path),
    }

    report_path = out / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write real-data supplementary report when --real-data is active
    if args.real_data:
        real_data_meta: dict[str, Any] = {}
        for domain in ("industrial", "healthcare"):
            track_obj = next((t for t in active_tracks if t.domain_name == domain), None)
            using_real = getattr(track_obj, "using_real_data", False) if track_obj else False
            pr = domain_proof_reports.get(domain, {})
            real_data_meta[domain] = {
                "source":          "real" if using_real else "calibrated-synthetic",
                "evidence_pass":   pr.get("evidence_pass", False),
                "baseline_tsvr":   pr.get("baseline_tsvr_mean", 0.0),
                "orius_tsvr":      pr.get("orius_tsvr_mean", 0.0),
                "reduction_pct":   pr.get("orius_reduction_pct", 0.0),
                "failure_reasons": pr.get("failure_reasons", []),
            }
        real_data_report_path = out / "real_data_report.json"
        with open(real_data_report_path, "w") as f:
            json.dump({
                "real_data_mode":  True,
                "seeds":           args.seeds,
                "horizon":         args.horizon,
                "domains":         real_data_meta,
                "all_pass":        all(v["evidence_pass"] for v in real_data_meta.values()),
            }, f, indent=2)
        print(f"  Real-data report      → {real_data_report_path}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("=== Universal ORIUS Multi-Domain Validation ===")
    print(f"  Domains run:             {len(active_tracks)}")
    print(f"  Harness pass:            {harness_pass}")
    print(f"  Evidence pass (defended): {evidence_pass}")
    print(f"  All defended domains pass: {all_proof_pass}  {DEFENDED_DOMAINS}")
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
    for d, dr in portability_reports.items():
        ep = dr.get("portability_pass", False)
        symbol = "✓" if ep else "✗"
        reasons_str = ""
        if not ep:
            reasons_str = "  ← " + ", ".join(dr.get("failure_reasons", []))
        print(f"  Support tier:  {symbol} {d}{reasons_str}")
    print(f"  Report                → {report_path}")
    print(f"  OASG table            → {csv_path}")
    print(f"  Domain summary        → {summary_csv_path}")
    print(f"  Per-controller CSV    → {per_ctrl_csv_path}")
    print(f"  Proof-domain report   → {proof_report_path}")
    print(f"  Portability report    → {portability_report_path}")

    if not args.no_fail and not report["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
