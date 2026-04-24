#!/usr/bin/env python3
"""Build canonical healthcare runtime artifacts for the promoted MIMIC row."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.adapters.healthcare import HealthcareDomainAdapter, HealthcareTrackAdapter
from orius.certos.verification import (
    REQUIRED_CERTIFICATE_FIELDS,
    certificate_intervention_semantics_valid,
    count_present_required_certificate_fields,
    extract_certificate_validity_horizon,
    formal_validity_predicate,
    load_certificates_from_duckdb,
    missing_required_certificate_fields,
    verify_certificates,
)
from orius.dc3s.certificate import recompute_certificate_hash, store_certificates_batch
from orius.orius_bench.controller_api import DC3SController, DomainAwareController, NominalController
from orius.orius_bench.fault_engine import active_faults, generate_fault_schedule
from orius.orius_bench.metrics_engine import StepRecord, compute_all_metrics
from orius.universal_theory.domain_runtime_contracts import (
    HEALTHCARE_FAIL_SAFE_CONTRACT_ID,
    witness_trace_fields_from_result,
)
from orius.universal_framework import run_universal_step


DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "healthcare" / "mimic3" / "processed" / "mimic3_healthcare_orius.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "healthcare"
TABLE_NAME = "dispatch_certificates"
HEALTHCARE_CFG = {
    "expected_cadence_s": 1.0,
    "runtime_surface": "mimic_monitoring_replay",
    "closure_tier": "runtime_contract_closed",
}


def _write_equal_domain_artifacts(out_dir: Path) -> dict[str, str]:
    script_path = REPO_ROOT / "scripts" / "build_equal_domain_artifact_discipline.py"
    spec = importlib.util.spec_from_file_location("build_equal_domain_artifact_discipline", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load equal-domain artifact builder from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.write_runtime_comparator_artifacts_for_domain("healthcare", out_dir=out_dir)


def _constraint_payload() -> dict[str, float]:
    return {
        "spo2_min_pct": 90.0,
        "hr_min_bpm": 40.0,
        "hr_max_bpm": 120.0,
        "rr_min": 8.0,
        "rr_max": 30.0,
    }


def _patient_seed(patient_id: str, offset: int = 0) -> int:
    total = 0
    for index, char in enumerate(str(patient_id)):
        total += (index + 1) * ord(char)
    return (total + offset) % 100_000


def _finite_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _certificate_reliability_weight(certificate: Mapping[str, Any]) -> float:
    for key in ("reliability_w", "w_t"):
        numeric = _finite_float_or_none(certificate.get(key))
        if numeric is not None:
            return float(numeric)
    reliability = certificate.get("reliability")
    if isinstance(reliability, Mapping):
        for key in ("w_t", "w", "reliability_w"):
            numeric = _finite_float_or_none(reliability.get(key))
            if numeric is not None:
                return float(numeric)
        return 0.0
    numeric = _finite_float_or_none(reliability)
    return float(numeric) if numeric is not None else 0.0


def _predict_certificate_validity(
    certificate: Mapping[str, Any] | None,
    previous_certificate: Mapping[str, Any] | None,
) -> bool:
    if certificate is None or missing_required_certificate_fields(certificate):
        return False
    if not certificate_intervention_semantics_valid(certificate):
        return False
    current_prev_hash = certificate.get("prev_hash")
    if previous_certificate is None:
        if current_prev_hash not in (None, ""):
            return False
    elif current_prev_hash != previous_certificate.get("certificate_hash"):
        return False
    horizon_value = extract_certificate_validity_horizon(certificate)
    return horizon_value > 0 and _certificate_reliability_weight(certificate) >= 0.0


def _independent_certificate_validity(
    certificate: Mapping[str, Any] | None,
    previous_certificate: Mapping[str, Any] | None,
) -> bool:
    if certificate is None or missing_required_certificate_fields(certificate):
        return False
    if not certificate_intervention_semantics_valid(certificate):
        return False
    verdict = formal_validity_predicate(certificate, previous_certificate, w_min=0.0)
    return bool(verdict.valid)


def _fault_family(faults: list[Any]) -> str:
    if not faults:
        return "nominal"
    return str(faults[0].kind)


def _summary_row(controller_name: str, records: list[StepRecord]) -> dict[str, Any]:
    metrics = compute_all_metrics(records)
    n_steps = max(metrics.n_steps, 1)
    fallback_activation_rate = sum(1 for record in records if record.fallback_used) / n_steps
    useful_work_total = sum(float(record.useful_work) for record in records)
    useful_work_mean = useful_work_total / n_steps
    max_alert_rate = sum(
        1
        for record in records
        if float(record.action.get("alert_level", 0.0) or 0.0) >= 0.999
    ) / n_steps
    return {
        "controller": controller_name,
        "tsvr": metrics.tsvr,
        "oasg": metrics.oasg,
        "cva": metrics.cva,
        "gdq": metrics.gdq,
        "intervention_rate": metrics.intervention_rate,
        "fallback_activation_rate": fallback_activation_rate,
        "max_alert_rate": max_alert_rate,
        "useful_work_total": useful_work_total,
        "useful_work_mean": useful_work_mean,
        "audit_completeness": metrics.audit_completeness,
        "recovery_latency": metrics.recovery_latency,
        "n_steps": metrics.n_steps,
    }


def _trace_row(
    *,
    controller: str,
    patient_id: str,
    step: int,
    ts_utc: str,
    fault_family: str,
    candidate_action: Mapping[str, Any],
    safe_action: Mapping[str, Any],
    contract_state: Mapping[str, Any],
    observed_state: Mapping[str, Any],
    reliability_w: float,
    inflation: float,
    intervened: bool,
    fallback_used: bool,
    certificate_valid: bool,
    certificate_predicted_valid: bool,
    audit_fields_present: int,
    audit_fields_required: int,
    true_constraint_violated: bool,
    observed_constraint_satisfied: bool | None,
    true_margin: float | None,
    observed_margin: float | None,
    validity_horizon_H_t: int | None,
    half_life_steps: int | None,
    expires_at_step: int | None,
    validity_status: str,
    useful_work: float,
    latency_us: float,
    intervention_reason: str,
    repair_mode: str,
    fallback_reason: str,
    release_requires_max_alert: bool,
    theorem_contract: str,
    projected_release: bool = False,
    projected_release_margin: float | None = None,
    runtime_policy_family: str = "",
    contract_id: str = HEALTHCARE_FAIL_SAFE_CONTRACT_ID,
    source_theorem: str = "T11",
    t11_status: str = "missing",
    t11_failed_obligations: str = "",
    domain_postcondition_passed: bool = False,
    domain_postcondition_failure: str = "theorem_contract_not_evaluated",
    validity_scope: str = "not_certified",
    validity_theorem_id: str = "",
    validity_theorem_contract: str = "",
    certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    certificate_payload = dict(certificate or {})
    return {
        "trace_id": f"healthcare-{controller}-{patient_id}-{step}",
        "controller": controller,
        "patient_id": str(patient_id),
        "ts_utc": str(ts_utc),
        "step_index": int(step),
        "fault_family": fault_family,
        "candidate_alert_level": float(candidate_action.get("alert_level", 0.0)),
        "safe_alert_level": float(safe_action.get("alert_level", 0.0)),
        "intervened": bool(intervened),
        "fallback_used": bool(fallback_used),
        "repair_mode": repair_mode,
        "fallback_mode": "max_alert" if repair_mode == "fallback" else "hold",
        "intervention_reason": intervention_reason,
        "fallback_reason": fallback_reason,
        "release_requires_max_alert": bool(release_requires_max_alert),
        "projected_release": bool(projected_release),
        "projected_release_margin": projected_release_margin,
        "runtime_policy_family": runtime_policy_family,
        "certificate_valid": bool(certificate_valid),
        "certificate_predicted_valid": bool(certificate_predicted_valid),
        "audit_fields_present": int(audit_fields_present),
        "audit_fields_required": int(audit_fields_required),
        "true_constraint_violated": bool(true_constraint_violated),
        "observed_constraint_satisfied": observed_constraint_satisfied,
        "true_margin": true_margin,
        "observed_margin": observed_margin,
        "true_spo2_pct": float(contract_state.get("spo2_pct", float("nan"))),
        "observed_spo2_pct": float(observed_state.get("spo2_pct", float("nan"))),
        "true_forecast_spo2_pct": float(contract_state.get("forecast_spo2_pct", float("nan"))),
        "observed_forecast_spo2_pct": float(observed_state.get("forecast_spo2_pct", float("nan"))),
        "true_hr_bpm": float(contract_state.get("hr_bpm", float("nan"))),
        "observed_hr_bpm": float(observed_state.get("hr_bpm", float("nan"))),
        "true_respiratory_rate": float(contract_state.get("respiratory_rate", float("nan"))),
        "observed_respiratory_rate": float(observed_state.get("respiratory_rate", float("nan"))),
        "true_reliability": float(contract_state.get("reliability", 1.0)),
        "observed_reliability": float(observed_state.get("reliability", 1.0)),
        "telemetry_missing_count": int(observed_state.get("telemetry_missing_count", 0) or 0),
        "telemetry_missing_fields": str(observed_state.get("telemetry_missing_fields", "")),
        "reliability_w": float(reliability_w),
        "inflation": float(inflation),
        "validity_horizon_H_t": validity_horizon_H_t,
        "half_life_steps": half_life_steps,
        "expires_at_step": expires_at_step,
        "validity_status": validity_status,
        "certificate_schema_version": str(certificate_payload.get("certificate_schema_version", "")),
        "certificate_hash": str(certificate_payload.get("certificate_hash", "")),
        "prev_hash": str(certificate_payload.get("prev_hash", "") or ""),
        "issuer": str(certificate_payload.get("issuer", "")),
        "domain": str(certificate_payload.get("domain", "")),
        "action": json.dumps(certificate_payload.get("action", {}), sort_keys=True) if certificate_payload else "",
        "theorem_contracts": json.dumps(certificate_payload.get("theorem_contracts", {}), sort_keys=True) if certificate_payload else "",
        "validity_scope": validity_scope,
        "validity_theorem_id": validity_theorem_id,
        "validity_theorem_contract": validity_theorem_contract,
        "runtime_surface": HEALTHCARE_CFG["runtime_surface"],
        "closure_tier": HEALTHCARE_CFG["closure_tier"],
        "theorem_contract": theorem_contract,
        "contract_id": contract_id,
        "source_theorem": source_theorem,
        "t11_status": t11_status,
        "t11_failed_obligations": t11_failed_obligations,
        "domain_postcondition_passed": bool(domain_postcondition_passed),
        "domain_postcondition_failure": domain_postcondition_failure,
        "useful_work": float(useful_work),
        "latency_us": float(latency_us),
    }


def _observed_vitals_unsafe(state: Mapping[str, Any]) -> bool:
    try:
        spo2 = float(state.get("spo2_pct"))
        hr = float(state.get("hr_bpm"))
        rr = float(state.get("respiratory_rate"))
        forecast_spo2 = float(state.get("forecast_spo2_pct", spo2))
    except (TypeError, ValueError):
        return True
    if not all(math.isfinite(v) for v in (spo2, hr, rr, forecast_spo2)):
        return True
    return bool(spo2 < 90.0 or forecast_spo2 < 90.0 or hr < 40.0 or hr > 120.0 or rr < 8.0 or rr > 30.0)


def _healthcare_policy_action(
    policy_name: str,
    observed_state: Mapping[str, Any],
    candidate_action: Mapping[str, Any],
) -> tuple[dict[str, float], str]:
    if policy_name == "baseline":
        return {"alert_level": float(candidate_action.get("alert_level", 0.0) or 0.0)}, "nominal_deterministic_controller"
    if policy_name == "predictor_only_no_runtime":
        return {"alert_level": float(candidate_action.get("alert_level", 0.0) or 0.0)}, "predictor_only_no_runtime"
    if policy_name == "fixed_conservative_alert":
        return {"alert_level": 0.65}, "fixed_conservative_alert"
    if policy_name == "ews_threshold":
        reliability = _finite_float_or_none(observed_state.get("reliability"))
        alert = 1.0 if _observed_vitals_unsafe(observed_state) or (reliability is not None and reliability < 0.2) else 0.25
        return {"alert_level": float(alert)}, "ews_threshold"
    if policy_name == "conformal_alert_only":
        try:
            spo2 = float(observed_state.get("spo2_pct"))
            forecast_spo2 = float(observed_state.get("forecast_spo2_pct", spo2))
        except (TypeError, ValueError):
            return {"alert_level": 1.0}, "conformal_alert_only"
        alert = 0.85 if min(spo2, forecast_spo2) < 94.0 else 0.15
        return {"alert_level": float(alert)}, "conformal_alert_only"
    if policy_name == "stale_certificate_no_temporal_guard":
        alert = 1.0 if _observed_vitals_unsafe(observed_state) else 0.35
        return {"alert_level": float(alert)}, "stale_certificate_no_temporal_guard"
    raise ValueError(f"Unknown healthcare runtime policy: {policy_name}")


def _run_policy_episode(
    track: HealthcareTrackAdapter,
    patient_id: str,
    horizon: int,
    *,
    policy_name: str,
    seed_offset: int,
) -> tuple[list[StepRecord], list[dict[str, Any]]]:
    controller = DomainAwareController(NominalController(), track.domain_name)
    schedule = generate_fault_schedule(_patient_seed(patient_id, offset=seed_offset), horizon)
    track.load_episode(patient_id)
    records: list[StepRecord] = []
    trace_rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []

    for step in range(horizon):
        contract_state = dict(track.true_state())
        faults = active_faults(schedule, step)
        fault_dict = {"kind": faults[0].kind, **faults[0].params} if faults else None
        observed_state = dict(track.observe(contract_state, fault_dict))
        candidate_action = controller.propose_action(observed_state, certificate_state=None)
        latency_start = time.perf_counter_ns()
        safe_action, policy_family = _healthcare_policy_action(policy_name, observed_state, candidate_action)
        latency_us = (time.perf_counter_ns() - latency_start) / 1_000.0

        release_requires = bool(_observed_vitals_unsafe(observed_state) and float(safe_action.get("alert_level", 0.0)) >= 0.999)
        contract_state = {
            **contract_state,
            **safe_action,
            "validity_status": policy_name,
            "release_requires_max_alert": release_requires,
            "runtime_policy_family": policy_family,
        }
        violation = track.check_violation(contract_state)
        observed_safe = track.observed_constraint_satisfied({**observed_state, **safe_action, "release_requires_max_alert": release_requires})
        true_margin = track.constraint_margin(contract_state)
        observed_margin = track.constraint_margin({**observed_state, **safe_action, "release_requires_max_alert": release_requires})
        trajectory.append(dict(contract_state))
        useful_work = track.compute_useful_work(trajectory[-2:] if len(trajectory) >= 2 else [trajectory[-1]])
        track.step(safe_action)
        fallback_used = float(safe_action.get("alert_level", 0.0)) >= 0.999 and policy_name in {
            "ews_threshold",
            "conformal_alert_only",
            "stale_certificate_no_temporal_guard",
        }

        records.append(
            StepRecord(
                step=step,
                true_state=contract_state,
                observed_state=observed_state,
                action=safe_action,
                true_constraint_violated=bool(violation["violated"]),
                observed_constraint_satisfied=observed_safe,
                true_margin=true_margin,
                observed_margin=observed_margin,
                intervened=safe_action != dict(candidate_action),
                fallback_used=fallback_used,
                certificate_valid=False,
                certificate_predicted_valid=False,
                useful_work=0.0 if math.isnan(useful_work) else useful_work,
                audit_fields_present=0,
                audit_fields_required=len(REQUIRED_CERTIFICATE_FIELDS),
                latency_us=float(latency_us),
            )
        )
        trace_rows.append(
            _trace_row(
                controller=policy_name,
                patient_id=patient_id,
                ts_utc=str(contract_state.get("ts_utc", "")),
                step=step,
                fault_family=_fault_family(faults),
                candidate_action=candidate_action,
                safe_action=safe_action,
                contract_state=contract_state,
                observed_state=observed_state,
                reliability_w=float(contract_state.get("reliability", 1.0)),
                inflation=1.0,
                intervened=safe_action != dict(candidate_action),
                fallback_used=fallback_used,
                certificate_valid=False,
                certificate_predicted_valid=False,
                audit_fields_present=0,
                audit_fields_required=len(REQUIRED_CERTIFICATE_FIELDS),
                true_constraint_violated=bool(violation["violated"]),
                observed_constraint_satisfied=observed_safe,
                true_margin=true_margin,
                observed_margin=observed_margin,
                validity_horizon_H_t=None,
                half_life_steps=None,
                expires_at_step=None,
                validity_status=policy_name,
                useful_work=0.0 if math.isnan(useful_work) else useful_work,
                latency_us=float(latency_us),
                intervention_reason=policy_family if safe_action != dict(candidate_action) else "",
                repair_mode="policy",
                fallback_reason=policy_family if fallback_used else "",
                release_requires_max_alert=release_requires,
                theorem_contract="healthcare_fail_safe_release",
                domain_postcondition_passed=not bool(violation["violated"]),
                domain_postcondition_failure="non_orius_controller",
                runtime_policy_family=policy_family,
            )
        )

    return records, trace_rows


def _policy_summary_from_trace(policy_name: str, trace_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_steps = max(len(trace_rows), 1)
    violations = 0
    fallback_count = 0
    interventions = 0
    useful_work_total = 0.0
    for row in trace_rows:
        observed_state = {
            "spo2_pct": row.get("observed_spo2_pct"),
            "forecast_spo2_pct": row.get("observed_forecast_spo2_pct"),
            "hr_bpm": row.get("observed_hr_bpm"),
            "respiratory_rate": row.get("observed_respiratory_rate"),
            "reliability": row.get("observed_reliability"),
        }
        candidate_action = {"alert_level": _finite_float_or_none(row.get("candidate_alert_level")) or 0.0}
        safe_action, _policy_family = _healthcare_policy_action(policy_name, observed_state, candidate_action)
        alert = max(0.0, min(1.0, float(safe_action.get("alert_level", 0.0))))
        true_state = {
            "spo2_pct": row.get("true_spo2_pct"),
            "forecast_spo2_pct": row.get("true_forecast_spo2_pct"),
            "hr_bpm": row.get("true_hr_bpm"),
            "respiratory_rate": row.get("true_respiratory_rate"),
        }
        violated = bool(_observed_vitals_unsafe(true_state) and alert < 0.999)
        violations += int(violated)
        fallback_count += int(alert >= 0.999)
        interventions += int(abs(alert - float(candidate_action["alert_level"])) > 1e-9)
        if not violated:
            useful_work_total += 1.0 - alert
    tsvr = violations / n_steps
    fallback_rate = fallback_count / n_steps
    return {
        "controller": policy_name,
        "tsvr": tsvr,
        "oasg": max(0.0, 1.0 - tsvr),
        "cva": 0.0,
        "gdq": 0.0,
        "intervention_rate": interventions / n_steps,
        "fallback_activation_rate": fallback_rate,
        "max_alert_rate": fallback_rate,
        "useful_work_total": useful_work_total,
        "useful_work_mean": useful_work_total / n_steps,
        "audit_completeness": 0.0,
        "recovery_latency": 0.0,
        "n_steps": len(trace_rows),
    }


def _run_baseline_episode(track: HealthcareTrackAdapter, patient_id: str, horizon: int) -> tuple[list[StepRecord], list[dict[str, Any]]]:
    controller = DomainAwareController(NominalController(), track.domain_name)
    schedule = generate_fault_schedule(_patient_seed(patient_id), horizon)
    track.load_episode(patient_id)
    records: list[StepRecord] = []
    trace_rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []

    for step in range(horizon):
        contract_state = dict(track.true_state())
        faults = active_faults(schedule, step)
        fault_dict = {"kind": faults[0].kind, **faults[0].params} if faults else None
        observed_state = dict(track.observe(contract_state, fault_dict))
        candidate_action = controller.propose_action(observed_state, certificate_state=None)
        latency_start = time.perf_counter_ns()
        safe_action = dict(candidate_action)
        latency_us = (time.perf_counter_ns() - latency_start) / 1_000.0

        contract_state = {
            **contract_state,
            **safe_action,
            "validity_status": "baseline_no_certificate",
            "release_requires_max_alert": track.release_contract_status({**contract_state, **safe_action})["requires_max_alert"],
        }
        violation = track.check_violation(contract_state)
        observed_safe = track.observed_constraint_satisfied(observed_state)
        true_margin = track.constraint_margin(contract_state)
        observed_margin = track.constraint_margin(observed_state)
        trajectory.append(dict(contract_state))
        useful_work = track.compute_useful_work(trajectory[-2:] if len(trajectory) >= 2 else [trajectory[-1]])
        track.step(safe_action)

        records.append(
            StepRecord(
                step=step,
                true_state=contract_state,
                observed_state=observed_state,
                action=safe_action,
                true_constraint_violated=bool(violation["violated"]),
                observed_constraint_satisfied=observed_safe,
                true_margin=true_margin,
                observed_margin=observed_margin,
                intervened=False,
                fallback_used=False,
                certificate_valid=False,
                certificate_predicted_valid=False,
                useful_work=0.0 if math.isnan(useful_work) else useful_work,
                audit_fields_present=0,
                audit_fields_required=len(REQUIRED_CERTIFICATE_FIELDS),
                latency_us=float(latency_us),
            )
        )
        trace_rows.append(
            _trace_row(
                controller="baseline",
                patient_id=patient_id,
                ts_utc=str(contract_state.get("ts_utc", "")),
                step=step,
                fault_family=_fault_family(faults),
                candidate_action=candidate_action,
                safe_action=safe_action,
                contract_state=contract_state,
                observed_state=observed_state,
                reliability_w=float(contract_state.get("reliability", 1.0)),
                inflation=1.0,
                intervened=False,
                fallback_used=False,
                certificate_valid=False,
                certificate_predicted_valid=False,
                audit_fields_present=0,
                audit_fields_required=len(REQUIRED_CERTIFICATE_FIELDS),
                true_constraint_violated=bool(violation["violated"]),
                observed_constraint_satisfied=observed_safe,
                true_margin=true_margin,
                observed_margin=observed_margin,
                validity_horizon_H_t=None,
                half_life_steps=None,
                expires_at_step=None,
                validity_status="baseline_no_certificate",
                useful_work=0.0 if math.isnan(useful_work) else useful_work,
                latency_us=float(latency_us),
                intervention_reason="",
                repair_mode="hold",
                fallback_reason="",
                release_requires_max_alert=bool(contract_state.get("release_requires_max_alert", False)),
                theorem_contract="healthcare_fail_safe_release",
                domain_postcondition_passed=not bool(violation["violated"]),
                domain_postcondition_failure="non_orius_controller",
            )
        )

    return records, trace_rows


def _run_orius_episode(
    track: HealthcareTrackAdapter,
    patient_id: str,
    horizon: int,
    *,
    previous_certificate: dict[str, Any] | None = None,
) -> tuple[list[StepRecord], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    controller = DomainAwareController(DC3SController(), track.domain_name)
    adapter = HealthcareDomainAdapter(HEALTHCARE_CFG)
    schedule = generate_fault_schedule(_patient_seed(patient_id, offset=17), horizon)
    track.load_episode(patient_id)

    history: list[dict[str, Any]] = []
    records: list[StepRecord] = []
    trace_rows: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for step in range(horizon):
        contract_state = dict(track.true_state())
        faults = active_faults(schedule, step)
        fault_dict = {"kind": faults[0].kind, **faults[0].params} if faults else None
        observed_state = dict(track.observe(contract_state, fault_dict))
        raw_telemetry = dict(observed_state)
        if history:
            previous_state = history[-1]
            for key in ("hr_bpm", "spo2_pct", "respiratory_rate", "reliability", "forecast_spo2_pct"):
                raw_telemetry.setdefault(f"_hold_{key}", previous_state.get(key, 0.0))

        candidate_action = controller.propose_action(observed_state, certificate_state=None)
        latency_start = time.perf_counter_ns()
        result = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=raw_telemetry,
            history=history,
            candidate_action=candidate_action,
            constraints=_constraint_payload(),
            quantile=5.0,
            cfg={**HEALTHCARE_CFG, "step_index": step, "planning_horizon_steps": horizon},
            prev_cert_hash=(previous_certificate or {}).get("certificate_hash"),
            device_id="mimic-monitor-0",
            zone_id="healthcare",
            controller="orius_healthcare_runtime",
        )
        latency_us = (time.perf_counter_ns() - latency_start) / 1_000.0
        safe_action = dict(result["safe_action"])
        certificate = dict(result["certificate"])
        if certificate.get("validity_horizon_H_t") in (None, "") and certificate.get("certificate_horizon_steps") not in (None, ""):
            certificate["validity_horizon_H_t"] = int(certificate["certificate_horizon_steps"])
        certificate["certificate_hash"] = recompute_certificate_hash(certificate)

        certificate_predicted_valid = _predict_certificate_validity(certificate, previous_certificate)
        certificate_valid = _independent_certificate_validity(certificate, previous_certificate)
        previous_certificate = dict(certificate)
        certificates.append(certificate)

        repair_meta = dict(result["repair_meta"])
        fallback_used = str(repair_meta.get("mode", "")) == "fallback"
        projected_release = bool(repair_meta.get("projected_release", False))
        contract_state = {
            **contract_state,
            **safe_action,
            "certificate_valid": certificate_valid,
            "validity_status": str(certificate.get("validity_status", "nominal")),
            "release_requires_max_alert": bool(repair_meta.get("fallback_required", fallback_used)),
            "projected_release_valid": bool(projected_release and certificate_valid),
            "projected_release_margin": float(repair_meta.get("projected_release_margin", 0.0)),
        }
        observed_contract_state = {
            **observed_state,
            **safe_action,
            "certificate_valid": certificate_valid,
            "validity_status": str(certificate.get("validity_status", "nominal")),
            "release_requires_max_alert": bool(repair_meta.get("fallback_required", fallback_used)),
            "projected_release_valid": bool(projected_release and certificate_valid),
            "projected_release_margin": float(repair_meta.get("projected_release_margin", 0.0)),
        }
        violation = track.check_violation(contract_state)
        observed_safe = track.observed_constraint_satisfied(observed_contract_state)
        true_margin = track.constraint_margin(contract_state)
        observed_margin = track.constraint_margin(observed_contract_state)
        audit_fields_present = count_present_required_certificate_fields(certificate)
        audit_fields_required = len(REQUIRED_CERTIFICATE_FIELDS)
        trajectory.append(dict(contract_state))
        useful_work = track.compute_useful_work(trajectory[-2:] if len(trajectory) >= 2 else [trajectory[-1]])
        reliability_w = float(result["reliability_w"])
        inflation = float(result["uncertainty_set"].get("meta", {}).get("inflation", 1.0))
        track.step(safe_action)
        trace_id = f"healthcare-orius-{patient_id}-{step}"
        theorem_trace_fields = witness_trace_fields_from_result(
            domain="healthcare",
            trace_id=trace_id,
            theorem_contracts=result["theorem_contracts"],
            certificate_valid=certificate_valid,
            postcondition_passed=not bool(violation["violated"]),
            post_margin=true_margin,
        )

        records.append(
            StepRecord(
                step=step,
                true_state=contract_state,
                observed_state=observed_contract_state,
                action=safe_action,
                true_constraint_violated=bool(violation["violated"]),
                observed_constraint_satisfied=observed_safe,
                true_margin=true_margin,
                observed_margin=observed_margin,
                intervened=bool(repair_meta.get("repaired", False)),
                fallback_used=fallback_used,
                certificate_valid=certificate_valid,
                certificate_predicted_valid=certificate_predicted_valid,
                useful_work=0.0 if math.isnan(useful_work) else useful_work,
                audit_fields_present=audit_fields_present,
                audit_fields_required=audit_fields_required,
                latency_us=float(latency_us),
            )
        )
        trace_rows.append(
            _trace_row(
                controller="orius",
                patient_id=patient_id,
                ts_utc=str(contract_state.get("ts_utc", "")),
                step=step,
                fault_family=_fault_family(faults),
                candidate_action=candidate_action,
                safe_action=safe_action,
                contract_state=contract_state,
                observed_state=observed_contract_state,
                reliability_w=reliability_w,
                inflation=inflation,
                intervened=bool(repair_meta.get("repaired", False)),
                fallback_used=fallback_used,
                certificate_valid=certificate_valid,
                certificate_predicted_valid=certificate_predicted_valid,
                audit_fields_present=audit_fields_present,
                audit_fields_required=audit_fields_required,
                true_constraint_violated=bool(violation["violated"]),
                observed_constraint_satisfied=observed_safe,
                true_margin=true_margin,
                observed_margin=observed_margin,
                validity_horizon_H_t=certificate.get("validity_horizon_H_t"),
                half_life_steps=certificate.get("half_life_steps"),
                expires_at_step=certificate.get("expires_at_step"),
                validity_status=str(certificate.get("validity_status", "")),
                validity_scope=str(certificate.get("validity_scope", "")),
                validity_theorem_id=str(certificate.get("validity_theorem_id", "")),
                validity_theorem_contract=str(certificate.get("validity_theorem_contract", "")),
                useful_work=0.0 if math.isnan(useful_work) else useful_work,
                latency_us=float(latency_us),
                intervention_reason=str(repair_meta.get("intervention_reason", "")),
                repair_mode=str(repair_meta.get("mode", "hold")),
                fallback_reason=str(repair_meta.get("intervention_reason", "")),
                release_requires_max_alert=bool(repair_meta.get("fallback_required", fallback_used)),
                theorem_contract=str(repair_meta.get("theorem_contract", "healthcare_fail_safe_release")),
                projected_release=projected_release,
                projected_release_margin=float(repair_meta.get("projected_release_margin", 0.0)),
                runtime_policy_family="orius_full_stack",
                certificate=certificate,
                **theorem_trace_fields,
            )
        )
        history.append(dict(result["state"]))

    return records, trace_rows, certificates, previous_certificate


def _run_always_alert_episode(track: HealthcareTrackAdapter, patient_id: str, horizon: int) -> tuple[list[StepRecord], list[dict[str, Any]]]:
    schedule = generate_fault_schedule(_patient_seed(patient_id, offset=31), horizon)
    track.load_episode(patient_id)
    records: list[StepRecord] = []
    trace_rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []

    for step in range(horizon):
        contract_state = dict(track.true_state())
        faults = active_faults(schedule, step)
        fault_dict = {"kind": faults[0].kind, **faults[0].params} if faults else None
        observed_state = dict(track.observe(contract_state, fault_dict))
        candidate_action = {"alert_level": 1.0}
        latency_start = time.perf_counter_ns()
        safe_action = {"alert_level": 1.0}
        latency_us = (time.perf_counter_ns() - latency_start) / 1_000.0

        contract_state = {
            **contract_state,
            **safe_action,
            "validity_status": "always_alert",
            "release_requires_max_alert": True,
        }
        violation = track.check_violation(contract_state)
        observed_safe = track.observed_constraint_satisfied({**observed_state, "release_requires_max_alert": True})
        true_margin = track.constraint_margin(contract_state)
        observed_margin = track.constraint_margin(observed_state)
        trajectory.append(dict(contract_state))
        useful_work = track.compute_useful_work(trajectory[-2:] if len(trajectory) >= 2 else [trajectory[-1]])
        track.step(safe_action)

        records.append(
            StepRecord(
                step=step,
                true_state=contract_state,
                observed_state=observed_state,
                action=safe_action,
                true_constraint_violated=bool(violation["violated"]),
                observed_constraint_satisfied=observed_safe,
                true_margin=true_margin,
                observed_margin=observed_margin,
                intervened=True,
                fallback_used=True,
                certificate_valid=True,
                certificate_predicted_valid=True,
                useful_work=0.0 if math.isnan(useful_work) else useful_work,
                audit_fields_present=len(REQUIRED_CERTIFICATE_FIELDS),
                audit_fields_required=len(REQUIRED_CERTIFICATE_FIELDS),
                latency_us=float(latency_us),
            )
        )
        trace_rows.append(
            _trace_row(
                controller="always_alert",
                patient_id=patient_id,
                ts_utc=str(contract_state.get("ts_utc", "")),
                step=step,
                fault_family=_fault_family(faults),
                candidate_action=candidate_action,
                safe_action=safe_action,
                contract_state=contract_state,
                observed_state=observed_state,
                reliability_w=float(contract_state.get("reliability", 1.0)),
                inflation=1.0,
                intervened=True,
                fallback_used=True,
                certificate_valid=True,
                certificate_predicted_valid=True,
                audit_fields_present=len(REQUIRED_CERTIFICATE_FIELDS),
                audit_fields_required=len(REQUIRED_CERTIFICATE_FIELDS),
                true_constraint_violated=bool(violation["violated"]),
                observed_constraint_satisfied=observed_safe,
                true_margin=true_margin,
                observed_margin=observed_margin,
                validity_horizon_H_t=0,
                half_life_steps=0,
                expires_at_step=step,
                validity_status="always_alert",
                validity_scope="degenerate_comparator",
                validity_theorem_id="",
                validity_theorem_contract="",
                useful_work=0.0 if math.isnan(useful_work) else useful_work,
                latency_us=float(latency_us),
                intervention_reason="degenerate_always_alert",
                repair_mode="fallback",
                fallback_reason="degenerate_always_alert",
                release_requires_max_alert=True,
                theorem_contract="healthcare_fail_safe_release",
                domain_postcondition_passed=not bool(violation["violated"]),
                domain_postcondition_failure="non_orius_controller",
            )
        )

    return records, trace_rows


def build_healthcare_runtime_artifacts(
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    seeds: int | None = None,
    horizon: int | None = None,
    start_seed: int = 2000,
) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing promoted healthcare runtime dataset: {dataset_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "healthcare_runtime.duckdb"
    if db_path.exists():
        db_path.unlink()

    baseline_track = HealthcareTrackAdapter(dataset_path=dataset_path)
    orius_track = HealthcareTrackAdapter(dataset_path=dataset_path)
    always_alert_track = HealthcareTrackAdapter(dataset_path=dataset_path)
    policy_names = (
        "ews_threshold",
        "conformal_alert_only",
        "predictor_only_no_runtime",
        "fixed_conservative_alert",
        "stale_certificate_no_temporal_guard",
    )

    patient_ids = baseline_track.episode_ids
    if not patient_ids:
        raise RuntimeError("Promoted healthcare runtime surface is empty.")
    if seeds is None:
        selected_patients = patient_ids
    else:
        selected_patients = [
            patient_ids[index % len(patient_ids)]
            for index in range(start_seed, start_seed + max(int(seeds), 0))
        ]

    all_trace_rows: list[dict[str, Any]] = []
    baseline_trace_rows_all: list[dict[str, Any]] = []
    all_certificates: list[dict[str, Any]] = []
    baseline_records_all: list[StepRecord] = []
    orius_records_all: list[StepRecord] = []
    always_alert_records_all: list[StepRecord] = []
    previous_certificate: dict[str, Any] | None = None

    for patient_id in selected_patients:
        episode_horizon = baseline_track.episode_length(patient_id)
        if horizon is not None:
            episode_horizon = min(episode_horizon, int(horizon))

        baseline_records, baseline_traces = _run_baseline_episode(baseline_track, patient_id, episode_horizon)
        orius_records, orius_traces, orius_certs, previous_certificate = _run_orius_episode(
            orius_track,
            patient_id,
            episode_horizon,
            previous_certificate=previous_certificate,
        )
        always_alert_records, always_alert_traces = _run_always_alert_episode(
            always_alert_track,
            patient_id,
            episode_horizon,
        )

        baseline_records_all.extend(baseline_records)
        orius_records_all.extend(orius_records)
        always_alert_records_all.extend(always_alert_records)
        baseline_trace_rows_all.extend(baseline_traces)
        all_trace_rows.extend(baseline_traces)
        all_trace_rows.extend(orius_traces)
        all_trace_rows.extend(always_alert_traces)
        all_certificates.extend(orius_certs)

    trace_df = pd.DataFrame(all_trace_rows)
    trace_path = out_dir / "runtime_traces.csv"
    trace_df.to_csv(trace_path, index=False)

    summary_rows = [
        _summary_row("baseline", baseline_records_all),
        *[
            _policy_summary_from_trace(name, baseline_trace_rows_all)
            for name in policy_names
        ],
        _summary_row("always_alert", always_alert_records_all),
        _summary_row("orius", orius_records_all),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "runtime_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    store_certificates_batch(all_certificates, duckdb_path=str(db_path), table_name=TABLE_NAME)
    certificates = load_certificates_from_duckdb(db_path, table_name=TABLE_NAME)
    certos_summary, failure_df, expiry_df, governance_df = verify_certificates(certificates)

    certos_summary_path = out_dir / "certos_verification_summary.json"
    failures_path = out_dir / "certos_verification_failures.csv"
    expiry_path = out_dir / "certificate_expiry_trace.csv"
    governance_path = out_dir / "runtime_governance_summary.csv"

    certos_summary_path.write_text(json.dumps(certos_summary, indent=2) + "\n", encoding="utf-8")
    failure_df.to_csv(failures_path, index=False)
    expiry_df.to_csv(expiry_path, index=False)
    governance_df.to_csv(governance_path, index=False)

    conn = duckdb.connect(str(db_path))
    try:
        conn.register("runtime_trace_df", trace_df)
        conn.register("runtime_summary_df", summary_df)
        conn.execute("CREATE OR REPLACE TABLE healthcare_runtime_traces AS SELECT * FROM runtime_trace_df")
        conn.execute("CREATE OR REPLACE TABLE healthcare_runtime_summary AS SELECT * FROM runtime_summary_df")
    finally:
        conn.close()

    comparison_rows = [
        {
            "controller": "orius",
            "tsvr": float(summary_df.loc[summary_df["controller"] == "orius", "tsvr"].iloc[0]),
            "max_alert_rate": float(summary_df.loc[summary_df["controller"] == "orius", "max_alert_rate"].iloc[0]),
            "useful_work_total": float(summary_df.loc[summary_df["controller"] == "orius", "useful_work_total"].iloc[0]),
        },
        {
            "controller": "always_alert",
            "tsvr": float(summary_df.loc[summary_df["controller"] == "always_alert", "tsvr"].iloc[0]),
            "max_alert_rate": float(summary_df.loc[summary_df["controller"] == "always_alert", "max_alert_rate"].iloc[0]),
            "useful_work_total": float(summary_df.loc[summary_df["controller"] == "always_alert", "useful_work_total"].iloc[0]),
        },
    ]
    comparison_path = out_dir / "runtime_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    equal_domain_artifacts = _write_equal_domain_artifacts(out_dir)

    return {
        "runtime_traces_csv": str(trace_path),
        "runtime_summary_csv": str(summary_path),
        "healthcare_runtime_duckdb": str(db_path),
        "certos_verification_summary_json": str(certos_summary_path),
        "runtime_governance_summary_csv": str(governance_path),
        "runtime_comparison_csv": str(comparison_path),
        **equal_domain_artifacts,
        "trace_rows": int(len(trace_df)),
        "certificate_rows": int(len(certificates)),
        "baseline_tsvr": float(summary_df.loc[summary_df["controller"] == "baseline", "tsvr"].iloc[0]),
        "orius_tsvr": float(summary_df.loc[summary_df["controller"] == "orius", "tsvr"].iloc[0]),
        "always_alert_tsvr": float(summary_df.loc[summary_df["controller"] == "always_alert", "tsvr"].iloc[0]),
        "orius_cva": float(summary_df.loc[summary_df["controller"] == "orius", "cva"].iloc[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build promoted healthcare runtime artifacts.")
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--start-seed", type=int, default=2000)
    args = parser.parse_args()

    report = build_healthcare_runtime_artifacts(
        dataset_path=args.dataset_path,
        out_dir=args.out_dir,
        seeds=args.seeds,
        horizon=args.horizon,
        start_seed=args.start_seed,
    )
    print(
        "[healthcare-runtime] "
        f"rows={report['trace_rows']} certs={report['certificate_rows']} "
        f"baseline_tsvr={report['baseline_tsvr']:.4f} "
        f"orius_tsvr={report['orius_tsvr']:.4f} "
        f"always_alert_tsvr={report['always_alert_tsvr']:.4f} "
        f"orius_cva={report['orius_cva']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
