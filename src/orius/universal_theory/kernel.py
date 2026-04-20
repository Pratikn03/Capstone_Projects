"""Pure orchestration helpers for the universal degraded-observation kernel."""
from __future__ import annotations

import uuid
from typing import Any, Mapping, Sequence

from orius.dc3s.drift import PageHinkleyDetector

from .contracts import (
    ContractVerifier,
    DomainInstantiation,
    ObservationConsistentStateSet,
    ObservationPacket,
    ReliabilityAssessment,
    RepairDecision,
    SafeActionSet,
    SafetyCertificate,
    SafetySpec,
    UniversalStepResult,
    _bool_from_payload,
)
from .risk_bounds import (
    build_t3a_contract_summary,
    compute_episode_risk_bound,
    compute_step_risk_bound,
    verify_inflation_geq_one,
)


def _extract_interval_bounds(uncertainty: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    lower_bounds: dict[str, float] = {}
    upper_bounds: dict[str, float] = {}
    for key, value in uncertainty.items():
        if key == "meta":
            continue
        if key.endswith("_lower") or key.endswith("_lo"):
            lower_bounds[key.rsplit("_", 1)[0]] = float(value)
        elif key.endswith("_upper") or key.endswith("_hi"):
            upper_bounds[key.rsplit("_", 1)[0]] = float(value)
        elif "_lower_" in key:
            base, _, suffix = key.partition("_lower_")
            lower_bounds[f"{base}_{suffix}"] = float(value)
        elif "_upper_" in key:
            base, _, suffix = key.partition("_upper_")
            upper_bounds[f"{base}_{suffix}"] = float(value)
    return lower_bounds, upper_bounds


def build_observation_packet(
    *,
    raw_telemetry: Mapping[str, Any],
    parsed_state: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None,
    domain_id: str = "",
) -> ObservationPacket:
    timestamp = str(
        parsed_state.get("ts_utc")
        or raw_telemetry.get("ts_utc")
        or raw_telemetry.get("timestamp")
        or ""
    )
    return ObservationPacket(
        raw_telemetry=dict(raw_telemetry),
        parsed_state=dict(parsed_state),
        timestamp_utc=timestamp,
        history_length=len(history or ()),
        domain_id=str(domain_id),
    )


def build_reliability_assessment(
    *,
    weight: float,
    flags: Mapping[str, Any],
    drift_meta: Mapping[str, Any],
) -> ReliabilityAssessment:
    return ReliabilityAssessment(
        weight=float(weight),
        flags=dict(flags),
        drift_flag=bool(drift_meta.get("drift", False)),
        drift_meta=dict(drift_meta),
    )


def build_observation_consistent_state_set(
    *,
    observed_state: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    quantile: float,
    reliability: ReliabilityAssessment,
    calibration_meta: Mapping[str, Any] | None = None,
) -> ObservationConsistentStateSet:
    meta = dict(uncertainty.get("meta", {}))
    if calibration_meta:
        meta.update(dict(calibration_meta))
    raw_inflation = float(meta.get("inflation", 1.0))
    # Validate BEFORE clamping: callers must not supply sub-unit inflation.
    verify_inflation_geq_one(raw_inflation)
    inflation = max(1.0, raw_inflation)
    meta["inflation"] = inflation
    lower_bounds, upper_bounds = _extract_interval_bounds(uncertainty)
    return ObservationConsistentStateSet(
        observed_state=dict(observed_state),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        raw_uncertainty={**dict(uncertainty), "meta": meta},
        inflation=inflation,
        quantile=float(quantile),
        reliability_weight=float(reliability.weight),
    )


def build_safety_spec(
    *,
    constraints: Mapping[str, Any],
    fallback_action: Mapping[str, float] | None = None,
) -> SafetySpec:
    return SafetySpec(
        constraints=dict(constraints),
        fallback_action=None if fallback_action is None else dict(fallback_action),
    )


def derive_safe_action_set(
    *,
    tightened_set: Mapping[str, Any],
    safety_spec: SafetySpec,
) -> SafeActionSet:
    representation = dict(tightened_set)
    viable = not bool(
        representation.get("empty", False)
        or representation.get("is_empty", False)
        or representation.get("viable") is False
    )
    fallback = representation.get("fallback_action", safety_spec.fallback_action)
    fallback_action = None if fallback is None else {str(k): float(v) for k, v in dict(fallback).items()}
    return SafeActionSet(
        representation=representation,
        viable=bool(viable),
        fallback_action=fallback_action,
    )


def build_repair_decision(
    *,
    proposed_action: Mapping[str, Any],
    safe_action: Mapping[str, Any],
    repair_meta: Mapping[str, Any] | None,
    safe_action_set: SafeActionSet,
) -> RepairDecision:
    meta = dict(repair_meta or {})
    repaired = _bool_from_payload(meta, "repaired", default=dict(proposed_action) != dict(safe_action))
    if not safe_action and safe_action_set.fallback_action is not None:
        safe_action = dict(safe_action_set.fallback_action)
        repaired = True
        meta["intervention_reason"] = "fallback_action"
        meta["mode"] = "fallback"
    return RepairDecision(
        proposed_action=dict(proposed_action),
        safe_action=dict(safe_action),
        repaired=bool(repaired),
        mode=str(meta.get("mode", "projection")),
        reason=str(meta.get("intervention_reason")) if meta.get("intervention_reason") not in (None, "") else None,
        metadata=meta,
    )


def build_safety_certificate(
    *,
    payload: Mapping[str, Any],
    source_domain: str,
    assumption_tags: Sequence[str],
) -> SafetyCertificate:
    return SafetyCertificate.from_payload(
        payload,
        assumptions_checked=tuple(dict.fromkeys(str(tag) for tag in assumption_tags)),
        source_domain=source_domain,
    )


def _extract_reliability_history(
    history: Sequence[Mapping[str, Any]] | None,
    *,
    current_weight: float,
) -> list[float]:
    weights: list[float] = []
    for item in history or ():
        if not isinstance(item, Mapping):
            continue
        candidate = item.get("reliability_w")
        if candidate is None:
            candidate = item.get("w_t")
        if candidate is None and isinstance(item.get("reliability"), Mapping):
            reliability_payload = item.get("reliability")
            candidate = reliability_payload.get("w_t", reliability_payload.get("weight", reliability_payload.get("w")))
        if candidate is None:
            continue
        try:
            scalar = float(candidate)
        except (TypeError, ValueError):
            continue
        if not 0.0 <= scalar <= 1.0:
            raise ValueError(f"Reliability history value must lie in [0, 1]. Got {scalar!r}.")
        weights.append(scalar)
    if not 0.0 <= float(current_weight) <= 1.0:
        raise ValueError(f"current_weight must lie in [0, 1]. Got {current_weight!r}.")
    weights.append(float(current_weight))
    return weights


def execute_universal_step(
    *,
    domain_adapter: DomainInstantiation,
    raw_telemetry: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] | None,
    candidate_action: Mapping[str, Any],
    constraints: Mapping[str, Any],
    quantile: float = 50.0,
    cfg: Mapping[str, Any] | None = None,
    drift_detector: PageHinkleyDetector | None = None,
    residual: float | None = None,
    prev_cert_hash: str | None = None,
    device_id: str = "device-0",
    zone_id: str = "zone-0",
    controller: str = "orius-universal",
) -> UniversalStepResult:
    """Execute one Detect-Calibrate-Constrain-Shield-Certify step."""
    ContractVerifier.check(domain_adapter)

    dcfg = dict(cfg or {})
    state = dict(domain_adapter.ingest_telemetry(raw_telemetry))
    domain_id = getattr(domain_adapter, "domain_id", domain_adapter.__class__.__name__.lower())
    observation = build_observation_packet(
        raw_telemetry=raw_telemetry,
        parsed_state=state,
        history=history,
        domain_id=str(domain_id),
    )

    weight, flags = domain_adapter.compute_oqe(state, history)
    drift_meta: dict[str, Any] = {"drift": False, "score": 0.0}
    if drift_detector is not None and residual is not None:
        drift_meta = dict(drift_detector.update(abs(float(residual))))
    reliability = build_reliability_assessment(
        weight=float(weight),
        flags=flags,
        drift_meta=drift_meta,
    )

    uncertainty, cal_meta = domain_adapter.build_uncertainty_set(
        state=state,
        reliability_w=float(reliability.weight),
        quantile=quantile,
        cfg=dcfg,
        drift_flag=reliability.drift_flag,
        prev_meta=None,
    )
    state_set = build_observation_consistent_state_set(
        observed_state=state,
        uncertainty=uncertainty,
        quantile=quantile,
        reliability=reliability,
        calibration_meta=cal_meta,
    )

    tightened = domain_adapter.tighten_action_set(
        uncertainty=state_set.raw_uncertainty,
        constraints=constraints,
        cfg=dcfg,
    )
    safety_spec = build_safety_spec(constraints=constraints)
    safe_action_set = derive_safe_action_set(
        tightened_set=tightened,
        safety_spec=safety_spec,
    )

    safe_action, repair_meta = domain_adapter.repair_action(
        candidate_action=candidate_action,
        tightened_set=tightened,
        state=state,
        uncertainty=state_set.raw_uncertainty,
        constraints=constraints,
        cfg=dcfg,
    )
    repair_decision = build_repair_decision(
        proposed_action=candidate_action,
        safe_action=safe_action,
        repair_meta=repair_meta,
        safe_action_set=safe_action_set,
    )

    command_id = str(uuid.uuid4())
    certificate_payload = domain_adapter.emit_certificate(
        command_id=command_id,
        device_id=device_id,
        zone_id=zone_id,
        controller=controller,
        proposed_action=dict(candidate_action),
        safe_action=dict(repair_decision.safe_action),
        uncertainty=state_set.raw_uncertainty,
        reliability={"w_t": float(reliability.weight), **dict(reliability.flags)},
        drift=dict(reliability.drift_meta),
        cfg=dcfg,
        prev_hash=prev_cert_hash,
        repair_meta=repair_decision.metadata,
        guarantee_meta={
            "step_risk_bound": compute_step_risk_bound(
                float(reliability.weight),
                alpha=float(dcfg.get("alpha", dcfg.get("miscoverage_alpha", 0.10))),
            ),
        },
    )
    assumption_tags = tuple(
        dict.fromkeys(
            list(reliability.assumption_tags)
            + list(state_set.assumption_tags)
            + list(safety_spec.assumption_tags)
            + list(safe_action_set.assumption_tags)
            + list(repair_decision.assumption_tags)
        )
    )
    certificate = build_safety_certificate(
        payload=certificate_payload,
        source_domain=str(domain_id),
        assumption_tags=assumption_tags,
    )

    alpha = float(dcfg.get("alpha", dcfg.get("miscoverage_alpha", 0.10)))
    step_bound = compute_step_risk_bound(float(reliability.weight), alpha=alpha)
    planned_horizon = int(
        dcfg.get(
            "episode_horizon_steps",
            dcfg.get("planning_horizon_steps", dcfg.get("horizon_steps", 1)),
        )
    )
    reliability_history = _extract_reliability_history(
        history,
        current_weight=float(reliability.weight),
    )
    episode_bound = compute_episode_risk_bound(reliability_history, alpha=alpha)
    episode_bound["scope"] = "observed_prefix" if len(reliability_history) > 1 else "current_step_only"
    episode_bound["observed_reliability_samples"] = float(len(reliability_history))
    episode_bound["planned_horizon_steps"] = float(planned_horizon)

    contract_checks = ContractVerifier.validate_runtime_step(
        adapter=domain_adapter,
        state=state,
        constraints=constraints,
        quantile=quantile,
        cfg=dcfg,
        reliability=reliability,
        uncertainty_set=state_set,
        safe_action_set=safe_action_set,
        repair_decision=repair_decision,
        certificate=certificate,
        step_risk_bound=step_bound,
        episode_risk_bound=episode_bound,
        alpha=alpha,
    )
    theorem_contracts = {
        "T3a": build_t3a_contract_summary(
            reliability_w=float(reliability.weight),
            step_risk_bound=step_bound,
            episode_risk_bound=episode_bound,
            alpha=alpha,
            contract_checks=contract_checks,
            calibration_meta=cal_meta,
        ),
        "T11": ContractVerifier.build_transfer_theorem_summary(
            adapter=domain_adapter,
            reliability=reliability,
            uncertainty_set=state_set,
            safe_action_set=safe_action_set,
            repair_decision=repair_decision,
            certificate=certificate,
            contract_checks=contract_checks,
        ),
    }
    certificate.extras["semantic_checks"] = dict(contract_checks)
    certificate.extras["risk_bound_scope"] = str(episode_bound.get("scope", "current_step_only"))
    certificate.extras["theorem_contracts"] = theorem_contracts

    return UniversalStepResult(
        certificate=certificate,
        safe_action=dict(repair_decision.safe_action),
        reliability=reliability,
        uncertainty_set=state_set,
        safe_action_set=safe_action_set,
        repair_decision=repair_decision,
        observation=observation,
        state=state,
        step_risk_bound=step_bound,
        episode_risk_bound=episode_bound,
        contract_checks=contract_checks,
        theorem_contracts=theorem_contracts,
    )
