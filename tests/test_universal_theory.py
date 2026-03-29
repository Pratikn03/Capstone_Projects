"""Tests for the typed universal degraded-observation theory kernel."""
from __future__ import annotations

import pytest

from orius.adapters.battery import BatteryDomainAdapter
from orius.adapters.healthcare import HealthcareDomainAdapter
from orius.adapters.industrial import IndustrialDomainAdapter
from orius.adapters.vehicle import VehicleDomainAdapter
from orius.universal_theory import (
    ContractVerifier,
    ContractViolation,
    SafetyCertificate,
    UniversalStepResult,
    build_observation_consistent_state_set,
    build_reliability_assessment,
    build_repair_decision,
    build_safety_certificate,
    build_safety_spec,
    derive_safe_action_set,
)
from orius.universal_framework import get_adapter, run_universal_step


def test_contract_verifier_accepts_canonical_domain_instantiations() -> None:
    adapters = [
        BatteryDomainAdapter(),
        VehicleDomainAdapter({"expected_cadence_s": 0.25}),
        IndustrialDomainAdapter({"expected_cadence_s": 3600.0}),
        HealthcareDomainAdapter({"expected_cadence_s": 1.0}),
    ]
    for adapter in adapters:
        summary = ContractVerifier.summary(adapter)
        assert summary["contract_passed"] is True, summary
        ContractVerifier.check(adapter)


def test_run_universal_step_returns_structured_result() -> None:
    adapter = get_adapter("industrial", {})
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "temp_c": 25.0,
            "pressure_mbar": 1010.0,
            "power_mw": 450.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=[
            {
                "temp_c": 24.0,
                "pressure_mbar": 1008.0,
                "power_mw": 445.0,
                "ts_utc": "2025-12-31T23:00:00Z",
                "w_t": 0.7,
            }
        ],
        candidate_action={"power_setpoint_mw": 480.0},
        constraints={"power_max_mw": 500.0},
        quantile=30.0,
    )
    assert isinstance(result, UniversalStepResult)
    assert isinstance(result["certificate"], SafetyCertificate)
    assert result["certificate"]["controller"] == "orius-universal"
    assert result["step_risk_bound"] >= 0.0
    assert result["contract_checks"]["contract_passed"] is True
    assert result["episode_risk_bound"]["scope"] in {"current_step_only", "observed_prefix"}
    assert result["certificate"]["semantic_checks"]["contract_passed"] is True
    assert result["episode_risk_bound"]["scope"] == "observed_prefix"
    assert result["episode_risk_bound"]["horizon"] == 2.0
    assert result["contract_checks"]["contract_passed"] is True
    assert result["certificate"]["semantic_checks"]["contract_passed"] is True
    assert "reliability_range" in result["contract_checks"]["checked_invariants"]


def test_contract_verifier_rejects_tampered_runtime_certificate() -> None:
    adapter = get_adapter("industrial", {})
    constraints = {"power_max_mw": 500.0}
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "temp_c": 25.0,
            "pressure_mbar": 1010.0,
            "power_mw": 450.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=None,
        candidate_action={"power_setpoint_mw": 480.0},
        constraints=constraints,
        quantile=30.0,
    )
    result.certificate.safe_action = {"power_setpoint_mw": -1.0}
    try:
        ContractVerifier.validate_runtime_step(
            adapter=adapter,
            state=result.state,
            constraints=constraints,
            quantile=30.0,
            cfg={},
            reliability=result.reliability,
            uncertainty_set=result.uncertainty_set,
            safe_action_set=result.safe_action_set,
            repair_decision=result.repair_decision,
            certificate=result.certificate,
            step_risk_bound=result.step_risk_bound,
            episode_risk_bound=result.episode_risk_bound,
            alpha=result.episode_risk_bound["alpha"],
        )
    except ContractViolation as exc:
        assert "certificate_matches_repair" in str(exc)
    else:
        raise AssertionError("Tampered certificate should fail runtime semantic validation.")


def test_contract_verifier_rejects_dishonest_episode_scope() -> None:
    adapter = get_adapter("industrial", {})
    constraints = {"power_max_mw": 500.0}
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "temp_c": 25.0,
            "pressure_mbar": 1010.0,
            "power_mw": 450.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=None,
        candidate_action={"power_setpoint_mw": 480.0},
        constraints=constraints,
        quantile=30.0,
    )
    dishonest_episode_bound = dict(result.episode_risk_bound)
    dishonest_episode_bound["scope"] = "constant_reliability_proxy"
    with pytest.raises(ContractViolation, match="risk_bound_semantics"):
        ContractVerifier.validate_runtime_step(
            adapter=adapter,
            state=result.state,
            constraints=constraints,
            quantile=30.0,
            cfg={},
            reliability=result.reliability,
            uncertainty_set=result.uncertainty_set,
            safe_action_set=result.safe_action_set,
            repair_decision=result.repair_decision,
            certificate=result.certificate,
            step_risk_bound=result.step_risk_bound,
            episode_risk_bound=dishonest_episode_bound,
            alpha=result.episode_risk_bound["alpha"],
        )


def test_observation_consistent_state_set_tracks_monotone_inflation() -> None:
    reliability = build_reliability_assessment(
        weight=0.4,
        flags={"flags": {"dropout": True}},
        drift_meta={"drift": True, "score": 2.0},
    )
    state_set = build_observation_consistent_state_set(
        observed_state={"speed_mps": 5.0},
        uncertainty={
            "speed_lower_mps": 3.0,
            "speed_upper_mps": 8.0,
            "meta": {"inflation": 2.0},
        },
        quantile=0.9,
        reliability=reliability,
    )
    assert state_set.inflation == 2.0
    assert state_set.lower_bounds["speed_mps"] == 3.0
    assert state_set.upper_bounds["speed_mps"] == 8.0
    assert state_set.reliability_weight == 0.4


def test_observation_gap_counterexample_can_exist_while_observed_state_looks_safe() -> None:
    reliability = build_reliability_assessment(
        weight=0.2,
        flags={"flags": {"stale": True}},
        drift_meta={"drift": False, "score": 0.0},
    )
    state_set = build_observation_consistent_state_set(
        observed_state={"speed_mps": 9.5},
        uncertainty={
            "speed_lower_mps": 9.0,
            "speed_upper_mps": 12.0,
            "meta": {"inflation": 1.8},
        },
        quantile=0.9,
        reliability=reliability,
    )
    observed_safe = state_set.observed_state["speed_mps"] <= 10.0
    true_state_can_violate = state_set.upper_bounds["speed_mps"] > 10.0
    assert observed_safe is True
    assert true_state_can_violate is True


def test_repair_decision_is_idempotent_for_safe_actions() -> None:
    safety_spec = build_safety_spec(constraints={"power_max_mw": 500.0})
    safe_action_set = derive_safe_action_set(
        tightened_set={"viable": True},
        safety_spec=safety_spec,
    )
    decision = build_repair_decision(
        proposed_action={"power_setpoint_mw": 200.0},
        safe_action={"power_setpoint_mw": 200.0},
        repair_meta={"repaired": False, "mode": "projection"},
        safe_action_set=safe_action_set,
    )
    assert decision.repaired is False
    assert decision.safe_action == {"power_setpoint_mw": 200.0}


def test_repair_decision_falls_back_when_safe_action_is_empty() -> None:
    safety_spec = build_safety_spec(
        constraints={"min_soc_mwh": 0.0, "max_soc_mwh": 1.0},
        fallback_action={"charge_mw": 0.0, "discharge_mw": 0.0},
    )
    safe_action_set = derive_safe_action_set(
        tightened_set={"viable": False},
        safety_spec=safety_spec,
    )
    decision = build_repair_decision(
        proposed_action={"charge_mw": 2.0, "discharge_mw": 0.0},
        safe_action={},
        repair_meta={"repaired": False, "mode": "projection"},
        safe_action_set=safe_action_set,
    )
    assert decision.repaired is True
    assert decision.mode == "fallback"
    assert decision.reason == "fallback_action"
    assert decision.safe_action == {"charge_mw": 0.0, "discharge_mw": 0.0}


def test_certificate_preserves_causal_payload_and_horizon_metadata() -> None:
    cert = build_safety_certificate(
        payload={
            "command_id": "cmd-1",
            "certificate_id": "cmd-1",
            "created_at": "2026-01-01T00:00:00Z",
            "device_id": "device-1",
            "zone_id": "zone-1",
            "controller": "orius-universal",
            "proposed_action": {"ax": 1.0},
            "safe_action": {"ax": 0.0},
            "uncertainty": {"meta": {"inflation": 1.5}},
            "reliability": {"w_t": 0.5},
            "drift": {"drift": False},
            "certificate_hash": "hash-1",
            "validity_horizon_H_t": 5,
        },
        source_domain="navigation",
        assumption_tags=("A2", "A5", "A7"),
    )
    assert cert["command_id"] == "cmd-1"
    assert cert["source_domain"] == "navigation"
    assert cert.certificate_horizon_steps == 5
    assert cert["assumptions_checked"] == ["A2", "A5", "A7"]
