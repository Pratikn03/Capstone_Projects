"""Tests for the typed universal degraded-observation theory kernel."""
from __future__ import annotations

import pytest

from orius.adapters.battery import BatteryDomainAdapter
from orius.adapters.healthcare import HealthcareDomainAdapter
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
from orius.universal_theory.risk_bounds import calibration_contract_verify, compute_step_risk_bound


def test_contract_verifier_accepts_canonical_domain_instantiations() -> None:
    adapters = [
        BatteryDomainAdapter(),
        VehicleDomainAdapter({"expected_cadence_s": 0.25}),
        HealthcareDomainAdapter({"expected_cadence_s": 1.0}),
    ]
    for adapter in adapters:
        summary = ContractVerifier.summary(adapter)
        assert summary["contract_passed"] is True, summary
        ContractVerifier.check(adapter)


def test_run_universal_step_returns_structured_result() -> None:
    adapter = get_adapter("healthcare", {})
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "hr_bpm": 72.0,
            "spo2_pct": 97.0,
            "respiratory_rate": 14.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=[
            {
                "hr_bpm": 71.0,
                "spo2_pct": 97.0,
                "respiratory_rate": 14.0,
                "ts_utc": "2025-12-31T23:00:00Z",
                "w_t": 0.7,
            }
        ],
        candidate_action={"alert_level": 0.3},
        constraints={"spo2_min_pct": 90.0},
        quantile=5.0,
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
    assert set(result["theorem_contracts"]) == {"T3a", "T11"}
    assert result["theorem_contracts"]["T3a"]["all_executable_checks_passed"] is True
    assert result["theorem_contracts"]["T11"]["all_executable_checks_passed"] is True
    assert result["theorem_contracts"]["T11"]["forward_only"] is True
    assert result["certificate"]["theorem_contracts"]["T3a"]["status"] == "runtime_linked"
    assert result["certificate"]["theorem_contracts"]["T11"]["status"] == "runtime_linked"


def test_contract_verifier_rejects_tampered_runtime_certificate() -> None:
    adapter = get_adapter("healthcare", {})
    constraints = {"spo2_min_pct": 90.0}
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "hr_bpm": 72.0,
            "spo2_pct": 97.0,
            "respiratory_rate": 14.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=None,
        candidate_action={"alert_level": 0.3},
        constraints=constraints,
        quantile=5.0,
    )
    result.certificate.safe_action = {"alert_level": 2.0}
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
    adapter = get_adapter("healthcare", {})
    constraints = {"spo2_min_pct": 90.0}
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry={
            "hr_bpm": 72.0,
            "spo2_pct": 97.0,
            "respiratory_rate": 14.0,
            "ts_utc": "2026-01-01T00:00:00Z",
        },
        history=None,
        candidate_action={"alert_level": 0.3},
        constraints=constraints,
        quantile=5.0,
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


# ── PAC Validity Horizon, Conservative Horizon, Assumption Register ─────────

from orius.universal_theory.risk_bounds import pac_validity_horizon_bound
from orius.dc3s.half_life import compute_conservative_horizon
from orius.universal_theory.contracts import ASSUMPTION_REGISTER, verify_assumption_coverage


def test_pac_validity_horizon_bound() -> None:
    result = pac_validity_horizon_bound(
        n_cal=500, alpha=0.1, delta=0.05,
        sigma_d=10.0, margin=100.0, w_min=0.5,
    )
    assert result["pac_holds"] is True
    assert result["H_conservative"] > 0
    assert result["total_failure_prob"] < 1.0
    assert result["validity_probability_lower_bound"] > 0.0
    assert result["epsilon_conformal"] > 0


def test_invalid_reliability_weight_rejected() -> None:
    with pytest.raises(ValueError, match="weight"):
        build_reliability_assessment(
            weight=1.2,
            flags={},
            drift_meta={"drift": False},
        )


def test_invalid_inflation_rejected() -> None:
    reliability = build_reliability_assessment(
        weight=0.4,
        flags={},
        drift_meta={"drift": False},
    )
    with pytest.raises(ValueError, match="inflation"):
        build_observation_consistent_state_set(
            observed_state={"speed_mps": 5.0},
            uncertainty={
                "speed_lower_mps": 4.0,
                "speed_upper_mps": 6.0,
                "meta": {"inflation": 0.8},
            },
            quantile=0.9,
            reliability=reliability,
        )


def test_invalid_step_risk_reliability_rejected() -> None:
    with pytest.raises(ValueError, match="reliability_w"):
        compute_step_risk_bound(1.2, alpha=0.1)


def test_calibration_contract_rejects_empty_bins_as_supported_coverage() -> None:
    result = calibration_contract_verify(
        oqe_scores=[0.5, 0.5, 0.5],
        coverage_indicators=[1.0, 1.0, 1.0],
        n_bins=3,
    )
    assert result["contract_satisfied"] is False
    assert sum(1 for row in result["bins"] if row["n"] == 0 and row["passed"] is False) == 2


def test_conservative_horizon_vs_heuristic() -> None:
    from orius.dc3s.half_life import compute_validity_horizon
    conservative = compute_conservative_horizon(margin=100.0, sigma_d=10.0, delta=0.05)
    heuristic = compute_validity_horizon(
        observed_state={"current_soc_mwh": 500.0},
        quality_score=1.0,
        safety_margin_mwh=100.0,
        constraints={"min_soc_mwh": 0.0, "max_soc_mwh": 1000.0},
        sigma_d=10.0,
    )
    assert conservative["H_conservative"] <= heuristic["tau_t"]


def test_assumption_register_completeness() -> None:
    assert set(ASSUMPTION_REGISTER.keys()) == {
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10a",
        "A10b",
        "A11",
        "A12",
        "A13",
    }
    for tag, entry in ASSUMPTION_REGISTER.items():
        assert entry["tag"] == tag
        assert len(entry["formal"]) > 20
        assert len(entry["role"]) > 10

    cov = verify_assumption_coverage({"A1", "A2", "A5", "A6"})
    assert cov["fraction"] == pytest.approx(4 / 14)
    assert "A3" in cov["missing"]
