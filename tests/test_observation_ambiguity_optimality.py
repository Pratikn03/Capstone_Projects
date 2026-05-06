from __future__ import annotations

import pytest

from orius.universal_theory.observation_ambiguity import (
    build_observation_ambiguity_contract_summary,
    common_safe_core,
    observation_only_bayes_lower_bound,
    verify_covered_orius_release,
)

SAFE_ACTION_SETS = {
    "x_safe_left": {"hold", "brake"},
    "x_safe_right": {"hold", "slow"},
    "x_conflict_left": {"brake"},
    "x_conflict_right": {"accelerate"},
}


def test_different_safe_sets_do_not_imply_unavoidable_violation() -> None:
    core = common_safe_core(["x_safe_left", "x_safe_right"], SAFE_ACTION_SETS)
    assert core == frozenset({"hold"})

    result = observation_only_bayes_lower_bound(
        observation_groups={"same_obs": ["x_safe_left", "x_safe_right"]},
        action_space={"hold", "brake", "slow"},
        safe_action_sets=SAFE_ACTION_SETS,
        probabilities={"x_safe_left": 0.5, "x_safe_right": 0.5},
    )

    assert result["lower_bound"] == pytest.approx(0.0)
    assert result["per_observation"][0]["common_safe_core_empty"] is False
    assert result["per_observation"][0]["best_observation_only_action"] == "hold"


def test_empty_common_safe_core_produces_positive_lower_bound() -> None:
    core = common_safe_core(["x_conflict_left", "x_conflict_right"], SAFE_ACTION_SETS)
    assert core == frozenset()

    result = observation_only_bayes_lower_bound(
        observation_groups={"ambiguous_boundary": ["x_conflict_left", "x_conflict_right"]},
        action_space={"brake", "accelerate"},
        safe_action_sets=SAFE_ACTION_SETS,
        probabilities={"x_conflict_left": 0.5, "x_conflict_right": 0.5},
    )

    assert result["lower_bound"] == pytest.approx(0.5)
    assert result["per_observation"][0]["common_safe_core_empty"] is True


def test_orius_covered_release_certifies_zero_violation() -> None:
    result = verify_covered_orius_release(
        true_state="x_safe_left",
        uncertainty_set=["x_safe_left", "x_safe_right"],
        action="hold",
        safe_action_sets=SAFE_ACTION_SETS,
    )

    assert result["true_state_covered"] is True
    assert result["action_safe_for_uncertainty_set"] is True
    assert result["deterministic_zero_violation_certified"] is True
    assert result["violation_probability_upper_bound"] == pytest.approx(0.0)


def test_probabilistic_coverage_is_alpha_bounded_not_unconditional_zero() -> None:
    result = verify_covered_orius_release(
        true_state="x_conflict_left",
        uncertainty_set=["x_safe_left", "x_safe_right"],
        action="hold",
        safe_action_sets=SAFE_ACTION_SETS,
        coverage_miss_probability=0.1,
    )

    assert result["true_state_covered"] is False
    assert result["action_safe_for_uncertainty_set"] is True
    assert result["deterministic_zero_violation_certified"] is False
    assert result["violation_probability_upper_bound"] == pytest.approx(0.1)


def test_contract_summary_links_t10_lower_bound_and_t11_upper_bound() -> None:
    summary = build_observation_ambiguity_contract_summary(
        observation_groups={"same_obs": ["x_safe_left", "x_safe_right"]},
        action_space={"hold", "brake", "slow"},
        safe_action_sets=SAFE_ACTION_SETS,
        probabilities={"x_safe_left": 0.5, "x_safe_right": 0.5},
        true_state="x_safe_left",
        uncertainty_set=["x_safe_left", "x_safe_right"],
        action="hold",
        coverage_miss_probability=0.05,
    )

    assert summary["theorem_id"] == "T10_T11_ObservationAmbiguitySandwich"
    assert summary["source_theorems"] == ["T10", "T11"]
    assert summary["theorem_type"] == "supporting_optimality_corollary"
    assert summary["all_executable_checks_passed"] is True
    assert "not a global" in summary["claim_boundary"]
