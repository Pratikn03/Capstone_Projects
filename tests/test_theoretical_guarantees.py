"""Tests for the current ORIUS T9--T11 theorem helper surface."""
from __future__ import annotations

import pytest

from orius.dc3s.theoretical_guarantees import (
    THEOREM_REGISTER,
    TransferContractResult,
    compute_stylized_frontier_lower_bound,
    compute_universal_impossibility_bound,
    evaluate_structural_transfer,
)


class TestUniversalImpossibilityBound:
    """T9 helper tests."""

    def test_expected_lower_bound_matches_linear_formula(self) -> None:
        result = compute_universal_impossibility_bound(
            horizon=1000,
            fault_rate=0.2,
            sensitivity_constant=0.3,
        )
        assert result["expected_lower_bound"] == pytest.approx(60.0)
        assert result["linear_rate_lower_bound"] == pytest.approx(0.06)

    def test_zero_fault_rate_gives_zero_lower_bound(self) -> None:
        result = compute_universal_impossibility_bound(
            horizon=1000,
            fault_rate=0.0,
            sensitivity_constant=0.3,
        )
        assert result["expected_lower_bound"] == pytest.approx(0.0)

    def test_buffer_discount_reduces_effective_horizon(self) -> None:
        full = compute_universal_impossibility_bound(
            horizon=1000,
            fault_rate=0.2,
            sensitivity_constant=0.3,
            usable_horizon_fraction=1.0,
        )
        discounted = compute_universal_impossibility_bound(
            horizon=1000,
            fault_rate=0.2,
            sensitivity_constant=0.3,
            usable_horizon_fraction=0.5,
        )
        assert discounted["effective_horizon"] == pytest.approx(500.0)
        assert discounted["expected_lower_bound"] == pytest.approx(full["expected_lower_bound"] / 2.0)

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            compute_universal_impossibility_bound(0, 0.1, 0.2)
        with pytest.raises(ValueError, match="fault_rate"):
            compute_universal_impossibility_bound(10, -0.1, 0.2)
        with pytest.raises(ValueError, match="sensitivity_constant"):
            compute_universal_impossibility_bound(10, 0.1, -0.2)


class TestStylizedFrontierLowerBound:
    """T10 helper tests."""

    def test_general_formula_matches_half_sum_p_one_minus_w(self) -> None:
        result = compute_stylized_frontier_lower_bound(
            [1.0, 0.75, 0.25],
            boundary_mass=[0.2, 0.3, 0.4],
            alpha=0.10,
        )
        expected = 0.5 * (0.2 * 0.0 + 0.3 * 0.25 + 0.4 * 0.75)
        assert result["expected_lower_bound"] == pytest.approx(expected)
        assert result["special_case_active"] is True

    def test_special_case_lower_bound_activates_when_boundary_mass_large_enough(self) -> None:
        result = compute_stylized_frontier_lower_bound(
            [1.0, 0.8, 0.6, 0.4],
            boundary_mass=0.2,
            alpha=0.10,
        )
        expected = 0.25 * 0.10 * ((1.0 - 1.0) + (1.0 - 0.8) + (1.0 - 0.6) + (1.0 - 0.4))
        assert result["special_case_active"] is True
        assert result["special_case_lower_bound"] == pytest.approx(expected)

    def test_special_case_does_not_activate_when_boundary_mass_is_too_small(self) -> None:
        result = compute_stylized_frontier_lower_bound(
            [1.0, 0.8, 0.6],
            boundary_mass=0.01,
            alpha=0.10,
        )
        assert result["special_case_active"] is False
        assert result["special_case_lower_bound"] is None

    def test_more_degradation_increases_lower_bound(self) -> None:
        mild = compute_stylized_frontier_lower_bound([0.9, 0.9, 0.9], boundary_mass=0.2)
        severe = compute_stylized_frontier_lower_bound([0.4, 0.4, 0.4], boundary_mass=0.2)
        assert severe["expected_lower_bound"] > mild["expected_lower_bound"]


class TestStructuralTransfer:
    """T11 helper tests."""

    def test_all_obligations_true_gives_one_step_transfer(self) -> None:
        result = evaluate_structural_transfer(
            coverage_holds=True,
            sound_safe_action_set=True,
            repair_membership_holds=True,
            fallback_exists=True,
            alpha=0.10,
            per_step_risk_budget=[0.01, 0.02, 0.03],
        )
        assert isinstance(result, TransferContractResult)
        assert result.one_step_transfer_holds is True
        assert result.safety_probability_lower_bound == pytest.approx(0.90)
        assert result.episode_bound_available is True
        assert result.episode_bound == pytest.approx(0.06)

    def test_one_step_transfer_does_not_imply_episode_bound_without_budget(self) -> None:
        result = evaluate_structural_transfer(
            coverage_holds=True,
            sound_safe_action_set=True,
            repair_membership_holds=True,
            fallback_exists=True,
            alpha=0.10,
            per_step_risk_budget=None,
        )
        assert result.one_step_transfer_holds is True
        assert result.episode_bound_available is False
        assert result.episode_bound is None

    @pytest.mark.parametrize(
        ("kwargs", "failed_name"),
        [
            ({"coverage_holds": False, "sound_safe_action_set": True, "repair_membership_holds": True, "fallback_exists": True}, "coverage"),
            ({"coverage_holds": True, "sound_safe_action_set": False, "repair_membership_holds": True, "fallback_exists": True}, "sound_safe_action_set"),
            ({"coverage_holds": True, "sound_safe_action_set": True, "repair_membership_holds": False, "fallback_exists": True}, "repair_membership"),
            ({"coverage_holds": True, "sound_safe_action_set": True, "repair_membership_holds": True, "fallback_exists": False}, "fallback"),
        ],
    )
    def test_missing_any_obligation_returns_counterexample(self, kwargs: dict, failed_name: str) -> None:
        result = evaluate_structural_transfer(alpha=0.10, **kwargs)
        assert result.one_step_transfer_holds is False
        assert result.failed_obligations[0] == failed_name
        assert result.counterexample is not None

    def test_invalid_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="per_step_risk_budget"):
            evaluate_structural_transfer(
                coverage_holds=True,
                sound_safe_action_set=True,
                repair_membership_holds=True,
                fallback_exists=True,
                per_step_risk_budget=[],
            )


class TestTheoremRegister:
    """Register consistency checks for the current theorem numbering."""

    def test_all_theorems_present(self) -> None:
        assert "T9" in THEOREM_REGISTER
        assert "T10" in THEOREM_REGISTER
        assert "T11" in THEOREM_REGISTER

    def test_register_points_to_current_witnesses(self) -> None:
        assert THEOREM_REGISTER["T9"]["code_witness"] == "compute_universal_impossibility_bound"
        assert THEOREM_REGISTER["T10"]["code_witness"] == "compute_stylized_frontier_lower_bound"
        assert THEOREM_REGISTER["T11"]["code_witness"] == "evaluate_structural_transfer"

    def test_theorem_types_match_current_surface(self) -> None:
        assert THEOREM_REGISTER["T9"]["type"] == "impossibility"
        assert THEOREM_REGISTER["T10"]["type"] == "lower_bound"
        assert THEOREM_REGISTER["T11"]["type"] == "transfer_theorem"
