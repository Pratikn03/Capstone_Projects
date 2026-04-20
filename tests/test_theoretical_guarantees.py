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

    def test_general_formula_matches_sum_p_one_minus_w(self) -> None:
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


# ── Byzantine Bound (T11_Byzantine) ─────────────────────────────────────────


from orius.dc3s.theoretical_guarantees import (
    prove_byzantine_bound,
    verify_byzantine_bound_empirical,
    stale_decay_bound,
    verify_stale_decay_sufficiency,
    stale_decay_episode_risk,
)


class TestByzantineBound:
    def test_bound_holds_for_small_f(self) -> None:
        result = prove_byzantine_bound(W=20, f=0.2, sigma_honest=1.0)
        assert result["holds"] is True
        assert result["bound"] < 1.0
        assert result["W_effective"] == pytest.approx(20 * 0.6)

    def test_bound_fails_at_one_third(self) -> None:
        result = prove_byzantine_bound(W=20, f=1 / 3, sigma_honest=1.0)
        assert result["holds"] is False
        assert result["bound"] == float("inf")

    def test_empirical_within_theory(self) -> None:
        import numpy as np
        rng = np.random.default_rng(42)
        honest = rng.normal(10.0, 1.0, size=18).tolist()
        adversarial = [100.0, -100.0]
        signal = honest + adversarial
        result = verify_byzantine_bound_empirical(
            signal_history=signal,
            n_adversarial=2,
            true_mean=10.0,
            trim_frac=0.15,
        )
        assert result["within_bound"] is True
        assert result["empirical_error"] < 2.0


# ── Stale-Decay (T_stale_decay) ─────────────────────────────────────────────


class TestStaleDecay:
    def test_reaches_epsilon(self) -> None:
        result = stale_decay_bound(w_0=1.0, gamma=0.85, tau_max=3, epsilon=0.05)
        assert result["holds"] is True
        assert result["N_to_epsilon"] > 3
        assert result["schedule"][-1] < 0.05

    def test_sufficiency_for_typical_domain(self) -> None:
        result = verify_stale_decay_sufficiency(
            gamma=0.80, w_min=0.05, T_phys=20, tau_max=3,
        )
        assert result["sufficient"] is True
        marginal = verify_stale_decay_sufficiency(
            gamma=0.85, w_min=0.05, T_phys=20, tau_max=3,
        )
        assert marginal["sufficient"] is False

    def test_episode_risk_degrades_gracefully(self) -> None:
        r_fresh = stale_decay_episode_risk(w_0=1.0, gamma=0.85, tau_max=3, T=20, alpha=0.1)
        r_stale = stale_decay_episode_risk(w_0=0.5, gamma=0.85, tau_max=3, T=20, alpha=0.1)
        assert r_stale["tsvr_bound"] > r_fresh["tsvr_bound"]
        assert r_fresh["tsvr_bound"] > 0


class TestNewTheoremRegisterEntries:
    def test_byzantine_in_register(self) -> None:
        assert "T11_Byzantine" in THEOREM_REGISTER
        assert THEOREM_REGISTER["T11_Byzantine"]["type"] == "robustness_bound"

    def test_stale_decay_in_register(self) -> None:
        assert "T_stale_decay" in THEOREM_REGISTER
        assert THEOREM_REGISTER["T_stale_decay"]["type"] == "decay_bound"


# ── Tight Minimax Bound (T_minimax) ─────────────────────────────────────────


from orius.dc3s.theoretical_guarantees import (
    compute_tight_impossibility_bound,
    verify_minimax_gap,
    sensor_quality_converse,
    compute_minimum_w_for_tsvr,
    verify_complete_characterization,
    complete_oasg_characterization,
)
from orius.universal_theory.risk_bounds import pac_trajectory_safety_certificate


class TestTightMinimaxBound:

    def test_lower_bound_uses_alpha_and_w_bar(self) -> None:
        result = compute_tight_impossibility_bound([0.6] * 100, alpha=0.10, K_factor=2.0)
        assert result["episode_lower_bound_rate"] == pytest.approx(0.02)
        assert result["upper_bound_rate"] == pytest.approx(0.04)
        assert result["minimax_gap_factor"] == pytest.approx(2.0)

    def test_perfect_reliability_gives_zero_lower_bound(self) -> None:
        result = compute_tight_impossibility_bound([1.0] * 50, alpha=0.10)
        assert result["episode_lower_bound"] == pytest.approx(0.0)

    def test_lower_bound_increases_with_degradation(self) -> None:
        mild = compute_tight_impossibility_bound([0.9] * 100, alpha=0.10)
        severe = compute_tight_impossibility_bound([0.3] * 100, alpha=0.10)
        assert severe["episode_lower_bound"] > mild["episode_lower_bound"]

    def test_verify_minimax_gap_reports_constant_factor(self) -> None:
        result = verify_minimax_gap([0.5] * 200, alpha=0.10, K_factor=2.0)
        assert result["gap_factor"] == pytest.approx(2.0)
        assert result["lower_bound_rate"] < result["upper_bound_rate"]

    def test_invalid_k_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="K_factor"):
            compute_tight_impossibility_bound([0.5] * 10, alpha=0.10, K_factor=0.5)


# ── Sensor Quality Converse (T_sensor_converse) ─────────────────────────────


class TestSensorConverse:

    def test_converse_holds_when_w_sufficient(self) -> None:
        result = sensor_quality_converse(w_mean=0.7, alpha=0.10, epsilon=0.04)
        assert result["converse_holds"] is True
        assert result["w_required"] == pytest.approx(0.6)

    def test_converse_fails_when_w_insufficient(self) -> None:
        result = sensor_quality_converse(w_mean=0.5, alpha=0.10, epsilon=0.04)
        assert result["converse_holds"] is False

    def test_minimum_w_for_tsvr_matches_inverse(self) -> None:
        result = compute_minimum_w_for_tsvr(target_tsvr=0.04, alpha=0.10)
        assert result["w_min_required"] == pytest.approx(0.6)

    def test_complete_characterization_closes_gap(self) -> None:
        result = verify_complete_characterization([0.7] * 100, alpha=0.10, K_factor=2.0)
        assert result["characterization_complete"] is False
        assert result["gap_factor"] == pytest.approx(2.0)
        assert result["defended_status"] == "open_converse_gap"


# ── Trajectory PAC Certificate (T_trajectory_PAC) ───────────────────────────


class TestTrajectoryPAC:

    def test_perfect_reliability_gives_high_safety(self) -> None:
        result = pac_trajectory_safety_certificate(
            H=10, n_cal=500, alpha=0.10, delta=0.05,
            w_sequence=[1.0] * 10, margin=5.0, sigma_d=0.1,
        )
        assert result["trajectory_safety_prob"] > 0.90

    def test_H_max_certifiable_is_finite(self) -> None:
        result = pac_trajectory_safety_certificate(
            H=50, n_cal=500, alpha=0.10, delta=0.05,
            w_sequence=[0.8] * 50, margin=5.0, sigma_d=0.1,
        )
        assert result["H_max_certifiable"] > 0
        assert result["H_max_certifiable"] < 10000

    def test_degraded_reliability_reduces_safety_prob(self) -> None:
        good = pac_trajectory_safety_certificate(
            H=20, n_cal=500, alpha=0.10, delta=0.05,
            w_sequence=[0.9] * 20, margin=5.0, sigma_d=0.1,
        )
        bad = pac_trajectory_safety_certificate(
            H=20, n_cal=500, alpha=0.10, delta=0.05,
            w_sequence=[0.4] * 20, margin=5.0, sigma_d=0.1,
        )
        assert good["trajectory_safety_prob"] > bad["trajectory_safety_prob"]

    def test_martingale_flag_set_correctly(self) -> None:
        result = pac_trajectory_safety_certificate(
            H=10, n_cal=500, alpha=0.10, delta=0.05,
            w_sequence=[0.8] * 10, use_martingale=True,
        )
        assert result["uses_martingale"] is True
        assert result["bound_style"] == "bonferroni_union_bound"
        assert "does not claim a separate Ville certificate" in result["martingale_note"]


# ── Grand Unification ────────────────────────────────────────────────────────


class TestGrandUnification:

    def test_all_three_depth_theorems_in_register(self) -> None:
        assert "T_minimax" in THEOREM_REGISTER
        assert "T_sensor_converse" in THEOREM_REGISTER
        assert "T_trajectory_PAC" in THEOREM_REGISTER
        assert THEOREM_REGISTER["T_minimax"]["type"] == "minimax_optimality"
        assert THEOREM_REGISTER["T_sensor_converse"]["type"] == "converse_bound"
        assert THEOREM_REGISTER["T_trajectory_PAC"]["type"] == "pac_trajectory"

    def test_depth_theorem_dependency_graph_matches_bounded_posture(self) -> None:
        assert THEOREM_REGISTER["T3"]["parent_law"] == "T3a"
        assert THEOREM_REGISTER["T3a"]["parent_law"] is None
        assert THEOREM_REGISTER["T3b"]["parent_law"] is None
        assert THEOREM_REGISTER["T9"]["parent_law"] is None
        assert THEOREM_REGISTER["T10"]["parent_law"] is None
        assert THEOREM_REGISTER["T_minimax"]["parent_law"] is None
        assert THEOREM_REGISTER["T_sensor_converse"]["parent_law"] is None
        assert THEOREM_REGISTER["T_trajectory_PAC"]["parent_law"] is None

    def test_complete_characterization_assembles_all_paths(self) -> None:
        result = complete_oasg_characterization(
            [0.7] * 100, alpha=0.10, n_cal=500, delta=0.05,
            margin=5.0, sigma_d=0.1,
        )
        assert result["gap_closed"] is False
        assert "path_a_minimax" in result
        assert "path_b_trajectory_pac" in result
        assert "path_c_sensor_converse" in result
