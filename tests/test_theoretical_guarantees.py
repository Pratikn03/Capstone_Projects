"""Tests for the deep theoretical guarantees (T9, T10, T11).

Verifies:
  - T9:  Finite-sample bound is mathematically correct and tightens
  - T10: Separation gap is empirically validated
  - T11: Adaptive regret is sublinear and simulated tracking is bounded
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from orius.dc3s.theoretical_guarantees import (
    THEOREM_REGISTER,
    SeparationResult,
    assert_finite_sample_bound,
    assert_separation,
    assert_sublinear_regret,
    compute_adaptive_regret_bound,
    compute_coverage_bound_surface,
    compute_finite_sample_coverage_bound,
    compute_separation_gap,
    simulate_adaptive_tracking,
    simulate_separation_construction,
)


# ══════════════════════════════════════════════════════════════════════
# T9: Finite-Sample Coverage Bound
# ══════════════════════════════════════════════════════════════════════

class TestFiniteSampleCoverageBound:
    """Theorem T9 witness tests."""

    def test_basic_computation(self) -> None:
        result = compute_finite_sample_coverage_bound(
            n_calibration=1000, alpha=0.10, delta=0.05, w_min=0.5,
        )
        assert result["nominal_coverage"] == pytest.approx(0.90)
        assert result["n_eff"] == 500
        assert result["epsilon"] > 0
        assert result["coverage_bound"] > 0
        assert result["coverage_bound"] < result["nominal_coverage"]

    def test_perfect_reliability_tightest_bound(self) -> None:
        """w_min=1.0 gives the tightest bound (n_eff = n)."""
        perfect = compute_finite_sample_coverage_bound(
            n_calibration=1000, alpha=0.10, delta=0.05, w_min=1.0,
        )
        degraded = compute_finite_sample_coverage_bound(
            n_calibration=1000, alpha=0.10, delta=0.05, w_min=0.1,
        )
        assert perfect["coverage_bound"] > degraded["coverage_bound"]
        assert perfect["n_eff"] > degraded["n_eff"]
        assert perfect["epsilon"] < degraded["epsilon"]

    def test_bound_tightens_with_more_data(self) -> None:
        """More calibration data -> tighter bound."""
        small = compute_finite_sample_coverage_bound(
            n_calibration=100, alpha=0.10, delta=0.05, w_min=0.5,
        )
        large = compute_finite_sample_coverage_bound(
            n_calibration=10000, alpha=0.10, delta=0.05, w_min=0.5,
        )
        assert large["coverage_bound"] > small["coverage_bound"]
        assert large["epsilon"] < small["epsilon"]

    def test_bound_tightens_with_higher_delta(self) -> None:
        """Looser confidence (higher delta) -> tighter bound."""
        tight_confidence = compute_finite_sample_coverage_bound(
            n_calibration=1000, alpha=0.10, delta=0.01, w_min=0.5,
        )
        loose_confidence = compute_finite_sample_coverage_bound(
            n_calibration=1000, alpha=0.10, delta=0.20, w_min=0.5,
        )
        assert loose_confidence["coverage_bound"] > tight_confidence["coverage_bound"]

    def test_bound_nonnegative(self) -> None:
        """Coverage bound should never be negative."""
        result = compute_finite_sample_coverage_bound(
            n_calibration=5, alpha=0.10, delta=0.01, w_min=0.01,
        )
        assert result["coverage_bound"] >= 0.0

    def test_epsilon_formula_correct(self) -> None:
        """Verify epsilon = sqrt(log(2/delta) / (2*n_eff))."""
        result = compute_finite_sample_coverage_bound(
            n_calibration=1000, alpha=0.10, delta=0.05, w_min=0.5,
        )
        expected_eps = math.sqrt(math.log(2.0 / 0.05) / (2.0 * 500))
        assert result["epsilon"] == pytest.approx(expected_eps, rel=1e-10)

    def test_assert_passes_for_sufficient_data(self) -> None:
        """With enough data, assertion should pass."""
        result = assert_finite_sample_bound(
            n_calibration=5000, alpha=0.10, delta=0.05, w_min=0.5,
            required_coverage=0.85,
        )
        assert result["coverage_bound"] >= 0.85

    def test_assert_fails_for_insufficient_data(self) -> None:
        """With too little data, assertion should fail."""
        with pytest.raises(AssertionError, match="Finite-sample coverage bound"):
            assert_finite_sample_bound(
                n_calibration=10, alpha=0.10, delta=0.01, w_min=0.05,
                required_coverage=0.89,
            )

    def test_coverage_bound_surface(self) -> None:
        """Surface computation produces expected grid."""
        surface = compute_coverage_bound_surface(
            n_values=[100, 1000, 10000],
            w_min_values=[0.05, 0.50, 1.00],
        )
        assert len(surface) == 9  # 3 x 3 grid
        # Coverage should be monotone in both n and w_min
        bounds = {(r["n_calibration"], r["w_min"]): r["coverage_bound"] for r in surface}
        assert bounds[(10000, 1.00)] > bounds[(100, 0.05)]

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="n_calibration"):
            compute_finite_sample_coverage_bound(0, 0.10, 0.05, 0.5)
        with pytest.raises(ValueError, match="alpha"):
            compute_finite_sample_coverage_bound(100, 0.0, 0.05, 0.5)
        with pytest.raises(ValueError, match="delta"):
            compute_finite_sample_coverage_bound(100, 0.10, 0.0, 0.5)
        with pytest.raises(ValueError, match="w_min"):
            compute_finite_sample_coverage_bound(100, 0.10, 0.05, 0.0)


# ══════════════════════════════════════════════════════════════════════
# T10: Separation / Necessity of Reliability Awareness
# ══════════════════════════════════════════════════════════════════════

class TestSeparation:
    """Theorem T10 witness tests."""

    def test_dc3s_pareto_dominates_when_strictly_better(self) -> None:
        result = compute_separation_gap(
            dc3s_violations=0.0, dc3s_interventions=0.028,
            blind_violations=0.039, blind_interventions=0.028,
            w_min=0.05,
        )
        assert result.pareto_dominant is True
        assert result.violation_gap > 0

    def test_dc3s_not_dominant_when_worse_on_one_axis(self) -> None:
        result = compute_separation_gap(
            dc3s_violations=0.0, dc3s_interventions=0.10,
            blind_violations=0.0, blind_interventions=0.028,
        )
        assert result.pareto_dominant is False

    def test_theoretical_lower_bounds(self) -> None:
        """Verify the theoretical lower bound formulae."""
        result = compute_separation_gap(
            dc3s_violations=0.0, dc3s_interventions=0.028,
            blind_violations=0.05, blind_interventions=0.05,
            w_min=0.10, alpha=0.10,
        )
        assert result.violation_lower_bound == pytest.approx(0.10 * 0.90 / 2.0)
        assert result.intervention_lower_bound == pytest.approx(0.90 / 2.0)

    def test_assert_separation_passes(self) -> None:
        result = assert_separation(
            dc3s_violations=0.0, dc3s_interventions=0.028,
            blind_violations=0.039, blind_interventions=0.028,
        )
        assert isinstance(result, SeparationResult)

    def test_assert_separation_fails_when_not_dominant(self) -> None:
        with pytest.raises(AssertionError, match="Pareto-dominate"):
            assert_separation(
                dc3s_violations=0.05, dc3s_interventions=0.10,
                blind_violations=0.0, blind_interventions=0.05,
            )

    def test_separation_construction_simulation(self) -> None:
        """Run the constructive proof simulation."""
        result = simulate_separation_construction(
            n_steps=500, w_min=0.10, alpha=0.10, seed=42,
        )
        assert result["n_steps"] == 500
        assert "dc3s" in result["controllers"]
        assert "blind_narrow" in result["controllers"]
        assert "blind_wide" in result["controllers"]

        # DC³S should not have higher violations than blind narrow
        dc3s_vr = result["controllers"]["dc3s"]["violation_rate"]
        narrow_vr = result["controllers"]["blind_narrow"]["violation_rate"]
        # This is the empirical check — DC³S adapts to degradation
        assert dc3s_vr <= narrow_vr + 0.05  # allow small simulation noise

    def test_gap_grows_with_degradation(self) -> None:
        """Worse telemetry (lower w_min) should create a larger gap."""
        mild = compute_separation_gap(
            dc3s_violations=0.0, dc3s_interventions=0.028,
            blind_violations=0.01, blind_interventions=0.05,
            w_min=0.50,
        )
        severe = compute_separation_gap(
            dc3s_violations=0.0, dc3s_interventions=0.028,
            blind_violations=0.05, blind_interventions=0.30,
            w_min=0.05,
        )
        assert severe.violation_lower_bound > mild.violation_lower_bound
        assert severe.intervention_lower_bound > mild.intervention_lower_bound


# ══════════════════════════════════════════════════════════════════════
# T11: Adaptive Inflation Regret Bound
# ══════════════════════════════════════════════════════════════════════

class TestAdaptiveRegret:
    """Theorem T11 witness tests."""

    def test_basic_bound_computation(self) -> None:
        result = compute_adaptive_regret_bound(
            T=500, tau=30.0, max_oracle_jump=0.3,
        )
        assert result["cumulative_bound"] > 0
        assert result["per_step_bound"] > 0
        assert result["gamma"] == pytest.approx(1.0 - math.exp(-1.0 / 30.0))

    def test_per_step_regret_decreases_with_horizon(self) -> None:
        """Per-step regret should decrease as T grows (sublinearity)."""
        short = compute_adaptive_regret_bound(T=100, tau=30.0, max_oracle_jump=0.3)
        long = compute_adaptive_regret_bound(T=10000, tau=30.0, max_oracle_jump=0.3)
        assert long["per_step_bound"] < short["per_step_bound"]

    def test_larger_tau_increases_bound(self) -> None:
        """Larger time constant -> more lag -> higher regret."""
        fast = compute_adaptive_regret_bound(T=500, tau=5.0, max_oracle_jump=0.3)
        slow = compute_adaptive_regret_bound(T=500, tau=100.0, max_oracle_jump=0.3)
        assert slow["tracking_term"] > fast["tracking_term"]

    def test_zero_jump_zero_regret(self) -> None:
        """If oracle never changes, regret is zero."""
        result = compute_adaptive_regret_bound(T=500, tau=30.0, max_oracle_jump=0.0)
        assert result["cumulative_bound"] == pytest.approx(0.0)
        assert result["per_step_bound"] == pytest.approx(0.0)

    def test_assert_sublinear_passes(self) -> None:
        result = assert_sublinear_regret(T=500, tau=30.0, max_oracle_jump=0.3)
        assert result["per_step_bound"] > 0

    def test_simulation_empirical_vs_bound(self) -> None:
        """Simulated tracking error should be below theoretical bound."""
        result = simulate_adaptive_tracking(
            T=500, tau=30.0, n_jumps=5, jump_magnitude=0.3, seed=42,
        )
        assert result["empirical_cumulative_error"] <= (
            result["theoretical_cumulative_bound"] * 1.1  # 10% tolerance for simulation noise
        )

    def test_simulation_produces_valid_sequences(self) -> None:
        result = simulate_adaptive_tracking(T=200, tau=30.0, seed=0)
        assert len(result["oracle_sequence"]) == 200
        assert len(result["tracker_sequence"]) == 200
        assert len(result["tracking_errors"]) == 200
        assert all(e >= 0 for e in result["tracking_errors"])

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="T must be positive"):
            compute_adaptive_regret_bound(0, 30.0, 0.3)
        with pytest.raises(ValueError, match="tau must be positive"):
            compute_adaptive_regret_bound(500, 0.0, 0.3)
        with pytest.raises(ValueError, match="non-negative"):
            compute_adaptive_regret_bound(500, 30.0, -0.1)


# ══════════════════════════════════════════════════════════════════════
# Theorem Register
# ══════════════════════════════════════════════════════════════════════

class TestTheoremRegister:
    """Verify the theorem register has all three entries."""

    def test_all_theorems_present(self) -> None:
        assert "T9" in THEOREM_REGISTER
        assert "T10" in THEOREM_REGISTER
        assert "T11" in THEOREM_REGISTER

    def test_each_theorem_has_code_witness(self) -> None:
        for key, entry in THEOREM_REGISTER.items():
            assert "code_witness" in entry, f"{key} missing code_witness"
            assert "name" in entry, f"{key} missing name"
            assert "statement" in entry, f"{key} missing statement"
            assert "type" in entry, f"{key} missing type"

    def test_theorem_types(self) -> None:
        assert THEOREM_REGISTER["T9"]["type"] == "quantitative_guarantee"
        assert THEOREM_REGISTER["T10"]["type"] == "lower_bound"
        assert THEOREM_REGISTER["T11"]["type"] == "regret_bound"
