"""Tests for supporting lemmas, propositions, and corollaries.

Each test exercises the code witness for a supporting result in the
thesis surface register, verifying the mathematical claim holds.
"""

from __future__ import annotations

import numpy as np
import pytest

from orius.dc3s.supporting_results import (
    SUPPORTING_RESULTS_REGISTER,
    verify_admissible_fault_sequence_existence,
    verify_aggregation_under_predictable_budget,
    verify_av_promotion_routes,
    verify_boundary_proximity_under_arbitrage,
    verify_conditional_conservatism,
    verify_constraint_class_mismatch_barrier,
    verify_dc3s_feasibility_guarantee,
    verify_episode_aggregation,
    verify_illusion_under_dropout,
    verify_inflated_set_contains_state,
    verify_insufficiency_of_observed_evaluation,
    verify_intervention_lead_time,
    verify_intervention_safety_tradeoff,
    verify_intervention_sufficiency,
    verify_no_margin_compensation,
    verify_oasg_rate_lower_bound,
    verify_oasg_severity,
    verify_observation_gap_under_dropout,
    verify_perfect_telemetry_collapse,
    verify_reliability_awareness_necessary,
    verify_reliability_proportional_safety,
    verify_safe_budget_monotonicity,
    verify_tightened_feasibility,
    verify_transfer_failure_breaks_pattern,
    verify_zero_violation_regime,
)

# ═══════════════════════════════════════════════════════════════════════
# Register completeness
# ═══════════════════════════════════════════════════════════════════════


class TestRegister:
    def test_register_has_all_entries(self):
        assert len(SUPPORTING_RESULTS_REGISTER) == 25

    def test_every_entry_has_code_witness(self):
        for key, entry in SUPPORTING_RESULTS_REGISTER.items():
            assert "code_witness" in entry, f"{key} missing code_witness"
            assert "name" in entry, f"{key} missing name"
            assert "kind" in entry, f"{key} missing kind"


# ═══════════════════════════════════════════════════════════════════════
# S1 / S2: Precursor theorems
# ═══════════════════════════════════════════════════════════════════════


class TestS1IllusionUnderDropout:
    def test_illusion_exists_positive_dropout(self):
        r = verify_illusion_under_dropout(0.5, dropout_fraction=0.3)
        assert r["illusion_exists"] is True
        assert r["observation_gap"] == pytest.approx(0.3)

    def test_no_illusion_at_zero_dropout(self):
        r = verify_illusion_under_dropout(0.5, dropout_fraction=0.0)
        assert r["illusion_exists"] is False

    def test_gap_scales_with_signal_range(self):
        r = verify_illusion_under_dropout(0.5, 0.2, signal_range=10.0)
        assert r["observation_gap"] == pytest.approx(2.0)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError):
            verify_illusion_under_dropout(0.5, dropout_fraction=1.5)


class TestS2FeasibilityGuarantee:
    def test_feasible_interior_state(self):
        r = verify_dc3s_feasibility_guarantee(inflation=1.2, soc=0.5)
        assert r["feasibility_guaranteed"] is True

    def test_infeasible_at_boundary(self):
        r = verify_dc3s_feasibility_guarantee(inflation=1.2, soc=0.0)
        assert r["feasibility_guaranteed"] is False

    def test_infeasible_without_repair(self):
        r = verify_dc3s_feasibility_guarantee(inflation=1.2, soc=0.5, action_repair_available=False)
        assert r["feasibility_guaranteed"] is False


# ═══════════════════════════════════════════════════════════════════════
# Lemmas
# ═══════════════════════════════════════════════════════════════════════


class TestLemmaObservationGap:
    def test_gap_proportional_to_dropout(self):
        r = verify_observation_gap_under_dropout(0.4, signal_range=2.0)
        assert r["gap_lower_bound"] == pytest.approx(0.8)
        assert r["holds"] is True

    def test_zero_dropout_zero_gap(self):
        r = verify_observation_gap_under_dropout(0.0)
        assert r["gap_lower_bound"] == 0.0


class TestLemmaBoundaryProximity:
    def test_arbitrage_approaches_boundary(self):
        # Simulate charge/discharge oscillation approaching boundaries
        soc = [0.5, 0.9, 0.1, 0.95, 0.05, 0.98, 0.02]
        r = verify_boundary_proximity_under_arbitrage(soc, proximity_threshold=0.06)
        assert r["holds"] is True
        assert r["boundary_proximity_rate"] > 0.0

    def test_stable_interior_low_proximity(self):
        soc = [0.5] * 20
        r = verify_boundary_proximity_under_arbitrage(soc, proximity_threshold=0.05)
        assert r["boundary_proximity_rate"] == 0.0


class TestLemmaAdmissibleFault:
    def test_fault_sequence_exists(self):
        r = verify_admissible_fault_sequence_existence(n_steps=100, fault_rate=0.3)
        assert r["admissible_sequence_exists"] is True
        assert r["total_faults"] > 0

    def test_zero_fault_rate(self):
        r = verify_admissible_fault_sequence_existence(n_steps=20, fault_rate=0.0)
        assert r["total_faults"] == 0
        assert r["admissible_sequence_exists"] is False


class TestLemmaNoMarginCompensation:
    def test_fixed_margin_fails_under_degradation(self):
        # Quality drops to 0.3 → required margin = 0.10*(1-0.3) = 0.07
        w = [0.3, 0.4, 0.5, 0.9, 0.2]
        r = verify_no_margin_compensation(fixed_margin=0.03, quality_sequence=w)
        assert r["holds"] is True
        assert r["fraction_under_margined"] > 0.0

    def test_large_margin_never_fails(self):
        w = [0.5, 0.6, 0.7]
        r = verify_no_margin_compensation(fixed_margin=1.0, quality_sequence=w)
        assert r["fraction_under_margined"] == 0.0


class TestLemmaAggregation:
    def test_episode_bound_is_sum(self):
        risks = [0.01, 0.02, 0.03, 0.04]
        r = verify_aggregation_under_predictable_budget(risks)
        assert r["episode_risk_bound"] == pytest.approx(0.10)
        assert r["holds"] is True

    def test_all_zero_risks(self):
        r = verify_aggregation_under_predictable_budget([0.0, 0.0, 0.0])
        assert r["episode_risk_bound"] == 0.0


# ═══════════════════════════════════════════════════════════════════════
# Propositions
# ═══════════════════════════════════════════════════════════════════════


class TestPropInsufficiency:
    def test_true_exceeds_observed(self):
        r = verify_insufficiency_of_observed_evaluation(violations_observed=3, violations_true=8)
        assert r["holds"] is True
        assert r["evaluation_gap"] == 5

    def test_no_gap_when_equal(self):
        r = verify_insufficiency_of_observed_evaluation(5, 5)
        assert r["holds"] is False


class TestPropInflatedSet:
    def test_contained_with_sufficient_inflation(self):
        r = verify_inflated_set_contains_state(
            x_true=0.55, x_obs=0.50, inflation=2.0, interval_half_width=0.1
        )
        assert r["state_contained"] is True

    def test_not_contained_insufficient_inflation(self):
        r = verify_inflated_set_contains_state(x_true=0.8, x_obs=0.5, inflation=1.0, interval_half_width=0.1)
        assert r["state_contained"] is False

    def test_minimum_inflation_computed(self):
        r = verify_inflated_set_contains_state(x_true=0.6, x_obs=0.5, inflation=2.0, interval_half_width=0.1)
        assert r["minimum_inflation_needed"] == pytest.approx(1.0)


class TestPropTightenedFeasibility:
    def test_tightened_action_is_true_feasible(self):
        r = verify_tightened_feasibility(
            action=0.3,
            tightened_lower=0.2,
            tightened_upper=0.4,
            true_lower=0.0,
            true_upper=1.0,
        )
        assert r["holds"] is True

    def test_fails_when_not_subset(self):
        r = verify_tightened_feasibility(
            action=0.3,
            tightened_lower=-0.1,
            tightened_upper=0.4,
            true_lower=0.0,
            true_upper=1.0,
        )
        assert r["holds"] is False


class TestPropConditionalConservatism:
    def test_coverage_meets_target_all_groups(self):
        rng = np.random.default_rng(42)
        n = 500
        w = rng.uniform(0, 1, n)
        y = rng.normal(0, 1, n)
        # Wide intervals guaranteed to cover
        lo = y - 5.0
        up = y + 5.0
        r = verify_conditional_conservatism(y, lo, up, w, alpha=0.10, n_groups=3)
        assert r["all_groups_meet_target"] is True

    def test_narrow_intervals_may_fail(self):
        rng = np.random.default_rng(42)
        n = 500
        w = rng.uniform(0, 1, n)
        y = rng.normal(0, 1, n)
        # Offset intervals so they rarely cover
        lo = y + 0.5
        up = y + 1.5
        r = verify_conditional_conservatism(y, lo, up, w, alpha=0.10, n_groups=3)
        # Deliberately mis-centered intervals fail coverage
        assert r["marginal_coverage"] < 0.5


class TestPropInterventionLeadTime:
    def test_immediate_intervention(self):
        w = [0.9, 0.3, 0.2, 0.8, 0.9]
        interventions = [1, 2]  # Intervene at the same steps
        r = verify_intervention_lead_time(w, interventions, threshold=0.5)
        assert r["max_lead_time"] == 0

    def test_delayed_intervention(self):
        w = [0.3, 0.3, 0.3, 0.9, 0.9]
        interventions = [2]
        r = verify_intervention_lead_time(w, interventions, threshold=0.5)
        assert r["n_drops"] == 3
        assert r["holds"] is True


class TestPropBudgetMonotonicity:
    def test_monotone_decreasing(self):
        w = [0.1, 0.3, 0.5, 0.7, 0.9]
        r = verify_safe_budget_monotonicity(w)
        assert r["monotone_decreasing_in_w"] is True

    def test_single_value(self):
        r = verify_safe_budget_monotonicity([0.5])
        assert r["holds"] is True


class TestPropTransferFailure:
    def test_all_obligations_met(self):
        r = verify_transfer_failure_breaks_pattern(True, True, True, True)
        assert r["pattern_transfers"] is True

    def test_one_failure_breaks(self):
        r = verify_transfer_failure_breaks_pattern(True, False, True, True)
        assert r["pattern_transfers"] is False
        assert "sound_safe_set" in r["failed_obligations"]

    def test_all_fail(self):
        r = verify_transfer_failure_breaks_pattern(False, False, False, False)
        assert len(r["failed_obligations"]) == 4


class TestPropMismatchBarrier:
    def test_mismatch_exists(self):
        r = verify_constraint_class_mismatch_barrier(action_dim=1, constraint_dim=2)
        assert r["mismatch_exists"] is True

    def test_no_mismatch(self):
        r = verify_constraint_class_mismatch_barrier(action_dim=2, constraint_dim=2)
        assert r["mismatch_exists"] is False


# ═══════════════════════════════════════════════════════════════════════
# Corollaries
# ═══════════════════════════════════════════════════════════════════════


class TestCorOASGRate:
    def test_rate_proportional_to_fault_rate(self):
        r = verify_oasg_rate_lower_bound(fault_rate=0.2, n_steps=100)
        assert r["expected_oasg_events"] == pytest.approx(20.0)


class TestCorOASGSeverity:
    def test_severity_bounded(self):
        errors = [0.1, -0.2, 0.3, -0.05]
        r = verify_oasg_severity(errors)
        assert r["max_severity"] == pytest.approx(0.3)
        assert r["mean_severity"] == pytest.approx(0.1625)


class TestCorZeroViolation:
    def test_high_reliability_zero_regime(self):
        r = verify_zero_violation_regime(w_bar=0.99, T=100)
        assert r["in_zero_regime"] is True
        assert r["expected_violations"] == pytest.approx(0.1)

    def test_low_reliability_above_zero(self):
        r = verify_zero_violation_regime(w_bar=0.5, T=100)
        assert r["in_zero_regime"] is False


class TestCorInterventionSafety:
    def test_negative_correlation(self):
        ir = [0.0, 0.1, 0.2, 0.3, 0.4]
        vr = [0.4, 0.3, 0.2, 0.1, 0.0]
        r = verify_intervention_safety_tradeoff(ir, vr)
        assert r["negative_correlation"] is True
        assert r["monotone_decreasing"] is True


class TestCorPerfectTelemetry:
    def test_collapses_to_zero(self):
        r = verify_perfect_telemetry_collapse()
        assert r["collapses_to_zero"] is True
        assert r["expected_violations"] == 0.0

    def test_any_alpha_and_T(self):
        r = verify_perfect_telemetry_collapse(alpha=0.5, T=10000)
        assert r["collapses_to_zero"] is True


class TestCorReliabilityProportional:
    def test_proportionality(self):
        r = verify_reliability_proportional_safety([0.8, 0.9, 0.7])
        assert r["violation_rate"] > 0.0
        assert r["holds"] is True


class TestCorInterventionSufficiency:
    def test_safe_intervention(self):
        r = verify_intervention_sufficiency(
            soc=0.5,
            repaired_action=0.1,
            capacity_mwh=1.0,
            efficiency=0.95,
            dt_h=1.0,
        )
        assert r["within_bounds"] is True

    def test_violation_intervention(self):
        r = verify_intervention_sufficiency(
            soc=0.95,
            repaired_action=0.5,
            capacity_mwh=1.0,
            efficiency=0.95,
            dt_h=1.0,
        )
        assert r["within_bounds"] is False


class TestCorReliabilityAwareness:
    def test_awareness_necessary(self):
        w = [0.3, 0.4, 0.5, 0.2, 0.6]
        r = verify_reliability_awareness_necessary(w, fixed_margin=0.03)
        assert r["awareness_necessary"] is True

    def test_high_quality_fixed_works(self):
        w = [0.95, 0.98, 0.99]
        r = verify_reliability_awareness_necessary(w, fixed_margin=0.10)
        assert r["fraction_fixed_fails"] == 0.0


class TestCorEpisodeAggregation:
    def test_sum_is_episode_bound(self):
        risks = [0.01] * 100
        r = verify_episode_aggregation(risks)
        assert r["episode_bound"] == pytest.approx(1.0)


class TestCorAVPromotion:
    def test_mismatch_gives_three_routes(self):
        r = verify_av_promotion_routes(action_dim=1, constraint_dim=2)
        assert r["n_promotion_routes"] == 3
        assert r["mismatch_exists"] is True

    def test_no_mismatch_no_routes(self):
        r = verify_av_promotion_routes(action_dim=3, constraint_dim=2)
        assert r["n_promotion_routes"] == 0
