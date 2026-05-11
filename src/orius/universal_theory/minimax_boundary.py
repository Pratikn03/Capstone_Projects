"""Finite ambiguity-class minimax lower bound helpers (Tminimax)."""

from __future__ import annotations

from orius.universal_theory.boundary_indistinguishability import two_state_lower_bound


def finite_ambiguity_minimax_lower_bound(epsilon: float) -> float:
    return two_state_lower_bound(epsilon, disjoint_safe_sets=True)


def evaluate_obs_policy_risk(epsilon: float, empirical_risk: float) -> dict[str, float]:
    lb = finite_ambiguity_minimax_lower_bound(epsilon)
    return {
        "lower_bound": lb,
        "empirical_risk": empirical_risk,
        "satisfies_lower_bound": float(empirical_risk) >= lb,
    }


def evaluate_orius_upper_bound(coverage_miss_alpha: float) -> float:
    return max(0.0, min(1.0, float(coverage_miss_alpha)))
