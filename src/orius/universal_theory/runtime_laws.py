"""Runtime monotonicity law suite (L1-L4)."""

from __future__ import annotations


def verify_inflation_monotonicity(q_t: float, w_low: float, w_high: float, epsilon: float = 1e-6) -> bool:
    m_low = q_t / (w_low + epsilon)
    m_high = q_t / (w_high + epsilon)
    return w_high >= w_low and m_high <= m_low


def verify_safe_set_antitonicity(
    x1_safe_intersection: set[object], x2_safe_intersection: set[object]
) -> bool:
    return set(x2_safe_intersection).issubset(set(x1_safe_intersection))


def verify_intervention_threshold(candidate_action: object, common_safe_core: set[object]) -> bool:
    return candidate_action not in set(common_safe_core)


def verify_ambiguity_sandwich(
    empty_core: bool, mandatory_risk_lb: float, covered_release_risk_ub: float
) -> bool:
    if not empty_core:
        return True
    return (
        mandatory_risk_lb >= 0.0
        and covered_release_risk_ub >= 0.0
        and mandatory_risk_lb >= min(covered_release_risk_ub, 0.0)
    )
