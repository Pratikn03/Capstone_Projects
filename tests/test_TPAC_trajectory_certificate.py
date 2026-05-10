import pytest

from orius.universal_theory.risk_bounds import (
    pac_trajectory_budget,
    trajectory_union_bound_certificate,
    validate_pac_certificate,
)


def test_sum_of_budgets_below_delta_passes() -> None:
    result = trajectory_union_bound_certificate([0.005] * 5, delta=0.05)

    assert result["passes"] is True
    assert result["violation_probability_upper_bound"] == 0.025
    assert result["bound_style"] == "bonferroni_union_bound"


def test_sum_of_budgets_above_delta_fails() -> None:
    result = trajectory_union_bound_certificate([0.02] * 5, delta=0.05)

    assert result["passes"] is False


def test_empirical_violation_must_respect_bound_when_supplied() -> None:
    result = validate_pac_certificate([0.01] * 10, delta=0.2, empirical_violation_rate=0.08)

    assert result["empirical_within_bound"] is True
    assert result["passes"] is True


def test_pac_budget_reports_horizon_and_sum() -> None:
    result = pac_trajectory_budget([0.01, 0.02, 0.03])

    assert result["horizon"] == 3
    assert result["budget_sum"] == pytest.approx(0.06)
