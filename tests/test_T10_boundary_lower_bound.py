from orius.universal_theory.boundary_indistinguishability import (
    estimate_total_variation,
    two_state_lower_bound,
)


def test_tv_extremes():
    assert estimate_total_variation([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert two_state_lower_bound(0.0, True) == 0.5
    assert two_state_lower_bound(1.0, True) == 0.0


def test_non_disjoint_safe_sets_no_trigger():
    assert two_state_lower_bound(0.2, False) == 0.0
