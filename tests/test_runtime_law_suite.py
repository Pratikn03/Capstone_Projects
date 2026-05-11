from orius.universal_theory.runtime_laws import (
    verify_ambiguity_sandwich,
    verify_inflation_monotonicity,
    verify_intervention_threshold,
    verify_safe_set_antitonicity,
)


def test_runtime_laws_smoke():
    assert verify_inflation_monotonicity(1.0, 0.2, 0.8)
    assert verify_safe_set_antitonicity({"a", "b"}, {"a"})
    assert verify_intervention_threshold("c", {"a", "b"})
    assert verify_ambiguity_sandwich(True, 0.2, 0.1)
