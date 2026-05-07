from orius.universal_theory.minimax_boundary import finite_ambiguity_minimax_lower_bound


def test_minimax_scoped_lower_bound():
    assert finite_ambiguity_minimax_lower_bound(0.0) == 0.5
    assert finite_ambiguity_minimax_lower_bound(1.0) == 0.0
