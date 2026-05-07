from orius.universal_theory.stale_decay import stale_certificate_expiry, stale_uncertainty_growth


def test_stale_growth_linear_and_expiry():
    assert stale_uncertainty_growth(0.1, 0.05, 4) == 0.30000000000000004
    assert stale_certificate_expiry(6, 2) == 4
    assert stale_certificate_expiry(2, 8) == 0
