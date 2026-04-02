from __future__ import annotations

import orius.universal_theory as universal_theory


def test_universal_theory_root_exports_are_domain_neutral() -> None:
    exported = set(getattr(universal_theory, "__all__", []))
    forbidden = {
        "forward_tube",
        "certificate_validity_horizon",
        "certificate_expiration_bound",
        "zero_dispatch_fallback",
        "validate_battery_fallback",
        "evaluate_graceful_degradation_dominance",
        "certificate_half_life",
        "should_renew_certificate",
        "should_expire_certificate",
    }
    assert exported.isdisjoint(forbidden)
