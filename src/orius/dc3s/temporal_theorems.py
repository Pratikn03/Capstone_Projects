"""Backward-compatible imports for battery-specific temporal helpers."""

from __future__ import annotations

from orius.universal_theory.battery_instantiation import (
    certificate_expiration_bound,
    certificate_half_life,
    certificate_validity_horizon,
    evaluate_graceful_degradation_dominance,
    forward_tube,
    should_expire_certificate,
    should_renew_certificate,
    zero_dispatch_fallback,
)
from orius.universal_theory.battery_instantiation import (
    validate_battery_fallback as certify_fallback_existence,
)

__all__ = [
    "certificate_expiration_bound",
    "certificate_half_life",
    "certificate_validity_horizon",
    "certify_fallback_existence",
    "evaluate_graceful_degradation_dominance",
    "forward_tube",
    "should_expire_certificate",
    "should_renew_certificate",
    "zero_dispatch_fallback",
]
