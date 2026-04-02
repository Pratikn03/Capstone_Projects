"""Battery-only theorem helpers.

This module is the domain-scoped home for battery temporal and fallback
helpers that were previously exported from ``orius.universal_theory``.
Core code should not import these symbols through a universal package root.
"""
from __future__ import annotations

from orius.universal_theory.battery_instantiation import (
    certificate_expiration_bound,
    certificate_half_life,
    certificate_validity_horizon,
    evaluate_graceful_degradation_dominance,
    forward_tube,
    should_expire_certificate,
    should_renew_certificate,
    validate_battery_fallback,
    zero_dispatch_fallback,
)

__all__ = [
    "forward_tube",
    "certificate_validity_horizon",
    "certificate_expiration_bound",
    "zero_dispatch_fallback",
    "validate_battery_fallback",
    "evaluate_graceful_degradation_dominance",
    "certificate_half_life",
    "should_renew_certificate",
    "should_expire_certificate",
]
