"""Explicit domain governance policies for CertOS portability checks."""
from __future__ import annotations

from typing import Any, Mapping

from .runtime import DomainGovernancePolicy


class _BaseDomainPolicy(DomainGovernancePolicy):
    def required_certificate_fields(self) -> set[str]:
        return {
            "status",
            "op",
            "safe_action",
            "proposed_action",
            "runtime_surface",
            "closure_tier",
        }


class VehicleGovernancePolicy(_BaseDomainPolicy):
    def fallback_action(self, constraints=None, state=None) -> dict[str, float]:
        return {"acceleration_mps2": -5.0}


class IndustrialGovernancePolicy(_BaseDomainPolicy):
    def fallback_action(self, constraints=None, state=None) -> dict[str, float]:
        return {"power_setpoint_mw": 0.0}


class HealthcareGovernancePolicy(_BaseDomainPolicy):
    def fallback_action(self, constraints=None, state=None) -> dict[str, float]:
        return {"alert_level": 0.0}


class NavigationGovernancePolicy(_BaseDomainPolicy):
    def fallback_action(self, constraints=None, state=None) -> dict[str, float]:
        return {"ax": 0.0, "ay": 0.0}


class AerospaceGovernancePolicy(_BaseDomainPolicy):
    def fallback_action(self, constraints=None, state=None) -> dict[str, float]:
        return {"throttle": 0.0, "bank_deg": 0.0}


def policy_for_domain(domain: str) -> DomainGovernancePolicy:
    normalized = str(domain).strip().lower()
    if normalized in {"vehicle", "av"}:
        return VehicleGovernancePolicy()
    if normalized == "industrial":
        return IndustrialGovernancePolicy()
    if normalized == "healthcare":
        return HealthcareGovernancePolicy()
    if normalized == "navigation":
        return NavigationGovernancePolicy()
    if normalized == "aerospace":
        return AerospaceGovernancePolicy()
    return DomainGovernancePolicy()
