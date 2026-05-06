"""Explicit domain governance policies for CertOS portability checks."""

from __future__ import annotations

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


class BatteryGovernancePolicy(_BaseDomainPolicy):
    def fallback_action(self, constraints=None, state=None) -> dict[str, float]:
        return {"charge_mw": 0.0, "discharge_mw": 0.0}


class HealthcareGovernancePolicy(_BaseDomainPolicy):
    def fallback_action(self, constraints=None, state=None) -> dict[str, float]:
        return {"alert_level": 0.0}


def policy_for_domain(domain: str) -> DomainGovernancePolicy:
    normalized = str(domain).strip().lower()
    if normalized in {"battery", "energy"}:
        return BatteryGovernancePolicy()
    if normalized in {"vehicle", "av"}:
        return VehicleGovernancePolicy()
    if normalized == "healthcare":
        return HealthcareGovernancePolicy()
    return DomainGovernancePolicy()
