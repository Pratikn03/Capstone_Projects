"""Domain registry for ORIUS Universal Framework."""
from __future__ import annotations

from typing import Any, Callable, Mapping

_REGISTRY: dict[str, Callable[[Mapping[str, Any] | None], Any]] = {}


def register_domain(domain_id: str, adapter_factory: Callable[[Mapping[str, Any] | None], Any]) -> None:
    """Register a domain adapter factory."""
    _REGISTRY[domain_id] = adapter_factory


def get_adapter(domain_id: str, cfg: Mapping[str, Any] | None = None) -> Any:
    """Get a domain adapter instance."""
    if domain_id not in _REGISTRY:
        raise KeyError(f"Unknown domain: {domain_id}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[domain_id](cfg)


def list_domains() -> list[str]:
    """List registered domain IDs."""
    return list(_REGISTRY.keys())


def _register_builtins() -> None:
    """Register built-in domain adapters."""
    try:
        from orius.dc3s.battery_adapter import BatteryDomainAdapter
        register_domain("energy", lambda cfg: BatteryDomainAdapter())
    except ImportError:
        pass
    try:
        from orius.vehicles.vehicle_adapter import VehicleDomainAdapter
        register_domain("av", lambda cfg: VehicleDomainAdapter(cfg))
    except ImportError:
        pass
    try:
        from orius.universal_framework.industrial_adapter import IndustrialDomainAdapter
        register_domain("industrial", lambda cfg: IndustrialDomainAdapter(cfg))
    except ImportError:
        pass
    try:
        from orius.universal_framework.healthcare_adapter import HealthcareDomainAdapter
        register_domain("healthcare", lambda cfg: HealthcareDomainAdapter(cfg))
        register_domain("surgical_robotics", lambda cfg: HealthcareDomainAdapter(cfg))
    except ImportError:
        pass
    try:
        from orius.universal_framework.aerospace_adapter import AerospaceDomainAdapter
        register_domain("aerospace", lambda cfg: AerospaceDomainAdapter(cfg))
    except ImportError:
        pass


_register_builtins()
