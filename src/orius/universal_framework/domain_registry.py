"""Domain registry for ORIUS Universal Framework."""
from __future__ import annotations

from typing import Any, Callable, Mapping

_REGISTRY: dict[str, Callable[[Mapping[str, Any] | None], Any]] = {}


def _normalize_domain_id(domain_id: str) -> str:
    return str(domain_id).strip().lower()


def register_domain(domain_id: str, adapter_factory: Callable[[Mapping[str, Any] | None], Any]) -> None:
    """Register a domain adapter factory."""
    _REGISTRY[_normalize_domain_id(domain_id)] = adapter_factory


def get_adapter(domain_id: str, cfg: Mapping[str, Any] | None = None) -> Any:
    """Get a domain adapter instance."""
    _register_builtins()
    normalized = _normalize_domain_id(domain_id)
    if normalized not in _REGISTRY:
        raise KeyError(f"Unknown domain: {domain_id}. Available: {list_domains()}")
    return _REGISTRY[normalized](cfg)


def list_domains() -> list[str]:
    """List registered domain IDs."""
    _register_builtins()
    return sorted(_REGISTRY.keys())


def _register_builtins() -> None:
    """Register built-in domain adapters. Uses canonical orius.adapters.* paths.

    This function is intentionally idempotent. Some call paths import
    ``orius.universal_framework`` while a canonical adapter package is still
    initializing, which can temporarily hide that adapter behind a partial
    import. Re-running registration after imports settle restores the missing
    domain without changing external behavior.
    """
    try:
        from orius.adapters.battery import BatteryDomainAdapter
        register_domain("energy", lambda cfg: BatteryDomainAdapter())
    except ImportError:
        pass
    try:
        from orius.adapters.vehicle import VehicleDomainAdapter
        register_domain("av", lambda cfg: VehicleDomainAdapter(cfg))
    except ImportError:
        pass
    try:
        from orius.adapters.industrial import IndustrialDomainAdapter
        register_domain("industrial", lambda cfg: IndustrialDomainAdapter(cfg))
    except ImportError:
        pass
    try:
        from orius.adapters.healthcare import HealthcareDomainAdapter
        register_domain("healthcare", lambda cfg: HealthcareDomainAdapter(cfg))
        register_domain("surgical_robotics", lambda cfg: HealthcareDomainAdapter(cfg))
    except ImportError:
        pass
    try:
        from orius.adapters.aerospace import AerospaceDomainAdapter
        register_domain("aerospace", lambda cfg: AerospaceDomainAdapter(cfg))
    except ImportError:
        pass


_register_builtins()
