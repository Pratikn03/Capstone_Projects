"""Domain registry for ORIUS Universal Framework."""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DomainRegistration:
    """Factory plus conservative capability metadata for one domain."""

    factory: Callable[[Mapping[str, Any] | None], Any]
    capabilities: Mapping[str, Any]


_REGISTRY: dict[str, DomainRegistration] = {}


def _normalize_domain_id(domain_id: str) -> str:
    return str(domain_id).strip().lower()


def register_domain(
    domain_id: str,
    adapter_factory: Callable[[Mapping[str, Any] | None], Any],
    *,
    capabilities: Mapping[str, Any] | None = None,
) -> None:
    """Register a domain adapter factory."""
    _REGISTRY[_normalize_domain_id(domain_id)] = DomainRegistration(
        factory=adapter_factory,
        capabilities=dict(capabilities or {}),
    )


def get_adapter(domain_id: str, cfg: Mapping[str, Any] | None = None) -> Any:
    """Get a domain adapter instance."""
    _register_builtins()
    normalized = _normalize_domain_id(domain_id)
    if normalized not in _REGISTRY:
        raise KeyError(f"Unknown domain: {domain_id}. Available: {list_domains()}")
    return _REGISTRY[normalized].factory(cfg)


def get_domain_capabilities(domain_id: str, cfg: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Return capability metadata for a registered domain."""
    _register_builtins()
    adapter = get_adapter(domain_id, cfg)
    if hasattr(adapter, "capability_profile"):
        try:
            return dict(adapter.capability_profile())
        except Exception:  # noqa: BLE001
            logger.warning("Adapter capability profile failed for %s; using registry defaults", domain_id)
    normalized = _normalize_domain_id(domain_id)
    return dict(_REGISTRY[normalized].capabilities)


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
        register_domain(
            "energy",
            lambda cfg: BatteryDomainAdapter(),
            capabilities={
                "safety_surface_type": "soc_power_envelope",
                "repair_mode": "one_dim_projection",
                "fallback_mode": "safe_hold",
                "supports_multi_agent_eval": True,
                "supports_certos_eval": True,
            },
        )
    except ImportError as e:
        logger.warning("Failed to register energy domain adapter: %s", e)
    try:
        from orius.adapters.vehicle import VehicleDomainAdapter
        register_domain(
            "av",
            lambda cfg: VehicleDomainAdapter(cfg),
            capabilities={
                "safety_surface_type": "ttc_entry_barrier",
                "repair_mode": "one_dim_projection",
                "fallback_mode": "full_brake",
                "supports_multi_agent_eval": False,
                "supports_certos_eval": True,
            },
        )
    except ImportError as e:
        logger.warning("Failed to register av domain adapter: %s", e)
    try:
        from orius.adapters.navigation import NavigationDomainAdapter
        register_domain(
            "navigation",
            lambda cfg: NavigationDomainAdapter(cfg),
            capabilities={
                "safety_surface_type": "arena_obstacle_bounds",
                "repair_mode": "vector_projection",
                "fallback_mode": "hold_position",
                "supports_multi_agent_eval": False,
                "supports_certos_eval": False,
            },
        )
    except ImportError as e:
        logger.warning("Failed to register navigation domain adapter: %s", e)
    try:
        from orius.universal_framework.industrial_adapter import IndustrialDomainAdapter
        register_domain(
            "industrial",
            lambda cfg: IndustrialDomainAdapter(cfg),
            capabilities={
                "safety_surface_type": "power_temperature_envelope",
                "repair_mode": "one_dim_projection",
                "fallback_mode": "power_cap",
                "supports_multi_agent_eval": True,
                "supports_certos_eval": True,
            },
        )
    except ImportError as e:
        logger.warning("Failed to register industrial domain adapter: %s", e)
    try:
        from orius.adapters.healthcare import HealthcareDomainAdapter
        register_domain(
            "healthcare",
            lambda cfg: HealthcareDomainAdapter(cfg),
            capabilities={
                "safety_surface_type": "vital_alert_envelope",
                "repair_mode": "one_dim_projection",
                "fallback_mode": "max_alert",
                "supports_multi_agent_eval": False,
                "supports_certos_eval": True,
            },
        )
        register_domain(
            "surgical_robotics",
            lambda cfg: HealthcareDomainAdapter(cfg),
            capabilities={
                "safety_surface_type": "vital_alert_envelope",
                "repair_mode": "one_dim_projection",
                "fallback_mode": "max_alert",
                "supports_multi_agent_eval": False,
                "supports_certos_eval": True,
            },
        )
    except ImportError as e:
        logger.warning("Failed to register healthcare domain adapter: %s", e)
    try:
        from orius.adapters.aerospace import AerospaceDomainAdapter
        register_domain(
            "aerospace",
            lambda cfg: AerospaceDomainAdapter(cfg),
            capabilities={
                "safety_surface_type": "approach_energy_envelope_placeholder",
                "repair_mode": "bounded_projection",
                "fallback_mode": "envelope_hold",
                "supports_multi_agent_eval": False,
                "supports_certos_eval": False,
            },
        )
    except ImportError as e:
        logger.warning("Failed to register aerospace domain adapter: %s", e)


_register_builtins()
