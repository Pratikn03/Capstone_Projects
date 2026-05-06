"""Domain registry for the canonical three-domain ORIUS runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

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
        except Exception:
            logger.warning("Adapter capability profile failed for %s; using registry defaults", domain_id)
    normalized = _normalize_domain_id(domain_id)
    return dict(_REGISTRY[normalized].capabilities)


def list_domains() -> list[str]:
    """List registered domain IDs."""
    _register_builtins()
    return sorted(_REGISTRY.keys())


def _register_builtins() -> None:
    """Register the canonical three-domain ORIUS runtime adapters."""
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
    except ImportError as e:
        logger.warning("Failed to register healthcare domain adapter: %s", e)
