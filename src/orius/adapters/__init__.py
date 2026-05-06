"""ORIUS canonical adapter layer.

This package intentionally resolves exports lazily.

Importing ``orius.adapters.<domain>`` first loads this package, so eagerly
re-exporting every domain module here can create circular imports while the
canonical adapter packages are still initializing.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BatteryDomainAdapter": ("orius.adapters.battery", "BatteryDomainAdapter"),
    "BatteryTrackAdapter": ("orius.adapters.battery", "BatteryTrackAdapter"),
    "VehicleDomainAdapter": ("orius.adapters.vehicle", "VehicleDomainAdapter"),
    "VehicleTrackAdapter": ("orius.adapters.vehicle", "VehicleTrackAdapter"),
    "HealthcareDomainAdapter": ("orius.adapters.healthcare", "HealthcareDomainAdapter"),
    "HealthcareTrackAdapter": ("orius.adapters.healthcare", "HealthcareTrackAdapter"),
    "WaymoAVDomainAdapter": ("orius.adapters.av_waymo", "WaymoAVDomainAdapter"),
    "WaymoReplayTrackAdapter": ("orius.adapters.av_waymo", "WaymoReplayTrackAdapter"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
