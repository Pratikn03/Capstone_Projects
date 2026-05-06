"""ORIUS vehicles domain — DEPRECATED prototype extension.

1D longitudinal control. Not part of locked battery thesis claims.
Use ``orius.universal_framework`` adapters for new development.
"""

import warnings as _w

_w.warn(
    "orius.vehicles is deprecated; use orius.universal_framework adapters",
    DeprecationWarning,
    stacklevel=2,
)

from .plant import VehiclePlant  # noqa: E402
from .vehicle_adapter import VehicleDomainAdapter  # noqa: E402
from .vehicle_runner import compute_vehicle_metrics, run_vehicle_episode  # noqa: E402

__all__ = [
    "VehicleDomainAdapter",
    "VehiclePlant",
    "compute_vehicle_metrics",
    "run_vehicle_episode",
]
