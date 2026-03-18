"""Vehicle domain adapter — canonical entrypoint.

Re-exports from current implementations. New code should import from here:

    from orius.adapters.vehicle import VehicleDomainAdapter, VehicleTrackAdapter
"""
from __future__ import annotations

from orius.vehicles.vehicle_adapter import VehicleDomainAdapter
from orius.orius_bench.vehicle_track import VehicleTrackAdapter

__all__ = ["VehicleDomainAdapter", "VehicleTrackAdapter"]
