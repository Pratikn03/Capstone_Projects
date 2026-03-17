"""ORIUS vehicles domain — prototype extension.

1D longitudinal control. Not part of locked battery thesis claims.
"""
from .plant import VehiclePlant
from .vehicle_adapter import VehicleDomainAdapter
from .vehicle_runner import run_vehicle_episode, compute_vehicle_metrics

__all__ = [
    "VehiclePlant",
    "VehicleDomainAdapter",
    "run_vehicle_episode",
    "compute_vehicle_metrics",
]
