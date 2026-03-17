"""Multi-agent shared-constraint safety (Paper 5).

Infrastructure-aware safety: local certificates do not auto-compose
when agents share feeder capacity or transformer limits.
"""
from .plant import SharedFeederPlant
from .protocol import (
    IndependentLocalProtocol,
    CentralizedCoordinatorProtocol,
    DistributedNegotiationProtocol,
)
from .margin_allocation import allocate_margins
from .scenarios import run_transformer_capacity_scenario

__all__ = [
    "SharedFeederPlant",
    "IndependentLocalProtocol",
    "CentralizedCoordinatorProtocol",
    "DistributedNegotiationProtocol",
    "allocate_margins",
    "run_transformer_capacity_scenario",
]
