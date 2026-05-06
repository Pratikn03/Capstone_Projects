"""Multi-agent shared-constraint safety (Paper 5).

Infrastructure-aware safety: local certificates do not auto-compose
when agents share feeder capacity or transformer limits.
"""

from .margin_allocation import allocate_margins
from .plant import SharedFeederPlant
from .protocol import (
    CentralizedCoordinatorProtocol,
    DistributedNegotiationProtocol,
    IndependentLocalProtocol,
)
from .scenarios import run_transformer_capacity_scenario

__all__ = [
    "CentralizedCoordinatorProtocol",
    "DistributedNegotiationProtocol",
    "IndependentLocalProtocol",
    "SharedFeederPlant",
    "allocate_margins",
    "run_transformer_capacity_scenario",
]
