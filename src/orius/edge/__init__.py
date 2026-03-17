"""Edge/HIL evidence layer (Paper 6 extension).

Hardware abstraction, replay-to-hardware bridge, fault injector.
"""
from .hil_hooks import (
    HardwareAbstraction,
    FaultInjector,
    ReplayToHardwareBridge,
)

__all__ = [
    "HardwareAbstraction",
    "FaultInjector",
    "ReplayToHardwareBridge",
]
