"""Edge/HIL evidence layer (Paper 6 extension).

Hardware abstraction, replay-to-hardware bridge, fault injector.
"""

from .hil_hooks import (
    FaultInjector,
    HardwareAbstraction,
    ReplayToHardwareBridge,
)

__all__ = [
    "FaultInjector",
    "HardwareAbstraction",
    "ReplayToHardwareBridge",
]
