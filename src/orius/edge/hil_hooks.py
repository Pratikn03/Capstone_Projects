"""HIL-ready hooks: hardware abstraction, replay-to-hardware bridge, fault injector."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class HardwareAbstraction(ABC):
    """Abstract hardware interface for HIL testing."""

    @abstractmethod
    def read_sensor(self, channel: str) -> float:
        """Read a sensor channel (e.g. soc_mwh, power_mw)."""

    @abstractmethod
    def write_actuator(self, channel: str, value: float) -> None:
        """Write to an actuator channel."""

    @abstractmethod
    def get_state(self) -> Mapping[str, Any]:
        """Return current hardware state snapshot."""


class FaultInjector:
    """Inject faults for HIL testing: packet drop, bias, delay."""

    def __init__(self, drop_rate: float = 0.0, seed: int = 42):
        import numpy as np

        self._rng = np.random.default_rng(seed)
        self._drop_rate = max(0.0, min(1.0, drop_rate))

    def should_drop(self) -> bool:
        """Simulate packet drop: True = drop this packet."""
        return self._rng.random() < self._drop_rate

    def inject_bias(self, value: float, magnitude: float = 1.0) -> float:
        """Add random bias to a value."""
        return value + float(self._rng.normal(0, magnitude))

    def inject_delay_steps(self, max_delay: int = 3) -> int:
        """Return simulated delay in steps."""
        return int(self._rng.integers(0, max_delay + 1))


class ReplayToHardwareBridge:
    """Bridge: replay recorded trace to hardware (or mock)."""

    def __init__(
        self,
        trace: Sequence[Mapping[str, Any]],
        hardware: HardwareAbstraction | None = None,
        on_step: Callable[[int, Mapping[str, Any]], None] | None = None,
    ):
        self._trace = list(trace)
        self._hardware = hardware
        self._on_step = on_step

    def replay_step(self, step: int) -> Mapping[str, Any] | None:
        """Replay one step; optionally write to hardware."""
        if step >= len(self._trace):
            return None
        record = self._trace[step]
        if self._hardware is not None:
            for ch, val in record.items():
                if isinstance(val, int | float):
                    self._hardware.write_actuator(str(ch), float(val))
        if self._on_step is not None:
            self._on_step(step, record)
        return record
