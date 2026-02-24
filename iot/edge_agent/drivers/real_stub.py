"""Placeholder real-device driver interface for future hardware integration."""
from __future__ import annotations

from typing import Any


class RealDeviceDriverStub:
    """Interface-compatible placeholder for BMS/inverter hardware integration."""

    def apply_command(self, *, charge_mw: float, discharge_mw: float) -> dict[str, Any]:
        raise NotImplementedError(
            "Real hardware driver is not implemented yet. Replace with Modbus/BMS/inverter integration."
        )

