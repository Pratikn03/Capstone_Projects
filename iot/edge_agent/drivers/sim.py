"""Digital-twin battery driver used by the IoT closed-loop simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SimBatteryDriver:
    """Simple deterministic battery state transition model with safety clipping."""

    capacity_mwh: float = 10.0
    max_power_mw: float = 5.0
    min_soc_mwh: float = 0.5
    max_soc_mwh: float = 9.5
    efficiency: float = 0.95
    current_soc_mwh: float = 1.0

    def apply_command(self, *, charge_mw: float, discharge_mw: float) -> dict[str, Any]:
        charge = max(0.0, min(float(charge_mw), self.max_power_mw))
        discharge = max(0.0, min(float(discharge_mw), self.max_power_mw))
        if charge > 0.0 and discharge > 0.0:
            if discharge >= charge:
                charge = 0.0
            else:
                discharge = 0.0

        soc_before = float(self.current_soc_mwh)
        charge = min(charge, max(0.0, (self.max_soc_mwh - self.current_soc_mwh) / self.efficiency))
        discharge = min(discharge, max(0.0, (self.current_soc_mwh - self.min_soc_mwh) * self.efficiency))
        soc_after = self.current_soc_mwh + self.efficiency * charge - discharge / self.efficiency
        soc_after = min(self.max_soc_mwh, max(self.min_soc_mwh, soc_after))
        self.current_soc_mwh = float(soc_after)

        violation = (
            charge_mw > self.max_power_mw
            or discharge_mw > self.max_power_mw
            or soc_after < self.min_soc_mwh - 1e-9
            or soc_after > self.max_soc_mwh + 1e-9
        )
        return {
            "accepted": not violation,
            "violation": bool(violation),
            "applied_charge_mw": float(charge),
            "applied_discharge_mw": float(discharge),
            "soc_before_mwh": float(soc_before),
            "soc_after_mwh": float(soc_after),
        }
