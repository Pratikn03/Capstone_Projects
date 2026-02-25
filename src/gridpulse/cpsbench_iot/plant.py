from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BatteryPlant:
    """Truth plant: evolves SOC using physics. NEVER clamps SOC."""
    soc_mwh: float
    min_soc_mwh: float
    max_soc_mwh: float
    charge_eff: float
    discharge_eff: float
    dt_hours: float = 1.0

    def step(self, charge_mw: float, discharge_mw: float) -> float:
        c = max(0.0, float(charge_mw))
        d = max(0.0, float(discharge_mw))

        # no simultaneous charge/discharge
        if c > 0.0 and d > 0.0:
            if d >= c:
                c = 0.0
            else:
                d = 0.0

        # physics (truth). NO CLAMP.
        self.soc_mwh = self.soc_mwh + (self.charge_eff * c * self.dt_hours) - (d * self.dt_hours / max(self.discharge_eff, 1e-9))
        return float(self.soc_mwh)

    def violation(self) -> Dict[str, float | bool]:
        below = self.soc_mwh < self.min_soc_mwh
        above = self.soc_mwh > self.max_soc_mwh
        severity = 0.0
        if below:
            severity = float(self.min_soc_mwh - self.soc_mwh)
        if above:
            severity = float(self.soc_mwh - self.max_soc_mwh)
        return {
            "violated": bool(below or above),
            "below": bool(below),
            "above": bool(above),
            "severity_mwh": float(severity),
        }
