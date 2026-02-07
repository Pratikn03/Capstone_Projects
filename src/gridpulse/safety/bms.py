"""Virtual battery management safety layer."""
from datetime import datetime
import logging

from pydantic import BaseModel, Field

# Use a dedicated logger name so audit events can be filtered or routed.
audit_logger = logging.getLogger("audit_log")


class SafetyViolation(Exception):
    """Raised when a command violates physical safety limits."""


class BatteryState(BaseModel):
    soc_mwh: float = Field(..., ge=0)
    capacity_mwh: float
    max_charge_mw: float
    max_discharge_mw: float
    last_heartbeat: datetime


class SafetyLayer:
    def __init__(
        self,
        capacity_mwh: float,
        max_power_mw: float,
        min_soc_pct: float = 0.05,
        max_soc_pct: float = 0.95,
    ):
        self.capacity_mwh = capacity_mwh
        self.max_power_mw = max_power_mw
        # Hard Safety Limits (Physical)
        self.min_soc_limit = min_soc_pct * capacity_mwh  # Never go below configured minimum
        self.max_soc_limit = max_soc_pct * capacity_mwh  # Never go above configured maximum

    def validate_dispatch(self, current_soc: float, charge_mw: float, discharge_mw: float) -> bool:
        """Return True if safe, raise SafetyViolation if unsafe."""
        # 1. Power Limit Check
        if charge_mw > self.max_power_mw or discharge_mw > self.max_power_mw:
            audit_logger.error(
                "SAFETY TRIP: Power command %sMW exceeds rating %sMW",
                max(charge_mw, discharge_mw),
                self.max_power_mw,
            )
            raise SafetyViolation("Command exceeds inverter power rating.")

        # 2. Simultaneous Charge/Discharge Check (Physics Impossibility)
        if charge_mw > 0 and discharge_mw > 0:
            audit_logger.error("SAFETY TRIP: Simultaneous charge and discharge command received.")
            raise SafetyViolation("Cannot charge and discharge simultaneously.")

        # 3. State of Charge (SOC) Lookahead Check
        # Assuming 1-hour dispatch interval
        next_soc = current_soc + (charge_mw * 1.0) - (discharge_mw * 1.0)

        if next_soc < self.min_soc_limit:
            audit_logger.critical("SAFETY TRIP: Deep Discharge Risk. Predicted SOC: %s MWh", next_soc)
            raise SafetyViolation("Dispatch causes SOC to drop below safety floor (5%).")

        if next_soc > self.max_soc_limit:
            audit_logger.critical("SAFETY TRIP: Overcharge Risk. Predicted SOC: %s MWh", next_soc)
            raise SafetyViolation("Dispatch causes SOC to exceed safety ceiling (95%).")

        audit_logger.info("Command Validated: Charge %sMW | Discharge %sMW", charge_mw, discharge_mw)
        return True
