"""Streaming schemas for telemetry events."""
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class OPSDTelemetryEvent(BaseModel):
    """
    Data contract for a single telemetry event.
    Keep this minimal and stable; add versioning if you evolve the schema.
    """
    utc_timestamp: str = Field(..., description="ISO timestamp (UTC)")

    # Raw OPSD signals (use your exact column names if you keep them)
    DE_load_actual_entsoe_transparency: Optional[float] = None
    DE_wind_generation_actual: Optional[float] = None
    DE_solar_generation_actual: Optional[float] = None

    model_config = ConfigDict(extra="allow")
