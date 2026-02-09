"""
Streaming: Pydantic Schemas for Kafka Message Contracts.

This module defines the data contracts for messages flowing through the
streaming pipeline. Using Pydantic models ensures:

1. **Runtime validation**: Invalid messages are rejected early
2. **Documentation**: Schema is self-documenting with Field descriptions
3. **Type safety**: IDE autocomplete and static analysis support
4. **Serialization**: Easy JSON encoding/decoding

Message Evolution:
    When updating schemas, follow these guidelines:
    - New optional fields are backwards-compatible
    - Never remove or rename existing fields
    - Use versioned topic names for breaking changes
    
Usage:
    >>> from gridpulse.streaming.schemas import OPSDTelemetryEvent
    >>> event = OPSDTelemetryEvent.model_validate(json_payload)
    >>> print(event.DE_load_actual_entsoe_transparency)
    
See Also:
    - consumer.py: Kafka consumer that validates against these schemas
    - producer.py: Kafka producer that serializes these schemas
"""
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class OPSDTelemetryEvent(BaseModel):
    """
    Data contract for a single OPSD telemetry event.
    
    This schema represents a single hourly observation from the German
    grid (OPSD/ENTSO-E). All power values are in MW.
    
    Schema Versioning:
        Keep this minimal and stable. Add new optional fields for
        backwards-compatible extensions. For breaking changes,
        create a new topic (e.g., gridpulse.opsd.v2).
        
    Attributes:
        utc_timestamp: ISO 8601 timestamp in UTC
        DE_load_actual_entsoe_transparency: Actual load (MW)
        DE_wind_generation_actual: Wind generation (MW)
        DE_solar_generation_actual: Solar generation (MW)
    """
    utc_timestamp: str = Field(..., description="ISO timestamp (UTC)")

    # Raw OPSD signals - using exact column names from OPSD export
    # Optional fields allow partial updates and graceful degradation
    DE_load_actual_entsoe_transparency: Optional[float] = Field(
        None, description="Actual grid load in MW"
    )
    DE_wind_generation_actual: Optional[float] = Field(
        None, description="Wind generation in MW"
    )
    DE_solar_generation_actual: Optional[float] = Field(
        None, description="Solar generation in MW"
    )

    # Allow additional fields for extensibility
    model_config = ConfigDict(extra="allow")
