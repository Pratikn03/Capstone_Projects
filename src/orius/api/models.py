"""Pydantic request/response models for the ORIUS /step endpoint."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StepRequest(BaseModel):
    """Request body for POST /step."""

    domain: str = Field(
        ...,
        description="Domain name: battery, vehicle, healthcare",
        examples=["battery"],
    )
    state: dict[str, Any] = Field(
        ...,
        description="Current observed state from the domain adapter (key-value pairs).",
        examples=[{"power_mw": 490.0, "temp_c": 112.0, "pressure_mbar": 1010.0}],
    )
    candidate_action: dict[str, Any] = Field(
        default_factory=dict,
        description="Proposed action before DC3S repair.",
        examples=[{"power_setpoint_mw": 520.0}],
    )
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific safety constraints (e.g. power_max_mw).",
        examples=[{"power_max_mw": 500.0}],
    )
    cfg: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional pipeline configuration overrides.",
    )
    quantile: float = Field(
        default=50.0,
        ge=1.0,
        le=99.0,
        description="Conformal quantile level (50 = median, 90 = 90th percentile).",
    )


class StepResponse(BaseModel):
    """Response body for POST /step."""

    domain: str
    safe_action: dict[str, Any]
    reliability_w: float
    repaired: bool
    certificate: dict[str, Any]
    uncertainty_set: dict[str, Any]
