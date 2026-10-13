"""kga.gateway.modes -- Operational modes for the Autonomous Safety Gateway."""

from __future__ import annotations

import enum


class DeploymentMode(str, enum.Enum):
    """Operational deployment modes for the safety gateway."""
    
    INLINE_GATE = "inline_gate"
    """Active synchronous safety gate: routes traffic to adapted model when certified,
    falls back to frozen base model otherwise."""

    SHADOW_CANARY = "shadow_canary"
    """100% of live traffic is served by the frozen base model (f0).
    Candidate adaptation and safety certifications are evaluated in shadow mode,
    recording telemetry and drift statistics without exposing users to risk."""

    AUDIT_ONLY = "audit_only"
    """Passive monitoring mode: records predictions and evidence without triggering
    online updates or model switching."""
