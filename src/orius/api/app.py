"""ORIUS FastAPI production endpoint.

POST /step — Execute one DC3S safety step for any supported domain.
GET  /health — Liveness check.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from orius.universal_framework.domain_registry import get_adapter
from orius.universal_framework.pipeline import run_universal_step

from .models import StepRequest, StepResponse
from .serialization import api_jsonable

app = FastAPI(
    title="ORIUS DC3S API",
    description=(
        "Production endpoint for the ORIUS DC3S five-stage safety pipeline. "
        "POST /step receives a domain state and candidate action, applies the "
        "Detect-Calibrate-Constrain-Shield-Certify pipeline, and returns the "
        "repaired safe action with a certificate."
    ),
    version="1.0.0",
)

# Maps user-facing domain names to domain registry IDs
_DOMAIN_MAP: dict[str, str] = {
    "battery":    "energy",
    "energy":     "energy",
    "vehicle":    "av",
    "av":         "av",
    "healthcare": "healthcare",
}
_SUPPORTED_DOMAINS = frozenset(_DOMAIN_MAP.keys())


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — returns status=ok."""
    return {"status": "ok", "service": "orius-dc3s"}


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    """Execute one DC3S safety step.

    Stages (in order):
        1. Detect  — OQE / reliability scoring
        2. Calibrate — conformal uncertainty set with inflation
        3. Constrain — tighten safe action set
        4. Shield  — repair candidate action into safe set
        5. Certify — emit dispatch certificate

    Returns the repaired safe action and a signed certificate.
    """
    domain = request.domain.lower().strip()
    if domain not in _SUPPORTED_DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported domain '{domain}'. "
                   f"Supported: {sorted(_SUPPORTED_DOMAINS)}",
        )

    registry_id = _DOMAIN_MAP[domain]
    try:
        adapter = get_adapter(registry_id, cfg=request.cfg or None)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialise domain adapter for '{domain}': {exc}",
        ) from exc

    try:
        result: dict[str, Any] = run_universal_step(
            domain_adapter=adapter,
            raw_telemetry=request.state,
            history=None,
            candidate_action=request.candidate_action,
            constraints=request.constraints,
            quantile=request.quantile,
            cfg=request.cfg,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"DC3S pipeline error for domain '{domain}': {exc}",
        ) from exc

    safe_action: dict[str, Any] = result.get("safe_action", {})
    candidate = dict(request.candidate_action)
    repaired = safe_action != candidate

    return StepResponse(
        domain=domain,
        safe_action=api_jsonable(safe_action),
        reliability_w=float(result.get("reliability_w", 1.0)),
        repaired=repaired,
        certificate=api_jsonable(result.get("certificate", {})),
        uncertainty_set=api_jsonable(result.get("uncertainty_set", {})),
    )
