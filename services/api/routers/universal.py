"""API router for the ORIUS universal framework runtime."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, Field

from orius.api.serialization import api_jsonable
from orius.dc3s.drift import PageHinkleyDetector
from orius.universal_framework import get_adapter, list_domains, run_universal_step
from orius.universal_framework.pipeline import PIPELINE_STAGES
from orius.universal_framework.tables import (
    DOMAIN_SAFETY_TABLE,
    DOMAIN_STATE_TABLE,
    FAULT_TAXONOMY_TABLE,
    PIPELINE_STAGES_TABLE,
)
from services.api.security import get_api_key, verify_scope

router = APIRouter()


class UniversalDomainsResponse(BaseModel):
    domains: list[str]
    pipeline_stages: list[str]
    tables: dict[str, str]


class UniversalStepRequest(BaseModel):
    domain_id: str = Field(..., min_length=1)
    raw_telemetry: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, Any]] = Field(default_factory=list)
    candidate_action: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)
    quantile: float = Field(default=50.0, ge=0.0)
    cfg: dict[str, Any] = Field(default_factory=dict)
    residual: float | None = None
    prev_cert_hash: str | None = None
    device_id: str = "device-0"
    zone_id: str = "zone-0"
    controller: str = "orius-universal"


class UniversalStepResponse(BaseModel):
    domain_id: str
    pipeline_stages: list[str]
    certificate: dict[str, Any]
    safe_action: dict[str, Any]
    reliability_w: float
    reliability_flags: dict[str, Any]
    drift_flag: bool
    drift_meta: dict[str, Any]
    uncertainty_set: dict[str, Any]
    repair_meta: dict[str, Any]
    state: dict[str, Any]
    theorem_contracts: dict[str, Any]


def _build_drift_detector(cfg: dict[str, Any], residual: float | None) -> PageHinkleyDetector | None:
    if residual is None:
        return None
    return PageHinkleyDetector.from_state(None, cfg=cfg.get("drift", {}))


@router.get("/domains", response_model=UniversalDomainsResponse)
def universal_domains(api_key: str = Security(get_api_key)) -> UniversalDomainsResponse:
    verify_scope("read", api_key)
    return UniversalDomainsResponse(
        domains=list_domains(),
        pipeline_stages=list(PIPELINE_STAGES),
        tables={
            "domain_state": DOMAIN_STATE_TABLE.strip(),
            "domain_safety": DOMAIN_SAFETY_TABLE.strip(),
            "fault_taxonomy": FAULT_TAXONOMY_TABLE.strip(),
            "pipeline_stages": PIPELINE_STAGES_TABLE.strip(),
        },
    )


@router.post("/step", response_model=UniversalStepResponse)
def universal_step(req: UniversalStepRequest, api_key: str = Security(get_api_key)) -> UniversalStepResponse:
    verify_scope("write", api_key)
    try:
        adapter = get_adapter(req.domain_id, req.cfg)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    drift_detector = _build_drift_detector(req.cfg, req.residual)
    result = run_universal_step(
        domain_adapter=adapter,
        raw_telemetry=req.raw_telemetry,
        history=req.history,
        candidate_action=req.candidate_action,
        constraints=req.constraints,
        quantile=req.quantile,
        cfg=req.cfg,
        drift_detector=drift_detector,
        residual=req.residual,
        prev_cert_hash=req.prev_cert_hash,
        device_id=req.device_id,
        zone_id=req.zone_id,
        controller=req.controller,
    )

    return UniversalStepResponse(
        domain_id=req.domain_id.strip().lower(),
        pipeline_stages=list(PIPELINE_STAGES),
        certificate=api_jsonable(result["certificate"]),
        safe_action=api_jsonable(result["safe_action"]),
        reliability_w=float(result["reliability_w"]),
        reliability_flags=api_jsonable(result["reliability_flags"]),
        drift_flag=bool(result["drift_flag"]),
        drift_meta=api_jsonable(result["drift_meta"]),
        uncertainty_set=api_jsonable(result["uncertainty_set"]),
        repair_meta=api_jsonable(result["repair_meta"]),
        state=api_jsonable(result["state"]),
        theorem_contracts=api_jsonable(result["theorem_contracts"]),
    )
