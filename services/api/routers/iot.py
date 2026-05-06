"""API router: IoT closed-loop telemetry, queue, and acknowledgement endpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import APIRouter, HTTPException, Query, Security
from pydantic import BaseModel, Field

from orius.dc3s.certificate import get_certificate
from orius.dc3s.quality import compute_reliability
from orius.iot.store import IoTLoopStore
from orius.security.device_identity import verify_device_request
from services.api.security import get_api_key, verify_scope

router = APIRouter()


class TelemetryEvent(BaseModel):
    ts_utc: datetime
    load_mw: float = Field(..., ge=0.0, le=200000.0)
    renewables_mw: float = Field(..., ge=0.0, le=120000.0)
    soc_mwh: float | None = Field(default=None, ge=0.0)

    class Config:
        extra = "allow"


class IoTTelemetryRequest(BaseModel):
    device_id: str
    zone_id: Literal["DE", "US"] = "DE"
    telemetry_event: TelemetryEvent
    device_key_id: str | None = None
    device_ts_utc: str | None = None
    device_nonce: str | None = None
    device_signature: str | None = None


class IoTTelemetryResponse(BaseModel):
    status: str
    device_id: str
    ts_utc: str
    reliability_w: float
    reliability_flags: dict[str, Any]


class IoTCommandNextResponse(BaseModel):
    status: str
    command: dict[str, Any] | None = None
    hold_reason: str | None = None


class IoTAckRequest(BaseModel):
    device_id: str
    command_id: str
    status: Literal["acked", "nacked"]
    certificate_id: str | None = None
    reason: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    device_key_id: str | None = None
    device_ts_utc: str | None = None
    device_nonce: str | None = None
    device_signature: str | None = None


class IoTAckResponse(BaseModel):
    status: str
    ack_id: str
    command_id: str


class IoTHoldResetRequest(BaseModel):
    device_id: str
    reason: str | None = None


class IoTHoldResetResponse(BaseModel):
    status: str
    device_id: str
    hold_active: bool
    hold_reason: str | None = None


def _load_reliability_cfg() -> tuple[float, dict[str, Any]]:
    cfg_path = Path("configs/dc3s.yaml")
    if not cfg_path.exists():
        return 3600.0, {}
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    dc3s = payload.get("dc3s", {}) if isinstance(payload, dict) else {}
    return float(dc3s.get("expected_cadence_s", 3600.0)), dict(dc3s.get("reliability", {}))


def _load_dc3s_audit_cfg() -> tuple[str, str]:
    cfg_path = Path("configs/dc3s.yaml")
    if not cfg_path.exists():
        return "data/audit/dc3s_audit.duckdb", "dispatch_certificates"
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    dc3s = payload.get("dc3s", {}) if isinstance(payload, dict) else {}
    audit = dc3s.get("audit", {}) if isinstance(dc3s, dict) else {}
    return str(audit.get("duckdb_path", "data/audit/dc3s_audit.duckdb")), str(
        audit.get("table_name", "dispatch_certificates")
    )


def _telemetry_payload(event: TelemetryEvent) -> dict[str, Any]:
    payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else event.dict()
    ts_val = payload.get("ts_utc")
    if isinstance(ts_val, datetime):
        payload["ts_utc"] = ts_val.astimezone(UTC).isoformat()
    return dict(payload)


def _request_payload(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return dict(model.model_dump(mode="json", exclude_none=True))
    return dict(model.dict(exclude_none=True))


def _verify_device_identity_or_raise(payload: dict[str, Any], store: IoTLoopStore) -> None:
    verification = verify_device_request(payload)
    if not verification["valid"]:
        status_code = 403
        raise HTTPException(status_code=status_code, detail=str(verification["reason"]))
    if verification.get("verified"):
        recorded = store.record_device_nonce(
            device_id=str(verification["device_id"]),
            device_key_id=str(verification["device_key_id"]),
            device_nonce=str(verification["device_nonce"]),
        )
        if not recorded:
            raise HTTPException(status_code=409, detail="device nonce replay detected")


@router.post("/telemetry", response_model=IoTTelemetryResponse)
def post_telemetry(req: IoTTelemetryRequest, api_key: str = Security(get_api_key)) -> IoTTelemetryResponse:
    verify_scope("write", api_key)
    cadence_s, reliability_cfg = _load_reliability_cfg()
    store = IoTLoopStore()
    try:
        _verify_device_identity_or_raise(_request_payload(req), store)
        current = _telemetry_payload(req.telemetry_event)
        current.setdefault("device_id", req.device_id)
        current.setdefault("zone_id", req.zone_id)
        current["ts_utc"] = str(current.get("ts_utc") or datetime.now(UTC).isoformat())

        last = store.get_last_telemetry(req.device_id)
        w_t, flags = compute_reliability(
            current,
            last,
            expected_cadence_s=cadence_s,
            reliability_cfg=reliability_cfg,
        )
        store.record_telemetry(
            device_id=req.device_id,
            ts_utc=current["ts_utc"],
            payload=current,
            reliability_w=float(w_t),
            reliability_flags=flags,
        )
        return IoTTelemetryResponse(
            status="ok",
            device_id=req.device_id,
            ts_utc=current["ts_utc"],
            reliability_w=float(w_t),
            reliability_flags=flags,
        )
    finally:
        store.close()


@router.get("/command/next", response_model=IoTCommandNextResponse)
def get_command_next(
    device_id: str = Query(...),
    peek: bool = Query(default=False),
    device_key_id: str | None = Query(default=None),
    device_ts_utc: str | None = Query(default=None),
    device_nonce: str | None = Query(default=None),
    device_signature: str | None = Query(default=None),
    api_key: str = Security(get_api_key),
) -> IoTCommandNextResponse:
    verify_scope("read", api_key)
    store = IoTLoopStore()
    try:
        _verify_device_identity_or_raise(
            {
                "device_id": device_id,
                "peek": bool(peek),
                "device_key_id": device_key_id,
                "device_ts_utc": device_ts_utc,
                "device_nonce": device_nonce,
                "device_signature": device_signature,
            },
            store,
        )
        store.expire_stale_commands(device_id=device_id)
        state = store.get_state(device_id) or {}
        if bool(state.get("hold_active", False)):
            return IoTCommandNextResponse(status="hold", command=None, hold_reason=state.get("hold_reason"))
        command = store.get_next_command(device_id=device_id, peek=peek)
        if command is None:
            return IoTCommandNextResponse(status="empty", command=None)
        return IoTCommandNextResponse(status="ok", command=command)
    finally:
        store.close()


@router.post("/ack", response_model=IoTAckResponse)
def post_ack(req: IoTAckRequest, api_key: str = Security(get_api_key)) -> IoTAckResponse:
    verify_scope("write", api_key)
    store = IoTLoopStore()
    try:
        _verify_device_identity_or_raise(_request_payload(req), store)
        ack = store.record_ack(
            device_id=req.device_id,
            command_id=req.command_id,
            status=req.status,
            certificate_id=req.certificate_id,
            reason=req.reason,
            payload=req.payload,
        )
        return IoTAckResponse(status="ok", ack_id=str(ack["ack_id"]), command_id=req.command_id)
    finally:
        store.close()


@router.get("/state")
def get_state(device_id: str = Query(...), api_key: str = Security(get_api_key)) -> dict[str, Any]:
    verify_scope("read", api_key)
    store = IoTLoopStore()
    try:
        state = store.get_state(device_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"No state found for device_id={device_id}")
        return state
    finally:
        store.close()


@router.get("/audit/{command_id}")
def get_audit(command_id: str, api_key: str = Security(get_api_key)) -> dict[str, Any]:
    verify_scope("read", api_key)
    duckdb_path, table_name = _load_dc3s_audit_cfg()
    cert = get_certificate(command_id=command_id, duckdb_path=duckdb_path, table_name=table_name)
    if cert is None:
        raise HTTPException(status_code=404, detail=f"No certificate found for command_id={command_id}")
    return cert


@router.post("/control/reset-hold", response_model=IoTHoldResetResponse)
def reset_hold(req: IoTHoldResetRequest, api_key: str = Security(get_api_key)) -> IoTHoldResetResponse:
    verify_scope("write", api_key)
    store = IoTLoopStore()
    try:
        state = store.get_state(req.device_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"No state found for device_id={req.device_id}")
        store.reset_hold(device_id=req.device_id, reason=req.reason)
        return IoTHoldResetResponse(
            status="ok",
            device_id=req.device_id,
            hold_active=False,
            hold_reason=req.reason,
        )
    finally:
        store.close()
