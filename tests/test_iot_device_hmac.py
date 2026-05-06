"""Strict device HMAC identity tests for IoT routes."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from orius.security.device_identity import sign_device_payload
from services.api.config import get_api_keys
from services.api.main import app


def _configure_security(monkeypatch, tmp_path) -> dict[str, str]:
    monkeypatch.setenv("ORIUS_ENV", "production")
    monkeypatch.setenv("ORIUS_REQUIRE_DEVICE_SIGNATURE", "1")
    monkeypatch.setenv("ORIUS_IOT_DUCKDB_PATH", str(tmp_path / "iot_hmac.duckdb"))
    monkeypatch.setenv("ORIUS_API_KEYS", json.dumps({"rw-key": ["read", "write"]}))
    monkeypatch.setenv(
        "ORIUS_DEVICE_KEYS",
        json.dumps({"iot-dev-1": {"device-key-1": "device-secret-with-enough-length-123"}}),
    )
    get_api_keys.cache_clear()
    return {"X-ORIUS-Key": "rw-key"}


def _signed_payload(payload: dict, *, nonce: str) -> dict:
    signed = {
        **payload,
        "device_key_id": "device-key-1",
        "device_ts_utc": datetime.now(UTC).isoformat(),
        "device_nonce": nonce,
    }
    signed["device_signature"] = sign_device_payload(
        signed,
        "device-secret-with-enough-length-123",
    )
    return signed


def _telemetry_payload() -> dict:
    return {
        "device_id": "iot-dev-1",
        "zone_id": "DE",
        "telemetry_event": {
            "ts_utc": "2026-05-03T12:00:00+00:00",
            "load_mw": 52.0,
            "renewables_mw": 12.0,
            "soc_mwh": 1.0,
        },
    }


def test_unsigned_iot_telemetry_fails_when_device_signature_required(monkeypatch, tmp_path):
    headers = _configure_security(monkeypatch, tmp_path)
    client = TestClient(app)

    response = client.post("/iot/telemetry", json=_telemetry_payload(), headers=headers)

    assert response.status_code == 403
    assert "device signature required" in response.text


def test_valid_iot_hmac_passes_and_nonce_replay_fails(monkeypatch, tmp_path):
    headers = _configure_security(monkeypatch, tmp_path)
    client = TestClient(app)
    payload = _signed_payload(_telemetry_payload(), nonce="nonce-1")

    ok = client.post("/iot/telemetry", json=payload, headers=headers)
    replay = client.post("/iot/telemetry", json=payload, headers=headers)

    assert ok.status_code == 200, ok.text
    assert replay.status_code == 409
    assert "nonce replay" in replay.text


def test_bad_iot_hmac_fails(monkeypatch, tmp_path):
    headers = _configure_security(monkeypatch, tmp_path)
    client = TestClient(app)
    payload = _signed_payload(_telemetry_payload(), nonce="nonce-2")
    payload["device_signature"] = "bad-signature"

    response = client.post("/iot/telemetry", json=payload, headers=headers)

    assert response.status_code == 403
    assert "device signature invalid" in response.text


def test_signed_iot_ack_passes_in_strict_mode(monkeypatch, tmp_path):
    headers = _configure_security(monkeypatch, tmp_path)
    client = TestClient(app)
    ack = _signed_payload(
        {
            "device_id": "iot-dev-1",
            "command_id": "cmd-1",
            "status": "acked",
            "certificate_id": "cert-1",
            "payload": {"applied": True},
        },
        nonce="ack-nonce-1",
    )

    response = client.post("/iot/ack", json=ack, headers=headers)

    assert response.status_code == 200, response.text
