"""Auth and scope enforcement tests for /iot endpoints."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from services.api.config import get_api_keys
from services.api.main import app


def _configure_keys(monkeypatch, tmp_path) -> dict[str, list[str]]:
    keys = {
        "rw-key": ["read", "write"],
        "read-key": ["read"],
        "write-key": ["write"],
    }
    monkeypatch.setenv("GRIDPULSE_API_KEYS", json.dumps(keys))
    monkeypatch.setenv("GRIDPULSE_IOT_DUCKDB_PATH", str(tmp_path / "iot_auth.duckdb"))
    get_api_keys.cache_clear()
    return keys


def _telemetry_payload(device_id: str) -> dict:
    return {
        "device_id": device_id,
        "zone_id": "DE",
        "telemetry_event": {
            "ts_utc": "2026-02-22T00:00:00+00:00",
            "load_mw": 52.0,
            "renewables_mw": 12.0,
            "soc_mwh": 1.0,
        },
    }


def test_iot_endpoints_require_api_key(monkeypatch, tmp_path):
    _configure_keys(monkeypatch, tmp_path)
    client = TestClient(app)

    no_key_read = client.get("/iot/state", params={"device_id": "d1"})
    no_key_write = client.post("/iot/telemetry", json=_telemetry_payload("d1"))

    assert no_key_read.status_code == 403
    assert no_key_write.status_code == 403


def test_iot_scope_enforcement(monkeypatch, tmp_path):
    _configure_keys(monkeypatch, tmp_path)
    client = TestClient(app)

    read_headers = {"X-GridPulse-Key": "read-key"}
    write_headers = {"X-GridPulse-Key": "write-key"}
    rw_headers = {"X-GridPulse-Key": "rw-key"}

    write_with_read_scope = client.post("/iot/telemetry", json=_telemetry_payload("scope-d1"), headers=read_headers)
    read_with_write_scope = client.get("/iot/state", params={"device_id": "scope-d1"}, headers=write_headers)
    assert write_with_read_scope.status_code == 401
    assert read_with_write_scope.status_code == 401

    ok_write = client.post("/iot/telemetry", json=_telemetry_payload("scope-d1"), headers=rw_headers)
    assert ok_write.status_code == 200, ok_write.text

    ok_read = client.get("/iot/state", params={"device_id": "scope-d1"}, headers=rw_headers)
    assert ok_read.status_code == 200, ok_read.text

    # Authorized request reaches handler; command_id does not exist so this should be not found, not auth failure.
    audit = client.get("/iot/audit/missing-command-id", headers=rw_headers)
    assert audit.status_code == 404
