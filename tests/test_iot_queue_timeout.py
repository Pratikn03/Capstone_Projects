"""Timeout -> hold -> reset lifecycle tests for IoT command queue."""
from __future__ import annotations

import json

import duckdb
from fastapi.testclient import TestClient

from gridpulse.iot.store import IoTLoopStore
from services.api.config import get_api_keys
from services.api.main import app


def _setup(monkeypatch, tmp_path) -> tuple[TestClient, dict[str, str], str]:
    db_path = tmp_path / "iot_timeout.duckdb"
    key = "rw-timeout-key"
    monkeypatch.setenv("GRIDPULSE_IOT_DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("GRIDPULSE_API_KEYS", json.dumps({key: ["read", "write"]}))
    get_api_keys.cache_clear()
    return TestClient(app), {"X-GridPulse-Key": key}, str(db_path)


def test_queue_timeout_triggers_hold_and_reset(monkeypatch, tmp_path):
    client, headers, db_path = _setup(monkeypatch, tmp_path)
    device_id = "timeout-device"

    telemetry = {
        "device_id": device_id,
        "zone_id": "DE",
        "telemetry_event": {
            "ts_utc": "2026-02-22T00:00:00+00:00",
            "load_mw": 60.0,
            "renewables_mw": 10.0,
            "soc_mwh": 2.0,
        },
    }
    resp = client.post("/iot/telemetry", json=telemetry, headers=headers)
    assert resp.status_code == 200, resp.text

    store = IoTLoopStore(db_path)
    try:
        store.enqueue_command(
            device_id=device_id,
            zone_id="DE",
            command_id="expired-command-1",
            certificate_id="expired-command-1",
            command={"safe_action": {"charge_mw": 0.0, "discharge_mw": 1.0}},
            ttl_seconds=0,
        )
    finally:
        store.close()

    next_cmd = client.get("/iot/command/next", params={"device_id": device_id, "peek": "false"}, headers=headers)
    assert next_cmd.status_code == 200, next_cmd.text
    next_payload = next_cmd.json()
    assert next_payload["status"] == "hold"
    assert next_payload["hold_reason"] == "ack_timeout"

    state = client.get("/iot/state", params={"device_id": device_id}, headers=headers)
    assert state.status_code == 200, state.text
    state_payload = state.json()
    assert state_payload["hold_active"] is True
    assert state_payload["hold_reason"] == "ack_timeout"

    reset = client.post(
        "/iot/control/reset-hold",
        json={"device_id": device_id, "reason": "operator_clearance"},
        headers=headers,
    )
    assert reset.status_code == 200, reset.text
    reset_payload = reset.json()
    assert reset_payload["hold_active"] is False

    after = client.get("/iot/state", params={"device_id": device_id}, headers=headers)
    assert after.status_code == 200, after.text
    after_payload = after.json()
    assert after_payload["hold_active"] is False

    empty = client.get("/iot/command/next", params={"device_id": device_id, "peek": "false"}, headers=headers)
    assert empty.status_code == 200, empty.text
    assert empty.json()["status"] == "empty"

    conn = duckdb.connect(db_path)
    try:
        row = conn.execute(
            "SELECT status, timeout_reason FROM iot_command_queue WHERE command_id = ?",
            ["expired-command-1"],
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "timeout"
    assert row[1] == "ack_timeout"
