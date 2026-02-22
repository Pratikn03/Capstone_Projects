"""Smoke test for IoT telemetry-command-ack closed-loop persistence."""
from __future__ import annotations

import math
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from services.api.config import get_api_keys
from services.api.main import app
from services.api.routers import dc3s as dc3s_router


def _predict_target(*, target: str, horizon: int, features_df: pd.DataFrame, forecast_cfg: dict, required: bool):
    idx = np.arange(horizon, dtype=float)
    if target == "load_mw":
        y = 52.0 + 4.0 * np.sin((2.0 * math.pi * idx / 24.0) - 0.5)
    elif target == "wind_mw":
        y = 8.0 + 1.8 * np.sin((2.0 * math.pi * (idx + 3.0) / 24.0))
    else:
        y = np.maximum(0.0, 4.0 * np.sin(math.pi * ((idx % 24.0) - 6.0) / 12.0))
    return np.asarray(y, dtype=float), Path(f"iot_test_{target}.bin")


def test_iot_closed_loop_smoke(monkeypatch, tmp_path):
    db_path = tmp_path / "iot_loop.duckdb"
    api_key = "iot-test-key"
    monkeypatch.setenv("GRIDPULSE_IOT_DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("GRIDPULSE_API_KEYS", json.dumps({api_key: ["read", "write"]}))
    get_api_keys.cache_clear()
    monkeypatch.setattr(dc3s_router, "_load_features_df", lambda _cfg: pd.DataFrame({"price_eur_mwh": [60.0], "carbon_kg_per_mwh": [400.0]}))
    monkeypatch.setattr(dc3s_router, "_predict_target", _predict_target)
    monkeypatch.setattr(dc3s_router, "_resolve_conformal_q", lambda target, horizon: np.full(horizon, 4.0, dtype=float))

    client = TestClient(app)
    headers = {"X-GridPulse-Key": api_key}
    device_id = "iot-test-device"
    telemetry_payload = {
        "device_id": device_id,
        "zone_id": "DE",
        "telemetry_event": {
            "ts_utc": "2026-02-22T00:00:00+00:00",
            "load_mw": 52.0,
            "renewables_mw": 12.0,
        },
    }
    telemetry_resp = client.post("/iot/telemetry", json=telemetry_payload, headers=headers)
    assert telemetry_resp.status_code == 200, telemetry_resp.text

    dc3s_req = {
        "device_id": device_id,
        "zone_id": "DE",
        "current_soc_mwh": 1.0,
        "telemetry_event": telemetry_payload["telemetry_event"],
        "last_actual_load_mw": 52.0,
        "last_pred_load_mw": 50.0,
        "controller": "deterministic",
        "horizon": 24,
        "enqueue_iot": True,
        "queue_ttl_seconds": 30,
        "include_certificate": True,
    }
    step_resp = client.post("/dc3s/step", json=dc3s_req)
    assert step_resp.status_code == 200, step_resp.text
    step_json = step_resp.json()
    command_id = step_json["command_id"]
    assert step_json["queued"] is True
    assert step_json["queue_status"] == "queued"

    next_resp = client.get("/iot/command/next", params={"device_id": device_id, "peek": "false"}, headers=headers)
    assert next_resp.status_code == 200, next_resp.text
    next_payload = next_resp.json()
    assert next_payload["status"] == "ok"
    assert next_payload["command"]["command_id"] == command_id

    ack_resp = client.post(
        "/iot/ack",
        json={
            "device_id": device_id,
            "command_id": command_id,
            "status": "acked",
            "certificate_id": step_json.get("certificate_id"),
            "payload": {"accepted": True, "violation": False},
        },
        headers=headers,
    )
    assert ack_resp.status_code == 200, ack_resp.text

    state_resp = client.get("/iot/state", params={"device_id": device_id}, headers=headers)
    assert state_resp.status_code == 200, state_resp.text
    state_payload = state_resp.json()
    assert state_payload["last_command_id"] == command_id
    assert state_payload["last_ack"]["status"] == "acked"

    conn = duckdb.connect(str(db_path))
    try:
        telemetry_count = conn.execute("SELECT COUNT(*) FROM iot_telemetry").fetchone()[0]
        ack_count = conn.execute("SELECT COUNT(*) FROM iot_ack").fetchone()[0]
    finally:
        conn.close()
    assert telemetry_count >= 1
    assert ack_count >= 1
