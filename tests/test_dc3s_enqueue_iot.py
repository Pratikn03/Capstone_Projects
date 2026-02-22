"""Tests for additive DC3S enqueue_iot request/response behavior."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from gridpulse.iot.store import IoTLoopStore
from services.api.config import get_api_keys
from services.api.main import app
from services.api.routers import dc3s as dc3s_router


def _predict_target(*, target: str, horizon: int, features_df: pd.DataFrame, forecast_cfg: dict, required: bool):
    idx = np.arange(horizon, dtype=float)
    if target == "load_mw":
        y = 52.0 + 4.0 * np.sin((2.0 * math.pi * idx / 24.0) - 0.5)
    elif target == "wind_mw":
        y = 8.0 + 1.8 * np.sin((2.0 * math.pi * (idx + 4.0) / 24.0))
    else:
        y = np.maximum(0.0, 4.0 * np.sin(math.pi * ((idx % 24.0) - 6.0) / 12.0))
    return np.asarray(y, dtype=float), Path(f"enqueue_test_{target}.bin")


def test_dc3s_enqueue_iot_toggle(monkeypatch, tmp_path):
    db_path = tmp_path / "iot_enqueue.duckdb"
    api_key = "enqueue-rw-key"
    monkeypatch.setenv("GRIDPULSE_IOT_DUCKDB_PATH", str(db_path))
    monkeypatch.setenv("GRIDPULSE_API_KEYS", json.dumps({api_key: ["read", "write"]}))
    get_api_keys.cache_clear()

    monkeypatch.setattr(
        dc3s_router,
        "_load_features_df",
        lambda _cfg: pd.DataFrame({"price_eur_mwh": [60.0], "carbon_kg_per_mwh": [400.0]}),
    )
    monkeypatch.setattr(dc3s_router, "_predict_target", _predict_target)
    monkeypatch.setattr(dc3s_router, "_resolve_conformal_q", lambda target, horizon: np.full(horizon, 4.0, dtype=float))

    client = TestClient(app)
    headers = {"X-GridPulse-Key": api_key}
    base_req = {
        "device_id": "enqueue-device",
        "zone_id": "DE",
        "current_soc_mwh": 1.0,
        "telemetry_event": {
            "ts_utc": "2026-02-22T00:00:00+00:00",
            "load_mw": 52.0,
            "renewables_mw": 12.0,
        },
        "last_actual_load_mw": 52.0,
        "last_pred_load_mw": 50.0,
        "controller": "deterministic",
        "horizon": 24,
        "include_certificate": True,
    }

    with_enqueue = dict(base_req)
    with_enqueue["enqueue_iot"] = True
    with_enqueue["queue_ttl_seconds"] = 30
    step_queued = client.post("/dc3s/step", json=with_enqueue)
    assert step_queued.status_code == 200, step_queued.text
    queued_payload = step_queued.json()
    assert queued_payload["queued"] is True
    assert queued_payload["queue_status"] == "queued"

    peek = client.get(
        "/iot/command/next",
        params={"device_id": base_req["device_id"], "peek": "true"},
        headers=headers,
    )
    assert peek.status_code == 200, peek.text
    peek_payload = peek.json()
    assert peek_payload["status"] == "ok"
    assert peek_payload["command"]["command_id"] == queued_payload["command_id"]

    without_enqueue = dict(base_req)
    without_enqueue["enqueue_iot"] = False
    without_enqueue["queue_ttl_seconds"] = 30
    step_skipped = client.post("/dc3s/step", json=without_enqueue)
    assert step_skipped.status_code == 200, step_skipped.text
    skipped_payload = step_skipped.json()
    assert skipped_payload["queued"] is False
    assert skipped_payload["queue_status"] == "skipped"

    store = IoTLoopStore(str(db_path))
    try:
        queued_row = store.get_next_command(device_id=base_req["device_id"], peek=True)
    finally:
        store.close()
    assert queued_row is not None
    assert queued_row["command_id"] == queued_payload["command_id"]
