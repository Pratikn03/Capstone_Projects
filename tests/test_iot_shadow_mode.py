"""Shadow-mode ACK semantics test for edge-agent iteration helper."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from iot.edge_agent.run_agent import run_one_iteration
from services.api.config import get_api_keys
from services.api.main import app
from services.api.routers import dc3s as dc3s_router


class _ShadowOnlyDriver:
    def fetch_telemetry(self) -> dict:
        return {
            "ts_utc": "2026-02-22T00:00:00+00:00",
            "load_mw": 52.0,
            "renewables_mw": 12.0,
            "soc_mwh": 1.0,
        }

    def apply_command(self, *, charge_mw: float, discharge_mw: float) -> dict:
        raise AssertionError("apply_command must not be called in shadow mode")


class _TestApiClient:
    def __init__(self, client: TestClient, api_key: str) -> None:
        self.client = client
        self.headers = {"X-GridPulse-Key": api_key}

    def post(self, path: str, payload: dict) -> dict:
        resp = self.client.post(path, json=payload, headers=self.headers)
        resp.raise_for_status()
        return resp.json()

    def get(self, path: str, params: dict | None = None) -> dict:
        resp = self.client.get(path, params=params, headers=self.headers)
        resp.raise_for_status()
        return resp.json()


def _predict_target(*, target: str, horizon: int, features_df: pd.DataFrame, forecast_cfg: dict, required: bool):
    idx = np.arange(horizon, dtype=float)
    if target == "load_mw":
        y = 52.0 + 4.0 * np.sin((2.0 * math.pi * idx / 24.0) - 0.5)
    elif target == "wind_mw":
        y = 8.0 + 1.8 * np.sin((2.0 * math.pi * (idx + 4.0) / 24.0))
    else:
        y = np.maximum(0.0, 4.0 * np.sin(math.pi * ((idx % 24.0) - 6.0) / 12.0))
    return np.asarray(y, dtype=float), Path(f"shadow_test_{target}.bin")


def test_shadow_mode_ack_payload(monkeypatch, tmp_path):
    api_key = "shadow-rw-key"
    monkeypatch.setenv("GRIDPULSE_API_KEYS", json.dumps({api_key: ["read", "write"]}))
    monkeypatch.setenv("GRIDPULSE_IOT_DUCKDB_PATH", str(tmp_path / "iot_shadow.duckdb"))
    get_api_keys.cache_clear()
    monkeypatch.setattr(
        dc3s_router,
        "_load_features_df",
        lambda _cfg: pd.DataFrame({"price_eur_mwh": [60.0], "carbon_kg_per_mwh": [400.0]}),
    )
    monkeypatch.setattr(dc3s_router, "_predict_target", _predict_target)
    monkeypatch.setattr(dc3s_router, "_resolve_conformal_q", lambda target, horizon: np.full(horizon, 4.0, dtype=float))

    client = TestClient(app)
    api = _TestApiClient(client, api_key)
    driver = _ShadowOnlyDriver()

    out = run_one_iteration(
        api=api,
        driver=driver,
        device_id="shadow-device",
        zone_id="DE",
        mode="shadow",
        controller="deterministic",
        horizon=24,
        queue_ttl_seconds=30,
    )
    assert out["status"] == "ok"
    assert out["ack_status"] == "acked"
    assert out["shadow_mode"] is True

    state = api.get("/iot/state", params={"device_id": "shadow-device"})
    assert state["last_ack"]["status"] == "acked"
    assert state["last_ack"]["payload"]["shadow_mode"] is True
    assert state["last_ack"]["payload"]["applied"] is False
