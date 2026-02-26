"""Smoke tests for DC3S API endpoints with synthetic forecast patching."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

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
    return np.asarray(y, dtype=float), Path(f"test_{target}.bin")


def test_dc3s_step_and_audit_round_trip(monkeypatch):
    monkeypatch.setattr(dc3s_router, "_load_features_df", lambda _cfg: pd.DataFrame({"price_eur_mwh": [60.0], "carbon_kg_per_mwh": [400.0]}))
    monkeypatch.setattr(dc3s_router, "_predict_target", _predict_target)
    monkeypatch.setattr(dc3s_router, "_resolve_conformal_q", lambda target, horizon: np.full(horizon, 4.0, dtype=float))

    client = TestClient(app)
    req = {
        "device_id": "test-device",
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

    step = client.post("/dc3s/step", json=req)
    assert step.status_code == 200, step.text
    payload = step.json()
    assert "safe_action" in payload
    assert "command_id" in payload
    assert payload["certificate_id"] == payload["command_id"]
    assert "intervened" in payload
    assert "intervention_reason" in payload
    assert "reliability_w" in payload
    assert "drift_flag" in payload
    assert "inflation" in payload
    assert "guarantee_checks_passed" in payload
    assert "guarantee_fail_reasons" in payload
    assert 0.0 <= float(payload["reliability_w"]) <= 1.0
    assert float(payload["inflation"]) >= 1.0
    assert payload["guarantee_checks_passed"] in {True, False}
    assert isinstance(payload["guarantee_fail_reasons"], list) or payload["guarantee_fail_reasons"] is None

    audit = client.get(f"/dc3s/audit/{payload['command_id']}")
    assert audit.status_code == 200, audit.text
    cert = audit.json()
    assert cert["command_id"] == payload["command_id"]
    assert "certificate_hash" in cert
    assert cert.get("intervened") == payload["intervened"]
    assert cert.get("reliability_w") == payload["reliability_w"]
    assert cert.get("drift_flag") == payload["drift_flag"]
    assert cert.get("inflation") == payload["inflation"]
    assert cert.get("guarantee_checks_passed") == payload["guarantee_checks_passed"]


def test_dc3s_step_active_mode_blocks_failed_guarantees(monkeypatch):
    monkeypatch.setattr(dc3s_router, "_load_features_df", lambda _cfg: pd.DataFrame({"price_eur_mwh": [60.0], "carbon_kg_per_mwh": [400.0]}))
    monkeypatch.setattr(dc3s_router, "_predict_target", _predict_target)
    monkeypatch.setattr(dc3s_router, "_resolve_conformal_q", lambda target, horizon: np.full(horizon, 4.0, dtype=float))
    monkeypatch.setattr(dc3s_router, "_resolve_iot_mode", lambda _event: "active")
    monkeypatch.setattr(
        dc3s_router,
        "evaluate_guarantee_checks",
        lambda current_soc, action, constraints: (False, ["soc_invariance"], current_soc),
    )

    client = TestClient(app)
    req = {
        "device_id": "test-device-active",
        "zone_id": "DE",
        "current_soc_mwh": 1.0,
        "telemetry_event": {
            "ts_utc": "2026-02-22T00:00:00+00:00",
            "load_mw": 52.0,
            "renewables_mw": 12.0,
        },
        "controller": "deterministic",
        "horizon": 24,
        "include_certificate": False,
    }
    step = client.post("/dc3s/step", json=req)
    assert step.status_code == 400
    assert "guarantee checks failed" in step.text.lower()
