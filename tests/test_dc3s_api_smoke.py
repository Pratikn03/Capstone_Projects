"""Smoke tests for DC3S API endpoints with synthetic forecast patching."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from gridpulse.dc3s.state import DC3SStateStore
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


def test_dc3s_step_and_audit_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(dc3s_router, "_load_features_df", lambda _cfg: pd.DataFrame({"price_eur_mwh": [60.0], "carbon_kg_per_mwh": [400.0]}))
    monkeypatch.setattr(dc3s_router, "_predict_target", _predict_target)
    monkeypatch.setattr(dc3s_router, "_resolve_conformal_q", lambda target, horizon: np.full(horizon, 4.0, dtype=float))
    monkeypatch.setattr(
        dc3s_router,
        "_load_dc3s_cfg",
        lambda: {
            "dc3s": {
                "law": "ftit_ro",
                "expected_cadence_s": 3600.0,
                "reliability": {"min_w": 0.05, "lambda_delay": 0.002, "spike_beta": 0.25, "ooo_gamma": 0.35},
                "ftit": {
                    "decay": 0.98,
                    "decay_e": 0.95,
                    "dt_hours": 1.0,
                    "stale_k": 3,
                    "stale_tol": 1.0e-9,
                    "sigma2_init": 1.0,
                    "sigma2_decay": 0.95,
                    "sigma2_floor": 1.0e-6,
                    "delta": 0.05,
                    "eps_interval": 1.0e-6,
                },
                "shield": {"mode": "projection", "reserve_soc_pct_drift": 0.08},
                "drift": {},
                "audit": {
                    "duckdb_path": str(tmp_path / "dc3s_audit.duckdb"),
                    "table_name": "dispatch_certificates",
                    "state_table_name": "dc3s_online_state",
                },
            }
        },
    )

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
    assert "gamma_mw" in cert
    assert "e_t_mwh" in cert
    assert "soc_tube_lower_mwh" in cert
    assert "soc_tube_upper_mwh" in cert

    store = DC3SStateStore(str(tmp_path / "dc3s_audit.duckdb"), table_name="dc3s_online_state")
    try:
        state = store.get(zone_id="DE", device_id="test-device", target="load_mw")
    finally:
        store.close()
    assert state is not None
    assert state["adaptive_state"] != {}
    assert "ftit" in state["adaptive_state"]
    assert "sigma2" in state["adaptive_state"]["ftit"]
    assert "e_t_mwh" in state["adaptive_state"]["ftit"]
    assert "stale_tracker" in state["adaptive_state"]["ftit"]


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


def test_dc3s_step_persists_adaptive_state_across_repeated_events(monkeypatch, tmp_path):
    monkeypatch.setattr(dc3s_router, "_load_features_df", lambda _cfg: pd.DataFrame({"price_eur_mwh": [60.0], "carbon_kg_per_mwh": [400.0]}))
    monkeypatch.setattr(dc3s_router, "_predict_target", _predict_target)
    monkeypatch.setattr(dc3s_router, "_resolve_conformal_q", lambda target, horizon: np.full(horizon, 4.0, dtype=float))
    monkeypatch.setattr(
        dc3s_router,
        "_load_dc3s_cfg",
        lambda: {
            "dc3s": {
                "law": "ftit_ro",
                "expected_cadence_s": 3600.0,
                "reliability": {"min_w": 0.05, "lambda_delay": 0.002, "spike_beta": 0.25, "ooo_gamma": 0.35},
                "ftit": {
                    "decay": 0.98,
                    "decay_e": 0.95,
                    "dt_hours": 1.0,
                    "stale_k": 3,
                    "stale_tol": 1.0e-9,
                    "sigma2_init": 1.0,
                    "sigma2_decay": 0.95,
                    "sigma2_floor": 1.0e-6,
                    "delta": 0.05,
                    "eps_interval": 1.0e-6,
                },
                "shield": {"mode": "projection", "reserve_soc_pct_drift": 0.08},
                "drift": {},
                "audit": {
                    "duckdb_path": str(tmp_path / "dc3s_state.duckdb"),
                    "table_name": "dispatch_certificates",
                    "state_table_name": "dc3s_online_state",
                },
            }
        },
    )
    client = TestClient(app)
    req = {
        "device_id": "persist-device",
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
    first = client.post("/dc3s/step", json=req)
    assert first.status_code == 200, first.text
    req["telemetry_event"]["ts_utc"] = "2026-02-22T01:00:00+00:00"
    second = client.post("/dc3s/step", json=req)
    assert second.status_code == 200, second.text

    store = DC3SStateStore(str(tmp_path / "dc3s_state.duckdb"), table_name="dc3s_online_state")
    try:
        state = store.get(zone_id="DE", device_id="persist-device", target="load_mw")
    finally:
        store.close()
    assert state is not None
    tracker = state["adaptive_state"]["ftit"]["stale_tracker"]
    assert tracker["unchanged_counts"]["load_mw"] >= 1
