"""API integration tests for Phase 6 wiring."""
from __future__ import annotations

from fastapi.testclient import TestClient

from services.api.main import app
import services.api.routers.monitor as monitor_router
import services.api.routers.optimize as optimize_router


def test_optimize_defaults_to_robust_dispatch(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_robust_dispatch(**kwargs):
        calls.update(kwargs)
        return {
            "battery_charge_mw": [0.0, 0.0],
            "battery_discharge_mw": [0.0, 0.0],
            "feasible": True,
            "solver_status": "ok",
            "total_cost": 321.5,
        }

    monkeypatch.setattr(optimize_router, "_load_cfg", lambda: {"battery": {}, "grid": {}})
    monkeypatch.setattr(optimize_router, "optimize_robust_dispatch", fake_robust_dispatch)

    with TestClient(app) as client:
        resp = client.post(
            "/optimize",
            json={
                "forecast_load_mw": [100.0, 110.0],
                "forecast_renewables_mw": [20.0, 25.0],
                "forecast_price_eur_mwh": [60.0, 65.0],
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["expected_cost_usd"] == 321.5
    assert payload["dispatch_plan"]["feasible"] is True
    assert calls["load_lower_bound"] == [100.0, 110.0]
    assert calls["load_upper_bound"] == [100.0, 110.0]


def test_optimize_supports_deterministic_mode(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_det_dispatch(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {
            "expected_cost_usd": 123.0,
            "carbon_kg": 8.0,
            "carbon_cost_usd": 4.0,
        }

    def fail_robust(**kwargs):
        raise AssertionError("robust path should not run in deterministic mode")

    monkeypatch.setattr(optimize_router, "_load_cfg", lambda: {"battery": {}, "grid": {}})
    monkeypatch.setattr(optimize_router, "optimize_dispatch", fake_det_dispatch)
    monkeypatch.setattr(optimize_router, "optimize_robust_dispatch", fail_robust)

    with TestClient(app) as client:
        resp = client.post(
            "/optimize",
            json={
                "optimization_mode": "deterministic",
                "forecast_load_mw": [100.0, 90.0],
                "forecast_renewables_mw": [30.0, 25.0],
            },
        )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["expected_cost_usd"] == 123.0
    assert payload["carbon_kg"] == 8.0
    assert payload["carbon_cost_usd"] == 4.0
    assert calls["args"][0] == [100.0, 90.0]
    assert calls["args"][1] == [30.0, 25.0]


def test_monitor_research_metrics_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        monitor_router,
        "_load_latest_research_summary",
        lambda path: {"row_type": "run_summary", "vss": 1.5, "source_csv": str(path)},
    )
    monkeypatch.setattr(
        monitor_router,
        "_load_frozen_metrics_snapshot",
        lambda: {"frozen_run_id": "20260216_202050"},
    )

    with TestClient(app) as client:
        resp = client.get("/monitor/research-metrics")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["available"] is True
    assert payload["datasets"]["de"]["row_type"] == "run_summary"
    assert payload["datasets"]["us"]["row_type"] == "run_summary"
    assert payload["frozen_metrics_snapshot"]["frozen_run_id"] == "20260216_202050"
