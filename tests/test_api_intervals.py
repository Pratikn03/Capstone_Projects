"""Interval API tests with patched dependencies."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from services.api.main import app
import services.api.routers.forecast_intervals as intervals
from gridpulse.forecasting.uncertainty.conformal import ConformalConfig, ConformalInterval


def _setup_monkeypatch(monkeypatch, tmp_path: Path, horizon: int) -> None:
    features_path = tmp_path / "features.parquet"
    model_path = tmp_path / "model.pkl"
    conformal_path = tmp_path / "load_mw_conformal.json"

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=48, freq="h", tz="UTC"),
            "load_mw": np.zeros(48, dtype=float),
        }
    )
    features_path.write_text("stub", encoding="utf-8")
    model_path.write_text("stub", encoding="utf-8")
    conformal_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        intervals,
        "_load_cfg",
        lambda: {"data": {"features_path": str(features_path)}, "models": {"load_mw": str(model_path)}},
    )
    monkeypatch.setattr(intervals, "_resolve_model_path", lambda target, cfg: model_path)
    monkeypatch.setattr(intervals, "_cached_bundle", lambda path: {"model_type": "gbm"})
    monkeypatch.setattr(intervals, "load_uncertainty_config", lambda: {"enabled": True})
    monkeypatch.setattr(intervals, "get_conformal_path", lambda target, cfg: conformal_path)
    monkeypatch.setattr(intervals.pd, "read_parquet", lambda path: df)

    ci = ConformalInterval(ConformalConfig(alpha=0.1, horizon_wise=True, rolling=False))
    ci.q_h = np.ones(horizon, dtype=float)
    ci.q_global = 2.0
    monkeypatch.setattr(intervals, "load_conformal", lambda path: ci)

    def _fake_predict(df, bundle, horizon: int = 24):
        return {"forecast": [1.0] * horizon}

    monkeypatch.setattr(intervals, "predict_next_24h", _fake_predict)


def test_forecast_with_intervals_success(monkeypatch, tmp_path):
    _setup_monkeypatch(monkeypatch, tmp_path, horizon=24)
    with TestClient(app) as client:
        resp = client.get("/forecast/with-intervals?target=load_mw&horizon=24")
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["yhat"]) == 24
    assert len(payload["pi90_lower"]) == 24
    assert len(payload["pi90_upper"]) == 24


def test_forecast_with_intervals_horizon_global_fallback(monkeypatch, tmp_path):
    _setup_monkeypatch(monkeypatch, tmp_path, horizon=24)
    with TestClient(app) as client:
        resp = client.get("/forecast/with-intervals?target=load_mw&horizon=48")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["pi90_lower"][0] == -1.0
    assert payload["pi90_upper"][0] == 3.0
