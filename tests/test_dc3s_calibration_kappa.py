from __future__ import annotations

import numpy as np
import pytest

from gridpulse.cpsbench_iot import baselines as cps_baselines
from gridpulse.cpsbench_iot import runner as cps_runner
from gridpulse.dc3s.calibration import build_uncertainty_set_kappa


def _kappa_cfg(**overrides):
    cfg = {
        "reliability": {"min_w": 0.05},
        "infl_max": 3.0,
        "kappa_drift_penalty": 0.5,
        "cooldown_smoothing": 0.0,
    }
    cfg.update(overrides)
    return cfg


def test_kappa_increases_when_reliability_decreases() -> None:
    cfg = _kappa_cfg()
    _, _, meta_hi = build_uncertainty_set_kappa(100.0, 10.0, w_t=1.0, drift_flag=False, cfg=cfg, sigma_sq=4.0)
    _, _, meta_lo = build_uncertainty_set_kappa(100.0, 10.0, w_t=0.2, drift_flag=False, cfg=cfg, sigma_sq=4.0)
    assert meta_lo["kappa"] > meta_hi["kappa"]


def test_kappa_increases_when_sigma_sq_increases() -> None:
    cfg = _kappa_cfg()
    _, _, meta_low = build_uncertainty_set_kappa(100.0, 10.0, w_t=0.8, drift_flag=False, cfg=cfg, sigma_sq=1.0)
    _, _, meta_high = build_uncertainty_set_kappa(100.0, 10.0, w_t=0.8, drift_flag=False, cfg=cfg, sigma_sq=9.0)
    assert meta_high["kappa"] > meta_low["kappa"]


def test_kappa_is_clipped_by_infl_max() -> None:
    cfg = _kappa_cfg(infl_max=1.5)
    _, _, meta = build_uncertainty_set_kappa(100.0, 10.0, w_t=0.05, drift_flag=True, cfg=cfg, sigma_sq=1.0e8)
    assert meta["kappa"] == pytest.approx(1.5)
    assert meta["inflation"] == pytest.approx(1.5)


def test_drift_flag_triggers_wider_inflation() -> None:
    cfg = _kappa_cfg(kappa_drift_penalty=0.5)
    lo_base, hi_base, meta_base = build_uncertainty_set_kappa(
        100.0, 10.0, w_t=0.8, drift_flag=False, cfg=cfg, sigma_sq=4.0
    )
    lo_drift, hi_drift, meta_drift = build_uncertainty_set_kappa(
        100.0, 10.0, w_t=0.8, drift_flag=True, cfg=cfg, sigma_sq=4.0
    )
    assert meta_drift["kappa"] > meta_base["kappa"]
    assert lo_drift[0] < lo_base[0]
    assert hi_drift[0] > hi_base[0]


def test_eps_floor_prevents_blowup_when_q_is_tiny() -> None:
    cfg = _kappa_cfg()
    lo_small, hi_small, meta_small = build_uncertainty_set_kappa(
        100.0, 1.0e-9, w_t=0.5, drift_flag=False, cfg=cfg, sigma_sq=100.0, eps_floor=50.0
    )
    _, _, meta_large = build_uncertainty_set_kappa(
        100.0, 1.0e-9, w_t=0.5, drift_flag=False, cfg=cfg, sigma_sq=100.0, eps_floor=100.0
    )
    assert np.isfinite(lo_small[0])
    assert np.isfinite(hi_small[0])
    assert meta_small["kappa"] <= cfg["infl_max"]
    assert meta_large["kappa"] <= meta_small["kappa"]


def test_prev_inflation_smoothing_blends_as_expected() -> None:
    cfg = _kappa_cfg(cooldown_smoothing=0.8)
    _, _, meta_raw = build_uncertainty_set_kappa(
        100.0, 10.0, w_t=0.9, drift_flag=False, cfg={**cfg, "cooldown_smoothing": 0.0}, sigma_sq=1.0
    )
    _, _, meta_smoothed = build_uncertainty_set_kappa(
        100.0, 10.0, w_t=0.9, drift_flag=False, cfg=cfg, sigma_sq=1.0, prev_inflation=2.5
    )
    expected = np.clip(0.8 * 2.5 + 0.2 * meta_raw["kappa"], 1.0, cfg["infl_max"])
    assert meta_smoothed["kappa"] == pytest.approx(expected)


def test_dc3s_wrapped_dispatch_kappa_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise_dispatch(*args, **kwargs):
        raise RuntimeError("skip solver")

    monkeypatch.setattr(cps_baselines, "optimize_dispatch", _raise_dispatch)

    load_forecast = np.array([100.0, 105.0, 110.0], dtype=float)
    load_true = np.array([103.0, 102.0, 112.0], dtype=float)
    renewables = np.array([10.0, 10.0, 10.0], dtype=float)
    price = np.array([50.0, 55.0, 52.0], dtype=float)
    telemetry_events = [
        {"ts_utc": "2026-01-01T00:00:00Z", "load_mw": 100.0, "device_id": "d1", "zone_id": "z1"},
        {"ts_utc": "2026-01-01T01:00:00Z", "load_mw": 105.0, "device_id": "d1", "zone_id": "z1"},
        {"ts_utc": "2026-01-01T02:00:00Z", "load_mw": 110.0, "device_id": "d1", "zone_id": "z1"},
    ]
    dc3s_cfg = {
        "law": "linear",
        "inflation_law": "kappa",
        "reliability": {"min_w": 0.05},
        "infl_max": 3.0,
        "kappa_drift_penalty": 0.5,
        "expected_cadence_s": 3600.0,
        "drift": {},
        "shield": {"mode": "projection"},
        "ftit": {"law": "linear", "decay": 0.98, "decay_e": 0.95, "dt_hours": 1.0},
    }

    monkeypatch.setattr(cps_baselines, "_load_dc3s_cfg", lambda: dc3s_cfg)

    res = cps_baselines.dc3s_wrapped_dispatch(
        load_forecast=load_forecast,
        renewables_forecast=renewables,
        load_true=load_true,
        telemetry_events=telemetry_events,
        price=price,
        optimization_cfg={},
        command_prefix="kappa-test",
        variant="kappa",
    )

    assert np.all(res["interval_upper"] > res["interval_lower"])
    assert len(res["certificates"]) == len(load_forecast)
    assert res["certificates"][0]["uncertainty"]["meta"]["inflation_law"] == "kappa"


def test_runner_rac_bounds_kappa_wiring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cps_runner,
        "_cqr_bounds",
        lambda load_window: (np.asarray(load_window, dtype=float) - 10.0, np.asarray(load_window, dtype=float) + 10.0),
    )
    monkeypatch.setattr(cps_runner, "_load_rac_cert", lambda target: None)

    load_window = np.array([100.0, 102.0, 104.0], dtype=float)
    kappa_cfg = {
        "law": "linear",
        "inflation_law": "kappa",
        "reliability": {"min_w": 0.05},
        "infl_max": 3.0,
        "kappa_drift_penalty": 0.5,
        "ambiguity": {},
        "rac_cert": {},
    }
    linear_cfg = {**kappa_cfg, "inflation_law": "linear"}

    lo_k, hi_k, meta_k = cps_runner._rac_bounds(
        load_window=load_window,
        dc3s_cfg=kappa_cfg,
        w_t=0.5,
        drift_flag=True,
        sensitivity_t=0.0,
        sensitivity_norm=0.0,
        sigma_sq=9.0,
    )
    lo_l, hi_l, meta_l = cps_runner._rac_bounds(
        load_window=load_window,
        dc3s_cfg=linear_cfg,
        w_t=0.5,
        drift_flag=True,
        sensitivity_t=0.0,
        sensitivity_norm=0.0,
        sigma_sq=9.0,
    )

    assert meta_k["inflation_law"] == "kappa"
    assert meta_l["inflation_rule"] == "linear"
    assert not np.allclose(lo_k, lo_l)
    assert not np.allclose(hi_k, hi_l)
