"""Unit tests for DC3S safety shield repair behavior."""
from __future__ import annotations

from gridpulse.dc3s.shield import repair_action


def test_projection_repair_enforces_soc_and_power_limits():
    safe, meta = repair_action(
        a_star={"charge_mw": 9000.0, "discharge_mw": 8000.0},
        state={"current_soc_mwh": 1200.0},
        uncertainty_set={"meta": {"drift_flag": True}, "lower": [48000.0], "upper": [52000.0]},
        constraints={
            "capacity_mwh": 20000.0,
            "min_soc_mwh": 1000.0,
            "max_soc_mwh": 19000.0,
            "max_power_mw": 5000.0,
            "max_charge_mw": 5000.0,
            "max_discharge_mw": 5000.0,
            "ramp_mw": 2500.0,
            "last_net_mw": 0.0,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.95,
        },
        cfg={"shield": {"mode": "projection", "reserve_soc_pct_drift": 0.08}},
    )
    assert 0.0 <= safe["charge_mw"] <= 5000.0
    assert 0.0 <= safe["discharge_mw"] <= 5000.0
    assert not (safe["charge_mw"] > 0 and safe["discharge_mw"] > 0)
    assert meta["mode"] == "projection"


def test_projection_repair_respects_ftit_soc_min_override():
    safe, meta = repair_action(
        a_star={"charge_mw": 0.0, "discharge_mw": 20.0},
        state={"current_soc_mwh": 15.0},
        uncertainty_set={"meta": {"drift_flag": True}, "lower": [1.0], "upper": [2.0]},
        constraints={
            "capacity_mwh": 100.0,
            "min_soc_mwh": 10.0,
            "max_soc_mwh": 90.0,
            "ftit_soc_min_mwh": 14.0,
            "max_power_mw": 50.0,
            "max_charge_mw": 50.0,
            "max_discharge_mw": 50.0,
            "charge_efficiency": 1.0,
            "discharge_efficiency": 1.0,
        },
        cfg={"shield": {"mode": "projection", "reserve_soc_pct_drift": 0.50}},
    )
    assert safe["discharge_mw"] == 1.0
    assert meta["effective_min_soc_mwh"] == 14.0


def test_projection_repair_respects_ftit_soc_max_override():
    safe, meta = repair_action(
        a_star={"charge_mw": 20.0, "discharge_mw": 0.0},
        state={"current_soc_mwh": 79.0},
        uncertainty_set={"meta": {"drift_flag": False}, "lower": [1.0], "upper": [2.0]},
        constraints={
            "capacity_mwh": 100.0,
            "min_soc_mwh": 10.0,
            "max_soc_mwh": 90.0,
            "ftit_soc_max_mwh": 80.0,
            "max_power_mw": 50.0,
            "max_charge_mw": 50.0,
            "max_discharge_mw": 50.0,
            "charge_efficiency": 1.0,
            "discharge_efficiency": 1.0,
        },
        cfg={"shield": {"mode": "projection", "reserve_soc_pct_drift": 0.0}},
    )
    assert safe["charge_mw"] == 1.0
    assert meta["effective_max_soc_mwh"] == 80.0
