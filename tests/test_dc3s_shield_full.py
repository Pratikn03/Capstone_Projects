"""Comprehensive tests for DC3S safety shield."""
from __future__ import annotations

import pytest

from orius.dc3s.shield import repair_action


def _constraints(**overrides):
    base = {
        "capacity_mwh": 100.0,
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "max_power_mw": 50.0,
        "max_charge_mw": 50.0,
        "max_discharge_mw": 50.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "ramp_mw": 0.0,
        "last_net_mw": 0.0,
        "time_step_hours": 1.0,
    }
    base.update(overrides)
    return base


def _uset(w_t=1.0, drift=False):
    return {"meta": {"w_t": w_t, "drift_flag": drift}, "lower": [48000.0], "upper": [52000.0]}


_PROJ_CFG = {"shield": {"mode": "projection", "reserve_soc_pct_drift": 0.0}}


class TestProjectionPassthrough:
    def test_feasible_action_unchanged(self):
        safe, meta = repair_action(
            {"charge_mw": 10.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 50.0}, _uset(), _constraints(), _PROJ_CFG,
        )
        assert safe["charge_mw"] == pytest.approx(10.0)
        assert safe["discharge_mw"] == 0.0
        assert meta["repaired"] is False

    def test_discharge_feasible_unchanged(self):
        safe, meta = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 10.0},
            {"current_soc_mwh": 50.0}, _uset(), _constraints(), _PROJ_CFG,
        )
        assert safe["discharge_mw"] == pytest.approx(10.0)
        assert meta["repaired"] is False


class TestProjectionClipping:
    def test_charge_exceeding_max_clipped(self):
        safe, _ = repair_action(
            {"charge_mw": 100.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 50.0}, _uset(), _constraints(), _PROJ_CFG,
        )
        assert safe["charge_mw"] <= 50.0

    def test_discharge_exceeding_max_clipped(self):
        safe, _ = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 100.0},
            {"current_soc_mwh": 50.0}, _uset(), _constraints(), _PROJ_CFG,
        )
        assert safe["discharge_mw"] <= 50.0

    def test_simultaneous_charge_discharge_resolved(self):
        safe, _ = repair_action(
            {"charge_mw": 20.0, "discharge_mw": 30.0},
            {"current_soc_mwh": 50.0}, _uset(), _constraints(), _PROJ_CFG,
        )
        assert not (safe["charge_mw"] > 0.01 and safe["discharge_mw"] > 0.01)


class TestProjectionSOCBoundary:
    def test_cant_discharge_below_min_soc(self):
        safe, _ = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 50.0},
            {"current_soc_mwh": 11.0}, _uset(),
            _constraints(discharge_efficiency=1.0), _PROJ_CFG,
        )
        assert safe["discharge_mw"] <= 1.0 + 1e-6

    def test_cant_charge_above_max_soc(self):
        safe, _ = repair_action(
            {"charge_mw": 50.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 89.0}, _uset(),
            _constraints(charge_efficiency=1.0), _PROJ_CFG,
        )
        assert safe["charge_mw"] <= 1.0 + 1e-6


class TestProjectionRamp:
    def test_ramp_constraint_applied(self):
        safe, _ = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 50.0},
            {"current_soc_mwh": 50.0}, _uset(),
            _constraints(ramp_mw=5.0, last_net_mw=0.0), _PROJ_CFG,
        )
        assert safe["discharge_mw"] <= 5.0 + 1e-6

    def test_large_ramp_allows_full_action(self):
        safe, _ = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 30.0},
            {"current_soc_mwh": 50.0}, _uset(),
            _constraints(ramp_mw=100.0, last_net_mw=0.0, discharge_efficiency=1.0), _PROJ_CFG,
        )
        assert safe["discharge_mw"] == pytest.approx(30.0)


class TestProjectionDriftReserve:
    def test_drift_reserve_tightens_min_soc(self):
        cfg = {"shield": {"mode": "projection", "reserve_soc_pct_drift": 0.10}}
        _, meta = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 5.0},
            {"current_soc_mwh": 50.0},
            _uset(drift=True), _constraints(), cfg,
        )
        assert meta["effective_min_soc_mwh"] > 10.0


class TestProjectionFTITOverrides:
    def test_ftit_soc_min_overrides(self):
        _, meta = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 5.0},
            {"current_soc_mwh": 50.0}, _uset(),
            _constraints(ftit_soc_min_mwh=20.0), _PROJ_CFG,
        )
        assert meta["effective_min_soc_mwh"] == 20.0

    def test_ftit_soc_max_overrides(self):
        _, meta = repair_action(
            {"charge_mw": 5.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 50.0}, _uset(),
            _constraints(ftit_soc_max_mwh=70.0), _PROJ_CFG,
        )
        assert meta["effective_max_soc_mwh"] == 70.0


class TestProjectionEfficiency:
    def test_charge_efficiency_affects_next_soc(self):
        safe, meta = repair_action(
            {"charge_mw": 10.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 50.0}, _uset(),
            _constraints(charge_efficiency=0.90), _PROJ_CFG,
        )
        assert meta["next_soc_mwh"] == pytest.approx(50.0 + 0.90 * 10.0, abs=0.1)


class TestSafeLanding:
    def test_moves_soc_toward_target_from_above(self):
        cfg = {"shield": {"mode": "safe_landing", "safe_landing": {"safe_margin_pct": 0.10}}}
        safe, meta = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 92.0}, _uset(w_t=0.1),
            _constraints(max_soc_mwh=95.0, charge_efficiency=1.0, discharge_efficiency=1.0), cfg,
        )
        assert meta["mode"] == "safe_landing"
        assert safe["discharge_mw"] > 0.0

    def test_moves_soc_toward_target_from_below(self):
        cfg = {"shield": {"mode": "safe_landing", "safe_landing": {"safe_margin_pct": 0.10}}}
        safe, meta = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 8.0}, _uset(w_t=0.1),
            _constraints(charge_efficiency=1.0, discharge_efficiency=1.0), cfg,
        )
        assert safe["charge_mw"] > 0.0

    def test_soc_in_safe_zone_no_action(self):
        cfg = {"shield": {"mode": "safe_landing", "safe_landing": {"safe_margin_pct": 0.10}}}
        safe, meta = repair_action(
            {"charge_mw": 5.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 50.0}, _uset(w_t=0.5),
            _constraints(charge_efficiency=1.0, discharge_efficiency=1.0), cfg,
        )
        assert safe["charge_mw"] == 0.0
        assert safe["discharge_mw"] == 0.0

    def test_auto_activate_under_low_w(self):
        cfg = {
            "shield": {
                "mode": "projection",
                "safe_landing": {"auto_activate": True, "w_threshold": 0.15, "safe_margin_pct": 0.10},
            }
        }
        _, meta = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 10.0},
            {"current_soc_mwh": 50.0}, _uset(w_t=0.05),
            _constraints(charge_efficiency=1.0, discharge_efficiency=1.0), cfg,
        )
        assert meta["mode"] == "safe_landing"

    def test_no_auto_activate_above_threshold(self):
        cfg = {
            "shield": {
                "mode": "projection",
                "safe_landing": {"auto_activate": True, "w_threshold": 0.10, "safe_margin_pct": 0.10},
            }
        }
        _, meta = repair_action(
            {"charge_mw": 0.0, "discharge_mw": 10.0},
            {"current_soc_mwh": 50.0}, _uset(w_t=0.5),
            _constraints(), cfg,
        )
        assert meta["mode"] == "projection"


class TestMetaFields:
    def test_mode_field_projection(self):
        _, meta = repair_action(
            {"charge_mw": 5.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 50.0}, _uset(), _constraints(), _PROJ_CFG,
        )
        assert meta["mode"] == "projection"

    def test_repaired_true_when_changed(self):
        _, meta = repair_action(
            {"charge_mw": 100.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 50.0}, _uset(), _constraints(), _PROJ_CFG,
        )
        assert meta["repaired"] is True

    def test_current_soc_in_meta(self):
        _, meta = repair_action(
            {"charge_mw": 5.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 42.0}, _uset(), _constraints(), _PROJ_CFG,
        )
        assert meta["current_soc_mwh"] == 42.0


class TestEdgeCases:
    def test_zero_capacity(self):
        safe, _ = repair_action(
            {"charge_mw": 5.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 0.0}, _uset(),
            _constraints(capacity_mwh=0.0, max_soc_mwh=0.0, min_soc_mwh=0.0), _PROJ_CFG,
        )
        assert safe["charge_mw"] == 0.0

    def test_zero_power(self):
        safe, _ = repair_action(
            {"charge_mw": 5.0, "discharge_mw": 0.0},
            {"current_soc_mwh": 50.0}, _uset(),
            _constraints(max_power_mw=0.0, max_charge_mw=0.0, max_discharge_mw=0.0), _PROJ_CFG,
        )
        assert safe["charge_mw"] == 0.0
