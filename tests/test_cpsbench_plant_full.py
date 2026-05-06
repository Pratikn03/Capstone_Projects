"""Comprehensive tests for CPSBench battery plant model."""

from __future__ import annotations

import pytest

from orius.cpsbench_iot.plant import BatteryPlant


def _plant(**kw):
    defaults = {
        "soc_mwh": 50.0,
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "charge_eff": 0.95,
        "discharge_eff": 0.95,
        "dt_hours": 1.0,
    }
    defaults.update(kw)
    return BatteryPlant(**defaults)


class TestPlantStep:
    def test_charge_increases_soc(self):
        p = _plant()
        soc = p.step(10.0, 0.0)
        assert soc == pytest.approx(50.0 + 0.95 * 10.0)

    def test_discharge_decreases_soc(self):
        p = _plant()
        soc = p.step(0.0, 10.0)
        assert soc == pytest.approx(50.0 - 10.0 / 0.95)

    def test_no_action_preserves_soc(self):
        p = _plant()
        soc = p.step(0.0, 0.0)
        assert soc == pytest.approx(50.0)

    def test_simultaneous_resolves_to_discharge(self):
        p = _plant()
        soc = p.step(5.0, 10.0)
        assert soc < 50.0

    def test_simultaneous_resolves_to_charge(self):
        p = _plant()
        soc = p.step(10.0, 5.0)
        assert soc > 50.0

    def test_no_clamping(self):
        p = _plant(soc_mwh=89.0)
        soc = p.step(50.0, 0.0)
        assert soc > 90.0

    def test_dt_hours_scaling(self):
        p = _plant(dt_hours=0.5, charge_eff=1.0)
        soc = p.step(10.0, 0.0)
        assert soc == pytest.approx(55.0)

    def test_negative_charge_clamped_to_zero(self):
        p = _plant()
        soc = p.step(-5.0, 0.0)
        assert soc == pytest.approx(50.0)


class TestPlantViolation:
    def test_no_violation(self):
        p = _plant(soc_mwh=50.0)
        v = p.violation()
        assert v["violated"] is False
        assert v["severity_mwh"] == 0.0

    def test_below_min(self):
        p = _plant(soc_mwh=5.0)
        v = p.violation()
        assert v["violated"] is True
        assert v["below"] is True
        assert v["severity_mwh"] == pytest.approx(5.0)

    def test_above_max(self):
        p = _plant(soc_mwh=95.0)
        v = p.violation()
        assert v["violated"] is True
        assert v["above"] is True
        assert v["severity_mwh"] == pytest.approx(5.0)

    def test_at_boundary_no_violation(self):
        p = _plant(soc_mwh=10.0)
        assert p.violation()["violated"] is False
        p2 = _plant(soc_mwh=90.0)
        assert p2.violation()["violated"] is False


class TestPlantSequential:
    def test_10_step_sequence(self):
        p = _plant(charge_eff=1.0, discharge_eff=1.0)
        for _ in range(5):
            p.step(2.0, 0.0)
        assert p.soc_mwh == pytest.approx(60.0)
        for _ in range(5):
            p.step(0.0, 2.0)
        assert p.soc_mwh == pytest.approx(50.0)
