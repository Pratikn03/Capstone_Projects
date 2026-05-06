"""Comprehensive tests for DC3S drift detectors."""

from __future__ import annotations

import numpy as np
import pytest

from orius.dc3s.drift import ADWINDetector, PageHinkleyDetector, make_detector


class TestPageHinkleyBasics:
    def test_no_drift_during_warmup(self):
        d = PageHinkleyDetector(warmup_steps=10, threshold=5.0, delta=0.01)
        for _ in range(10):
            out = d.update(10.0)
            assert out["drift"] is False

    def test_drift_triggers_with_large_shift(self):
        d = PageHinkleyDetector(warmup_steps=5, threshold=0.5, delta=0.01, cooldown_steps=2)
        for _ in range(6):
            d.update(0.1)
        out = d.update(10.0)
        assert out["drift"] is True

    def test_cooldown_prevents_retrigger(self):
        d = PageHinkleyDetector(warmup_steps=3, threshold=0.5, delta=0.01, cooldown_steps=5)
        for _ in range(4):
            d.update(0.1)
        d.update(10.0)  # trigger
        for _ in range(3):
            out = d.update(10.0)
            assert out["drift"] is False
            assert out["cooldown_remaining"] > 0

    def test_mean_tracking(self):
        d = PageHinkleyDetector(warmup_steps=100, threshold=100.0)
        for _ in range(100):
            d.update(5.0)
        assert d.mean == pytest.approx(5.0, abs=0.01)

    def test_count_increments(self):
        d = PageHinkleyDetector()
        for i in range(10):
            out = d.update(1.0)
            assert out["count"] == i + 1


class TestPageHinkleySerialization:
    def test_to_state_from_state_round_trip(self):
        d = PageHinkleyDetector(delta=0.02, threshold=3.0, warmup_steps=10, cooldown_steps=5)
        for _ in range(15):
            d.update(1.0)
        state = d.to_state()
        d2 = PageHinkleyDetector.from_state(state)
        assert d2.count == d.count
        assert d2.mean == pytest.approx(d.mean)
        assert d2.delta == d.delta
        assert d2.threshold == d.threshold

    def test_from_state_with_cfg_overrides(self):
        d = PageHinkleyDetector.from_state(
            None, cfg={"ph_delta": 0.05, "ph_lambda": 10.0, "warmup_steps": 20}
        )
        assert d.delta == 0.05
        assert d.threshold == 10.0
        assert d.warmup_steps == 20

    def test_from_state_with_empty_state(self):
        d = PageHinkleyDetector.from_state({})
        assert d.count == 0

    def test_state_fields_complete(self):
        d = PageHinkleyDetector()
        d.update(1.0)
        state = d.to_state()
        expected_keys = {
            "delta",
            "threshold",
            "warmup_steps",
            "cooldown_steps",
            "count",
            "mean",
            "cumulative_sum",
            "min_cumulative_sum",
            "cooldown_remaining",
        }
        assert set(state.keys()) == expected_keys


class TestADWINBasics:
    def test_no_drift_stationary(self):
        d = ADWINDetector(min_window=10, delta=0.002, max_window=200)
        rng = np.random.default_rng(42)
        for _ in range(100):
            out = d.update(rng.normal(5.0, 0.5))
        assert out["drift"] is False

    def test_drift_with_mean_shift(self):
        d = ADWINDetector(min_window=20, delta=0.01, max_window=500, cooldown_steps=2)
        rng = np.random.default_rng(42)
        for _ in range(100):
            d.update(rng.normal(1.0, 0.1))
        triggered = False
        for _ in range(100):
            out = d.update(rng.normal(10.0, 0.1))
            if out["drift"]:
                triggered = True
                break
        assert triggered

    def test_max_window_enforced(self):
        d = ADWINDetector(max_window=50, min_window=5)
        for i in range(200):
            out = d.update(float(i))
        assert out["window_size"] <= 50

    def test_cooldown_after_drift(self):
        d = ADWINDetector(min_window=10, delta=0.5, max_window=200, cooldown_steps=10)
        for _ in range(20):
            d.update(1.0)
        d.update(1000.0)
        out = d.update(1000.0)
        if out["cooldown_remaining"] > 0:
            next_out = d.update(1000.0)
            assert next_out["drift"] is False


class TestADWINSerialization:
    def test_to_state_from_state_round_trip(self):
        d = ADWINDetector(delta=0.01, max_window=100, min_window=5, cooldown_steps=3)
        for _ in range(30):
            d.update(1.0)
        state = d.to_state()
        d2 = ADWINDetector.from_state(state)
        assert d2._count == d._count
        assert list(d2._window) == list(d._window)

    def test_from_state_with_cfg(self):
        d = ADWINDetector.from_state(None, cfg={"adwin_delta": 0.05, "adwin_max_window": 500})
        assert d.delta == 0.05
        assert d.max_window == 500

    def test_state_contains_window(self):
        d = ADWINDetector()
        d.update(1.0)
        d.update(2.0)
        state = d.to_state()
        assert "window" in state
        assert len(state["window"]) == 2


class TestMakeDetector:
    def test_default_is_page_hinkley(self):
        d = make_detector()
        assert isinstance(d, PageHinkleyDetector)

    def test_page_hinkley_explicit(self):
        d = make_detector({"detector": "page_hinkley"})
        assert isinstance(d, PageHinkleyDetector)

    def test_adwin(self):
        d = make_detector({"detector": "adwin"})
        assert isinstance(d, ADWINDetector)

    def test_with_params(self):
        d = make_detector({"detector": "page_hinkley", "ph_delta": 0.05, "ph_lambda": 8.0})
        assert isinstance(d, PageHinkleyDetector)
        assert d.delta == 0.05
        assert d.threshold == 8.0


class TestStressTest:
    def test_page_hinkley_1000_samples(self):
        d = PageHinkleyDetector(warmup_steps=50, threshold=5.0)
        rng = np.random.default_rng(0)
        drifts = 0
        for i in range(1000):
            val = rng.normal(1.0, 0.2) if i < 500 else rng.normal(5.0, 0.2)
            out = d.update(val)
            if out["drift"]:
                drifts += 1
        assert drifts >= 1

    def test_adwin_1000_samples_no_crash(self):
        d = ADWINDetector(min_window=20, max_window=200)
        rng = np.random.default_rng(1)
        for _i in range(1000):
            d.update(rng.normal(0.0, 1.0))
        assert d._count == 1000
