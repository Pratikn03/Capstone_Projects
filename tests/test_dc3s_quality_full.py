"""Comprehensive tests for DC3S telemetry quality scoring."""
from __future__ import annotations

import math

import pytest

from orius.dc3s.quality import compute_reliability


def _event(ts="2026-01-01T01:00:00+00:00", load=50000.0, ren=10000.0, **kw):
    e = {"ts_utc": ts, "load_mw": load, "renewables_mw": ren}
    e.update(kw)
    return e


def _last(ts="2026-01-01T00:00:00+00:00", load=50000.0, ren=10000.0, **kw):
    return _event(ts=ts, load=load, ren=ren, **kw)


_CFG = {"lambda_delay": 0.002, "spike_beta": 0.25, "ooo_gamma": 0.35, "min_w": 0.05}
_CADENCE = 3600.0


class TestPerfectTelemetry:
    def test_clean_signals_w_near_one(self):
        w, f = compute_reliability(_event(), _last(), _CADENCE, _CFG)
        assert w == pytest.approx(1.0, abs=0.01)
        assert f["missing_fraction"] == 0.0

    def test_no_spike_on_small_change(self):
        w, f = compute_reliability(
            _event(load=50100.0), _last(load=50000.0), _CADENCE, _CFG
        )
        assert f["spike_detected"] is False


class TestMissingFraction:
    def test_one_signal_missing(self):
        w, f = compute_reliability(
            _event(ren=None), _last(), _CADENCE, _CFG
        )
        assert f["missing_fraction"] == pytest.approx(0.5, abs=0.01)
        assert w < 1.0

    def test_all_signals_missing(self):
        w, f = compute_reliability(
            {"ts_utc": "2026-01-01T01:00:00+00:00", "load_mw": None, "renewables_mw": None},
            _last(), _CADENCE, _CFG,
        )
        assert f["missing_fraction"] == pytest.approx(1.0, abs=0.01)

    def test_no_numeric_signals_gives_zero_fraction(self):
        w, f = compute_reliability(
            {"ts_utc": "2026-01-01T01:00:00+00:00"}, None, _CADENCE, _CFG,
        )
        assert f["missing_fraction"] == 0.0


class TestDelayPenalty:
    def test_on_time_no_penalty(self):
        w, f = compute_reliability(_event(), _last(), _CADENCE, _CFG)
        assert f["delay_seconds"] == 0.0

    def test_late_arrival_penalises(self):
        w, f = compute_reliability(
            _event(ts="2026-01-01T02:30:00+00:00"), _last(), _CADENCE, _CFG
        )
        assert f["delay_seconds"] > 0.0
        assert w < 1.0

    def test_very_large_delay_brings_w_low(self):
        w, f = compute_reliability(
            _event(ts="2026-01-02T00:00:00+00:00"), _last(), _CADENCE, _CFG
        )
        assert w < 0.5


class TestOutOfOrderPenalty:
    def test_ooo_detected(self):
        w, f = compute_reliability(
            _event(ts="2026-01-01T00:55:00+00:00"),
            _last(ts="2026-01-01T01:00:00+00:00"),
            _CADENCE, _CFG,
        )
        assert f["out_of_order"] is True
        assert w < 1.0

    def test_not_ooo_when_forward(self):
        _, f = compute_reliability(_event(), _last(), _CADENCE, _CFG)
        assert f["out_of_order"] is False


class TestSpikeDetection:
    def test_large_relative_jump_is_spike(self):
        _, f = compute_reliability(
            _event(load=80000.0), _last(load=50000.0), _CADENCE, _CFG
        )
        assert f["spike_detected"] is True

    def test_small_jump_no_spike(self):
        _, f = compute_reliability(
            _event(load=50500.0), _last(load=50000.0), _CADENCE, _CFG
        )
        assert f["spike_detected"] is False

    def test_spike_ratio_tracks_largest_jump(self):
        _, f = compute_reliability(
            _event(load=100000.0, ren=10000.0),
            _last(load=50000.0, ren=10000.0),
            _CADENCE, _CFG,
        )
        assert f["spike_ratio"] == pytest.approx(1.0, abs=0.01)


class TestStaleSensor:
    def test_repeated_values_detected_after_stale_k(self):
        ftit = {"law": "linear", "stale_k": 3, "stale_tol": 1e-9, "decay": 0.98}
        state = {}
        prev = _last()
        for i in range(4):
            ev = _event(ts=f"2026-01-01T{i+1:02d}:00:00+00:00", load=10.0, ren=5.0)
            w, f = compute_reliability(ev, prev, _CADENCE, _CFG, adaptive_state=state, ftit_cfg=ftit)
            state = {"ftit": {"stale_tracker": f["stale_tracker"]}}
            prev = ev
        assert f["fault_flags"]["stale_sensor"] is True


class TestExplicitFaultFlags:
    def test_explicit_dropout_overrides_detection(self):
        _, f = compute_reliability(
            _event(dropout=True), _last(), _CADENCE, _CFG
        )
        assert f["fault_flags"]["dropout"] is True

    def test_explicit_false_overrides(self):
        _, f = compute_reliability(
            _event(load=None, dropout=False), _last(), _CADENCE, _CFG
        )
        assert f["fault_flags"]["dropout"] is False

    def test_string_truthy_values(self):
        _, f = compute_reliability(
            _event(spikes="true"), _last(), _CADENCE, _CFG
        )
        assert f["fault_flags"]["spikes"] is True

    def test_int_truthy_values(self):
        _, f = compute_reliability(
            _event(delay_jitter=1), _last(), _CADENCE, _CFG
        )
        assert f["fault_flags"]["delay_jitter"] is True


class TestFTITMode:
    def test_ftit_ro_returns_different_w_than_linear(self):
        ftit_ro = {"law": "ftit_ro", "decay": 0.98}
        ftit_lin = {"law": "linear", "decay": 0.98}
        ev = _event(dropout=True)
        w_ro, _ = compute_reliability(ev, _last(), _CADENCE, _CFG, ftit_cfg=ftit_ro)
        w_lin, _ = compute_reliability(ev, _last(), _CADENCE, _CFG, ftit_cfg=ftit_lin)
        assert 0.05 <= w_ro <= 1.0
        assert 0.05 <= w_lin <= 1.0


class TestMinWFloor:
    def test_worst_case_above_min_w(self):
        ev = {"ts_utc": "2026-01-02T00:00:00+00:00", "load_mw": None, "renewables_mw": None}
        last = _last()
        w, _ = compute_reliability(ev, last, _CADENCE, {"min_w": 0.05})
        assert w >= 0.05


class TestEdgeCases:
    def test_no_last_event(self):
        w, f = compute_reliability(_event(), None, _CADENCE, _CFG)
        assert 0.05 <= w <= 1.0

    def test_no_timestamps(self):
        w, _ = compute_reliability({"load_mw": 100.0}, {"load_mw": 100.0}, _CADENCE, _CFG)
        assert 0.05 <= w <= 1.0

    def test_non_numeric_values_ignored(self):
        ev = {"ts_utc": "2026-01-01T01:00:00+00:00", "load_mw": "bad", "renewables_mw": 100.0}
        w, f = compute_reliability(ev, _last(), _CADENCE, _CFG)
        assert 0.05 <= w <= 1.0


class TestCombinedFaults:
    def test_multiple_faults_lower_w(self):
        ev = {
            "ts_utc": "2026-01-01T03:00:00+00:00",
            "load_mw": None,
            "renewables_mw": 80000.0,
        }
        w, f = compute_reliability(ev, _last(), _CADENCE, _CFG)
        assert w < 0.8
        assert f["missing_fraction"] > 0.0

    def test_custom_config_params(self):
        cfg = {"lambda_delay": 0.01, "spike_beta": 0.5, "ooo_gamma": 0.5, "min_w": 0.10}
        w, _ = compute_reliability(
            _event(ts="2026-01-01T02:30:00+00:00"), _last(), _CADENCE, cfg
        )
        assert w >= 0.10


class TestComponentsDict:
    def test_components_present(self):
        _, f = compute_reliability(_event(), _last(), _CADENCE, _CFG)
        c = f["components"]
        assert "missing_penalty" in c
        assert "delay_penalty" in c
        assert "ooo_penalty" in c
        assert "spike_penalty" in c

    def test_smooth_rates_present(self):
        _, f = compute_reliability(
            _event(), _last(), _CADENCE, _CFG,
            ftit_cfg={"law": "linear", "decay": 0.98}
        )
        assert "smooth_rates" in f
        assert set(f["smooth_rates"]) == {"dropout", "stale_sensor", "delay_jitter", "out_of_order", "spikes"}
