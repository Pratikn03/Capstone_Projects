"""Unit tests for DC3S telemetry reliability scoring."""
from __future__ import annotations

from gridpulse.dc3s.quality import compute_reliability


def test_reliability_bounds_and_penalties():
    last = {
        "ts_utc": "2026-02-22T00:00:00+00:00",
        "load_mw": 50000.0,
        "renewables_mw": 17000.0,
    }
    event = {
        "ts_utc": "2026-02-22T02:30:00+00:00",
        "load_mw": 65000.0,
        "renewables_mw": None,
    }
    w_t, flags = compute_reliability(
        event,
        last,
        expected_cadence_s=3600.0,
        reliability_cfg={"lambda_delay": 0.002, "spike_beta": 0.25, "ooo_gamma": 0.35, "min_w": 0.05},
    )
    assert 0.05 <= w_t <= 1.0
    assert flags["missing_fraction"] > 0.0
    assert flags["delay_seconds"] > 0.0
    assert flags["spike_detected"] is True


def test_reliability_out_of_order_penalty():
    last = {"ts_utc": "2026-02-22T01:00:00+00:00", "load_mw": 50000.0}
    event = {"ts_utc": "2026-02-22T00:55:00+00:00", "load_mw": 50010.0}
    w_t, flags = compute_reliability(event, last, expected_cadence_s=3600.0, reliability_cfg={"min_w": 0.05})
    assert 0.05 <= w_t <= 1.0
    assert flags["out_of_order"] is True

