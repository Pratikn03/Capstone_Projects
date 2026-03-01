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


def test_reliability_passthrough_explicit_fault_flags_and_smooth_rates():
    last = {"ts_utc": "2026-02-22T00:00:00+00:00", "load_mw": 100.0}
    event = {
        "ts_utc": "2026-02-22T01:00:00+00:00",
        "load_mw": 100.0,
        "dropout": True,
        "stale_sensor": True,
        "delay_jitter": 1,
        "out_of_order": False,
        "spikes": "true",
    }
    w_t, flags = compute_reliability(
        event,
        last,
        expected_cadence_s=3600.0,
        reliability_cfg={"min_w": 0.05},
        adaptive_state={},
        ftit_cfg={"law": "ftit_ro", "decay": 0.98},
    )
    assert 0.05 <= w_t <= 1.0
    assert flags["fault_flags"] == {
        "dropout": True,
        "stale_sensor": True,
        "delay_jitter": True,
        "out_of_order": False,
        "spikes": True,
    }
    assert {"p_drop", "p_stale", "p_delay", "p_ooo", "p_spike"} <= set(flags)
    assert flags["smooth_rates"]["dropout"] > 0.0
    assert flags["smooth_rates"]["stale_sensor"] > 0.0
    assert flags["smooth_rates"]["delay_jitter"] > 0.0
    assert flags["smooth_rates"]["spikes"] > 0.0


def test_reliability_detects_stale_sensor_from_repeated_values():
    adaptive_state = {}
    last = {"ts_utc": "2026-02-22T00:00:00+00:00", "load_mw": 10.0}
    event1 = {"ts_utc": "2026-02-22T01:00:00+00:00", "load_mw": 10.0}
    _, flags1 = compute_reliability(
        event1,
        last,
        expected_cadence_s=3600.0,
        reliability_cfg={"min_w": 0.05},
        adaptive_state=adaptive_state,
        ftit_cfg={"law": "linear", "stale_k": 3, "stale_tol": 1.0e-9, "decay": 0.98},
    )
    adaptive_state = {"ftit": {"stale_tracker": flags1["stale_tracker"]}}
    event2 = {"ts_utc": "2026-02-22T02:00:00+00:00", "load_mw": 10.0}
    _, flags2 = compute_reliability(
        event2,
        event1,
        expected_cadence_s=3600.0,
        reliability_cfg={"min_w": 0.05},
        adaptive_state=adaptive_state,
        ftit_cfg={"law": "linear", "stale_k": 3, "stale_tol": 1.0e-9, "decay": 0.98},
    )
    adaptive_state = {"ftit": {"stale_tracker": flags2["stale_tracker"]}}
    event3 = {"ts_utc": "2026-02-22T03:00:00+00:00", "load_mw": 10.0}
    _, flags3 = compute_reliability(
        event3,
        event2,
        expected_cadence_s=3600.0,
        reliability_cfg={"min_w": 0.05},
        adaptive_state=adaptive_state,
        ftit_cfg={"law": "linear", "stale_k": 3, "stale_tol": 1.0e-9, "decay": 0.98},
    )
    assert flags3["fault_flags"]["stale_sensor"] is True
    assert flags3["stale_tracker"]["unchanged_counts"]["load_mw"] >= 2
