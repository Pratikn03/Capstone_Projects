"""Unit tests for PowerGridInverterAdapter and ROS2RoboticsAdapter."""

from __future__ import annotations

import pytest

from orius.adapters.power_grid.adapter import PowerGridInverterAdapter
from orius.adapters.robotics.adapter import ROS2RoboticsAdapter


def test_power_grid_adapter_lifecycle() -> None:
    adapter = PowerGridInverterAdapter()

    # 1. Ingest nominal telemetry
    pkt = adapter.ingest_telemetry({"f_hz": 60.01, "v_pu": 1.01, "delay_ms": 10.0})
    assert pkt["frequency_hz"] == 60.01
    assert not adapter.true_constraint_violated(pkt)

    # 2. OQE score under fresh vs stale
    score_fresh, _ = adapter.compute_oqe([pkt])
    assert score_fresh == 1.0

    stale_pkt = adapter.ingest_telemetry({"f_hz": 60.0, "is_stale": True, "delay_ms": 120.0})
    score_stale, meta = adapter.compute_oqe([stale_pkt])
    assert score_stale < 0.5
    assert "telemetry_stale" in meta["flags"]

    # 3. Uncertainty set inflation
    u_fresh = adapter.build_uncertainty_set(pkt, score_fresh)
    u_stale = adapter.build_uncertainty_set(stale_pkt, score_stale)
    assert (u_stale["f_max"] - u_stale["f_min"]) > (u_fresh["f_max"] - u_fresh["f_min"])

    # 4. Action repair: candidate asking 15 MW curtailed to allowable bound
    admissible = adapter.tighten_action_set(u_fresh)
    safe_a, meta = adapter.repair_action({"p_mw": 15.0, "q_mvar": 0.0}, admissible)
    assert safe_a["p_mw"] <= 10.0
    assert meta["repaired"] is True

    # 5. Certificate emission
    cert = adapter.emit_certificate({"safe_a": safe_a})
    assert cert["certified"] is True
    assert cert["domain"] == "power_grid_inverter"


def test_robotics_adapter_lifecycle() -> None:
    adapter = ROS2RoboticsAdapter(v_max=1.5, a_brake=2.0, d_margin=0.3)

    # 1. Ingest nominal vs dropped telemetry
    pkt_norm = adapter.ingest_telemetry({"scan_min_dist": 5.0, "odom_v": 1.0})
    assert not adapter.true_constraint_violated(pkt_norm)

    pkt_dropped = adapter.ingest_telemetry({"scan_min_dist": 0.2, "odom_v": 1.5, "is_dropped": True})
    assert adapter.true_constraint_violated(pkt_dropped)

    # 2. OQE score under scan drop
    score, meta = adapter.compute_oqe([pkt_dropped])
    assert score <= 0.3
    assert "scan_dropped" in meta["flags"]

    # 3. Safe action tightening: obstacle at 0.6 m allows safe stopping velocity
    u = adapter.build_uncertainty_set(adapter.ingest_telemetry({"scan_min_dist": 0.6}), 1.0)
    admissible = adapter.tighten_action_set(u)
    assert admissible["v_max"] < 1.5  # curtailed below top speed (approx 1.09 m/s)

    # 4. Action repair
    repaired, meta = adapter.repair_action({"v_cmd": 1.5, "w_cmd": 0.0}, admissible)
    assert repaired["v_cmd"] <= admissible["v_max"]
    assert meta["repaired"] is True

    # 5. Fallback emergency brake-hold when obstacle within margin
    u_hazard = {"d_min_obs": 0.2}  # below d_margin 0.3
    admissible_hazard = adapter.tighten_action_set(u_hazard)
    assert admissible_hazard["is_empty"] is True
    fallback_a, meta_fb = adapter.repair_action({"v_cmd": 1.0, "w_cmd": 0.0}, admissible_hazard)
    assert fallback_a["v_cmd"] == 0.0
    assert meta_fb["fallback"] is True
