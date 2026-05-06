"""Tests for DC3S health metric computation and sustained triggers."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import duckdb

from orius.dc3s.certificate import make_certificate, store_certificate
from orius.monitoring.dc3s_health import compute_dc3s_health


def _cert(
    command_id: str, *, intervened: bool, reliability_w: float, drift_flag: bool, inflation: float
) -> dict:
    return make_certificate(
        command_id=command_id,
        device_id="dev",
        zone_id="DE",
        controller="deterministic",
        proposed_action={"charge_mw": 1.0, "discharge_mw": 0.0},
        safe_action={"charge_mw": 0.0, "discharge_mw": 0.0},
        uncertainty={"lower": [1.0], "upper": [2.0], "meta": {"inflation": inflation}},
        reliability={"w_t": reliability_w, "flags": {}},
        drift={"drift": drift_flag},
        model_hash="m",
        config_hash="c",
        intervened=intervened,
        intervention_reason="projection_clip" if intervened else None,
        reliability_w=reliability_w,
        drift_flag=drift_flag,
        inflation=inflation,
    )


def test_compute_dc3s_health_sustained_windows(tmp_path) -> None:
    db_path = tmp_path / "dc3s_health.duckdb"
    state_path = tmp_path / "monitoring_state.json"
    table_name = "dispatch_certificates"

    rows = [
        _cert("c1", intervened=True, reliability_w=0.40, drift_flag=True, inflation=2.6),
        _cert("c2", intervened=True, reliability_w=0.45, drift_flag=True, inflation=2.4),
        _cert("c3", intervened=True, reliability_w=0.55, drift_flag=False, inflation=2.2),
        _cert("c4", intervened=False, reliability_w=0.95, drift_flag=False, inflation=1.1),
        _cert("c5", intervened=False, reliability_w=0.92, drift_flag=False, inflation=1.0),
        _cert("c6", intervened=False, reliability_w=0.98, drift_flag=False, inflation=1.2),
    ]
    for cert in rows:
        store_certificate(cert, duckdb_path=str(db_path), table_name=table_name)

    thresholds = {
        "intervention_rate_threshold": 0.30,
        "low_reliability_w_threshold": 0.60,
        "low_reliability_rate_threshold": 0.25,
        "drift_flag_rate_threshold": 0.10,
        "inflation_p95_threshold": 2.0,
    }
    first = compute_dc3s_health(
        window_hours=24,
        min_commands=5,
        thresholds=thresholds,
        duckdb_path=str(db_path),
        table_name=table_name,
        sustained_windows=2,
        state_path=state_path,
        update_state=True,
    )
    assert first["commands_total"] == 6
    assert first["insufficient_data"] is False
    assert first["triggered"] is False
    assert abs(first["intervention_rate"] - 0.5) < 1e-9
    assert abs(first["low_reliability_rate"] - 0.5) < 1e-9
    assert abs(first["drift_flag_rate"] - (2 / 6)) < 1e-9
    assert first["inflation_p95"] > 2.0

    second = compute_dc3s_health(
        window_hours=24,
        min_commands=5,
        thresholds=thresholds,
        duckdb_path=str(db_path),
        table_name=table_name,
        sustained_windows=2,
        state_path=state_path,
        update_state=True,
    )
    assert second["triggered"] is True
    assert {"intervention_rate", "low_reliability_rate", "drift_flag_rate", "inflation_p95"} <= set(
        second["triggered_flags"]
    )


def test_compute_dc3s_health_falls_back_to_payload_json_for_legacy_rows(tmp_path) -> None:
    db_path = tmp_path / "dc3s_health_legacy.duckdb"
    table_name = "dispatch_certificates"
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(
            f"""
            CREATE TABLE {table_name} (
                command_id VARCHAR PRIMARY KEY,
                certificate_hash VARCHAR,
                prev_hash VARCHAR,
                created_at VARCHAR,
                payload_json VARCHAR
            )
            """
        )
        payload = {
            "reliability": {"w_t": 0.4},
            "drift": {"drift": True},
            "uncertainty": {"meta": {"inflation": 2.4}, "shield_repair": {"repaired": True}},
        }
        conn.execute(
            f"INSERT INTO {table_name} (command_id, certificate_hash, prev_hash, created_at, payload_json) VALUES (?, ?, ?, ?, ?)",
            [
                "legacy-1",
                "hash",
                None,
                datetime.now(UTC).isoformat(),
                json.dumps(payload),
            ],
        )
    finally:
        conn.close()

    out = compute_dc3s_health(
        window_hours=24,
        min_commands=1,
        thresholds={
            "intervention_rate_threshold": 0.3,
            "low_reliability_w_threshold": 0.6,
            "low_reliability_rate_threshold": 0.25,
            "drift_flag_rate_threshold": 0.1,
            "inflation_p95_threshold": 2.0,
        },
        duckdb_path=str(db_path),
        table_name=table_name,
        sustained_windows=1,
        state_path=tmp_path / "legacy_state.json",
        update_state=True,
    )
    assert out["commands_total"] == 1
    assert out["insufficient_data"] is False
    assert out["intervention_rate"] == 1.0
    assert out["low_reliability_rate"] == 1.0
    assert out["drift_flag_rate"] == 1.0
    assert out["inflation_p95"] >= 2.4
    assert out["triggered"] is True
