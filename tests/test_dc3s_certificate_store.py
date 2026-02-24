"""Tests for DC3S certificate storage schema and typed telemetry columns."""
from __future__ import annotations

import json

import duckdb

from gridpulse.dc3s.certificate import make_certificate, store_certificate


def _build_certificate(command_id: str) -> dict:
    return make_certificate(
        command_id=command_id,
        device_id="dev-1",
        zone_id="DE",
        controller="deterministic",
        proposed_action={"charge_mw": 1.0, "discharge_mw": 0.0},
        safe_action={"charge_mw": 0.0, "discharge_mw": 0.0},
        uncertainty={"lower": [1.0], "upper": [2.0], "meta": {"inflation": 1.4}},
        reliability={"w_t": 0.52, "flags": {"delay": True}},
        drift={"drift": True},
        model_hash="abc",
        config_hash="def",
        prev_hash=None,
        dispatch_plan=None,
        intervened=True,
        intervention_reason="projection_clip",
        reliability_w=0.52,
        drift_flag=True,
        inflation=1.4,
    )


def test_store_certificate_persists_typed_dc3s_columns(tmp_path) -> None:
    db_path = tmp_path / "dc3s_store.duckdb"
    table_name = "dispatch_certificates"
    cert = _build_certificate("cmd-typed-1")

    store_certificate(cert, duckdb_path=str(db_path), table_name=table_name)

    conn = duckdb.connect(str(db_path))
    try:
        cols = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        }
        assert {"intervened", "intervention_reason", "reliability_w", "drift_flag", "inflation"} <= cols

        row = conn.execute(
            f"""
            SELECT intervened, intervention_reason, reliability_w, drift_flag, inflation, payload_json
            FROM {table_name}
            WHERE command_id = ?
            """,
            [cert["command_id"]],
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert bool(row[0]) is True
    assert row[1] == "projection_clip"
    assert float(row[2]) == 0.52
    assert bool(row[3]) is True
    assert float(row[4]) == 1.4
    payload = json.loads(str(row[5]))
    assert payload["intervened"] is True


def test_store_certificate_migrates_legacy_table(tmp_path) -> None:
    db_path = tmp_path / "dc3s_legacy.duckdb"
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
    finally:
        conn.close()

    cert = _build_certificate("cmd-legacy-1")
    store_certificate(cert, duckdb_path=str(db_path), table_name=table_name)

    conn = duckdb.connect(str(db_path))
    try:
        cols = {
            row[1]
            for row in conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        }
        assert {"intervened", "intervention_reason", "reliability_w", "drift_flag", "inflation"} <= cols

        row = conn.execute(
            f"SELECT intervened, reliability_w, drift_flag, inflation FROM {table_name} WHERE command_id = ?",
            [cert["command_id"]],
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert bool(row[0]) is True
    assert float(row[1]) == 0.52
    assert bool(row[2]) is True
    assert float(row[3]) == 1.4
