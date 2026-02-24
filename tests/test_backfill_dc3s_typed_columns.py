"""Tests for DC3S typed-column backfill utility."""
from __future__ import annotations

import json

import duckdb

from scripts.backfill_dc3s_typed_columns import run_backfill


def test_backfill_populates_missing_typed_columns(tmp_path) -> None:
    db_path = tmp_path / "dc3s.duckdb"
    table = "dispatch_certificates"
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            f"""
            CREATE TABLE {table} (
                command_id VARCHAR PRIMARY KEY,
                payload_json VARCHAR,
                intervened BOOLEAN,
                intervention_reason VARCHAR,
                reliability_w DOUBLE,
                drift_flag BOOLEAN,
                inflation DOUBLE
            )
            """
        )
        payload = {
            "reliability": {"w_t": 0.42},
            "drift": {"drift": True},
            "uncertainty": {"meta": {"inflation": 2.1}, "shield_repair": {"repaired": True}},
        }
        con.execute(
            f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?)",
            ["cmd-1", json.dumps(payload), None, None, None, None, None],
        )
    finally:
        con.close()

    summary = run_backfill(duckdb_path=str(db_path), table_name=table)
    assert summary["rows_updated"] == 1
    assert summary["remaining_nulls"]["intervened"] == 0
    assert summary["remaining_nulls"]["reliability_w"] == 0
    assert summary["remaining_nulls"]["drift_flag"] == 0
    assert summary["remaining_nulls"]["inflation"] == 0
