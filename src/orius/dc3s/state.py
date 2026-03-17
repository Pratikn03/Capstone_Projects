"""Persistent online state for DC3S runtime."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import duckdb


class DC3SStateStore:
    def __init__(self, duckdb_path: str, table_name: str = "dc3s_online_state") -> None:
        self.duckdb_path = duckdb_path
        self.table_name = table_name
        db = Path(duckdb_path)
        db.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(db))
        self._init_table()

    def _init_table(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                state_key VARCHAR PRIMARY KEY,
                zone_id VARCHAR,
                device_id VARCHAR,
                target VARCHAR,
                last_timestamp VARCHAR,
                last_yhat DOUBLE,
                last_y_true DOUBLE,
                drift_state_json VARCHAR,
                adaptive_state_json VARCHAR,
                last_prev_hash VARCHAR,
                last_inflation DOUBLE,
                last_event_json VARCHAR,
                last_action_json VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    @staticmethod
    def _key(zone_id: str, device_id: str, target: str) -> str:
        return f"{zone_id}:{device_id}:{target}"

    def get(self, zone_id: str, device_id: str, target: str) -> dict[str, Any] | None:
        key = self._key(zone_id, device_id, target)
        row = self._conn.execute(
            f"""
            SELECT
                last_timestamp,
                last_yhat,
                last_y_true,
                drift_state_json,
                adaptive_state_json,
                last_prev_hash,
                last_inflation,
                last_event_json,
                last_action_json
            FROM {self.table_name}
            WHERE state_key = ?
            """,
            [key],
        ).fetchone()
        if row is None:
            return None

        return {
            "last_timestamp": row[0],
            "last_yhat": row[1],
            "last_y_true": row[2],
            "drift_state": json.loads(row[3]) if row[3] else {},
            "adaptive_state": json.loads(row[4]) if row[4] else {},
            "last_prev_hash": row[5],
            "last_inflation": row[6],
            "last_event": json.loads(row[7]) if row[7] else None,
            "last_action": json.loads(row[8]) if row[8] else None,
        }

    def upsert(
        self,
        *,
        zone_id: str,
        device_id: str,
        target: str,
        last_timestamp: str | None = None,
        last_yhat: float | None = None,
        last_y_true: float | None = None,
        drift_state: Mapping[str, Any] | None = None,
        adaptive_state: Mapping[str, Any] | None = None,
        last_prev_hash: str | None = None,
        last_inflation: float | None = None,
        last_event: Mapping[str, Any] | None = None,
        last_action: Mapping[str, Any] | None = None,
    ) -> None:
        key = self._key(zone_id, device_id, target)
        self._conn.execute(
            f"""
            INSERT OR REPLACE INTO {self.table_name} (
                state_key,
                zone_id,
                device_id,
                target,
                last_timestamp,
                last_yhat,
                last_y_true,
                drift_state_json,
                adaptive_state_json,
                last_prev_hash,
                last_inflation,
                last_event_json,
                last_action_json,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                key,
                zone_id,
                device_id,
                target,
                last_timestamp,
                last_yhat,
                last_y_true,
                json.dumps(dict(drift_state or {}), ensure_ascii=True, sort_keys=True),
                json.dumps(dict(adaptive_state or {}), ensure_ascii=True, sort_keys=True),
                last_prev_hash,
                last_inflation,
                json.dumps(dict(last_event or {}), ensure_ascii=True, sort_keys=True) if last_event else None,
                json.dumps(dict(last_action or {}), ensure_ascii=True, sort_keys=True) if last_action else None,
            ],
        )

    def close(self) -> None:
        self._conn.close()
