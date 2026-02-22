"""DuckDB-backed state and queue persistence for IoT closed-loop simulation."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import duckdb

_UNSET = object()


def get_iot_duckdb_path() -> str:
    """Resolve IoT loop store path with env override support."""
    return os.environ.get("GRIDPULSE_IOT_DUCKDB_PATH", "data/audit/iot_loop.duckdb")


def _json_dump(payload: Mapping[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(dict(payload), ensure_ascii=True, sort_keys=True)


def _json_load(payload: str | None) -> dict[str, Any] | None:
    if not payload:
        return None
    return json.loads(payload)


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IoTLoopStore:
    """Thin persistence layer for telemetry, command queue, ACKs, and latest device state."""

    def __init__(self, duckdb_path: str | None = None) -> None:
        self.duckdb_path = duckdb_path or get_iot_duckdb_path()
        db_path = Path(self.duckdb_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(db_path))
        self._init_tables()

    def close(self) -> None:
        self._conn.close()

    def _init_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS iot_telemetry (
                device_id VARCHAR,
                ts_utc VARCHAR,
                payload_json VARCHAR,
                reliability_w DOUBLE,
                reliability_flags_json VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS iot_command_queue (
                command_id VARCHAR PRIMARY KEY,
                device_id VARCHAR,
                zone_id VARCHAR,
                status VARCHAR,
                certificate_id VARCHAR,
                command_json VARCHAR,
                queued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dispatched_at TIMESTAMP,
                acked_at TIMESTAMP,
                expires_at TIMESTAMP,
                timeout_at TIMESTAMP,
                timeout_reason VARCHAR
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS iot_ack (
                ack_id VARCHAR PRIMARY KEY,
                device_id VARCHAR,
                command_id VARCHAR,
                certificate_id VARCHAR,
                status VARCHAR,
                reason VARCHAR,
                payload_json VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS iot_device_state (
                device_id VARCHAR PRIMARY KEY,
                latest_ts_utc VARCHAR,
                latest_telemetry_json VARCHAR,
                latest_reliability_w DOUBLE,
                latest_reliability_flags_json VARCHAR,
                last_command_id VARCHAR,
                last_command_json VARCHAR,
                last_ack_json VARCHAR,
                hold_active BOOLEAN DEFAULT FALSE,
                hold_reason VARCHAR,
                hold_since_utc VARCHAR,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Backward-compatible migrations for existing local DB files.
        self._conn.execute("ALTER TABLE iot_command_queue ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP")
        self._conn.execute("ALTER TABLE iot_command_queue ADD COLUMN IF NOT EXISTS timeout_at TIMESTAMP")
        self._conn.execute("ALTER TABLE iot_command_queue ADD COLUMN IF NOT EXISTS timeout_reason VARCHAR")
        self._conn.execute("ALTER TABLE iot_device_state ADD COLUMN IF NOT EXISTS hold_active BOOLEAN DEFAULT FALSE")
        self._conn.execute("ALTER TABLE iot_device_state ADD COLUMN IF NOT EXISTS hold_reason VARCHAR")
        self._conn.execute("ALTER TABLE iot_device_state ADD COLUMN IF NOT EXISTS hold_since_utc VARCHAR")

    def _state_upsert(
        self,
        *,
        device_id: str,
        latest_ts_utc: str | None = None,
        latest_telemetry: Mapping[str, Any] | None = None,
        latest_reliability_w: float | None = None,
        latest_reliability_flags: Mapping[str, Any] | None = None,
        last_command_id: str | None = None,
        last_command: Mapping[str, Any] | None = None,
        last_ack: Mapping[str, Any] | None = None,
        hold_active: bool | object = _UNSET,
        hold_reason: str | None | object = _UNSET,
        hold_since_utc: str | None | object = _UNSET,
    ) -> None:
        prior = self.get_state(device_id) or {}
        merged = {
            "latest_ts_utc": latest_ts_utc if latest_ts_utc is not None else prior.get("latest_ts_utc"),
            "latest_telemetry_json": _json_dump(latest_telemetry)
            if latest_telemetry is not None
            else _json_dump(prior.get("latest_telemetry")),
            "latest_reliability_w": float(latest_reliability_w)
            if latest_reliability_w is not None
            else prior.get("latest_reliability_w"),
            "latest_reliability_flags_json": _json_dump(latest_reliability_flags)
            if latest_reliability_flags is not None
            else _json_dump(prior.get("latest_reliability_flags")),
            "last_command_id": last_command_id if last_command_id is not None else prior.get("last_command_id"),
            "last_command_json": _json_dump(last_command)
            if last_command is not None
            else _json_dump(prior.get("last_command")),
            "last_ack_json": _json_dump(last_ack) if last_ack is not None else _json_dump(prior.get("last_ack")),
            "hold_active": bool(hold_active) if hold_active is not _UNSET else bool(prior.get("hold_active", False)),
            "hold_reason": hold_reason if hold_reason is not _UNSET else prior.get("hold_reason"),
            "hold_since_utc": hold_since_utc if hold_since_utc is not _UNSET else prior.get("hold_since_utc"),
        }
        self._conn.execute(
            """
            INSERT OR REPLACE INTO iot_device_state (
                device_id,
                latest_ts_utc,
                latest_telemetry_json,
                latest_reliability_w,
                latest_reliability_flags_json,
                last_command_id,
                last_command_json,
                last_ack_json,
                hold_active,
                hold_reason,
                hold_since_utc,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                device_id,
                merged["latest_ts_utc"],
                merged["latest_telemetry_json"],
                merged["latest_reliability_w"],
                merged["latest_reliability_flags_json"],
                merged["last_command_id"],
                merged["last_command_json"],
                merged["last_ack_json"],
                merged["hold_active"],
                merged["hold_reason"],
                merged["hold_since_utc"],
            ],
        )

    def get_last_telemetry(self, device_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT ts_utc, payload_json
            FROM iot_telemetry
            WHERE device_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [device_id],
        ).fetchone()
        if row is None:
            return None
        payload = _json_load(row[1]) or {}
        if row[0]:
            payload.setdefault("ts_utc", str(row[0]))
        return payload

    def record_telemetry(
        self,
        *,
        device_id: str,
        ts_utc: str,
        payload: Mapping[str, Any],
        reliability_w: float,
        reliability_flags: Mapping[str, Any],
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO iot_telemetry (
                device_id,
                ts_utc,
                payload_json,
                reliability_w,
                reliability_flags_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                device_id,
                ts_utc,
                _json_dump(payload),
                float(reliability_w),
                _json_dump(reliability_flags),
            ],
        )
        self._state_upsert(
            device_id=device_id,
            latest_ts_utc=ts_utc,
            latest_telemetry=payload,
            latest_reliability_w=float(reliability_w),
            latest_reliability_flags=reliability_flags,
        )

    def enqueue_command(
        self,
        *,
        device_id: str,
        command_id: str,
        command: Mapping[str, Any],
        zone_id: str | None = None,
        certificate_id: str | None = None,
        ttl_seconds: int = 30,
    ) -> None:
        ttl = max(0, int(ttl_seconds))
        self._conn.execute(
            """
            INSERT OR REPLACE INTO iot_command_queue (
                command_id,
                device_id,
                zone_id,
                status,
                certificate_id,
                command_json,
                queued_at,
                dispatched_at,
                acked_at,
                expires_at,
                timeout_at,
                timeout_reason
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, NULL, NULL, CURRENT_TIMESTAMP + (? * INTERVAL '1 second'), NULL, NULL)
            """,
            [
                command_id,
                device_id,
                zone_id,
                "queued",
                certificate_id,
                _json_dump(command),
                ttl,
            ],
        )
        self._state_upsert(
            device_id=device_id,
            last_command_id=command_id,
            last_command={
                "command_id": command_id,
                "device_id": device_id,
                "zone_id": zone_id,
                "status": "queued",
                "certificate_id": certificate_id,
                "command": dict(command),
            },
        )

    def expire_stale_commands(
        self,
        *,
        device_id: str,
        reason: str = "ack_timeout",
    ) -> int:
        stale_count = int(
            self._conn.execute(
                """
                SELECT COUNT(*)
                FROM iot_command_queue
                WHERE device_id = ?
                  AND status IN ('queued', 'dispatched')
                  AND expires_at IS NOT NULL
                  AND expires_at <= CURRENT_TIMESTAMP
                """,
                [device_id],
            ).fetchone()[0]
        )
        if stale_count <= 0:
            return 0

        self._conn.execute(
            """
            UPDATE iot_command_queue
            SET status = 'timeout', timeout_at = CURRENT_TIMESTAMP, timeout_reason = ?
            WHERE device_id = ?
              AND status IN ('queued', 'dispatched')
              AND expires_at IS NOT NULL
              AND expires_at <= CURRENT_TIMESTAMP
            """,
            [reason, device_id],
        )
        self._state_upsert(
            device_id=device_id,
            hold_active=True,
            hold_reason=reason,
            hold_since_utc=_utc_iso_now(),
        )
        return stale_count

    def reset_hold(self, *, device_id: str, reason: str | None = None) -> None:
        self._state_upsert(
            device_id=device_id,
            hold_active=False,
            hold_reason=reason,
            hold_since_utc=None,
        )

    def get_next_command(self, *, device_id: str, peek: bool = False) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT command_id, zone_id, status, certificate_id, command_json
            FROM iot_command_queue
            WHERE device_id = ?
              AND status = 'queued'
              AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            ORDER BY queued_at ASC, command_id ASC
            LIMIT 1
            """,
            [device_id],
        ).fetchone()
        if row is None:
            return None

        command = {
            "command_id": str(row[0]),
            "device_id": device_id,
            "zone_id": row[1],
            "status": str(row[2]),
            "certificate_id": row[3],
            "command": _json_load(row[4]) or {},
        }
        if not peek:
            self._conn.execute(
                """
                UPDATE iot_command_queue
                SET status = 'dispatched', dispatched_at = CURRENT_TIMESTAMP
                WHERE command_id = ?
                """,
                [command["command_id"]],
            )
            command["status"] = "dispatched"
            self._state_upsert(
                device_id=device_id,
                last_command_id=command["command_id"],
                last_command=command,
            )
        return command

    def record_ack(
        self,
        *,
        device_id: str,
        command_id: str,
        status: str,
        certificate_id: str | None = None,
        reason: str | None = None,
        payload: Mapping[str, Any] | None = None,
        ack_id: str | None = None,
    ) -> dict[str, Any]:
        ack = {
            "ack_id": ack_id or str(uuid4()),
            "device_id": device_id,
            "command_id": command_id,
            "certificate_id": certificate_id,
            "status": status,
            "reason": reason,
            "payload": dict(payload or {}),
        }
        self._conn.execute(
            """
            INSERT OR REPLACE INTO iot_ack (
                ack_id,
                device_id,
                command_id,
                certificate_id,
                status,
                reason,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ack["ack_id"],
                device_id,
                command_id,
                certificate_id,
                status,
                reason,
                _json_dump(payload or {}),
            ],
        )
        self._conn.execute(
            """
            UPDATE iot_command_queue
            SET status = ?, acked_at = CURRENT_TIMESTAMP
            WHERE command_id = ?
            """,
            [status, command_id],
        )
        self._state_upsert(device_id=device_id, last_ack=ack)
        return ack

    def get_state(self, device_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """
            SELECT
                latest_ts_utc,
                latest_telemetry_json,
                latest_reliability_w,
                latest_reliability_flags_json,
                last_command_id,
                last_command_json,
                last_ack_json,
                hold_active,
                hold_reason,
                hold_since_utc
            FROM iot_device_state
            WHERE device_id = ?
            """,
            [device_id],
        ).fetchone()
        if row is None:
            return None
        return {
            "device_id": device_id,
            "latest_ts_utc": row[0],
            "latest_telemetry": _json_load(row[1]),
            "latest_reliability_w": row[2],
            "latest_reliability_flags": _json_load(row[3]),
            "last_command_id": row[4],
            "last_command": _json_load(row[5]),
            "last_ack": _json_load(row[6]),
            "hold_active": bool(row[7]) if row[7] is not None else False,
            "hold_reason": row[8],
            "hold_since_utc": row[9],
        }
