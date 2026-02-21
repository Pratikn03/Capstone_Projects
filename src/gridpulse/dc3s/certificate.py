"""Dispatch certification and audit persistence for DC3S."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_model_hash(model_paths: Iterable[str | Path]) -> str:
    """
    Compute a stable model hash from model artifact bytes.
    Missing paths are included by name so the hash still captures configuration.
    """
    hasher = hashlib.sha256()
    for path in sorted({str(Path(p)) for p in model_paths if p is not None}):
        hasher.update(path.encode("utf-8"))
        p = Path(path)
        if p.exists() and p.is_file():
            hasher.update(p.read_bytes())
    return hasher.hexdigest()


def compute_config_hash(config_bytes: bytes) -> str:
    return _sha256_bytes(config_bytes)


def make_certificate(
    *,
    command_id: str,
    device_id: str,
    zone_id: str,
    controller: str,
    proposed_action: Mapping[str, Any],
    safe_action: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
    reliability: Mapping[str, Any],
    drift: Mapping[str, Any],
    model_hash: str,
    config_hash: str,
    prev_hash: str | None = None,
    dispatch_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "command_id": command_id,
        "certificate_id": command_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device_id": device_id,
        "zone_id": zone_id,
        "controller": controller,
        "proposed_action": dict(proposed_action),
        "safe_action": dict(safe_action),
        "uncertainty": dict(uncertainty),
        "reliability": dict(reliability),
        "drift": dict(drift),
        "model_hash": model_hash,
        "config_hash": config_hash,
        "prev_hash": prev_hash,
        "dispatch_plan": dict(dispatch_plan) if dispatch_plan is not None else None,
    }
    payload["certificate_hash"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def _ensure_store(conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            command_id VARCHAR PRIMARY KEY,
            certificate_hash VARCHAR,
            prev_hash VARCHAR,
            created_at VARCHAR,
            payload_json VARCHAR
        )
        """
    )


def store_certificate(certificate: Mapping[str, Any], duckdb_path: str, table_name: str) -> None:
    db_path = Path(duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        _ensure_store(conn, table_name)
        payload_json = json.dumps(certificate, ensure_ascii=True, sort_keys=True)
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {table_name} (
                command_id, certificate_hash, prev_hash, created_at, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                str(certificate.get("command_id")),
                str(certificate.get("certificate_hash")),
                certificate.get("prev_hash"),
                str(certificate.get("created_at")),
                payload_json,
            ],
        )
    finally:
        conn.close()


def get_certificate(command_id: str, duckdb_path: str, table_name: str) -> dict[str, Any] | None:
    db_path = Path(duckdb_path)
    if not db_path.exists():
        return None

    conn = duckdb.connect(str(db_path))
    try:
        _ensure_store(conn, table_name)
        row = conn.execute(
            f"SELECT payload_json FROM {table_name} WHERE command_id = ?",
            [command_id],
        ).fetchone()
        if row is None:
            return None
        return json.loads(str(row[0]))
    finally:
        conn.close()
