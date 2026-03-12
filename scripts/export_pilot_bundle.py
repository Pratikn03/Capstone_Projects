#!/usr/bin/env python3
"""Export a future-pilot evidence bundle from existing IoT and certificate stores."""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [json.dumps(_json_safe(row), sort_keys=True, ensure_ascii=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def export_bundle(
    *,
    out_dir: str,
    iot_db_path: str,
    audit_db_path: str,
    audit_table: str = "dispatch_certificates",
    device_id: str | None = None,
) -> Path:
    """Export telemetry, queue, ACK, and certificate artifacts into one bundle directory."""
    bundle_root = Path(out_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_dir = bundle_root / f"pilot_bundle_{device_id or 'all'}_{timestamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    iot_con = duckdb.connect(iot_db_path)
    try:
        params: list[Any] = []
        telemetry_query = "SELECT * FROM iot_telemetry"
        queue_query = "SELECT * FROM iot_command_queue"
        ack_query = "SELECT * FROM iot_ack"
        if device_id:
            telemetry_query += " WHERE device_id = ?"
            queue_query += " WHERE device_id = ?"
            ack_query += " WHERE device_id = ?"
            params = [device_id]
        telemetry = iot_con.execute(telemetry_query, params).df().to_dict(orient="records")
        queue = iot_con.execute(queue_query, params).df().to_dict(orient="records")
        ack = iot_con.execute(ack_query, params).df().to_dict(orient="records")
        state = None
        if device_id:
            state_row = iot_con.execute(
                "SELECT * FROM iot_device_state WHERE device_id = ?",
                [device_id],
            ).df()
            state = state_row.to_dict(orient="records")
    finally:
        iot_con.close()

    command_ids = [str(row["command_id"]) for row in queue if row.get("command_id")]
    audit_con = duckdb.connect(audit_db_path)
    try:
        if command_ids:
            placeholders = ",".join(["?"] * len(command_ids))
            cert_query = f"SELECT * FROM {audit_table} WHERE command_id IN ({placeholders})"
            certificates = audit_con.execute(cert_query, command_ids).df().to_dict(orient="records")
        else:
            certificates = []
    finally:
        audit_con.close()

    _write_ndjson(bundle_dir / "telemetry.ndjson", telemetry)
    _write_ndjson(bundle_dir / "command_queue.ndjson", queue)
    _write_ndjson(bundle_dir / "acks.ndjson", ack)
    _write_ndjson(bundle_dir / "certificates.ndjson", certificates)
    if state is not None:
        _write_ndjson(bundle_dir / "device_state.ndjson", state)

    manifest = {
        "bundle_version": 1,
        "bundle_name": bundle_dir.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_id": device_id,
        "source": {
            "iot_db_path": iot_db_path,
            "audit_db_path": audit_db_path,
            "audit_table": audit_table,
        },
        "files": {
            "telemetry": "telemetry.ndjson",
            "command_queue": "command_queue.ndjson",
            "acks": "acks.ndjson",
            "certificates": "certificates.ndjson",
            "device_state": "device_state.ndjson" if state is not None else None,
        },
        "counts": {
            "telemetry": len(telemetry),
            "command_queue": len(queue),
            "acks": len(ack),
            "certificates": len(certificates),
            "device_state": 0 if state is None else len(state),
        },
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return bundle_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a future-pilot evidence bundle")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iot-db", default="data/audit/iot_loop.duckdb")
    parser.add_argument("--audit-db", default="data/audit/dc3s_audit.duckdb")
    parser.add_argument("--audit-table", default="dispatch_certificates")
    parser.add_argument("--device-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    bundle_dir = export_bundle(
        out_dir=str(args.out_dir),
        iot_db_path=str(args.iot_db),
        audit_db_path=str(args.audit_db),
        audit_table=str(args.audit_table),
        device_id=args.device_id,
    )
    print(f"Wrote {bundle_dir}")


if __name__ == "__main__":
    main()
