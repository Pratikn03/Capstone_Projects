"""Backfill typed DC3S certificate columns from payload_json."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import duckdb
import yaml


def _load_dc3s_audit_cfg(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "data/audit/dc3s_audit.duckdb", "dispatch_certificates"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    dc3s = payload.get("dc3s", {}) if isinstance(payload, dict) else {}
    audit = dc3s.get("audit", {}) if isinstance(dc3s, dict) else {}
    return (
        str(audit.get("duckdb_path", "data/audit/dc3s_audit.duckdb")),
        str(audit.get("table_name", "dispatch_certificates")),
    )


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y"}:
            return True
        if low in {"0", "false", "no", "n"}:
            return False
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_payload(payload_json: Any) -> dict[str, Any]:
    if not isinstance(payload_json, str) or not payload_json:
        return {}
    try:
        obj = json.loads(payload_json)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _derive_fields(payload: dict[str, Any]) -> dict[str, Any]:
    uncertainty = payload.get("uncertainty", {}) if isinstance(payload.get("uncertainty"), dict) else {}
    meta = uncertainty.get("meta", {}) if isinstance(uncertainty.get("meta"), dict) else {}
    shield = uncertainty.get("shield_repair", {}) if isinstance(uncertainty.get("shield_repair"), dict) else {}
    reliability = payload.get("reliability", {}) if isinstance(payload.get("reliability"), dict) else {}
    drift = payload.get("drift", {}) if isinstance(payload.get("drift"), dict) else {}

    intervened = _safe_bool(payload.get("intervened"))
    if intervened is None:
        intervened = _safe_bool(shield.get("repaired"))

    reason = payload.get("intervention_reason")
    if not isinstance(reason, str) or not reason:
        robust_meta = shield.get("robust_meta")
        if isinstance(robust_meta, dict) and isinstance(robust_meta.get("reason"), str):
            reason = robust_meta.get("reason")
        elif intervened:
            reason = "projection_clip"
        else:
            reason = None

    reliability_w = _safe_float(payload.get("reliability_w"))
    if reliability_w is None:
        reliability_w = _safe_float(reliability.get("w_t"))

    drift_flag = _safe_bool(payload.get("drift_flag"))
    if drift_flag is None:
        drift_flag = _safe_bool(drift.get("drift"))

    inflation = _safe_float(payload.get("inflation"))
    if inflation is None:
        inflation = _safe_float(meta.get("inflation"))

    return {
        "intervened": intervened,
        "intervention_reason": reason if isinstance(reason, str) else None,
        "reliability_w": reliability_w,
        "drift_flag": drift_flag,
        "inflation": inflation,
    }


def _ensure_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS intervened BOOLEAN")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS intervention_reason VARCHAR")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS reliability_w DOUBLE")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS drift_flag BOOLEAN")
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS inflation DOUBLE")


def run_backfill(*, duckdb_path: str, table_name: str) -> dict[str, Any]:
    db = Path(duckdb_path)
    if not db.exists():
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duckdb_path": duckdb_path,
            "table_name": table_name,
            "exists": False,
            "rows_total": 0,
            "rows_scanned": 0,
            "rows_updated": 0,
        }

    conn = duckdb.connect(str(db))
    try:
        _ensure_columns(conn, table_name)
        rows_total = int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        rows = conn.execute(
            f"""
            SELECT command_id, payload_json, intervened, intervention_reason, reliability_w, drift_flag, inflation
            FROM {table_name}
            WHERE intervened IS NULL
               OR intervention_reason IS NULL
               OR reliability_w IS NULL
               OR drift_flag IS NULL
               OR inflation IS NULL
            """
        ).fetchall()
        updated = 0
        for row in rows:
            command_id, payload_json, intervened, intervention_reason, reliability_w, drift_flag, inflation = row
            payload = _load_payload(payload_json)
            derived = _derive_fields(payload)
            next_intervened = intervened if intervened is not None else derived["intervened"]
            next_reason = intervention_reason if intervention_reason is not None else derived["intervention_reason"]
            next_reliability_w = reliability_w if reliability_w is not None else derived["reliability_w"]
            next_drift_flag = drift_flag if drift_flag is not None else derived["drift_flag"]
            next_inflation = inflation if inflation is not None else derived["inflation"]

            if (
                next_intervened != intervened
                or next_reason != intervention_reason
                or next_reliability_w != reliability_w
                or next_drift_flag != drift_flag
                or next_inflation != inflation
            ):
                conn.execute(
                    f"""
                    UPDATE {table_name}
                    SET intervened = ?,
                        intervention_reason = ?,
                        reliability_w = ?,
                        drift_flag = ?,
                        inflation = ?
                    WHERE command_id = ?
                    """,
                    [
                        next_intervened,
                        next_reason,
                        next_reliability_w,
                        next_drift_flag,
                        next_inflation,
                        command_id,
                    ],
                )
                updated += 1

        nulls = conn.execute(
            f"""
            SELECT
                SUM(CASE WHEN intervened IS NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN intervention_reason IS NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN reliability_w IS NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN drift_flag IS NULL THEN 1 ELSE 0 END),
                SUM(CASE WHEN inflation IS NULL THEN 1 ELSE 0 END)
            FROM {table_name}
            """
        ).fetchone()
    finally:
        conn.close()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duckdb_path": duckdb_path,
        "table_name": table_name,
        "exists": True,
        "rows_total": rows_total,
        "rows_scanned": len(rows),
        "rows_updated": updated,
        "remaining_nulls": {
            "intervened": int(nulls[0] or 0),
            "intervention_reason": int(nulls[1] or 0),
            "reliability_w": int(nulls[2] or 0),
            "drift_flag": int(nulls[3] or 0),
            "inflation": int(nulls[4] or 0),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill typed DC3S certificate columns")
    parser.add_argument("--dc3s-config", default="configs/dc3s.yaml")
    parser.add_argument("--duckdb-path", default=None)
    parser.add_argument("--table-name", default=None)
    parser.add_argument("--out-json", default="reports/publish/dc3s_backfill_summary.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg_db, cfg_table = _load_dc3s_audit_cfg(Path(args.dc3s_config))
    summary = run_backfill(
        duckdb_path=str(args.duckdb_path or cfg_db),
        table_name=str(args.table_name or cfg_table),
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
