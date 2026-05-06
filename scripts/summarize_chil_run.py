#!/usr/bin/env python3
"""Summarize CHIL-style IoT/API runs into a stable JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


def _read_agent_log(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["status"])
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return pd.DataFrame(rows)


def summarize_run(
    *,
    iot_db_path: str,
    audit_db_path: str,
    audit_table: str = "dispatch_certificates",
    device_id: str | None = None,
    agent_log_path: str | None = None,
) -> dict[str, Any]:
    """Build a stable CHIL summary from IoT queue state, ACKs, and certificates."""
    iot_con = duckdb.connect(iot_db_path)
    try:
        telemetry_query = "SELECT * FROM iot_telemetry"
        queue_query = "SELECT * FROM iot_command_queue"
        ack_query = "SELECT * FROM iot_ack"
        params: list[Any] = []
        if device_id:
            telemetry_query += " WHERE device_id = ?"
            queue_query += " WHERE device_id = ?"
            ack_query += " WHERE device_id = ?"
            params = [device_id]
        telemetry = iot_con.execute(telemetry_query, params).df()
        queue = iot_con.execute(queue_query, params).df()
        ack = iot_con.execute(ack_query, params).df()
        state = None
        if device_id:
            state_row = iot_con.execute(
                "SELECT hold_active, hold_reason, hold_since_utc FROM iot_device_state WHERE device_id = ?",
                [device_id],
            ).fetchone()
            if state_row is not None:
                state = {
                    "hold_active": bool(state_row[0]) if state_row[0] is not None else False,
                    "hold_reason": state_row[1],
                    "hold_since_utc": state_row[2],
                }
    finally:
        iot_con.close()

    audit_con = duckdb.connect(audit_db_path)
    try:
        cert_query = f"SELECT * FROM {audit_table}"
        cert_params: list[Any] = []
        if device_id and not queue.empty:
            ids = queue["command_id"].astype(str).tolist()
            placeholders = ",".join(["?"] * len(ids))
            cert_query += f" WHERE command_id IN ({placeholders})"
            cert_params = ids
        certificates = audit_con.execute(cert_query, cert_params).df()
    finally:
        audit_con.close()

    for column in ("queued_at", "dispatched_at", "acked_at", "timeout_at"):
        if column in queue.columns:
            queue[column] = pd.to_datetime(queue[column], errors="coerce", utc=True)

    ack_latency_s: list[float] = []
    if {"dispatched_at", "acked_at"}.issubset(queue.columns):
        valid = queue.dropna(subset=["dispatched_at", "acked_at"]).copy()
        if not valid.empty:
            ack_latency_s = (
                ((valid["acked_at"] - valid["dispatched_at"]).dt.total_seconds()).astype(float).tolist()
            )

    certificate_required = [
        "command_id",
        "certificate_hash",
        "created_at",
        "payload_json",
        "reliability_w",
        "guarantee_checks_passed",
    ]
    completeness = 1.0
    if not certificates.empty:
        present = certificates[certificate_required].notna().all(axis=1)
        completeness = float(present.mean())

    bucket_rows: list[dict[str, Any]] = []
    if not certificates.empty and {"reliability_w", "intervened"}.issubset(certificates.columns):
        cert = certificates.dropna(subset=["reliability_w"]).copy()
        if not cert.empty:
            if cert["reliability_w"].nunique() == 1:
                cert["bucket"] = 0
                [float(cert["reliability_w"].iloc[0]), float(cert["reliability_w"].iloc[0])]
            else:
                cert["bucket"] = pd.qcut(
                    cert["reliability_w"],
                    q=min(5, cert["reliability_w"].nunique()),
                    labels=False,
                    duplicates="drop",
                )
            grouped = cert.groupby("bucket", dropna=False)
            for bucket, frame in grouped:
                bucket_rows.append(
                    {
                        "bucket": int(bucket),
                        "n": int(len(frame)),
                        "mean_reliability_w": float(frame["reliability_w"].mean()),
                        "intervention_rate": float(frame["intervened"].astype(bool).mean()),
                    }
                )

    agent_log = _read_agent_log(Path(agent_log_path) if agent_log_path else None)
    status_counts = (
        agent_log.get("status", pd.Series(dtype=str)).value_counts().to_dict() if not agent_log.empty else {}
    )

    def _dist(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"count": 0, "p50": None, "p95": None, "mean": None}
        series = pd.Series(values, dtype=float)
        return {
            "count": int(series.shape[0]),
            "p50": float(series.quantile(0.50)),
            "p95": float(series.quantile(0.95)),
            "mean": float(series.mean()),
        }

    return {
        "device_id": device_id,
        "telemetry_events": int(len(telemetry)),
        "queued_commands": int(len(queue)),
        "acks": int(len(ack)),
        "ack_status_counts": {
            str(k): int(v)
            for k, v in ack.get("status", pd.Series(dtype=str)).value_counts().to_dict().items()
        },
        "queue_status_counts": {
            str(k): int(v)
            for k, v in queue.get("status", pd.Series(dtype=str)).value_counts().to_dict().items()
        },
        "queue_expiry_events": int((queue.get("status", pd.Series(dtype=str)) == "timeout").sum())
        if not queue.empty
        else 0,
        "hold_events": int(status_counts.get("hold", 0))
        + int((queue.get("status", pd.Series(dtype=str)) == "timeout").sum())
        if not queue.empty
        else int(status_counts.get("hold", 0)),
        "empty_command_events": int(status_counts.get("empty", 0)),
        "error_events": int(status_counts.get("error", 0)),
        "ack_latency_seconds": _dist(ack_latency_s),
        "certificate_completeness": float(completeness),
        "intervention_rate_by_reliability_bucket": bucket_rows,
        "hold_state": state,
        "agent_log_status_counts": {str(k): int(v) for k, v in status_counts.items()},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CHIL IoT/API outputs into a stable JSON report")
    parser.add_argument("--iot-db", default="data/audit/iot_loop.duckdb")
    parser.add_argument("--audit-db", default="data/audit/dc3s_audit.duckdb")
    parser.add_argument("--audit-table", default="dispatch_certificates")
    parser.add_argument("--device-id", default=None)
    parser.add_argument("--agent-log", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = summarize_run(
        iot_db_path=str(args.iot_db),
        audit_db_path=str(args.audit_db),
        audit_table=str(args.audit_table),
        device_id=args.device_id,
        agent_log_path=args.agent_log,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
