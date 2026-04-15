"""Dispatch certification and audit persistence for DC3S."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import duckdb


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _to_json_safe(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    return obj


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    safe = _to_json_safe(payload)
    return json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


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
    intervened: bool | None = None,
    intervention_reason: str | None = None,
    reliability_w: float | None = None,
    drift_flag: bool | None = None,
    inflation: float | None = None,
    validity_score: float | None = None,
    adaptive_quantile: float | None = None,
    conditional_coverage_gap: float | None = None,
    runtime_interval_policy: str | None = None,
    coverage_group_key: str | None = None,
    shift_alert_flag: bool | None = None,
    guarantee_checks_passed: bool | None = None,
    guarantee_fail_reasons: list[str] | None = None,
    true_soc_violation_after_apply: bool | None = None,
    assumptions_version: str | None = None,
    gamma_mw: float | None = None,
    e_t_mwh: float | None = None,
    soc_tube_lower_mwh: float | None = None,
    soc_tube_upper_mwh: float | None = None,
    validity_horizon_H_t: int | None = None,
    half_life_steps: int | None = None,
    expires_at_step: int | None = None,
    validity_status: str | None = None,
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
        "intervened": bool(intervened) if intervened is not None else None,
        "intervention_reason": intervention_reason,
        "reliability_w": float(reliability_w) if reliability_w is not None else None,
        "drift_flag": bool(drift_flag) if drift_flag is not None else None,
        "inflation": float(inflation) if inflation is not None else None,
        "validity_score": float(validity_score) if validity_score is not None else None,
        "adaptive_quantile": float(adaptive_quantile) if adaptive_quantile is not None else None,
        "conditional_coverage_gap": float(conditional_coverage_gap) if conditional_coverage_gap is not None else None,
        "runtime_interval_policy": runtime_interval_policy,
        "coverage_group_key": coverage_group_key,
        "shift_alert_flag": bool(shift_alert_flag) if shift_alert_flag is not None else None,
        "guarantee_checks_passed": bool(guarantee_checks_passed) if guarantee_checks_passed is not None else None,
        "guarantee_fail_reasons": list(guarantee_fail_reasons or []),
        "true_soc_violation_after_apply": (
            bool(true_soc_violation_after_apply) if true_soc_violation_after_apply is not None else None
        ),
        "assumptions_version": assumptions_version,
        "gamma_mw": float(gamma_mw) if gamma_mw is not None else None,
        "e_t_mwh": float(e_t_mwh) if e_t_mwh is not None else None,
        "soc_tube_lower_mwh": float(soc_tube_lower_mwh) if soc_tube_lower_mwh is not None else None,
        "soc_tube_upper_mwh": float(soc_tube_upper_mwh) if soc_tube_upper_mwh is not None else None,
        "validity_horizon_H_t": int(validity_horizon_H_t) if validity_horizon_H_t is not None else None,
        "half_life_steps": int(half_life_steps) if half_life_steps is not None else None,
        "expires_at_step": int(expires_at_step) if expires_at_step is not None else None,
        "validity_status": validity_status,
    }
    payload["certificate_hash"] = _sha256_bytes(_canonical_bytes(payload))
    return _to_json_safe(payload)


def _table_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return {str(row[1]) for row in rows}


def _add_column_if_missing(conn: duckdb.DuckDBPyConnection, table_name: str, column_name: str, column_type: str) -> None:
    if column_name in _table_columns(conn, table_name):
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def _ensure_store(conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            command_id VARCHAR PRIMARY KEY,
            certificate_hash VARCHAR,
            prev_hash VARCHAR,
            created_at VARCHAR,
            payload_json VARCHAR,
            intervened BOOLEAN,
            intervention_reason VARCHAR,
            reliability_w DOUBLE,
            drift_flag BOOLEAN,
            inflation DOUBLE,
            validity_score DOUBLE,
            adaptive_quantile DOUBLE,
            conditional_coverage_gap DOUBLE,
            runtime_interval_policy VARCHAR,
            coverage_group_key VARCHAR,
            shift_alert_flag BOOLEAN,
            guarantee_checks_passed BOOLEAN,
            guarantee_fail_reasons VARCHAR,
            true_soc_violation_after_apply BOOLEAN,
            assumptions_version VARCHAR,
            gamma_mw DOUBLE,
            e_t_mwh DOUBLE,
            soc_tube_lower_mwh DOUBLE,
            soc_tube_upper_mwh DOUBLE
        )
        """
    )
    _add_column_if_missing(conn, table_name, "intervened", "BOOLEAN")
    _add_column_if_missing(conn, table_name, "intervention_reason", "VARCHAR")
    _add_column_if_missing(conn, table_name, "reliability_w", "DOUBLE")
    _add_column_if_missing(conn, table_name, "drift_flag", "BOOLEAN")
    _add_column_if_missing(conn, table_name, "inflation", "DOUBLE")
    _add_column_if_missing(conn, table_name, "validity_score", "DOUBLE")
    _add_column_if_missing(conn, table_name, "adaptive_quantile", "DOUBLE")
    _add_column_if_missing(conn, table_name, "conditional_coverage_gap", "DOUBLE")
    _add_column_if_missing(conn, table_name, "runtime_interval_policy", "VARCHAR")
    _add_column_if_missing(conn, table_name, "coverage_group_key", "VARCHAR")
    _add_column_if_missing(conn, table_name, "shift_alert_flag", "BOOLEAN")
    _add_column_if_missing(conn, table_name, "guarantee_checks_passed", "BOOLEAN")
    _add_column_if_missing(conn, table_name, "guarantee_fail_reasons", "VARCHAR")
    _add_column_if_missing(conn, table_name, "true_soc_violation_after_apply", "BOOLEAN")
    _add_column_if_missing(conn, table_name, "assumptions_version", "VARCHAR")
    _add_column_if_missing(conn, table_name, "gamma_mw", "DOUBLE")
    _add_column_if_missing(conn, table_name, "e_t_mwh", "DOUBLE")
    _add_column_if_missing(conn, table_name, "soc_tube_lower_mwh", "DOUBLE")
    _add_column_if_missing(conn, table_name, "soc_tube_upper_mwh", "DOUBLE")


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
                command_id,
                certificate_hash,
                prev_hash,
                created_at,
                payload_json,
                intervened,
                intervention_reason,
                reliability_w,
                drift_flag,
                inflation,
                validity_score,
                adaptive_quantile,
                conditional_coverage_gap,
                runtime_interval_policy,
                coverage_group_key,
                shift_alert_flag,
                guarantee_checks_passed,
                guarantee_fail_reasons,
                true_soc_violation_after_apply,
                assumptions_version,
                gamma_mw,
                e_t_mwh,
                soc_tube_lower_mwh,
                soc_tube_upper_mwh
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(certificate.get("command_id")),
                str(certificate.get("certificate_hash")),
                certificate.get("prev_hash"),
                str(certificate.get("created_at")),
                payload_json,
                certificate.get("intervened"),
                certificate.get("intervention_reason"),
                certificate.get("reliability_w"),
                certificate.get("drift_flag"),
                certificate.get("inflation"),
                certificate.get("validity_score"),
                certificate.get("adaptive_quantile"),
                certificate.get("conditional_coverage_gap"),
                certificate.get("runtime_interval_policy"),
                certificate.get("coverage_group_key"),
                certificate.get("shift_alert_flag"),
                certificate.get("guarantee_checks_passed"),
                json.dumps(certificate.get("guarantee_fail_reasons", []), ensure_ascii=True, sort_keys=True),
                certificate.get("true_soc_violation_after_apply"),
                certificate.get("assumptions_version"),
                certificate.get("gamma_mw"),
                certificate.get("e_t_mwh"),
                certificate.get("soc_tube_lower_mwh"),
                certificate.get("soc_tube_upper_mwh"),
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
