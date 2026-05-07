"""Dispatch certification and audit persistence for DC3S."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import duckdb
import numpy as np
import pandas as pd

from orius.security.policy import (
    get_active_certificate_key_id,
    get_certificate_key,
)
from orius.utils.sql import validate_column_type, validate_sql_identifier

CERTIFICATE_SCHEMA_VERSION = "orius.certificate.v1"
DEFAULT_CERTIFICATE_ISSUER = "orius.runtime"
CERTIFICATE_SIGNATURE_ALGORITHM = "HMAC-SHA256"
DEFAULT_CERTIFICATE_KEY_ID = "orius.local.hmac"
_SIGNATURE_FIELDS = {"signature", "signature_algorithm", "public_key_id"}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _to_json_safe(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating | np.integer):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_json_safe(x) for x in obj]
    return obj


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    safe = _to_json_safe(payload)
    return json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _certificate_hash_payload(certificate: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(_to_json_safe(dict(certificate)))
    payload.pop("certificate_hash", None)
    payload.pop("cert_hash", None)
    # The artifact identity must be stable before and after signing. Existing
    # unsigned certificates already hash these fields as null, so keep that
    # convention while making signature material tamper-evident separately.
    for field in _SIGNATURE_FIELDS:
        payload[field] = None
    return payload


def _certificate_signature_payload(certificate: Mapping[str, Any]) -> bytes:
    payload = dict(_to_json_safe(dict(certificate)))
    payload.pop("signature", None)
    payload.pop("cert_hash", None)
    return _canonical_bytes(payload)


def _resolve_signing_secret(secret: str | bytes | None = None, *, key_id: str | None = None) -> bytes:
    if isinstance(secret, bytes):
        key = secret
    else:
        configured_secret = get_certificate_key(key_id) if secret is None else None
        key_text = (
            secret
            if secret is not None
            else configured_secret or os.getenv("ORIUS_CERTIFICATE_SIGNING_KEY", "")
        )
        key = str(key_text).encode("utf-8")
    if not key:
        raise RuntimeError("certificate signing key is required")
    return key


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


def normalize_certificate_schema(
    certificate: Mapping[str, Any],
    *,
    issuer: str | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """Return a canonical certificate view while accepting legacy fields.

    Legacy CertOS surfaces used ``cert_hash``. The canonical public schema uses
    ``certificate_hash``. This helper preserves legacy data, promotes
    ``cert_hash`` to ``certificate_hash`` when needed, and fills non-security
    metadata fields used by runtime-table validators.
    """

    cert = dict(_to_json_safe(dict(certificate)))
    legacy_hash = cert.get("cert_hash")
    canonical_hash = cert.get("certificate_hash")
    if canonical_hash in (None, "") and legacy_hash not in (None, ""):
        cert["certificate_hash"] = str(legacy_hash)

    cert.setdefault("certificate_schema_version", CERTIFICATE_SCHEMA_VERSION)
    cert.setdefault("issuer", issuer or DEFAULT_CERTIFICATE_ISSUER)

    if domain not in (None, ""):
        cert.setdefault("domain", str(domain))
    else:
        cert.setdefault("domain", str(cert.get("domain") or cert.get("zone_id") or "unknown"))

    if "action" not in cert or cert.get("action") in (None, ""):
        action = cert.get("safe_action") if isinstance(cert.get("safe_action"), Mapping) else {}
        cert["action"] = dict(action)

    if cert.get("validity_horizon_H_t") in (None, ""):
        for key in ("certificate_horizon_steps", "validity_horizon", "tau_t"):
            if cert.get(key) not in (None, ""):
                try:
                    cert["validity_horizon_H_t"] = int(cert[key])
                    break
                except (TypeError, ValueError):
                    continue

    if cert.get("expires_at_step") in (None, "") and cert.get("validity_horizon_H_t") not in (None, ""):
        with contextlib.suppress(TypeError, ValueError):
            cert["expires_at_step"] = int(cert.get("validity_horizon_H_t"))

    cert.setdefault("theorem_contracts", {})
    cert.setdefault("signature", None)
    cert.setdefault("signature_algorithm", None)
    cert.setdefault("public_key_id", None)
    return cert


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
    runtime_surface: str | None = None,
    closure_tier: str | None = None,
    reliability_feature_basis: Mapping[str, Any] | None = None,
    certificate_schema_version: str = CERTIFICATE_SCHEMA_VERSION,
    issuer: str = DEFAULT_CERTIFICATE_ISSUER,
    domain: str | None = None,
    action: Mapping[str, Any] | None = None,
    theorem_contracts: Mapping[str, Any] | None = None,
    signature: str | None = None,
    signature_algorithm: str | None = None,
    public_key_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "certificate_schema_version": certificate_schema_version,
        "issuer": issuer,
        "domain": domain or zone_id,
        "action": dict(action) if action is not None else dict(safe_action),
        "theorem_contracts": dict(theorem_contracts or {}),
        "signature": signature,
        "signature_algorithm": signature_algorithm,
        "public_key_id": public_key_id,
        "command_id": command_id,
        "certificate_id": command_id,
        "created_at": datetime.now(UTC).isoformat(),
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
        "conditional_coverage_gap": float(conditional_coverage_gap)
        if conditional_coverage_gap is not None
        else None,
        "runtime_interval_policy": runtime_interval_policy,
        "coverage_group_key": coverage_group_key,
        "shift_alert_flag": bool(shift_alert_flag) if shift_alert_flag is not None else None,
        "guarantee_checks_passed": bool(guarantee_checks_passed)
        if guarantee_checks_passed is not None
        else None,
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
        "runtime_surface": None if runtime_surface in (None, "") else str(runtime_surface),
        "closure_tier": None if closure_tier in (None, "") else str(closure_tier),
        "reliability_feature_basis": dict(reliability_feature_basis or {}),
    }
    payload["certificate_hash"] = _sha256_bytes(_canonical_bytes(_certificate_hash_payload(payload)))
    return _to_json_safe(payload)


def recompute_certificate_hash(certificate: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes(_certificate_hash_payload(certificate)))


def sign_certificate(
    certificate: Mapping[str, Any],
    *,
    secret: str | bytes | None = None,
    key_id: str | None = None,
) -> dict[str, Any]:
    """Attach a tamper-evident HMAC signature to a valid certificate.

    This is deliberately symmetric-key signing because it is deployable in
    local HIL/predeployment environments without a PKI. Production systems can
    rotate the secret and key id through deployment secret management.
    """

    cert = normalize_certificate_schema(certificate)
    verification = verify_certificate(cert)
    if not verification["valid"]:
        raise RuntimeError(f"cannot sign invalid certificate: {verification['reason']}")

    resolved_key_id = key_id or get_active_certificate_key_id()
    cert["signature_algorithm"] = CERTIFICATE_SIGNATURE_ALGORITHM
    cert["public_key_id"] = resolved_key_id
    cert["signature"] = hmac.new(
        _resolve_signing_secret(secret, key_id=resolved_key_id),
        _certificate_signature_payload(cert),
        hashlib.sha256,
    ).hexdigest()
    return cert


def verify_certificate_signature(
    certificate: Mapping[str, Any],
    *,
    secret: str | bytes | None = None,
) -> dict[str, Any]:
    cert = normalize_certificate_schema(certificate)
    observed = cert.get("signature")
    if observed in (None, ""):
        return {
            "valid": False,
            "reason": "signature_missing",
            "expected_signature": None,
            "observed_signature": observed,
        }
    if cert.get("signature_algorithm") != CERTIFICATE_SIGNATURE_ALGORITHM:
        return {
            "valid": False,
            "reason": "unsupported_signature_algorithm",
            "expected_signature": None,
            "observed_signature": observed,
        }
    try:
        secret_bytes = _resolve_signing_secret(secret, key_id=str(cert.get("public_key_id") or ""))
    except RuntimeError:
        return {
            "valid": False,
            "reason": "signing_key_missing",
            "expected_signature": None,
            "observed_signature": observed,
        }
    expected = hmac.new(secret_bytes, _certificate_signature_payload(cert), hashlib.sha256).hexdigest()
    valid = hmac.compare_digest(str(observed), expected)
    return {
        "valid": bool(valid),
        "reason": None if valid else "signature_mismatch",
        "expected_signature": expected,
        "observed_signature": observed,
    }


def verify_certificate(
    certificate: Mapping[str, Any],
    *,
    require_signature: bool = False,
    signature_secret: str | bytes | None = None,
) -> dict[str, Any]:
    normalized = normalize_certificate_schema(certificate)
    observed_hash = normalized.get("certificate_hash")
    expected_hash = recompute_certificate_hash(certificate)
    if observed_hash != expected_hash:
        expected_hash = recompute_certificate_hash(normalized)
    valid = isinstance(observed_hash, str) and observed_hash == expected_hash
    signature_verification = None
    has_signature = normalized.get("signature") not in (None, "")
    if valid and (has_signature or require_signature):
        signature_verification = verify_certificate_signature(normalized, secret=signature_secret)
        valid = bool(signature_verification["valid"])
    return {
        "valid": bool(valid),
        "observed_hash": observed_hash,
        "expected_hash": expected_hash,
        "signature": signature_verification,
        "reason": None
        if valid
        else (
            str(signature_verification["reason"]) if signature_verification is not None else "hash_mismatch"
        ),
    }


def verify_certificate_chain(
    certificates: Iterable[Mapping[str, Any]],
    *,
    require_signature: bool = False,
    signature_secret: str | bytes | None = None,
) -> dict[str, Any]:
    checked = 0
    previous_hash: str | None = None
    for index, certificate in enumerate(certificates):
        normalized = normalize_certificate_schema(certificate)
        verification = verify_certificate(
            certificate,
            require_signature=require_signature,
            signature_secret=signature_secret,
        )
        if not verification["valid"]:
            return {
                "valid": False,
                "checked": checked,
                "failed_index": index,
                "reason": str(verification["reason"]),
                "expected_prev_hash": previous_hash,
                "observed_prev_hash": normalized.get("prev_hash"),
            }
        current_prev_hash = normalized.get("prev_hash")
        if index == 0:
            if current_prev_hash not in (None, ""):
                return {
                    "valid": False,
                    "checked": checked,
                    "failed_index": index,
                    "reason": "genesis_prev_hash_present",
                    "expected_prev_hash": "",
                    "observed_prev_hash": current_prev_hash,
                }
        elif current_prev_hash in (None, ""):
            return {
                "valid": False,
                "checked": checked,
                "failed_index": index,
                "reason": "prev_hash_missing",
                "expected_prev_hash": previous_hash,
                "observed_prev_hash": current_prev_hash,
            }
        elif current_prev_hash != previous_hash:
            return {
                "valid": False,
                "checked": checked,
                "failed_index": index,
                "reason": "prev_hash_mismatch",
                "expected_prev_hash": previous_hash,
                "observed_prev_hash": current_prev_hash,
            }
        previous_hash = str(normalized.get("certificate_hash"))
        checked += 1
    return {
        "valid": True,
        "checked": checked,
        "failed_index": None,
        "reason": None,
        "expected_prev_hash": previous_hash,
        "observed_prev_hash": previous_hash,
    }


def _table_columns(conn: duckdb.DuckDBPyConnection, table_name: str) -> set[str]:
    validate_sql_identifier(table_name, "table name")
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _add_column_if_missing(
    conn: duckdb.DuckDBPyConnection, table_name: str, column_name: str, column_type: str
) -> None:
    validate_sql_identifier(column_name, "column name")
    validate_column_type(column_type)
    if column_name in _table_columns(conn, table_name):  # _table_columns validates table_name
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def _ensure_store(conn: duckdb.DuckDBPyConnection, table_name: str) -> None:
    validate_sql_identifier(table_name, "table name")
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
            soc_tube_upper_mwh DOUBLE,
            validity_horizon_H_t INTEGER,
            half_life_steps INTEGER,
            expires_at_step INTEGER,
            validity_status VARCHAR,
            runtime_surface VARCHAR,
            closure_tier VARCHAR,
            reliability_feature_basis VARCHAR
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
    _add_column_if_missing(conn, table_name, "validity_horizon_H_t", "INTEGER")
    _add_column_if_missing(conn, table_name, "half_life_steps", "INTEGER")
    _add_column_if_missing(conn, table_name, "expires_at_step", "INTEGER")
    _add_column_if_missing(conn, table_name, "validity_status", "VARCHAR")
    _add_column_if_missing(conn, table_name, "runtime_surface", "VARCHAR")
    _add_column_if_missing(conn, table_name, "closure_tier", "VARCHAR")
    _add_column_if_missing(conn, table_name, "reliability_feature_basis", "VARCHAR")


def _event_table_name(table_name: str) -> str:
    return f"{table_name}_events"


def _ensure_event_store(conn: duckdb.DuckDBPyConnection, table_name: str) -> str:
    event_table = _event_table_name(table_name)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {event_table} (
            event_id VARCHAR PRIMARY KEY,
            command_id VARCHAR,
            certificate_hash VARCHAR,
            prev_hash VARCHAR,
            event_hash VARCHAR,
            prev_event_hash VARCHAR,
            payload_json VARCHAR,
            signature VARCHAR,
            public_key_id VARCHAR,
            created_at VARCHAR
        )
        """
    )
    return event_table


def _latest_event_hash(conn: duckdb.DuckDBPyConnection, event_table: str) -> str | None:
    row = conn.execute(
        f"""
        SELECT event_hash
        FROM {event_table}
        ORDER BY created_at DESC, event_id DESC
        LIMIT 1
        """
    ).fetchone()
    return None if row is None else str(row[0])


def _event_hash(
    *,
    command_id: str,
    certificate_hash: str,
    prev_event_hash: str | None,
    payload_json: str,
) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "command_id": command_id,
                "certificate_hash": certificate_hash,
                "prev_event_hash": prev_event_hash,
                "payload_json": payload_json,
            }
        )
    )


def _store_certificate_event(
    conn: duckdb.DuckDBPyConnection,
    *,
    certificate: Mapping[str, Any],
    table_name: str,
) -> None:
    event_table = _ensure_event_store(conn, table_name)
    command_id = str(certificate.get("command_id"))
    certificate_hash = str(certificate.get("certificate_hash"))
    existing = conn.execute(
        f"""
        SELECT certificate_hash
        FROM {event_table}
        WHERE command_id = ?
        """,
        [command_id],
    ).fetchall()
    if any(str(row[0]) == certificate_hash for row in existing):
        return
    if existing:
        raise RuntimeError(f"conflicting certificate overwrite rejected for command_id={command_id}")

    payload_json = json.dumps(certificate, ensure_ascii=True, sort_keys=True)
    prev_event_hash = _latest_event_hash(conn, event_table)
    event_hash = _event_hash(
        command_id=command_id,
        certificate_hash=certificate_hash,
        prev_event_hash=prev_event_hash,
        payload_json=payload_json,
    )
    conn.execute(
        f"""
        INSERT INTO {event_table} (
            event_id,
            command_id,
            certificate_hash,
            prev_hash,
            event_hash,
            prev_event_hash,
            payload_json,
            signature,
            public_key_id,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(uuid4()),
            command_id,
            certificate_hash,
            certificate.get("prev_hash"),
            event_hash,
            prev_event_hash,
            payload_json,
            certificate.get("signature"),
            certificate.get("public_key_id"),
            datetime.now(UTC).isoformat(),
        ],
    )


def store_certificate(certificate: Mapping[str, Any], duckdb_path: str, table_name: str) -> None:
    certificate = normalize_certificate_schema(certificate)
    db_path = Path(duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        _ensure_store(conn, table_name)
        _store_certificate_event(conn, certificate=certificate, table_name=table_name)
        conn.execute(_insert_sql(table_name), _certificate_row(certificate))
    finally:
        conn.close()


def _insert_sql(table_name: str) -> str:
    columns = ",\n            ".join(_INSERT_COLUMNS)
    placeholders = ", ".join("?" for _ in _INSERT_COLUMNS)
    return f"""
        INSERT OR REPLACE INTO {table_name} (
            {columns}
        ) VALUES ({placeholders})
    """


_INSERT_COLUMNS = [
    "command_id",
    "certificate_hash",
    "prev_hash",
    "created_at",
    "payload_json",
    "intervened",
    "intervention_reason",
    "reliability_w",
    "drift_flag",
    "inflation",
    "validity_score",
    "adaptive_quantile",
    "conditional_coverage_gap",
    "runtime_interval_policy",
    "coverage_group_key",
    "shift_alert_flag",
    "guarantee_checks_passed",
    "guarantee_fail_reasons",
    "true_soc_violation_after_apply",
    "assumptions_version",
    "gamma_mw",
    "e_t_mwh",
    "soc_tube_lower_mwh",
    "soc_tube_upper_mwh",
    "validity_horizon_H_t",
    "half_life_steps",
    "expires_at_step",
    "validity_status",
    "runtime_surface",
    "closure_tier",
    "reliability_feature_basis",
]


def _certificate_row(certificate: Mapping[str, Any]) -> list[Any]:
    certificate = normalize_certificate_schema(certificate)
    payload_json = json.dumps(certificate, ensure_ascii=True, sort_keys=True)
    return [
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
        certificate.get("validity_horizon_H_t"),
        certificate.get("half_life_steps"),
        certificate.get("expires_at_step"),
        certificate.get("validity_status"),
        certificate.get("runtime_surface"),
        certificate.get("closure_tier"),
        json.dumps(certificate.get("reliability_feature_basis", {}), ensure_ascii=True, sort_keys=True),
    ]


def store_certificates_batch(
    certificates: Iterable[Mapping[str, Any]], duckdb_path: str, table_name: str
) -> None:
    normalized = [normalize_certificate_schema(certificate) for certificate in certificates]
    rows = [_certificate_row(certificate) for certificate in normalized]
    if not rows:
        return
    db_path = Path(duckdb_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    try:
        _ensure_store(conn, table_name)
        event_table = _ensure_event_store(conn, table_name)
        existing_events = {
            str(command_id): str(certificate_hash)
            for command_id, certificate_hash in conn.execute(
                f"""
                SELECT command_id, certificate_hash
                FROM {event_table}
                """
            ).fetchall()
        }
        prev_event_hash = _latest_event_hash(conn, event_table)
        event_rows: list[dict[str, Any]] = []
        created_at = datetime.now(UTC).isoformat()
        for certificate in normalized:
            command_id = str(certificate.get("command_id"))
            certificate_hash = str(certificate.get("certificate_hash"))
            existing_hash = existing_events.get(command_id)
            if existing_hash == certificate_hash:
                continue
            if existing_hash is not None:
                raise RuntimeError(f"conflicting certificate overwrite rejected for command_id={command_id}")

            payload_json = json.dumps(certificate, ensure_ascii=True, sort_keys=True)
            event_hash = _event_hash(
                command_id=command_id,
                certificate_hash=certificate_hash,
                prev_event_hash=prev_event_hash,
                payload_json=payload_json,
            )
            event_rows.append(
                {
                    "event_id": str(uuid4()),
                    "command_id": command_id,
                    "certificate_hash": certificate_hash,
                    "prev_hash": certificate.get("prev_hash"),
                    "event_hash": event_hash,
                    "prev_event_hash": prev_event_hash,
                    "payload_json": payload_json,
                    "signature": certificate.get("signature"),
                    "public_key_id": certificate.get("public_key_id"),
                    "created_at": created_at,
                }
            )
            existing_events[command_id] = certificate_hash
            prev_event_hash = event_hash

        if event_rows:
            event_frame = pd.DataFrame(
                event_rows,
                columns=[
                    "event_id",
                    "command_id",
                    "certificate_hash",
                    "prev_hash",
                    "event_hash",
                    "prev_event_hash",
                    "payload_json",
                    "signature",
                    "public_key_id",
                    "created_at",
                ],
            )
            conn.register("certificate_batch_event_rows", event_frame)
            conn.execute(
                f"""
                INSERT INTO {event_table} (
                    event_id,
                    command_id,
                    certificate_hash,
                    prev_hash,
                    event_hash,
                    prev_event_hash,
                    payload_json,
                    signature,
                    public_key_id,
                    created_at
                )
                SELECT
                    event_id,
                    command_id,
                    certificate_hash,
                    prev_hash,
                    event_hash,
                    prev_event_hash,
                    payload_json,
                    signature,
                    public_key_id,
                    created_at
                FROM certificate_batch_event_rows
                """
            )
            conn.unregister("certificate_batch_event_rows")
        frame = pd.DataFrame(rows, columns=_INSERT_COLUMNS)
        conn.register("certificate_batch_rows", frame)
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {table_name} (
                {", ".join(_INSERT_COLUMNS)}
            )
            SELECT {", ".join(_INSERT_COLUMNS)}
            FROM certificate_batch_rows
            """
        )
        conn.unregister("certificate_batch_rows")
    finally:
        conn.close()


def get_certificate(command_id: str, duckdb_path: str, table_name: str) -> dict[str, Any] | None:
    db_path = Path(duckdb_path)
    if not db_path.exists():
        return None

    conn = duckdb.connect(str(db_path))
    try:
        _ensure_store(conn, table_name)
        event_table = _ensure_event_store(conn, table_name)
        row = conn.execute(
            f"""
            SELECT payload_json
            FROM {event_table}
            WHERE command_id = ?
            ORDER BY created_at DESC, event_id DESC
            LIMIT 1
            """,
            [command_id],
        ).fetchone()
        if row is not None:
            return json.loads(str(row[0]))
        row = conn.execute(
            f"SELECT payload_json FROM {table_name} WHERE command_id = ?",
            [command_id],
        ).fetchone()
        if row is None:
            return None
        return json.loads(str(row[0]))
    finally:
        conn.close()
