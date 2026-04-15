from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb
import pandas as pd

from orius.dc3s.certificate import recompute_certificate_hash


REQUIRED_CERTIFICATE_FIELDS = (
    "certificate_hash",
    "command_id",
    "controller",
    "created_at",
    "proposed_action",
    "safe_action",
    "uncertainty",
    "reliability",
)

NON_HASHED_EXTENSION_FIELDS = {
    "assumptions_checked",
    "controller_label",
    "coverage_lb_t",
    "delta_mw",
    "dispatch_regime",
    "ego_track_id",
    "fault_family",
    "intervention_trace_id",
    "interval_width",
    "lane",
    "lambda_mw_used",
    "neighbor_ids",
    "observed_margin",
    "q_eff",
    "q_multiplier",
    "risk_bound_scope",
    "scenario_id",
    "semantic_checks",
    "sensitivity_norm",
    "sensitivity_t",
    "shard_id",
    "shift_score",
    "solver_status",
    "source_domain",
    "true_margin",
    "w_t",
    "widening_factor",
}


def load_certificates_from_duckdb(duckdb_path: str | Path, table_name: str = "dispatch_certificates") -> list[dict[str, Any]]:
    db_path = Path(duckdb_path)
    if not db_path.exists():
        return []
    conn = duckdb.connect(str(db_path))
    try:
        tables = {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}
        if table_name not in tables:
            return []
        rows = conn.execute(
            f"""
            SELECT
                command_id,
                certificate_hash,
                prev_hash,
                created_at,
                payload_json
            FROM {table_name}
            ORDER BY created_at ASC, command_id ASC
            """
        ).fetchall()
    finally:
        conn.close()
    certificates: list[dict[str, Any]] = []
    for command_id, cert_hash, prev_hash, created_at, payload_json in rows:
        payload: dict[str, Any]
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except json.JSONDecodeError:
            payload = {}
        payload.setdefault("command_id", command_id)
        payload.setdefault("certificate_hash", cert_hash)
        payload.setdefault("prev_hash", prev_hash)
        payload.setdefault("created_at", created_at)
        certificates.append(payload)
    return certificates


def verify_certificates(certificates: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cert_list = [dict(cert) for cert in certificates]
    failure_rows: list[dict[str, Any]] = []
    expiry_rows: list[dict[str, Any]] = []
    hash_valid_flags: list[bool] = []

    for index, certificate in enumerate(cert_list):
        observed_hash = certificate.get("certificate_hash")
        expected_hash = recompute_certificate_hash(certificate)
        valid_hash = isinstance(observed_hash, str) and observed_hash == expected_hash
        if not valid_hash:
            stripped = {key: value for key, value in certificate.items() if key not in NON_HASHED_EXTENSION_FIELDS}
            expected_hash = recompute_certificate_hash(stripped)
            valid_hash = isinstance(observed_hash, str) and observed_hash == expected_hash
        missing_fields = sorted(field for field in REQUIRED_CERTIFICATE_FIELDS if field not in certificate or certificate.get(field) in (None, ""))
        intervention_reason = certificate.get("intervention_reason")
        intervened = bool(certificate.get("intervened", False))
        semantic_ok = not intervened or intervention_reason not in (None, "")
        if not valid_hash or missing_fields or not semantic_ok:
            failure_rows.append(
                {
                    "row_index": int(index),
                    "command_id": str(certificate.get("command_id", "")),
                    "failure_type": (
                        "hash_mismatch"
                        if not valid_hash
                        else "missing_required_fields"
                        if missing_fields
                        else "intervention_reason_missing"
                    ),
                    "missing_fields": ",".join(missing_fields),
                    "expected_hash": expected_hash,
                    "observed_hash": observed_hash,
                }
            )
        hash_valid_flags.append(bool(valid_hash))

        validity_horizon = certificate.get("validity_horizon_H_t")
        half_life = certificate.get("half_life_steps")
        expires_at = certificate.get("expires_at_step")
        status = certificate.get("validity_status")
        has_expiry = any(value not in (None, "") for value in (validity_horizon, half_life, expires_at, status))
        expiry_ok = True
        if has_expiry:
            try:
                horizon_value = None if validity_horizon in (None, "") else int(validity_horizon)
                half_life_value = None if half_life in (None, "") else int(half_life)
                expires_value = None if expires_at in (None, "") else int(expires_at)
                expiry_ok = all(
                    value is None or value >= 0
                    for value in (horizon_value, half_life_value, expires_value)
                )
                if horizon_value is not None and half_life_value is not None:
                    expiry_ok = expiry_ok and half_life_value <= max(horizon_value, half_life_value)
            except (TypeError, ValueError):
                expiry_ok = False
        expiry_rows.append(
            {
                "row_index": int(index),
                "command_id": str(certificate.get("command_id", "")),
                "has_expiry_metadata": bool(has_expiry),
                "validity_horizon_H_t": validity_horizon,
                "half_life_steps": half_life,
                "expires_at_step": expires_at,
                "validity_status": status,
                "expiry_consistent": bool(expiry_ok),
            }
        )
        if has_expiry and not expiry_ok:
            failure_rows.append(
                {
                    "row_index": int(index),
                    "command_id": str(certificate.get("command_id", "")),
                    "failure_type": "expiry_inconsistent",
                    "missing_fields": "",
                    "expected_hash": "",
                    "observed_hash": "",
                }
            )

    chain_valid = True
    failed_index: int | None = None
    failure_reason: str | None = None
    expected_prev_hash: str | None = None
    observed_prev_hash: str | None = None
    previous_hash: str | None = None
    for index, certificate in enumerate(cert_list):
        if not hash_valid_flags[index]:
            chain_valid = False
            failed_index = int(index)
            failure_reason = "hash_mismatch"
            break
        current_prev = certificate.get("prev_hash")
        if current_prev in (None, ""):
            previous_hash = str(certificate.get("certificate_hash"))
            continue
        if current_prev != previous_hash:
            chain_valid = False
            failed_index = int(index)
            failure_reason = "prev_hash_mismatch"
            expected_prev_hash = previous_hash
            observed_prev_hash = current_prev
            break
        previous_hash = str(certificate.get("certificate_hash"))
    if not chain_valid:
        failure_rows.append(
            {
                "row_index": int(failed_index or 0),
                "command_id": str(cert_list[int(failed_index or 0)].get("command_id", "")) if cert_list else "",
                "failure_type": str(failure_reason or "chain_invalid"),
                "missing_fields": "",
                "expected_hash": str(expected_prev_hash or ""),
                "observed_hash": str(observed_prev_hash or ""),
            }
        )

    failure_df = pd.DataFrame(
        failure_rows,
        columns=["row_index", "command_id", "failure_type", "missing_fields", "expected_hash", "observed_hash"],
    )
    expiry_df = pd.DataFrame(
        expiry_rows,
        columns=[
            "row_index",
            "command_id",
            "has_expiry_metadata",
            "validity_horizon_H_t",
            "half_life_steps",
            "expires_at_step",
            "validity_status",
            "expiry_consistent",
        ],
    )
    governance_df = pd.DataFrame(
        [
            {
                "metric": "certificate_rows",
                "value": int(len(cert_list)),
            },
            {
                "metric": "chain_valid",
                "value": 1 if chain_valid else 0,
            },
            {
                "metric": "failure_rows",
                "value": int(len(failure_df)),
            },
            {
                "metric": "required_payload_pass_rate",
                "value": float(1.0 - len(failure_df[failure_df["failure_type"] == "missing_required_fields"]) / max(len(cert_list), 1)),
            },
            {
                "metric": "expiry_metadata_presence_rate",
                "value": float(expiry_df["has_expiry_metadata"].mean()) if not expiry_df.empty else 0.0,
            },
            {
                "metric": "expiry_consistency_rate",
                "value": float(expiry_df["expiry_consistent"].mean()) if not expiry_df.empty else 1.0,
            },
            {
                "metric": "audit_completeness_rate",
                "value": float(max(0.0, 1.0 - len(failure_df) / max(len(cert_list), 1))),
            },
        ]
    )
    summary = {
        "certificate_rows": int(len(cert_list)),
        "chain_valid": bool(chain_valid),
        "checked": int(len(cert_list) if chain_valid else (failed_index or 0)),
        "failure_rows": int(len(failure_df)),
        "required_payload_pass_rate": float(governance_df.loc[governance_df["metric"] == "required_payload_pass_rate", "value"].iloc[0]) if not governance_df.empty else 0.0,
        "expiry_metadata_presence_rate": float(governance_df.loc[governance_df["metric"] == "expiry_metadata_presence_rate", "value"].iloc[0]) if not governance_df.empty else 0.0,
        "expiry_consistency_rate": float(governance_df.loc[governance_df["metric"] == "expiry_consistency_rate", "value"].iloc[0]) if not governance_df.empty else 1.0,
        "audit_completeness_rate": float(governance_df.loc[governance_df["metric"] == "audit_completeness_rate", "value"].iloc[0]) if not governance_df.empty else 1.0,
        "failed_index": failed_index,
        "reason": failure_reason,
    }
    return summary, failure_df, expiry_df, governance_df
