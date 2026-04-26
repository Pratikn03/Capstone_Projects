#!/usr/bin/env python3
"""Add canonical certificate-schema columns to promoted runtime traces.

AV and Healthcare certificates are stored in DuckDB audit ledgers.  Battery's
locked witness trace does not carry a certificate DB, so this script emits a
deterministic trace-level witness certificate hash for rows that declare a
valid certificate.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.dc3s.certificate import (  # noqa: E402
    CERTIFICATE_SCHEMA_VERSION,
    normalize_certificate_schema,
    recompute_certificate_hash,
)

SCHEMA_COLUMNS = [
    "certificate_schema_version",
    "certificate_hash",
    "prev_hash",
    "issuer",
    "domain",
    "action",
    "theorem_contracts",
]


def _read_certificates(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}
        if "dispatch_certificates" not in tables:
            return []
        rows = conn.execute(
            """
            SELECT command_id, certificate_hash, prev_hash, created_at, payload_json
            FROM dispatch_certificates
            ORDER BY created_at ASC, command_id ASC
            """
        ).fetchall()
    finally:
        conn.close()

    certificates: list[dict[str, Any]] = []
    for command_id, certificate_hash, prev_hash, created_at, payload_json in rows:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except json.JSONDecodeError:
            payload = {}
        payload.setdefault("command_id", command_id)
        payload.setdefault("certificate_hash", certificate_hash)
        payload.setdefault("prev_hash", prev_hash)
        payload.setdefault("created_at", created_at)
        normalized = normalize_certificate_schema(payload)
        if normalized.get("certificate_hash") in (None, ""):
            normalized["certificate_hash"] = recompute_certificate_hash(normalized)
        certificates.append(normalized)
    return certificates


def _cert_columns(cert: dict[str, Any] | None) -> dict[str, str]:
    if not cert:
        return {column: "" for column in SCHEMA_COLUMNS}
    normalized = normalize_certificate_schema(cert)
    return {
        "certificate_schema_version": str(normalized.get("certificate_schema_version", CERTIFICATE_SCHEMA_VERSION)),
        "certificate_hash": str(normalized.get("certificate_hash", "")),
        "prev_hash": str(normalized.get("prev_hash", "") or ""),
        "issuer": str(normalized.get("issuer", "")),
        "domain": str(normalized.get("domain", "")),
        "action": json.dumps(normalized.get("action", {}), sort_keys=True),
        "theorem_contracts": json.dumps(normalized.get("theorem_contracts", {}), sort_keys=True),
    }


def _battery_witness_cert(row: pd.Series, previous_hash: str | None) -> dict[str, Any]:
    safe_action = {
        "charge_mw": float(row.get("safe_charge_mw", 0.0) or 0.0),
        "discharge_mw": float(row.get("safe_discharge_mw", 0.0) or 0.0),
    }
    cert = normalize_certificate_schema(
        {
            "command_id": str(row.get("trace_id", "")),
            "controller": str(row.get("controller", "")),
            "created_at": "",
            "prev_hash": previous_hash,
            "proposed_action": {
                "charge_mw": float(row.get("candidate_charge_mw", 0.0) or 0.0),
                "discharge_mw": float(row.get("candidate_discharge_mw", 0.0) or 0.0),
            },
            "safe_action": safe_action,
            "uncertainty": {"interval_lower": row.get("interval_lower"), "interval_upper": row.get("interval_upper")},
            "reliability": {"w_t": float(row.get("reliability_w", 1.0) or 1.0)},
            "validity_horizon_H_t": 1,
            "theorem_contracts": {"T11": "battery_witness_runtime_certificate"},
        },
        issuer="orius.battery.runtime_trace",
        domain="battery",
    )
    cert["certificate_hash"] = recompute_certificate_hash(cert)
    return cert


def _sync_trace(trace_path: Path, *, domain: str, certificates: list[dict[str, Any]]) -> tuple[int, int]:
    if not trace_path.exists():
        return 0, 0
    df = pd.read_csv(trace_path)
    for column in SCHEMA_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    assigned = 0
    if domain in {"av", "healthcare"}:
        cert_iter = iter(certificates)
        mask = df["controller"].astype(str).eq("orius") if "controller" in df.columns else pd.Series(False, index=df.index)
        for idx in df.index[mask]:
            cert = next(cert_iter, None)
            if cert is None:
                break
            for column, value in _cert_columns(cert).items():
                df.at[idx, column] = value
            assigned += 1
    elif domain == "battery":
        previous_hash: str | None = None
        valid_mask = df.get("certificate_valid", pd.Series(False, index=df.index)).astype(str).str.lower().isin({"true", "1"})
        for idx in df.index[valid_mask]:
            cert = _battery_witness_cert(df.loc[idx], previous_hash)
            previous_hash = str(cert["certificate_hash"])
            for column, value in _cert_columns(cert).items():
                df.at[idx, column] = value
            assigned += 1

    df.to_csv(trace_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return int(len(df)), int(assigned)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    jobs = [
        (
            "battery",
            root / "reports" / "battery_av" / "battery" / "runtime_traces.csv",
            [],
        ),
        (
            "av",
            root / "reports" / "orius_av" / "full_corpus" / "runtime_traces.csv",
            _read_certificates(root / "reports" / "orius_av" / "full_corpus" / "dc3s_av_waymo_dryrun.duckdb"),
        ),
        (
            "healthcare",
            root / "reports" / "healthcare" / "runtime_traces.csv",
            _read_certificates(root / "reports" / "healthcare" / "healthcare_runtime.duckdb"),
        ),
    ]
    for domain, trace_path, certificates in jobs:
        rows, assigned = _sync_trace(trace_path, domain=domain, certificates=certificates)
        print(f"{domain}: rows={rows} certificate_schema_rows={assigned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
