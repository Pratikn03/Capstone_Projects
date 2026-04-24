#!/usr/bin/env python3
"""Validate canonical ORIUS certificate schema and runtime trace exposure."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.dc3s.certificate import (  # noqa: E402
    CERTIFICATE_SCHEMA_VERSION,
    make_certificate,
    normalize_certificate_schema,
    verify_certificate,
)

REQUIRED_SCHEMA_FIELDS = {
    "certificate_hash",
    "prev_hash",
    "certificate_schema_version",
    "issuer",
    "domain",
    "action",
    "validity_horizon_H_t",
    "expires_at_step",
    "theorem_contracts",
}

TRACE_SCHEMA_COLUMNS = {
    "certificate_schema_version",
    "certificate_hash",
    "prev_hash",
    "issuer",
    "domain",
    "action",
    "theorem_contracts",
}

TRACE_PATHS = {
    "battery": REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_traces.csv",
    "av": REPO_ROOT / "reports" / "orius_av" / "full_corpus" / "runtime_traces.csv",
    "healthcare": REPO_ROOT / "reports" / "healthcare" / "runtime_traces.csv",
}


def _sample_certificate() -> dict[str, object]:
    return make_certificate(
        command_id="schema-smoke",
        device_id="dev-1",
        zone_id="battery",
        controller="test",
        proposed_action={"charge_mw": 0.0},
        safe_action={"charge_mw": 0.0},
        uncertainty={"meta": {}},
        reliability={"w_t": 1.0},
        drift={"drift": False},
        model_hash="model",
        config_hash="config",
        validity_horizon_H_t=1,
        expires_at_step=1,
        theorem_contracts={"T11": "schema_smoke"},
    )


def _read_head(path: Path, limit: int = 5000) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for idx, row in enumerate(reader):
            rows.append(row)
            if idx + 1 >= limit:
                break
        return rows


def main() -> int:
    findings: list[str] = []

    cert = _sample_certificate()
    missing = sorted(field for field in REQUIRED_SCHEMA_FIELDS if field not in cert)
    if missing:
        findings.append(f"sample make_certificate missing fields: {missing}")
    verification = verify_certificate(cert)
    if not verification["valid"]:
        findings.append(f"sample make_certificate failed verification: {verification}")

    legacy = normalize_certificate_schema({"command_id": "legacy", "cert_hash": "abc123", "safe_action": {"x": 1}})
    if legacy.get("certificate_hash") != "abc123":
        findings.append("legacy cert_hash did not normalize to certificate_hash")
    if legacy.get("certificate_schema_version") != CERTIFICATE_SCHEMA_VERSION:
        findings.append("legacy certificate did not receive canonical schema version")

    for domain, path in TRACE_PATHS.items():
        if not path.exists():
            findings.append(f"{domain} runtime trace missing: {path.relative_to(REPO_ROOT)}")
            continue
        rows = _read_head(path)
        if not rows:
            findings.append(f"{domain} runtime trace is empty: {path.relative_to(REPO_ROOT)}")
            continue
        missing_columns = sorted(TRACE_SCHEMA_COLUMNS - set(rows[0]))
        if missing_columns:
            findings.append(f"{domain} runtime trace missing schema columns: {missing_columns}")
            continue
        certified_rows = [
            row for row in rows
            if (row.get("certificate_valid") or "").strip().lower() in {"true", "1"}
            and (row.get("controller") or "").strip() in {"orius", "dc3s_ftit", "dc3s_wrapped"}
        ]
        if certified_rows and not any((row.get("certificate_hash") or "").strip() for row in certified_rows):
            findings.append(f"{domain} certified trace sample has no certificate_hash values")

    if findings:
        print("[validate_certificate_schema] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_certificate_schema] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
