#!/usr/bin/env python3
"""Validate canonical ORIUS certificate schema and compact witness exposure.

The raw runtime traces can be hundreds of MB to multiple GB and are not a
GitHub-safe release surface. This validator therefore checks the certificate
constructor/normalizer plus a compact publication witness table that records
the canonical certificate fields for each promoted domain.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.dc3s.certificate import (
    CERTIFICATE_SCHEMA_VERSION,
    make_certificate,
    normalize_certificate_schema,
    sign_certificate,
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

EXPECTED_DOMAINS = {"battery", "av", "healthcare"}
SCHEMA_WITNESS_PATH = REPO_ROOT / "reports" / "publication" / "certificate_schema_witnesses.csv"
SUMMARY_PATH = REPO_ROOT / "reports" / "publication" / "domain_runtime_contract_summary.json"


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


def _is_sha256_hex(value: str) -> bool:
    text = str(value or "").strip()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text.lower())


def _validate_schema_witnesses(findings: list[str]) -> None:
    if not SCHEMA_WITNESS_PATH.exists():
        findings.append(f"certificate schema witness missing: {SCHEMA_WITNESS_PATH.relative_to(REPO_ROOT)}")
        return
    rows = _read_head(SCHEMA_WITNESS_PATH)
    if not rows:
        findings.append(f"certificate schema witness is empty: {SCHEMA_WITNESS_PATH.relative_to(REPO_ROOT)}")
        return
    missing_columns = sorted(
        (TRACE_SCHEMA_COLUMNS | {"validity_horizon_H_t", "expires_at_step"}) - set(rows[0])
    )
    if missing_columns:
        findings.append(f"certificate schema witness missing columns: {missing_columns}")
        return

    observed_domains = {str(row.get("domain", "")).strip().lower() for row in rows}
    missing_domains = sorted(EXPECTED_DOMAINS - observed_domains)
    if missing_domains:
        findings.append(f"certificate schema witness missing domains: {missing_domains}")

    for idx, row in enumerate(rows, start=2):
        domain = str(row.get("domain", "")).strip().lower()
        if domain not in EXPECTED_DOMAINS:
            findings.append(f"certificate schema witness row {idx} has unexpected domain: {domain!r}")
        if row.get("certificate_schema_version") != CERTIFICATE_SCHEMA_VERSION:
            findings.append(f"certificate schema witness row {idx} has stale schema version")
        if not _is_sha256_hex(str(row.get("certificate_hash", ""))):
            findings.append(f"certificate schema witness row {idx} has invalid certificate_hash")
        for field in ("issuer", "action", "theorem_contracts", "validity_horizon_H_t", "expires_at_step"):
            if str(row.get(field, "")).strip() == "":
                findings.append(f"certificate schema witness row {idx} missing {field}")
        if findings:
            return


def _validate_contract_summary(findings: list[str]) -> None:
    if not SUMMARY_PATH.exists():
        findings.append(f"domain runtime contract summary missing: {SUMMARY_PATH.relative_to(REPO_ROOT)}")
        return
    try:
        import json

        payload = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive validator path
        findings.append(f"domain runtime contract summary is malformed: {exc}")
        return
    domains = payload.get("domains")
    if not isinstance(domains, dict):
        findings.append("domain runtime contract summary has no domains object")
        return
    for domain in ("av", "healthcare"):
        if domain not in domains:
            findings.append(f"domain runtime contract summary missing {domain}")
            return


def main() -> int:
    findings: list[str] = []

    cert = _sample_certificate()
    missing = sorted(field for field in REQUIRED_SCHEMA_FIELDS if field not in cert)
    if missing:
        findings.append(f"sample make_certificate missing fields: {missing}")
    verification = verify_certificate(cert)
    if not verification["valid"]:
        findings.append(f"sample make_certificate failed verification: {verification}")
    signed = sign_certificate(cert, secret="schema-validator-secret-with-32-plus-chars")
    signed_verification = verify_certificate(
        signed,
        require_signature=True,
        signature_secret="schema-validator-secret-with-32-plus-chars",
    )
    if not signed_verification["valid"]:
        findings.append(f"sample signed certificate failed verification: {signed_verification}")
    tampered = dict(signed)
    tampered["action"] = {"charge_mw": 99.0}
    if verify_certificate(
        tampered,
        require_signature=True,
        signature_secret="schema-validator-secret-with-32-plus-chars",
    )["valid"]:
        findings.append("tampered signed certificate verified as valid")

    legacy = normalize_certificate_schema(
        {"command_id": "legacy", "cert_hash": "abc123", "safe_action": {"x": 1}}
    )
    if legacy.get("certificate_hash") != "abc123":
        findings.append("legacy cert_hash did not normalize to certificate_hash")
    if legacy.get("certificate_schema_version") != CERTIFICATE_SCHEMA_VERSION:
        findings.append("legacy certificate did not receive canonical schema version")

    _validate_schema_witnesses(findings)
    _validate_contract_summary(findings)

    if findings:
        print("[validate_certificate_schema] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_certificate_schema] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
