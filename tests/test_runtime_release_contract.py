"""Canonical runtime release contract tests."""

from __future__ import annotations

import json

import pytest

from orius.runtime.release_contract import (
    CANONICAL_RELEASE_EVIDENCE_FIELDS,
    RELEASE_CONTRACT_SCHEMA_VERSION,
    build_release_evidence,
    normalize_release_evidence,
    validate_release_evidence,
)


def _base_payload(domain: str) -> dict:
    return {
        "domain": domain,
        "trace_id": f"{domain}-trace-1",
        "command_id": f"{domain}-cmd-1",
        "reliability_score": 0.99,
        "uncertainty_summary": {"coverage_bucket": "mid", "width": 1.25},
        "proposed_action": {"raw": 1.0},
        "safe_action": {"safe": 1.0},
        "repair_status": "not_repaired",
        "fallback_status": "not_fallback",
        "release_decision": "released",
        "certificate_hash": "a" * 64,
        "prev_hash": "GENESIS",
        "certificate_signature_status": "unsigned_allowed",
        "append_only_audit_status": "append_only_supported",
        "t11_status": "runtime_linked",
        "postcondition_status": "passed",
        "source_artifact": "reports/publication/example.csv",
        "strict_runtime_gate": True,
        "evidence_scope_note": "compact promoted-domain witness",
    }


@pytest.mark.parametrize("domain", ["battery", "av", "healthcare"])
def test_build_release_evidence_exposes_identical_canonical_fields(domain: str) -> None:
    evidence = build_release_evidence(**_base_payload(domain))
    row = evidence.to_row()

    assert evidence.release_contract_schema_version == RELEASE_CONTRACT_SCHEMA_VERSION
    assert tuple(row) == CANONICAL_RELEASE_EVIDENCE_FIELDS
    assert set(row) == set(CANONICAL_RELEASE_EVIDENCE_FIELDS)
    json.loads(row["uncertainty_summary"])
    json.loads(row["proposed_action"])
    json.loads(row["safe_action"])
    assert validate_release_evidence(row) == []


def test_normalize_release_evidence_accepts_compact_rows() -> None:
    row = normalize_release_evidence(
        {
            **_base_payload("Battery Energy Storage"),
            "certificate_signature_status": "",
            "append_only_audit_status": "",
        }
    )

    assert row["domain"] == "battery"
    assert row["certificate_signature_status"] == "unsigned_allowed"
    assert row["append_only_audit_status"] == "append_only_supported"
    assert validate_release_evidence(row) == []


def test_release_contract_rejects_invalid_fallback_release() -> None:
    row = build_release_evidence(
        **{
            **_base_payload("healthcare"),
            "fallback_status": "invalid_fallback",
            "release_decision": "released",
        }
    ).to_row()

    findings = validate_release_evidence(row)

    assert any("invalid fallback" in finding for finding in findings)


def test_release_contract_strict_mode_requires_signed_append_only_release() -> None:
    row = build_release_evidence(**_base_payload("av")).to_row()

    findings = validate_release_evidence(row, strict=True)

    assert any("strict release requires signed certificate" in finding for finding in findings)
    assert any("strict release requires append-only audit" in finding for finding in findings)
