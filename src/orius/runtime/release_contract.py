"""Canonical ORIUS runtime release evidence contract.

This module is intentionally compact: it defines the shared publication-facing
evidence row that promoted Battery, AV, and Healthcare releases must expose.
It does not replace domain runtime code; it normalizes their evidence into one
auditable release grammar.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RELEASE_CONTRACT_SCHEMA_VERSION = "orius.release_contract.v1"
PROMOTED_RELEASE_DOMAINS = ("battery", "av", "healthcare")
RUNTIME_FLOW = "observe>reliability>uncertainty>repair_or_fallback>sign_certificate>append_only_audit"

CANONICAL_RELEASE_EVIDENCE_FIELDS = (
    "release_contract_schema_version",
    "domain",
    "trace_id",
    "command_id",
    "runtime_flow",
    "reliability_score",
    "uncertainty_summary",
    "proposed_action",
    "safe_action",
    "repair_status",
    "fallback_status",
    "release_decision",
    "certificate_hash",
    "prev_hash",
    "certificate_signature_status",
    "append_only_audit_status",
    "t11_status",
    "postcondition_status",
    "source_artifact",
    "strict_runtime_gate",
    "evidence_scope_note",
)

_SIGNED_STATUSES = {"signed", "signature_valid", "verified_signed"}
_APPEND_ONLY_STATUSES = {"append_only", "append_only_verified", "append_only_event_verified"}


def domain_key(domain: str) -> str:
    text = str(domain or "").strip().lower()
    if text in {"battery", "de", "battery energy storage", "battery_energy_storage", "energy_storage"}:
        return "battery"
    if text in {"av", "autonomous vehicles", "autonomous vehicle", "autonomous_vehicles", "orius_av"}:
        return "av"
    if text in {"healthcare", "medical and healthcare monitoring", "medical", "clinical_monitoring", "hc"}:
        return "healthcare"
    raise ValueError(f"unsupported promoted release-contract domain: {domain!r}")


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return json.dumps({"value": text}, sort_keys=True, separators=(",", ":"))
        return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    if value is None:
        return "{}"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "passed"}:
        return "True"
    return "False"


def _float_text(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    if not math.isfinite(numeric):
        numeric = 0.0
    return f"{numeric:.6f}"


def _is_sha256_hex(value: Any) -> bool:
    text = str(value or "").strip()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text.lower())


@dataclass(frozen=True, slots=True)
class RuntimeReleaseEvidence:
    release_contract_schema_version: str
    domain: str
    trace_id: str
    command_id: str
    runtime_flow: str
    reliability_score: str
    uncertainty_summary: str
    proposed_action: str
    safe_action: str
    repair_status: str
    fallback_status: str
    release_decision: str
    certificate_hash: str
    prev_hash: str
    certificate_signature_status: str
    append_only_audit_status: str
    t11_status: str
    postcondition_status: str
    source_artifact: str
    strict_runtime_gate: str
    evidence_scope_note: str

    def to_row(self) -> dict[str, str]:
        return {field: str(getattr(self, field)) for field in CANONICAL_RELEASE_EVIDENCE_FIELDS}


def build_release_evidence(
    *,
    domain: str,
    trace_id: str,
    command_id: str | None = None,
    reliability_score: Any = 0.0,
    uncertainty_summary: Any = None,
    proposed_action: Any = None,
    safe_action: Any = None,
    repair_status: str = "unknown",
    fallback_status: str = "unknown",
    release_decision: str = "released",
    certificate_hash: str,
    prev_hash: str | None = None,
    certificate_signature_status: str | None = None,
    append_only_audit_status: str | None = None,
    t11_status: str = "missing",
    postcondition_status: str | bool = "missing",
    source_artifact: str = "",
    strict_runtime_gate: Any = False,
    evidence_scope_note: str = "",
) -> RuntimeReleaseEvidence:
    key = domain_key(domain)
    post_status = str(postcondition_status).strip().lower()
    if post_status in {"1", "true", "t", "yes"}:
        post_status = "passed"
    elif post_status in {"0", "false", "f", "no"}:
        post_status = "failed"
    return RuntimeReleaseEvidence(
        release_contract_schema_version=RELEASE_CONTRACT_SCHEMA_VERSION,
        domain=key,
        trace_id=str(trace_id or command_id or ""),
        command_id=str(command_id or trace_id or ""),
        runtime_flow=RUNTIME_FLOW,
        reliability_score=_float_text(reliability_score),
        uncertainty_summary=_json_text(uncertainty_summary),
        proposed_action=_json_text(proposed_action),
        safe_action=_json_text(safe_action),
        repair_status=str(repair_status or "unknown"),
        fallback_status=str(fallback_status or "unknown"),
        release_decision=str(release_decision or "released"),
        certificate_hash=str(certificate_hash or ""),
        prev_hash=str(prev_hash or ""),
        certificate_signature_status=str(certificate_signature_status or "unsigned_allowed"),
        append_only_audit_status=str(append_only_audit_status or "append_only_supported"),
        t11_status=str(t11_status or "missing"),
        postcondition_status=post_status or "missing",
        source_artifact=str(source_artifact or ""),
        strict_runtime_gate=_bool_text(strict_runtime_gate),
        evidence_scope_note=str(evidence_scope_note or ""),
    )


def normalize_release_evidence(row: Mapping[str, Any]) -> dict[str, str]:
    if row.get("release_contract_schema_version") == RELEASE_CONTRACT_SCHEMA_VERSION:
        normalized = {field: str(row.get(field, "")) for field in CANONICAL_RELEASE_EVIDENCE_FIELDS}
        normalized["domain"] = domain_key(normalized["domain"])
        normalized["reliability_score"] = _float_text(normalized["reliability_score"])
        normalized["uncertainty_summary"] = _json_text(normalized["uncertainty_summary"])
        normalized["proposed_action"] = _json_text(normalized["proposed_action"])
        normalized["safe_action"] = _json_text(normalized["safe_action"])
        normalized["strict_runtime_gate"] = _bool_text(normalized["strict_runtime_gate"])
        normalized["certificate_signature_status"] = (
            normalized["certificate_signature_status"] or "unsigned_allowed"
        )
        normalized["append_only_audit_status"] = (
            normalized["append_only_audit_status"] or "append_only_supported"
        )
        return normalized
    return build_release_evidence(
        domain=str(row.get("domain", "")),
        trace_id=str(row.get("trace_id") or row.get("command_id") or ""),
        command_id=str(row.get("command_id") or row.get("trace_id") or ""),
        reliability_score=row.get("reliability_score", row.get("certificate_valid_release_rate", 0.0)),
        uncertainty_summary=row.get("uncertainty_summary", {}),
        proposed_action=row.get("proposed_action", {}),
        safe_action=row.get("safe_action", row.get("action", {})),
        repair_status=str(row.get("repair_status", "unknown")),
        fallback_status=str(row.get("fallback_status", "unknown")),
        release_decision=str(row.get("release_decision", "released")),
        certificate_hash=str(row.get("certificate_hash", "")),
        prev_hash=str(row.get("prev_hash", "")),
        certificate_signature_status=str(row.get("certificate_signature_status", "") or "unsigned_allowed"),
        append_only_audit_status=str(row.get("append_only_audit_status", "") or "append_only_supported"),
        t11_status=str(row.get("t11_status", "missing")),
        postcondition_status=row.get("postcondition_status", row.get("postcondition_passed", "missing")),
        source_artifact=str(row.get("source_artifact", "")),
        strict_runtime_gate=row.get("strict_runtime_gate", False),
        evidence_scope_note=str(row.get("evidence_scope_note", row.get("scope_note", ""))),
    ).to_row()


def validate_release_evidence(row: Mapping[str, Any], *, strict: bool = False) -> list[str]:
    findings: list[str] = []
    missing = [field for field in CANONICAL_RELEASE_EVIDENCE_FIELDS if field not in row]
    if missing:
        return [f"missing fields: {missing}"]
    try:
        normalized = normalize_release_evidence(row)
    except Exception as exc:
        return [f"could not normalize release evidence: {exc}"]
    if normalized["release_contract_schema_version"] != RELEASE_CONTRACT_SCHEMA_VERSION:
        findings.append("invalid release_contract_schema_version")
    if normalized["domain"] not in PROMOTED_RELEASE_DOMAINS:
        findings.append(f"unexpected domain: {normalized['domain']}")
    if not normalized["trace_id"]:
        findings.append("missing trace_id")
    if not normalized["command_id"]:
        findings.append("missing command_id")
    try:
        reliability = float(normalized["reliability_score"])
    except ValueError:
        reliability = -1.0
    if not (0.0 <= reliability <= 1.0):
        findings.append("reliability_score must be in [0, 1]")
    for json_field in ("uncertainty_summary", "proposed_action", "safe_action"):
        try:
            json.loads(normalized[json_field])
        except json.JSONDecodeError:
            findings.append(f"{json_field} must be valid JSON")
    if not _is_sha256_hex(normalized["certificate_hash"]):
        findings.append("invalid certificate_hash")
    if not normalized["t11_status"]:
        findings.append("missing t11_status")
    elif normalized["t11_status"] != "runtime_linked":
        findings.append("t11_status must be runtime_linked")
    if normalized["postcondition_status"] != "passed":
        findings.append("postcondition_status must be passed")
    if "invalid" in normalized["fallback_status"].lower() and normalized["release_decision"] == "released":
        findings.append("invalid fallback cannot be released")
    if strict and normalized["release_decision"] == "released":
        if normalized["certificate_signature_status"] not in _SIGNED_STATUSES:
            findings.append("strict release requires signed certificate")
        if normalized["append_only_audit_status"] not in _APPEND_ONLY_STATUSES:
            findings.append("strict release requires append-only audit")
    return findings
