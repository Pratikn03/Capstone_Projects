"""Runtime-linked domain contract witnesses for bounded T11 instantiations."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AV_BRAKE_HOLD_CONTRACT_ID = "AV.T11.brake_hold_runtime_lemma"
HEALTHCARE_FAIL_SAFE_CONTRACT_ID = "HC.T11.fail_safe_release_runtime_lemma"
SOURCE_THEOREM = "T11"

AV_SCOPE_NOTE = (
    "Bounded runtime lemma for promoted ORIUS AV replay rows: T11 forward "
    "obligations plus the true brake-hold postcondition in the longitudinal "
    "runtime trace."
)
HEALTHCARE_SCOPE_NOTE = (
    "Bounded runtime lemma for promoted ORIUS healthcare monitoring rows: T11 "
    "forward obligations plus the true fail-safe alert-release postcondition "
    "in the MIMIC monitoring trace."
)

AV_ASSUMPTIONS = (
    "T11.coverage",
    "T11.sound_safe_action_set",
    "T11.repair_membership",
    "T11.fallback_admissibility",
    "AV.brake_hold_runtime_postcondition",
)
HEALTHCARE_ASSUMPTIONS = (
    "T11.coverage",
    "T11.sound_safe_action_set",
    "T11.repair_membership",
    "T11.fallback_admissibility",
    "HC.fail_safe_alert_release_postcondition",
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return default


def _parse_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _parse_sequence(value: Any) -> tuple[str, ...]:
    if _is_missing(value):
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value if not _is_missing(item))
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return tuple(str(item) for item in parsed if not _is_missing(item))
    separator = "|" if "|" in text else ";" if ";" in text else ","
    return tuple(part.strip() for part in text.split(separator) if part.strip())


def _format_sequence(values: Sequence[str]) -> str:
    return "|".join(str(value) for value in values if str(value))


def _domain_key(domain: str) -> str:
    text = str(domain).strip().lower()
    if text in {"av", "orius_av", "autonomous_vehicle", "autonomous_vehicles"}:
        return "av"
    if text in {"hc", "healthcare", "mimic", "clinical_monitoring"}:
        return "healthcare"
    raise ValueError(f"Unsupported domain runtime contract witness domain: {domain!r}")


def contract_id_for_domain(domain: str) -> str:
    return AV_BRAKE_HOLD_CONTRACT_ID if _domain_key(domain) == "av" else HEALTHCARE_FAIL_SAFE_CONTRACT_ID


def assumptions_for_domain(domain: str) -> tuple[str, ...]:
    return AV_ASSUMPTIONS if _domain_key(domain) == "av" else HEALTHCARE_ASSUMPTIONS


def scope_note_for_domain(domain: str) -> str:
    return AV_SCOPE_NOTE if _domain_key(domain) == "av" else HEALTHCARE_SCOPE_NOTE


@dataclass(frozen=True)
class DomainRuntimeContractWitness:
    """Executable witness that a runtime row closes a bounded T11 domain lemma."""

    domain: str
    trace_id: str
    contract_id: str
    source_theorem: str
    t11_status: str
    failed_obligations: tuple[str, ...]
    certificate_valid: bool
    postcondition_passed: bool
    post_margin: float | None
    failure_reason: str
    assumptions_used: tuple[str, ...]

    @property
    def t11_passed(self) -> bool:
        return self.t11_status == "runtime_linked" and not self.failed_obligations

    @property
    def passed(self) -> bool:
        return self.t11_passed and self.certificate_valid and self.postcondition_passed

    @property
    def scope_note(self) -> str:
        return scope_note_for_domain(self.domain)

    def as_trace_fields(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "source_theorem": self.source_theorem,
            "t11_status": self.t11_status,
            "t11_failed_obligations": _format_sequence(self.failed_obligations),
            "domain_postcondition_passed": bool(self.postcondition_passed),
            "domain_postcondition_failure": self.failure_reason,
        }

    def as_publication_row(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "trace_id": self.trace_id,
            "contract_id": self.contract_id,
            "source_theorem": self.source_theorem,
            "t11_status": self.t11_status,
            "failed_obligations": _format_sequence(self.failed_obligations),
            "certificate_valid": bool(self.certificate_valid),
            "postcondition_passed": bool(self.postcondition_passed),
            "post_margin": "" if self.post_margin is None else float(self.post_margin),
            "failure_reason": self.failure_reason,
            "assumptions_used": _format_sequence(self.assumptions_used),
            "passed": bool(self.passed),
            "scope_note": self.scope_note,
        }


def _failure_reason(
    *,
    t11_status: str,
    failed_obligations: Sequence[str],
    certificate_valid: bool,
    postcondition_passed: bool,
) -> str:
    reasons: list[str] = []
    if t11_status != "runtime_linked":
        reasons.append("t11_not_runtime_linked")
    if failed_obligations:
        reasons.append("t11_failed_obligations:" + _format_sequence(failed_obligations))
    if not certificate_valid:
        reasons.append("certificate_invalid")
    if not postcondition_passed:
        reasons.append("postcondition_failed")
    return "none" if not reasons else ";".join(reasons)


def _postcondition_from_row(row: Mapping[str, Any]) -> bool:
    if "domain_postcondition_passed" in row and not _is_missing(row.get("domain_postcondition_passed")):
        return _parse_bool(row.get("domain_postcondition_passed"))
    return not _parse_bool(row.get("true_constraint_violated"), default=True)


def build_domain_runtime_contract_witness(
    *,
    domain: str,
    trace_id: str,
    t11_status: str,
    failed_obligations: Sequence[str] = (),
    certificate_valid: bool,
    postcondition_passed: bool,
    post_margin: float | None,
    failure_reason: str | None = None,
    assumptions_used: Sequence[str] | None = None,
) -> DomainRuntimeContractWitness:
    domain_key = _domain_key(domain)
    failed = tuple(str(item) for item in failed_obligations if str(item))
    reason = failure_reason or _failure_reason(
        t11_status=str(t11_status),
        failed_obligations=failed,
        certificate_valid=bool(certificate_valid),
        postcondition_passed=bool(postcondition_passed),
    )
    return DomainRuntimeContractWitness(
        domain=domain_key,
        trace_id=str(trace_id),
        contract_id=contract_id_for_domain(domain_key),
        source_theorem=SOURCE_THEOREM,
        t11_status=str(t11_status or "missing"),
        failed_obligations=failed,
        certificate_valid=bool(certificate_valid),
        postcondition_passed=bool(postcondition_passed),
        post_margin=post_margin,
        failure_reason=reason,
        assumptions_used=tuple(assumptions_used or assumptions_for_domain(domain_key)),
    )


def witness_from_runtime_trace_row(
    row: Mapping[str, Any],
    *,
    domain: str | None = None,
) -> DomainRuntimeContractWitness:
    if domain is None:
        if not _is_missing(row.get("contract_id")):
            contract_text = str(row.get("contract_id"))
            domain = "av" if contract_text.startswith("AV.") else "healthcare"
        elif "scenario_id" in row:
            domain = "av"
        else:
            domain = "healthcare"
    domain_key = _domain_key(domain)
    t11_status = str(row.get("t11_status", "missing") or "missing")
    failed_obligations = _parse_sequence(row.get("t11_failed_obligations", ()))
    certificate_valid = _parse_bool(row.get("certificate_valid"), default=False)
    postcondition_passed = _postcondition_from_row(row)
    post_margin = _parse_float(row.get("post_margin", row.get("true_margin")))
    return build_domain_runtime_contract_witness(
        domain=domain_key,
        trace_id=str(row.get("trace_id", "")),
        t11_status=t11_status,
        failed_obligations=failed_obligations,
        certificate_valid=certificate_valid,
        postcondition_passed=postcondition_passed,
        post_margin=post_margin,
    )


def witness_trace_fields_from_result(
    *,
    domain: str,
    trace_id: str,
    theorem_contracts: Mapping[str, Any] | None,
    certificate_valid: bool,
    postcondition_passed: bool,
    post_margin: float | None,
) -> dict[str, Any]:
    t11 = dict((theorem_contracts or {}).get("T11", {}))
    witness = build_domain_runtime_contract_witness(
        domain=domain,
        trace_id=trace_id,
        t11_status=str(t11.get("status", "missing") or "missing"),
        failed_obligations=_parse_sequence(t11.get("failed_obligations", ())),
        certificate_valid=certificate_valid,
        postcondition_passed=postcondition_passed,
        post_margin=post_margin,
    )
    return witness.as_trace_fields()


def summarize_witnesses(witnesses: Iterable[DomainRuntimeContractWitness]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[DomainRuntimeContractWitness]] = {}
    for witness in witnesses:
        grouped.setdefault(witness.domain, []).append(witness)

    summary: dict[str, dict[str, Any]] = {}
    for domain, rows in sorted(grouped.items()):
        n_steps = len(rows)
        contract_ids = sorted({row.contract_id for row in rows})
        summary[domain] = {
            "n_steps": int(n_steps),
            "t11_pass_rate": float(sum(row.t11_passed for row in rows) / n_steps) if n_steps else 0.0,
            "postcondition_pass_rate": float(sum(row.postcondition_passed for row in rows) / n_steps) if n_steps else 0.0,
            "certificate_valid_rate": float(sum(row.certificate_valid for row in rows) / n_steps) if n_steps else 0.0,
            "witness_pass_rate": float(sum(row.passed for row in rows) / n_steps) if n_steps else 0.0,
            "contract_id": contract_ids[0] if len(contract_ids) == 1 else "|".join(contract_ids),
            "scope_note": scope_note_for_domain(domain),
        }
    return summary


def write_domain_runtime_contract_artifacts(
    witnesses: Sequence[DomainRuntimeContractWitness],
    *,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    witness_path = out_dir / "domain_runtime_contract_witnesses.csv"
    summary_path = out_dir / "domain_runtime_contract_summary.json"
    fieldnames = [
        "domain",
        "trace_id",
        "contract_id",
        "source_theorem",
        "t11_status",
        "failed_obligations",
        "certificate_valid",
        "postcondition_passed",
        "post_margin",
        "failure_reason",
        "assumptions_used",
        "passed",
        "scope_note",
    ]
    with witness_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for witness in witnesses:
            writer.writerow(witness.as_publication_row())

    summary_payload = {
        "source_theorem": SOURCE_THEOREM,
        "scope_note": (
            "Battery remains the deepest witness row. AV and healthcare are "
            "bounded runtime-lemma rows under forward-only T11."
        ),
        "domains": summarize_witnesses(witnesses),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "domain_runtime_contract_witnesses_csv": str(witness_path),
        "domain_runtime_contract_summary_json": str(summary_path),
    }


__all__ = [
    "AV_BRAKE_HOLD_CONTRACT_ID",
    "HEALTHCARE_FAIL_SAFE_CONTRACT_ID",
    "DomainRuntimeContractWitness",
    "build_domain_runtime_contract_witness",
    "contract_id_for_domain",
    "scope_note_for_domain",
    "summarize_witnesses",
    "witness_from_runtime_trace_row",
    "witness_trace_fields_from_result",
    "write_domain_runtime_contract_artifacts",
]
