"""Validator tests for canonical runtime release contract artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_runtime_release_contract_witnesses import build_runtime_release_contract_witnesses
from scripts.validate_runtime_release_contract import validate

FIELDS = [
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
]


def _row(domain: str) -> dict[str, str]:
    return {
        "release_contract_schema_version": "orius.release_contract.v1",
        "domain": domain,
        "trace_id": f"{domain}-trace",
        "command_id": f"{domain}-command",
        "runtime_flow": "observe>reliability>uncertainty>repair_or_fallback>sign_certificate>append_only_audit",
        "reliability_score": "1.0",
        "uncertainty_summary": json.dumps({"coverage": 1.0}),
        "proposed_action": json.dumps({"raw": 0.0}),
        "safe_action": json.dumps({"safe": 0.0}),
        "repair_status": "not_repaired",
        "fallback_status": "not_fallback",
        "release_decision": "released",
        "certificate_hash": "b" * 64,
        "prev_hash": "GENESIS",
        "certificate_signature_status": "unsigned_allowed",
        "append_only_audit_status": "append_only_supported",
        "t11_status": "runtime_linked",
        "postcondition_status": "passed",
        "source_artifact": "reports/publication/test.csv",
        "strict_runtime_gate": "True",
        "evidence_scope_note": "test row",
    }


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


def _cert_rows() -> list[dict[str, str]]:
    return [
        {
            "domain": domain,
            "certificate_schema_version": "orius.certificate.v1",
            "certificate_hash": "c" * 64,
            "prev_hash": "GENESIS",
            "issuer": "orius.runtime",
            "action": json.dumps({"safe": domain}),
            "validity_horizon_H_t": "1",
            "expires_at_step": "1",
            "theorem_contracts": json.dumps({"T11": "runtime_linked"}),
            "source_artifact": f"reports/{domain}/runtime_summary.csv",
            "scope_note": "test cert",
        }
        for domain in ("battery", "av", "healthcare")
    ]


def _contract_rows() -> list[dict[str, str]]:
    return [
        {
            "domain": domain,
            "trace_id": f"{domain}-trace",
            "contract_id": f"{domain}.contract",
            "source_theorem": "T11",
            "t11_status": "runtime_linked",
            "failed_obligations": "",
            "certificate_valid": "True",
            "postcondition_passed": "True",
            "post_margin": "1.0",
            "failure_reason": "none",
            "assumptions_used": "T11.coverage",
            "passed": "True",
            "scope_note": "test contract",
        }
        for domain in ("battery", "av", "healthcare")
    ]


def _benchmark_rows() -> list[dict[str, str]]:
    names = {
        "battery": "Battery Energy Storage",
        "av": "Autonomous Vehicles",
        "healthcare": "Medical and Healthcare Monitoring",
    }
    return [
        {
            "domain": label,
            "certificate_valid_release_rate": "1.0",
            "intervention_rate": "0.1",
            "fallback_activation_rate": "0.0",
            "strict_runtime_gate": "True",
            "grouped_coverage_low": "0.9",
            "grouped_coverage_mid": "0.95",
            "grouped_coverage_high": "0.99",
            "grouped_width_low": "1.0",
            "grouped_width_mid": "2.0",
            "grouped_width_high": "3.0",
            "calibration_bucket_count": "3",
            "runtime_source": f"reports/{domain}/runtime_summary.csv",
        }
        for domain, label in names.items()
    ]


def test_validator_accepts_all_promoted_domains(tmp_path: Path) -> None:
    rows = [_row("battery"), _row("av"), _row("healthcare")]
    csv_path = tmp_path / "runtime_release_contract_witnesses.csv"
    json_path = tmp_path / "runtime_release_contract_witnesses.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)

    assert validate(csv_path=csv_path, json_path=json_path) == []


def test_builder_generates_valid_compact_release_contract_artifacts(tmp_path: Path) -> None:
    cert_path = tmp_path / "certs.csv"
    contract_path = tmp_path / "contracts.csv"
    benchmark_path = tmp_path / "benchmark.csv"
    out_csv = tmp_path / "out.csv"
    out_json = tmp_path / "out.json"
    _write_csv(cert_path, _cert_rows(), fields=list(_cert_rows()[0]))
    _write_csv(contract_path, _contract_rows(), fields=list(_contract_rows()[0]))
    _write_csv(benchmark_path, _benchmark_rows(), fields=list(_benchmark_rows()[0]))

    result = build_runtime_release_contract_witnesses(
        certificate_witness=cert_path,
        contract_witness=contract_path,
        benchmark=benchmark_path,
        out_csv=out_csv,
        out_json=out_json,
    )

    assert result["rows"] == 3
    assert validate(csv_path=out_csv, json_path=out_json) == []


def test_validator_rejects_missing_required_field(tmp_path: Path) -> None:
    fields = [field for field in FIELDS if field != "certificate_hash"]
    csv_path = tmp_path / "bad.csv"
    json_path = tmp_path / "bad.json"
    rows = [_row("battery"), _row("av"), _row("healthcare")]
    _write_csv(csv_path, rows, fields=fields)
    _write_json(json_path, rows)

    findings = validate(csv_path=csv_path, json_path=json_path)

    assert any("missing columns" in finding for finding in findings)


def test_validator_rejects_bad_certificate_hash_and_missing_t11(tmp_path: Path) -> None:
    rows = [_row("battery"), _row("av"), _row("healthcare")]
    rows[1]["certificate_hash"] = "not-a-hash"
    rows[2]["t11_status"] = ""
    csv_path = tmp_path / "bad.csv"
    json_path = tmp_path / "bad.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)

    findings = validate(csv_path=csv_path, json_path=json_path)

    assert any("invalid certificate_hash" in finding for finding in findings)
    assert any("missing t11_status" in finding for finding in findings)


def test_validator_rejects_invalid_postcondition_and_strict_unsigned(tmp_path: Path) -> None:
    rows = [_row("battery"), _row("av"), _row("healthcare")]
    rows[0]["postcondition_status"] = "failed"
    csv_path = tmp_path / "bad.csv"
    json_path = tmp_path / "bad.json"
    _write_csv(csv_path, rows)
    _write_json(json_path, rows)

    findings = validate(csv_path=csv_path, json_path=json_path, strict=True)

    assert any("postcondition_status must be passed" in finding for finding in findings)
    assert any("strict release requires signed certificate" in finding for finding in findings)
    assert any("strict release requires append-only audit" in finding for finding in findings)


def test_validator_rejects_corrupted_json_artifact(tmp_path: Path) -> None:
    rows = [_row("battery"), _row("av"), _row("healthcare")]
    csv_path = tmp_path / "ok.csv"
    json_path = tmp_path / "bad.json"
    _write_csv(csv_path, rows)
    json_path.write_text("{not-json", encoding="utf-8")

    findings = validate(csv_path=csv_path, json_path=json_path)

    assert any("json artifact is malformed" in finding for finding in findings)
