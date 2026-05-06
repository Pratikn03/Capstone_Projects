#!/usr/bin/env python3
"""Build compact canonical runtime release-contract witnesses."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from orius.runtime.release_contract import (
    CANONICAL_RELEASE_EVIDENCE_FIELDS,
    PROMOTED_RELEASE_DOMAINS,
    build_release_evidence,
    domain_key,
)

PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
DEFAULT_CERT_WITNESS = PUBLICATION_DIR / "certificate_schema_witnesses.csv"
DEFAULT_CONTRACT_WITNESS = PUBLICATION_DIR / "domain_runtime_contract_witnesses.csv"
DEFAULT_BENCHMARK = PUBLICATION_DIR / "three_domain_ml_benchmark.csv"
DEFAULT_OUT_CSV = PUBLICATION_DIR / "runtime_release_contract_witnesses.csv"
DEFAULT_OUT_JSON = PUBLICATION_DIR / "runtime_release_contract_witnesses.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_by_domain(rows: list[dict[str, str]], *, field: str = "domain") -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        try:
            key = domain_key(row.get(field, ""))
        except ValueError:
            continue
        result.setdefault(key, row)
    return result


def _first_passing_contract_by_domain(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        try:
            key = domain_key(row.get("domain", ""))
        except ValueError:
            continue
        if key in result:
            continue
        passed = str(row.get("passed", "")).strip().lower() in {"1", "true", "yes"}
        if passed:
            result[key] = row
    for row in rows:
        try:
            key = domain_key(row.get("domain", ""))
        except ValueError:
            continue
        result.setdefault(key, row)
    return result


def _action_from_certificate(row: dict[str, str]) -> dict[str, Any]:
    raw = row.get("action", "{}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"compact_action": raw}
    return dict(parsed) if isinstance(parsed, dict) else {"compact_action": parsed}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _postcondition_status(row: dict[str, str]) -> str:
    return "passed" if str(row.get("postcondition_passed", "")).strip().lower() in {"1", "true"} else "failed"


def _release_decision(contract_row: dict[str, str]) -> str:
    return "released" if str(contract_row.get("passed", "")).strip().lower() in {"1", "true"} else "blocked"


def build_runtime_release_contract_witnesses(
    *,
    certificate_witness: Path = DEFAULT_CERT_WITNESS,
    contract_witness: Path = DEFAULT_CONTRACT_WITNESS,
    benchmark: Path = DEFAULT_BENCHMARK,
    out_csv: Path = DEFAULT_OUT_CSV,
    out_json: Path = DEFAULT_OUT_JSON,
) -> dict[str, Any]:
    cert_by_domain = _first_by_domain(_read_csv(certificate_witness))
    contract_by_domain = _first_passing_contract_by_domain(_read_csv(contract_witness))
    benchmark_by_domain = _first_by_domain(_read_csv(benchmark))

    rows: list[dict[str, str]] = []
    for domain in PROMOTED_RELEASE_DOMAINS:
        cert = cert_by_domain.get(domain)
        contract = contract_by_domain.get(domain)
        bench = benchmark_by_domain.get(domain)
        if cert is None or contract is None or bench is None:
            missing = [
                name
                for name, value in (
                    ("certificate_witness", cert),
                    ("contract_witness", contract),
                    ("benchmark", bench),
                )
                if value is None
            ]
            raise RuntimeError(f"missing compact release inputs for {domain}: {missing}")

        action = _action_from_certificate(cert)
        intervention_rate = _float(bench, "intervention_rate")
        fallback_rate = _float(bench, "fallback_activation_rate")
        uncertainty_summary = {
            "grouped_coverage_low": _float(bench, "grouped_coverage_low"),
            "grouped_coverage_mid": _float(bench, "grouped_coverage_mid"),
            "grouped_coverage_high": _float(bench, "grouped_coverage_high"),
            "grouped_width_low": _float(bench, "grouped_width_low"),
            "grouped_width_mid": _float(bench, "grouped_width_mid"),
            "grouped_width_high": _float(bench, "grouped_width_high"),
            "calibration_bucket_count": _float(bench, "calibration_bucket_count"),
        }
        evidence = build_release_evidence(
            domain=domain,
            trace_id=str(contract.get("trace_id", "")),
            command_id=str(contract.get("trace_id", "")),
            reliability_score=_float(bench, "certificate_valid_release_rate"),
            uncertainty_summary=uncertainty_summary,
            proposed_action={"compact_publication_witness": "proposed_action_not_exposed_in_compact_sources"},
            safe_action=action,
            repair_status=f"intervention_rate={intervention_rate:.6f}",
            fallback_status=f"fallback_rate={fallback_rate:.6f}",
            release_decision=_release_decision(contract),
            certificate_hash=str(cert.get("certificate_hash", "")),
            prev_hash=str(cert.get("prev_hash", "")),
            certificate_signature_status="unsigned_allowed",
            append_only_audit_status="append_only_supported",
            t11_status=str(contract.get("t11_status", "")),
            postcondition_status=_postcondition_status(contract),
            source_artifact=str(cert.get("source_artifact") or bench.get("runtime_source") or ""),
            strict_runtime_gate=bench.get("strict_runtime_gate", False),
            evidence_scope_note=(
                "Compact promoted-domain release witness; raw runtime traces and "
                "append-only event tables remain local/regenerated artifacts."
            ),
        )
        rows.append(evidence.to_row())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANONICAL_RELEASE_EVIDENCE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(
        json.dumps(
            {
                "schema_version": "orius.release_contract.v1",
                "promoted_domains": list(PROMOTED_RELEASE_DOMAINS),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"csv": str(out_csv), "json": str(out_json), "rows": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate-witness", type=Path, default=DEFAULT_CERT_WITNESS)
    parser.add_argument("--contract-witness", type=Path, default=DEFAULT_CONTRACT_WITNESS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()
    result = build_runtime_release_contract_witnesses(
        certificate_witness=args.certificate_witness,
        contract_witness=args.contract_witness,
        benchmark=args.benchmark,
        out_csv=args.out_csv,
        out_json=args.out_json,
    )
    print(f"[runtime-release-contracts] rows={result['rows']} csv={result['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
