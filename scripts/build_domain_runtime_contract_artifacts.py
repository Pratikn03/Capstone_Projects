#!/usr/bin/env python3
"""Build publication-facing bounded T11 domain runtime contract witnesses."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import duckdb


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.universal_theory.domain_runtime_contracts import (  # noqa: E402
    DomainRuntimeContractWitness,
    contract_id_for_domain,
    witness_from_runtime_trace_row,
    write_domain_runtime_contract_artifacts,
)


DEFAULT_AV_RUNTIME_DIR = REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
DEFAULT_AV_TRACE = DEFAULT_AV_RUNTIME_DIR / "runtime_traces.csv"
DEFAULT_HEALTHCARE_TRACE = REPO_ROOT / "reports" / "healthcare" / "runtime_traces.csv"
DEFAULT_AV_CERT_DB = DEFAULT_AV_RUNTIME_DIR / "dc3s_av_waymo_dryrun.duckdb"
DEFAULT_HEALTHCARE_CERT_DB = REPO_ROOT / "reports" / "healthcare" / "healthcare_runtime.duckdb"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "publication"
TRACE_CONTRACT_FIELDS = [
    "contract_id",
    "source_theorem",
    "t11_status",
    "t11_failed_obligations",
    "domain_postcondition_passed",
    "domain_postcondition_failure",
]


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _is_orius_row(row: dict[str, Any]) -> bool:
    return str(row.get("controller", "")).strip().lower() == "orius"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "missing", "nan", "none", "null"}


def _load_t11_contracts_from_certificates(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        table_names = {row[0] for row in conn.execute("PRAGMA show_tables").fetchall()}
        if "dispatch_certificates" not in table_names:
            return []
        payload_rows = conn.execute(
            """
            SELECT payload_json
            FROM dispatch_certificates
            ORDER BY created_at, command_id
            """
        ).fetchall()
    finally:
        conn.close()

    contracts: list[dict[str, Any]] = []
    for (payload_json,) in payload_rows:
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError):
            contracts.append({})
            continue
        theorem_contracts = payload.get("theorem_contracts")
        contracts.append(dict(theorem_contracts.get("T11", {})) if isinstance(theorem_contracts, dict) else {})
    return contracts


def _apply_recovered_t11_contract(row: dict[str, Any], t11_contract: dict[str, Any]) -> None:
    if not _is_missing(row.get("t11_status")):
        return
    status = str(t11_contract.get("status", "missing") or "missing")
    failed = t11_contract.get("failed_obligations", [])
    if isinstance(failed, list):
        failed_text = "|".join(str(item) for item in failed if str(item))
    else:
        failed_text = str(failed or "")
    row["t11_status"] = status
    row["t11_failed_obligations"] = failed_text


def normalize_trace_rows(
    rows: list[dict[str, Any]],
    *,
    domain: str,
    certificate_t11_contracts: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[DomainRuntimeContractWitness]]:
    witnesses: list[DomainRuntimeContractWitness] = []
    contract_id = contract_id_for_domain(domain)
    certificate_contracts = certificate_t11_contracts or []
    orius_count = sum(1 for row in rows if _is_orius_row(row))
    can_recover_t11 = bool(certificate_contracts) and len(certificate_contracts) == orius_count
    orius_index = 0
    for row in rows:
        if _is_orius_row(row):
            if can_recover_t11:
                _apply_recovered_t11_contract(row, certificate_contracts[orius_index])
            orius_index += 1
            witness = witness_from_runtime_trace_row(row, domain=domain)
            row.update(witness.as_trace_fields())
            witnesses.append(witness)
        else:
            row.setdefault("contract_id", contract_id)
            row.setdefault("source_theorem", "T11")
            row.setdefault("t11_status", "missing")
            row.setdefault("t11_failed_obligations", "")
            row.setdefault(
                "domain_postcondition_passed",
                str(row.get("true_constraint_violated", "")).strip().lower() in {"false", "0"},
            )
            row.setdefault("domain_postcondition_failure", "non_orius_controller")
    return rows, witnesses


def build_domain_runtime_contract_artifacts(
    *,
    av_trace: Path = DEFAULT_AV_TRACE,
    healthcare_trace: Path = DEFAULT_HEALTHCARE_TRACE,
    av_cert_db: Path = DEFAULT_AV_CERT_DB,
    healthcare_cert_db: Path = DEFAULT_HEALTHCARE_CERT_DB,
    out_dir: Path = DEFAULT_OUT_DIR,
    normalize_traces: bool = True,
    recover_t11_from_certificates: bool = True,
) -> dict[str, Any]:
    witnesses: list[DomainRuntimeContractWitness] = []
    normalized_counts: dict[str, int] = {}

    for domain, trace_path, cert_db_path in (
        ("av", av_trace, av_cert_db),
        ("healthcare", healthcare_trace, healthcare_cert_db),
    ):
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing {domain} runtime trace: {trace_path}")
        if normalize_traces:
            rows = _read_rows(trace_path)
            certificate_contracts = (
                _load_t11_contracts_from_certificates(cert_db_path)
                if recover_t11_from_certificates
                else []
            )
            normalized, domain_witnesses = normalize_trace_rows(
                rows,
                domain=domain,
                certificate_t11_contracts=certificate_contracts,
            )
            witnesses.extend(domain_witnesses)
            normalized_counts[domain] = len(domain_witnesses)
            _write_rows(trace_path, normalized)
        else:
            domain_count = 0
            for row in _iter_rows(trace_path):
                if not _is_orius_row(row):
                    continue
                witnesses.append(witness_from_runtime_trace_row(row, domain=domain))
                domain_count += 1
            normalized_counts[domain] = domain_count

    outputs = write_domain_runtime_contract_artifacts(witnesses, out_dir=out_dir)
    return {
        **outputs,
        "n_witnesses": int(len(witnesses)),
        "normalized_witness_counts": normalized_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--av-trace", type=Path, default=DEFAULT_AV_TRACE)
    parser.add_argument("--healthcare-trace", type=Path, default=DEFAULT_HEALTHCARE_TRACE)
    parser.add_argument("--av-cert-db", type=Path, default=DEFAULT_AV_CERT_DB)
    parser.add_argument("--healthcare-cert-db", type=Path, default=DEFAULT_HEALTHCARE_CERT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-normalize-traces", action="store_true")
    parser.add_argument("--no-certificate-recovery", action="store_true")
    args = parser.parse_args()

    report = build_domain_runtime_contract_artifacts(
        av_trace=args.av_trace,
        healthcare_trace=args.healthcare_trace,
        av_cert_db=args.av_cert_db,
        healthcare_cert_db=args.healthcare_cert_db,
        out_dir=args.out_dir,
        normalize_traces=not args.no_normalize_traces,
        recover_t11_from_certificates=not args.no_certificate_recovery,
    )
    print(
        "[domain-runtime-contracts] "
        f"witnesses={report['n_witnesses']} "
        f"av={report['normalized_witness_counts'].get('av', 0)} "
        f"healthcare={report['normalized_witness_counts'].get('healthcare', 0)} "
        f"summary={report['domain_runtime_contract_summary_json']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
