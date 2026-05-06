#!/usr/bin/env python3
"""Validate canonical ORIUS runtime release-contract witness artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    normalize_release_evidence,
    validate_release_evidence,
)
from orius.security.policy import certificate_signature_required

DEFAULT_CSV = REPO_ROOT / "reports" / "publication" / "runtime_release_contract_witnesses.csv"
DEFAULT_JSON = REPO_ROOT / "reports" / "publication" / "runtime_release_contract_witnesses.json"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], [], [f"csv artifact missing: {path.relative_to(REPO_ROOT)}"]
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        return [], [], [f"csv artifact is malformed: {exc}"]
    return fields, rows, []


def _read_json(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"json artifact missing: {path.relative_to(REPO_ROOT)}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [f"json artifact is malformed: {exc}"]
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return [], ["json artifact must contain a rows list"]
    return [dict(row) for row in rows if isinstance(row, dict)], []


def _strict_default() -> bool:
    return certificate_signature_required() or os.getenv("ORIUS_REQUIRE_APPEND_ONLY_AUDIT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def validate(
    *,
    csv_path: Path = DEFAULT_CSV,
    json_path: Path = DEFAULT_JSON,
    strict: bool | None = None,
) -> list[str]:
    findings: list[str] = []
    strict_mode = _strict_default() if strict is None else bool(strict)
    fields, rows, csv_findings = _read_csv(csv_path)
    findings.extend(csv_findings)
    if fields:
        missing = [field for field in CANONICAL_RELEASE_EVIDENCE_FIELDS if field not in fields]
        if missing:
            findings.append(f"csv artifact missing columns: {missing}")
    if not rows and not csv_findings:
        findings.append("csv artifact has no rows")

    json_rows, json_findings = _read_json(json_path)
    findings.extend(json_findings)
    if json_rows and rows and len(json_rows) != len(rows):
        findings.append(f"json/csv row count mismatch: json={len(json_rows)} csv={len(rows)}")

    observed_domains: set[str] = set()
    for idx, row in enumerate(rows, start=2):
        row_findings = validate_release_evidence(row, strict=strict_mode)
        try:
            normalized = normalize_release_evidence(row)
            observed_domains.add(normalized["domain"])
        except Exception:
            normalized = {"domain": str(row.get("domain", ""))}
        for finding in row_findings:
            findings.append(f"row {idx} domain={normalized.get('domain')}: {finding}")

    missing_domains = sorted(set(PROMOTED_RELEASE_DOMAINS) - observed_domains)
    if missing_domains:
        findings.append(f"missing promoted domains: {missing_domains}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--strict", action="store_true", help="Require signed append-only releases.")
    parser.add_argument("--non-strict", action="store_true", help="Ignore strict env-derived release checks.")
    args = parser.parse_args()
    strict: bool | None
    if args.strict and args.non_strict:
        raise SystemExit("--strict and --non-strict are mutually exclusive")
    if args.strict:
        strict = True
    elif args.non_strict:
        strict = False
    else:
        strict = None
    findings = validate(csv_path=args.csv, json_path=args.json, strict=strict)
    if findings:
        print("[validate_runtime_release_contract] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_runtime_release_contract] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
