#!/usr/bin/env python3
"""Validate the T9/T10 research and proof-dependency package."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
REQUIRED_FAMILIES = {
    "lower_bounds",
    "mixing_processes",
    "runtime_assurance",
    "safety_filters",
    "conformal_shift",
    "av_validation",
    "battery_validation",
    "healthcare_validation",
}
REQUIRED_THEOREMS = {"T9", "T10"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_research_package(
    *,
    out_dir: Path = PUBLICATION_DIR,
    min_sources: int = 500,
) -> dict[str, Any]:
    findings: list[str] = []
    try:
        source_rows = _read_csv(out_dir / "t9_t10_research_source_matrix.csv")
        dependency_rows = _read_csv(out_dir / "t9_t10_proof_dependency_matrix.csv")
        scorecard = _read_json(out_dir / "t9_t10_research_scorecard.json")
    except FileNotFoundError as exc:
        return {"pass": False, "findings": [f"missing research package artifact: {exc.filename}"]}

    if len(source_rows) < min_sources:
        findings.append(f"source matrix has {len(source_rows)} rows; required at least {min_sources}")

    missing_provenance = [row.get("source_id", "") for row in source_rows if not row.get("provenance")]
    if missing_provenance:
        findings.append(f"source rows missing provenance: {missing_provenance[:10]}")

    missing_urls = [row.get("source_id", "") for row in source_rows if not row.get("doi_or_url")]
    if missing_urls:
        findings.append(f"source rows missing DOI/URL: {missing_urls[:10]}")

    topic_families = {row.get("topic_family", "") for row in source_rows}
    missing_families = sorted(REQUIRED_FAMILIES - topic_families)
    if missing_families:
        findings.append(f"missing required topic families: {missing_families}")

    keys = [(row.get("doi_or_url") or row.get("title", "")).strip().lower() for row in source_rows]
    unique_keys = {key for key in keys if key}
    if len(unique_keys) < int(0.9 * len(source_rows)):
        findings.append(
            "duplicate source padding suspected: "
            f"{len(unique_keys)} unique source keys for {len(source_rows)} rows"
        )

    source_ids = {row.get("source_id", "") for row in source_rows}
    dependency_theorems = {row.get("theorem_id", "") for row in dependency_rows}
    if not dependency_theorems >= REQUIRED_THEOREMS:
        findings.append(
            f"proof dependency matrix missing theorem rows: {sorted(REQUIRED_THEOREMS - dependency_theorems)}"
        )

    for row in dependency_rows:
        if not row.get("source_ids"):
            findings.append(f"{row.get('theorem_id')}:{row.get('proof_step')} has no source_ids")
            continue
        missing_refs = [item for item in row["source_ids"].split(";") if item and item not in source_ids]
        if missing_refs:
            findings.append(
                f"{row.get('theorem_id')}:{row.get('proof_step')} references unknown sources {missing_refs[:10]}"
            )
        if not row.get("code_anchor"):
            findings.append(f"{row.get('theorem_id')}:{row.get('proof_step')} missing code_anchor")
        if not row.get("artifact_anchor"):
            findings.append(f"{row.get('theorem_id')}:{row.get('proof_step')} missing artifact_anchor")

    if scorecard.get("source_count") != len(source_rows):
        findings.append("scorecard source_count does not match source matrix")
    if bool(scorecard.get("proof_dependency_complete")) is not True:
        findings.append("scorecard proof_dependency_complete must be true")
    if bool(scorecard.get("pass")) and findings:
        findings.append("scorecard pass=true conflicts with validator findings")

    return {
        "pass": not findings,
        "findings": findings,
        "source_count": len(source_rows),
        "unique_source_key_count": len(unique_keys),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=PUBLICATION_DIR)
    parser.add_argument("--min-sources", type=int, default=500)
    args = parser.parse_args()
    result = validate_research_package(out_dir=args.out_dir, min_sources=args.min_sources)
    print(
        "[validate_t9_t10_research_package] "
        f"{'PASS' if result['pass'] else 'FAIL'} source_count={result.get('source_count', 0)}"
    )
    for finding in result["findings"]:
        print(f"- {finding}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
