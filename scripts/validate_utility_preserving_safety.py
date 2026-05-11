#!/usr/bin/env python3
"""Validate the ORIUS utility-preserving safety scorecard."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "reports" / "publication" / "utility_preserving_safety_scorecard.csv"

REQUIRED_DOMAINS = {
    "Battery Energy Storage",
    "Autonomous Vehicles",
    "Medical and Healthcare Monitoring",
}


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in {None, ""}:
            return default
        if str(value).lower() == "inf":
            return math.inf
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def validate_scorecard(path: Path = DEFAULT_SCORECARD) -> list[str]:
    findings: list[str] = []
    rows = _rows(path)
    if not rows:
        return [f"missing or empty scorecard: {path}"]

    by_domain = {row.get("domain", ""): row for row in rows}
    missing = sorted(REQUIRED_DOMAINS - set(by_domain))
    if missing:
        findings.append(f"missing utility-preserving safety rows: {', '.join(missing)}")

    for domain in sorted(REQUIRED_DOMAINS & set(by_domain)):
        row = by_domain[domain]
        gate = str(row.get("utility_preserving_safety_gate", "")).lower() == "true"
        excess = _safe_float(row.get("excess_tsvr_over_safety_reference"), default=1.0)
        utility_delta = _safe_float(row.get("utility_delta_over_safety_reference"), default=-1.0)
        if not gate:
            findings.append(f"{domain}: utility_preserving_safety_gate is not True")
        if excess > 1e-3:
            findings.append(f"{domain}: excess TSVR over fail-safe reference is too high ({excess:.6f})")
        if utility_delta <= 0.0:
            findings.append(f"{domain}: ORIUS does not preserve more utility than fail-safe reference")

        if domain in {"Autonomous Vehicles", "Medical and Healthcare Monitoring"}:
            fallback_reduction = _safe_float(row.get("fallback_reduction_vs_safety_reference"), default=0.0)
            intervention_reduction = _safe_float(
                row.get("intervention_reduction_vs_safety_reference"), default=0.0
            )
            if fallback_reduction <= 0.0:
                findings.append(f"{domain}: fallback reduction vs fail-safe reference is not positive")
            if intervention_reduction <= 0.0:
                findings.append(f"{domain}: intervention reduction vs fail-safe reference is not positive")

        boundary = str(row.get("claim_boundary", "")).lower()
        forbidden = [
            "road deployment",
            "live clinical deployment",
            "field certification",
        ]
        if any(term in boundary for term in forbidden) and "not " not in boundary:
            findings.append(f"{domain}: claim boundary appears to overclaim deployment status")

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    args = parser.parse_args()

    path = args.scorecard if args.scorecard.is_absolute() else REPO_ROOT / args.scorecard
    findings = validate_scorecard(path)
    if findings:
        print("[validate_utility_preserving_safety] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_utility_preserving_safety] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
