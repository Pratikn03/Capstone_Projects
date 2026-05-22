#!/usr/bin/env python3
"""Validate the ORIUS utility-preserving safety scorecard."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORECARD = REPO_ROOT / "reports" / "publication" / "utility_preserving_safety_scorecard.csv"
DEFAULT_CLAIM_TABLE = REPO_ROOT / "reports" / "publication" / "utility_preserving_safety_claim_table.csv"
DEFAULT_ABLATION_SURFACES = (
    REPO_ROOT / "reports" / "publication" / "utility_preserving_safety_ablation_surfaces.csv"
)

REQUIRED_DOMAINS = {
    "Battery Energy Storage",
    "Autonomous Vehicles",
    "Medical and Healthcare Monitoring",
}

REQUIRED_CLAIM_COMPARISONS = {
    "predictor_only_safety",
    "shutdown_or_fallback_only_conservatism",
}

REQUIRED_ABLATION_SURFACES = {
    "no_reliability",
    "no_uncertainty",
    "no_repair",
    "no_fallback",
    "no_certificate_gate",
    "no_signature_hash_gate",
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


def validate_claim_table(path: Path = DEFAULT_CLAIM_TABLE) -> list[str]:
    findings: list[str] = []
    rows = _rows(path)
    if not rows:
        return [f"missing or empty claim comparison table: {path}"]

    by_key = {(row.get("domain", ""), row.get("comparison", "")): row for row in rows}
    missing = sorted(
        f"{domain}/{comparison}"
        for domain in REQUIRED_DOMAINS
        for comparison in REQUIRED_CLAIM_COMPARISONS
        if (domain, comparison) not in by_key
    )
    if missing:
        findings.append(f"missing claim comparison rows: {', '.join(missing)}")

    for domain in {"Autonomous Vehicles", "Medical and Healthcare Monitoring"}:
        row = by_key.get((domain, "predictor_only_safety"))
        if not row:
            continue
        if row.get("claim_relation") != "safer_than_predictor_only":
            findings.append(f"{domain}: predictor-only row does not show ORIUS as safer")
        if _safe_float(row.get("absolute_tsvr_reduction"), default=-1.0) <= 0.0:
            findings.append(f"{domain}: predictor-only TSVR reduction is not positive")

    battery = by_key.get(("Battery Energy Storage", "predictor_only_safety"))
    if battery and battery.get("claim_relation") == "safer_than_predictor_only":
        if _safe_float(battery.get("absolute_tsvr_reduction"), default=0.0) <= 0.0:
            findings.append("Battery Energy Storage: safer-than-predictor claim has no positive TSVR reduction")

    for domain in REQUIRED_DOMAINS:
        row = by_key.get((domain, "shutdown_or_fallback_only_conservatism"))
        if not row:
            continue
        if row.get("claim_relation") != "less_conservative_than_shutdown_or_fallback_only":
            findings.append(f"{domain}: fail-safe conservatism row is not claimable")
        if _safe_float(row.get("utility_delta"), default=-1.0) <= 0.0:
            findings.append(f"{domain}: fail-safe utility delta is not positive")

    return findings


def validate_ablation_surfaces(path: Path = DEFAULT_ABLATION_SURFACES) -> list[str]:
    findings: list[str] = []
    rows = _rows(path)
    if not rows:
        return [f"missing or empty ablation surface table: {path}"]

    present = {row.get("requested_surface", "") for row in rows}
    missing_surfaces = sorted(REQUIRED_ABLATION_SURFACES - present)
    if missing_surfaces:
        findings.append(f"missing ablation surfaces: {', '.join(missing_surfaces)}")

    rows_by_surface: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_surface.setdefault(row.get("requested_surface", ""), []).append(row)

    for surface in {"no_reliability", "no_repair"}:
        domains = {row.get("domain", "") for row in rows_by_surface.get(surface, [])}
        missing_domains = sorted(REQUIRED_DOMAINS - domains)
        if missing_domains:
            findings.append(f"{surface}: missing domain rows: {', '.join(missing_domains)}")
        for row in rows_by_surface.get(surface, []):
            if row.get("comparability") != "comparable_runtime_native":
                findings.append(f"{surface}/{row.get('domain', '')}: expected comparable runtime-native evidence")
            if _safe_float(row.get("absolute_tsvr_reduction"), default=-1.0) < 0.0:
                findings.append(f"{surface}/{row.get('domain', '')}: negative TSVR reduction")

    for surface in {"no_uncertainty", "no_fallback", "no_certificate_gate", "no_signature_hash_gate"}:
        for row in rows_by_surface.get(surface, []):
            if not row.get("comparability", "").startswith("non_comparable"):
                findings.append(f"{surface}/{row.get('domain', '')}: non-isolated surface must be marked non-comparable")

    signature_rows = rows_by_surface.get("no_signature_hash_gate", [])
    if signature_rows and all(row.get("evidence_surface") != "governance_fail_closed" for row in signature_rows):
        findings.append("no_signature_hash_gate: missing governance fail-closed evidence")

    return findings


def validate_utility_safety_outputs(
    scorecard: Path = DEFAULT_SCORECARD,
    claim_table: Path = DEFAULT_CLAIM_TABLE,
    ablation_surfaces: Path = DEFAULT_ABLATION_SURFACES,
) -> list[str]:
    findings = validate_scorecard(scorecard)
    findings.extend(validate_claim_table(claim_table))
    findings.extend(validate_ablation_surfaces(ablation_surfaces))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--claim-table", type=Path, default=DEFAULT_CLAIM_TABLE)
    parser.add_argument("--ablation-surfaces", type=Path, default=DEFAULT_ABLATION_SURFACES)
    args = parser.parse_args()

    scorecard = args.scorecard if args.scorecard.is_absolute() else REPO_ROOT / args.scorecard
    claim_table = args.claim_table if args.claim_table.is_absolute() else REPO_ROOT / args.claim_table
    ablation_surfaces = (
        args.ablation_surfaces if args.ablation_surfaces.is_absolute() else REPO_ROOT / args.ablation_surfaces
    )
    findings = validate_utility_safety_outputs(scorecard, claim_table, ablation_surfaces)
    if findings:
        print("[validate_utility_preserving_safety] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_utility_preserving_safety] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
