#!/usr/bin/env python3
"""Sync and verify paper assets against canonical data sources.

Catches stale values (e.g. legacy US row counts), verifies Table 13
controller coverage, checks claim-matrix gaps, and validates
metrics completeness.

Usage:
    python scripts/sync_paper_assets.py --check       # report mismatches
    python scripts/sync_paper_assets.py --check --json # machine-readable
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Known stale values that MUST NOT appear in paper assets
KNOWN_STALE_VALUES: dict[str, dict[str, Any]] = {
    "us_row_count_legacy": {
        "stale_variants": ["92,382", "92382", "92_382"],
        "description": "Legacy US EIA-930 row count (replaced by 13,638)",
    },
}

PAPER_ASSET_PATHS = [
    "paper/assets/tables/tbl05_dataset_summary.csv",
    "paper/metrics_manifest.json",
    "paper/claim_matrix.csv",
    "paper/PAPER_DRAFT.md",
    "paper/paper.tex",
]


def _check_file_for_stale(filepath: Path, stale_variants: list[str]) -> list[dict[str, Any]]:
    if not filepath.exists():
        return []
    content = filepath.read_text(encoding="utf-8")
    issues: list[dict[str, Any]] = []
    for i, line in enumerate(content.splitlines(), 1):
        for variant in stale_variants:
            if variant in line:
                issues.append({
                    "file": str(filepath.relative_to(REPO_ROOT)),
                    "line": i,
                    "stale_value": variant,
                    "line_content": line.strip()[:120],
                })
    return issues


def _check_table13_controller_coverage() -> list[dict[str, Any]]:
    main_table = REPO_ROOT / "reports" / "publication" / "dc3s_main_table.csv"
    summary_table = REPO_ROOT / "reports" / "publication" / "table13_dc3s_summary.csv"
    issues: list[dict[str, Any]] = []

    if not main_table.exists():
        issues.append({"check": "table13_coverage", "error": f"Not found: {main_table.relative_to(REPO_ROOT)}"})
        return issues

    with open(main_table, encoding="utf-8") as fh:
        main_controllers = {row.get("controller", "") for row in csv.DictReader(fh)}

    if not summary_table.exists():
        issues.append({
            "check": "table13_coverage",
            "error": f"Summary not found: {summary_table.relative_to(REPO_ROOT)}",
            "main_controllers": sorted(main_controllers),
        })
        return issues

    with open(summary_table, encoding="utf-8") as fh:
        summary_controllers = {row.get("controller", "") for row in csv.DictReader(fh)}

    missing = main_controllers - summary_controllers
    if missing:
        issues.append({
            "check": "table13_coverage",
            "error": f"Controllers in main but missing in summary: {sorted(missing)}",
            "main_controllers": sorted(main_controllers),
            "summary_controllers": sorted(summary_controllers),
        })
    return issues


def _check_claim_matrix_gaps() -> list[dict[str, Any]]:
    path = REPO_ROOT / "paper" / "claim_matrix.csv"
    if not path.exists():
        return [{"check": "claim_matrix", "error": "claim_matrix.csv not found"}]
    issues: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            status = row.get("status", "").strip()
            locations = row.get("manuscript_locations", "").strip()
            if status.lower() in ("unsupported", "needs citation", "needs_citation"):
                in_manuscript = locations.upper() != "NOT PRESENT"
                issues.append({
                    "check": "claim_matrix",
                    "claim_id": row.get("claim_id", ""),
                    "status": status,
                    "in_manuscript": in_manuscript,
                    "claim": row.get("claim_text", "")[:80],
                })
    return issues


def _check_picp_headline() -> list[dict[str, Any]]:
    """Flag the stale 33.8% → 38.3% PICP headline in paper.tex."""
    tex = REPO_ROOT / "paper" / "paper.tex"
    if not tex.exists():
        return []
    content = tex.read_text(encoding="utf-8")
    issues: list[dict[str, Any]] = []
    for i, line in enumerate(content.splitlines(), 1):
        if "33.8" in line and "38.3" in line:
            issues.append({
                "check": "picp_headline",
                "file": "paper/paper.tex",
                "line": i,
                "error": "Stale PICP headline (33.8% → 38.3%) must be updated from R1 run family",
                "line_content": line.strip()[:120],
            })
    return issues


def _check_metrics_completeness() -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    metrics_path = REPO_ROOT / "reports" / "week2_metrics.json"
    if not metrics_path.exists():
        issues.append({"check": "metrics_completeness", "error": "reports/week2_metrics.json not found"})
        return issues
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        issues.append({"check": "metrics_completeness", "error": "week2_metrics.json is invalid JSON"})
        return issues
    targets = list((metrics.get("targets") or {}).keys()) if isinstance(metrics.get("targets"), dict) else []
    expected_de = ["load_mw", "solar_mw", "wind_mw"]
    missing = [t for t in expected_de if t not in targets]
    if missing:
        issues.append({
            "check": "metrics_completeness",
            "error": f"Missing DE targets in week2_metrics.json: {missing}",
            "found_targets": targets,
        })
    return issues


def _check_uq_contract_consistency() -> list[dict[str, Any]]:
    """Verify that standard UQ metrics appear in report schemas."""
    required_metrics = {"picp_90", "picp_95", "mean_interval_width", "pinball_loss", "winkler_score"}
    issues: list[dict[str, Any]] = []
    # Check dc3s_main_table columns
    main_table = REPO_ROOT / "reports" / "publication" / "dc3s_main_table.csv"
    if main_table.exists():
        with open(main_table, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            available = set(reader.fieldnames or [])
        missing = {"picp_90", "mean_interval_width"} - available
        if missing:
            issues.append({
                "check": "uq_contract",
                "error": f"dc3s_main_table.csv missing standard UQ columns: {sorted(missing)}",
            })
    return issues


def run_checks() -> list[dict[str, Any]]:
    """Run all paper-asset consistency checks."""
    all_issues: list[dict[str, Any]] = []

    # Stale values
    for key, spec in KNOWN_STALE_VALUES.items():
        for asset_path in PAPER_ASSET_PATHS:
            for issue in _check_file_for_stale(REPO_ROOT / asset_path, spec["stale_variants"]):
                issue["stale_key"] = key
                issue["description"] = spec["description"]
                all_issues.append(issue)

    all_issues.extend(_check_table13_controller_coverage())
    all_issues.extend(_check_claim_matrix_gaps())
    all_issues.extend(_check_picp_headline())
    all_issues.extend(_check_metrics_completeness())
    all_issues.extend(_check_uq_contract_consistency())

    return all_issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync and verify paper assets")
    parser.add_argument("--check", action="store_true", default=True, help="Run checks")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    issues = run_checks()

    if args.json:
        print(json.dumps(issues, indent=2))
    else:
        if not issues:
            print("✅ All paper asset checks passed")
        else:
            print(f"⚠️  Found {len(issues)} issue(s):\n")
            for i, issue in enumerate(issues, 1):
                check = issue.get("check", issue.get("stale_key", "unknown"))
                error = issue.get("error", "")
                filepath = issue.get("file", "")
                line = issue.get("line", "")
                stale = issue.get("stale_value", "")
                if filepath and stale:
                    print(f"  {i}. [{check}] {filepath}:{line} — stale '{stale}'")
                    print(f"     {issue.get('line_content', '')}")
                elif error:
                    print(f"  {i}. [{check}] {error}")
                else:
                    print(f"  {i}. {issue}")
                print()

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
