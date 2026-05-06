#!/usr/bin/env python3
"""Validate that headline metric surfaces match the canonical runtime benchmark.

The publication-facing TSVR authority is
``reports/publication/three_domain_ml_benchmark.csv``.  Validation-harness rows
may remain as diagnostic surfaces, but they cannot appear in headline closure
tables or public summary prose.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = REPO_ROOT / "reports" / "publication" / "three_domain_ml_benchmark.csv"

DOMAIN_ALIASES = {
    "Battery Energy Storage": {"battery", "Battery", "Battery Energy Storage", "Battery (BESS)"},
    "Autonomous Vehicles": {"vehicle", "AV", "Autonomous Vehicles", "Vehicle (AV)"},
    "Medical and Healthcare Monitoring": {
        "healthcare",
        "Healthcare",
        "Medical and Healthcare Monitoring",
    },
}

HEADLINE_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "claim_ledger.md",
    REPO_ROOT / "docs" / "executive_summary.md",
    REPO_ROOT / "DATA.md",
    REPO_ROOT / "reports" / "universal_orius_validation" / "domain_validation_summary.csv",
    REPO_ROOT / "reports" / "universal_orius_validation" / "domain_closure_matrix.csv",
    REPO_ROOT / "reports" / "universal_orius_validation" / "tbl_domain_closure_matrix.tex",
    REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv",
    REPO_ROOT / "reports" / "publication" / "tbl_orius_domain_closure_matrix.tex",
    REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "domain_validation_summary.tex",
    REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "tbl_multi_domain_evidence_gate.tex",
    REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "tbl_all_domain_comparison.tex",
    REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "tbl_av_replay_surface.tex",
    REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "tbl_healthcare_replay_surface.tex",
]

FORBIDDEN_HEADLINE_PATTERNS = [
    (re.compile(r"\bTSVR\s+0\.1250?\s*(?:->|→)\s*0\.0417\b"), "old AV validation-harness TSVR headline"),
    (
        re.compile(r"\bTSVR\s+0\.2917\s*(?:->|→)\s*0\.0417\b"),
        "old healthcare validation-harness TSVR headline",
    ),
    (re.compile(r"\b0\.2276\s*(?:->|→)\s*0\.0000\b"), "old AV baseline headline"),
    (re.compile(r"\b0\.0393\s*(?:->|→)\s*0\.0000\b"), "old battery baseline headline"),
    (
        re.compile(r"\bbaseline TSVR 0\.1250\s*(?:->|→)\s*ORIUS TSVR 0\.0417\b", re.I),
        "old AV replay headline",
    ),
    (
        re.compile(r"\bbaseline TSVR 0\.2917\s*(?:->|→)\s*ORIUS TSVR 0\.0417\b", re.I),
        "old healthcare replay headline",
    ),
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _canonical_rows() -> dict[str, dict[str, str]]:
    if not BENCHMARK.exists():
        raise FileNotFoundError(BENCHMARK)
    rows = {row["domain"]: row for row in _read_csv(BENCHMARK)}
    missing = set(DOMAIN_ALIASES) - set(rows)
    if missing:
        raise ValueError(f"Missing benchmark domains: {sorted(missing)}")
    return rows


def _domain_for_name(name: str) -> str | None:
    normalized = (name or "").strip()
    for domain, aliases in DOMAIN_ALIASES.items():
        if normalized in aliases:
            return domain
    return None


def _check_domain_validation_summary(findings: list[str], canonical: dict[str, dict[str, str]]) -> None:
    path = REPO_ROOT / "reports" / "universal_orius_validation" / "domain_validation_summary.csv"
    if not path.exists():
        findings.append(f"{path.relative_to(REPO_ROOT)} missing")
        return
    rows = _read_csv(path)
    by_domain = {row.get("domain", ""): row for row in rows}
    expected = {
        "battery": canonical["Battery Energy Storage"],
        "vehicle": canonical["Autonomous Vehicles"],
        "healthcare": canonical["Medical and Healthcare Monitoring"],
    }
    for key, bench in expected.items():
        row = by_domain.get(key)
        if row is None:
            findings.append(f"{path.relative_to(REPO_ROOT)} missing {key} row")
            continue
        baseline = float(row.get("baseline_tsvr_mean", "nan"))
        orius = float(row.get("orius_tsvr_mean", "nan"))
        if abs(baseline - float(bench["baseline_tsvr_mean"])) > 1e-4:
            findings.append(
                f"{path.relative_to(REPO_ROOT)} {key} baseline {baseline:.6f} != "
                f"{float(bench['baseline_tsvr_mean']):.6f}"
            )
        if abs(orius - float(bench["orius_tsvr_mean"])) > 1e-6:
            findings.append(
                f"{path.relative_to(REPO_ROOT)} {key} ORIUS {orius:.6f} != "
                f"{float(bench['orius_tsvr_mean']):.6f}"
            )
        surface = row.get("metric_surface", "")
        expected_surface = bench["metric_surface"]
        if surface != expected_surface:
            findings.append(
                f"{path.relative_to(REPO_ROOT)} {key} metric_surface={surface!r} != {expected_surface!r}"
            )


def _check_publication_closure(findings: list[str], canonical: dict[str, dict[str, str]]) -> None:
    path = REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv"
    if not path.exists():
        findings.append(f"{path.relative_to(REPO_ROOT)} missing")
        return
    rows = {row["domain"]: row for row in _read_csv(path)}
    for domain, bench in canonical.items():
        row = rows.get(domain)
        if row is None:
            findings.append(f"{path.relative_to(REPO_ROOT)} missing {domain}")
            continue
        expected_baseline = f"{float(bench['baseline_tsvr_mean']):.4f}"
        expected_orius = f"{float(bench['orius_tsvr_mean']):.4f}"
        if row.get("baseline_tsvr") != expected_baseline:
            findings.append(
                f"{path.relative_to(REPO_ROOT)} {domain} baseline_tsvr={row.get('baseline_tsvr')!r} "
                f"!= {expected_baseline!r}"
            )
        if row.get("orius_tsvr") != expected_orius:
            findings.append(
                f"{path.relative_to(REPO_ROOT)} {domain} orius_tsvr={row.get('orius_tsvr')!r} "
                f"!= {expected_orius!r}"
            )
        status = row.get("current_status", "")
        if domain != "Battery Energy Storage" and (
            "runtime denominator" not in status or "secondary proxy" not in status
        ):
            findings.append(
                f"{path.relative_to(REPO_ROOT)} {domain} must disclose runtime denominator and secondary proxy"
            )


def _check_forbidden_headline_text(findings: list[str]) -> None:
    for path in HEADLINE_FILES:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern, label in FORBIDDEN_HEADLINE_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path.relative_to(REPO_ROOT)} contains {label}")


def main() -> int:
    try:
        canonical = _canonical_rows()
    except Exception as exc:  # pragma: no cover - defensive CLI surface
        print(f"[validate_metric_consistency] FAIL: {exc}")
        return 1

    findings: list[str] = []
    _check_domain_validation_summary(findings, canonical)
    _check_publication_closure(findings, canonical)
    _check_forbidden_headline_text(findings)

    if findings:
        print("[validate_metric_consistency] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_metric_consistency] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
