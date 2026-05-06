#!/usr/bin/env python3
"""Validate that Appendix B is the canonical A1-A13 assumption surface."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from orius.universal_theory.contracts import ASSUMPTION_REGISTER

APPENDIX_B = REPO_ROOT / "appendices" / "app_b_assumptions.tex"
DEFENDED_MAP = REPO_ROOT / "reports/publication/defended_assumption_map.csv"
THEOREM_REGISTRY = REPO_ROOT / "reports/publication/theorem_registry.yml"

EXPECTED_IDS = {
    "A1",
    "A2",
    "A3",
    "A4",
    "A5",
    "A6",
    "A7",
    "A8",
    "A9",
    "A10a",
    "A10b",
    "A11",
    "A12",
    "A13",
}

ASSUMPTION_RE = re.compile(r"\\begin\{assumption\}\[(A\d+[a-z]?)\s+[-\u2014]+")
STALE_NAMESPACE_RE = re.compile(r"A1\s*(?:--|-|\u2013)\s*A8")


def _appendix_ids() -> set[str]:
    return set(ASSUMPTION_RE.findall(APPENDIX_B.read_text(encoding="utf-8")))


def main() -> int:
    findings: list[str] = []
    appendix_ids = _appendix_ids()
    code_ids = set(ASSUMPTION_REGISTER)

    if appendix_ids != EXPECTED_IDS:
        findings.append(f"Appendix B assumption IDs drifted: {sorted(appendix_ids)}")
    if code_ids != EXPECTED_IDS:
        findings.append(f"Code assumption register drifted: {sorted(code_ids)}")

    for path in [APPENDIX_B, DEFENDED_MAP, THEOREM_REGISTRY]:
        if not path.exists():
            findings.append(f"Missing assumption surface: {path}")

    stale_surfaces = [
        REPO_ROOT / "chapters_merged/ch04_theoretical_foundations.tex",
        REPO_ROOT / "chapters_merged/ch06_main_results.tex",
        REPO_ROOT / "chapters_merged/ch07_universal_validation.tex",
        REPO_ROOT / "chapters/ch15_assumptions_notation_proof_discipline.tex",
        REPO_ROOT / "chapters/ch21_battery_to_universal_orius.tex",
        REPO_ROOT / "src/orius/universal_theory/contracts.py",
    ]
    for path in stale_surfaces:
        if path.exists() and STALE_NAMESPACE_RE.search(path.read_text(encoding="utf-8")):
            findings.append(f"{path}: stale A1-through-A8 assumption namespace remains")

    t4_sources = [
        REPO_ROOT / "reports/publication/active_theorem_audit.csv",
        REPO_ROOT / "reports/publication/theorem_registry.yml",
        REPO_ROOT / "chapters/ch19_no_free_safety_battery.tex",
        REPO_ROOT / "chapters_merged/ch04_theoretical_foundations.tex",
    ]
    for path in t4_sources:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if "boundary" not in text.lower() or "reachability" not in text.lower():
            findings.append(f"{path}: T4 surface must preserve boundary-reachability scope")

    if findings:
        print("[validate_assumption_consistency] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_assumption_consistency] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
