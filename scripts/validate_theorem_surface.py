#!/usr/bin/env python3
"""Validate the reconciled active T1--T11 theorem audit surface."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

from _active_theorem_program import (
    AUDIT_CSV,
    AUDIT_JSON,
    AUDIT_MD,
    REMEDIATION_MD,
    build_active_theorem_audit_payload,
    render_active_theorem_audit_csv,
    render_active_theorem_audit_json,
    render_active_theorem_audit_md,
    render_active_theorem_remediation_md,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

GENERAL_TARGET_FILES = [
    REPO_ROOT / "chapters_merged/ch04_theoretical_foundations.tex",
    REPO_ROOT / "chapters/ch37_universality_completeness.tex",
    REPO_ROOT / "appendices/app_b_assumptions.tex",
    REPO_ROOT / "appendices/app_c_full_proofs.tex",
    REPO_ROOT / "appendices/app_m_verified_theorems_and_gap_audit.tex",
    REPO_ROOT / "reports/publication/theorem_surface_register.csv",
    REPO_ROOT / "src/orius/universal_theory/risk_bounds.py",
    REPO_ROOT / "src/orius/universal_theory/contracts.py",
    REPO_ROOT / "src/orius/dc3s/theoretical_guarantees.py",
    REPO_ROOT / "src/orius/dc3s/coverage_theorem.py",
    REPO_ROOT / "src/orius/universal/contract.py",
    REPO_ROOT / "tests/test_conditional_coverage.py",
    REPO_ROOT / "tests/test_universal_contract.py",
    REPO_ROOT / "tests/test_unification.py",
]

BANNED_PATTERNS = {
    r"\bweighted exchangeability\b": "The active T3 surface no longer treats weighted exchangeability as the core justification.",
    r"\bexact conditional coverage\b": "The active theorem surface does not defend exact conditional coverage.",
    r"\bmathematically necessary\b": "The active T4/T9 surface is constructive and class-scoped, not a blanket necessity claim.",
}

FILE_BANNED_PATTERNS: dict[Path, dict[str, str]] = {
    REPO_ROOT / "src/orius/dc3s/coverage_theorem.py": {
        r"Theorem 9\s+[--—]": "Legacy auxiliary coverage helpers must not reuse the active T9 number.",
        r"Theorem 10\s+[--—]": "Legacy auxiliary coverage helpers must not reuse the active T10 number.",
    },
    REPO_ROOT / "tests/test_conditional_coverage.py": {
        r"Tests for Theorem 9": "The auxiliary coverage tests must stay out of the active theorem namespace.",
        r"Theorem 10": "The auxiliary coverage tests must stay out of the active theorem namespace.",
    },
    REPO_ROOT / "src/orius/universal/contract.py": {
        r"Any adapter satisfying the contract inherits the bound\.": "The five-invariant mini-harness is supporting evidence only.",
        r"needs no further\s+domain-specific safety proof": "The mini-harness does not discharge the full active T11 surface by itself.",
        r"reliability is a proper probability": "w_t is a bounded reliability score, not a probability by definition.",
    },
    REPO_ROOT / "tests/test_universal_contract.py": {
        r"five T11 invariants": "These are five supporting mini-harness invariants, not the full active T11 theorem.",
    },
    REPO_ROOT / "tests/test_unification.py": {
        r"five T11 invariant": "These are five supporting mini-harness invariants, not the full active T11 theorem.",
        r"Both prior frameworks satisfy T11": "Passing the supporting mini-harness must not be described as fully satisfying active T11.",
    },
}

SYNC_EXPECTATIONS = {
    REPO_ROOT / "chapters/ch37_universality_completeness.tex": [
        "Universal Impossibility, T9",
        "Boundary-indistinguishability lower bound, T10",
        "Typed structural transfer theorem, T11",
    ],
    REPO_ROOT / "appendices/app_c_full_proofs.tex": [
        "C.11\\quad Universal Impossibility (T9)",
        "C.12\\quad Boundary-Indistinguishability Lower Bound (T10)",
        "C.13\\quad Typed Structural Transfer and Failure-Mode Converse (T11)",
    ],
    REPO_ROOT / "appendices/app_b_assumptions.tex": [
        "A1 --- Bounded model error",
        "A8 --- Admissible fallback",
    ],
    REPO_ROOT / "src/orius/universal_theory/risk_bounds.py": [
        "w_t is a runtime reliability score, not a probability by definition.",
        "The envelope is marginal/expected-episode control, not a conditional coverage guarantee for every observation.",
    ],
}

EXPECTED_REGISTER = {
    "T9": ("impossibility", "compute_universal_impossibility_bound"),
    "T10": ("lower_bound", "compute_stylized_frontier_lower_bound"),
    "T11": ("transfer_theorem", "evaluate_structural_transfer"),
}


def _load_theorem_register() -> dict:
    module_path = REPO_ROOT / "src/orius/dc3s/theoretical_guarantees.py"
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "THEOREM_REGISTER":
                    return ast.literal_eval(node.value)
    raise RuntimeError("THEOREM_REGISTER assignment not found")


def main() -> int:
    findings: list[str] = []

    for path in GENERAL_TARGET_FILES:
        if not path.exists():
            findings.append(f"Missing target file: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, detail in BANNED_PATTERNS.items():
            if re.search(pattern, text):
                findings.append(f"{path}: banned pattern /{pattern}/ matched. {detail}")

    for path, patterns in FILE_BANNED_PATTERNS.items():
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for pattern, detail in patterns.items():
            if re.search(pattern, text, flags=re.MULTILINE):
                findings.append(f"{path}: banned pattern /{pattern}/ matched. {detail}")

    for path, required_phrases in SYNC_EXPECTATIONS.items():
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for phrase in required_phrases:
            if phrase not in text:
                findings.append(f"{path}: missing synced theory-surface phrase '{phrase}'")

    theorem_register = _load_theorem_register()
    for theorem_id, (expected_type, expected_witness) in EXPECTED_REGISTER.items():
        entry = theorem_register.get(theorem_id)
        if entry is None:
            findings.append(f"Theorem register missing entry {theorem_id}")
            continue
        if entry.get("type") != expected_type:
            findings.append(
                f"{theorem_id}: expected type '{expected_type}', found '{entry.get('type')}'"
            )
        if entry.get("code_witness") != expected_witness:
            findings.append(
                f"{theorem_id}: expected witness '{expected_witness}', found '{entry.get('code_witness')}'"
            )

    payload = build_active_theorem_audit_payload()
    theorem_ids = [row["theorem_id"] for row in payload["theorems"]]
    expected_ids = [f"T{i}" for i in range(1, 12)]
    if theorem_ids != expected_ids:
        findings.append(f"Active theorem audit IDs out of sync: expected {expected_ids}, found {theorem_ids}")

    for row in payload["theorems"]:
        required_keys = (
            "statement_location",
            "proof_location",
            "assumptions_used",
            "weakest_step",
            "rigor_rating",
            "code_correspondence",
            "severity_if_broken",
            "remediation_class",
            "code_anchors",
            "test_anchors",
        )
        for key in required_keys:
            value = row.get(key)
            if value in (None, "", []):
                findings.append(f"{row['theorem_id']}: missing required audit field '{key}'")

    t3 = next(row for row in payload["theorems"] if row["theorem_id"] == "T3")
    if "w_t is a runtime reliability score, not a probability by definition." not in t3["assumptions_used"]:
        findings.append("T3 audit row is missing the active reliability-score disclaimer.")

    t11 = next(row for row in payload["theorems"] if row["theorem_id"] == "T11")
    if not any(anchor["path"] == "src/orius/universal_theory/contracts.py" for anchor in t11["code_anchors"]):
        findings.append("T11 audit row must point at src/orius/universal_theory/contracts.py as the authoritative typed surface.")

    drift_surfaces = {entry["surface"] for entry in payload["namespace_drift"]}
    for required_surface in (
        "src/orius/dc3s/coverage_theorem.py and tests/test_conditional_coverage.py",
        "src/orius/universal/contract.py, tests/test_universal_contract.py, and tests/test_unification.py",
        "reports/publication/theorem_surface_register.csv",
    ):
        if required_surface not in drift_surfaces:
            findings.append(f"Missing namespace-drift entry for '{required_surface}'")

    expected_outputs = {
        AUDIT_JSON: render_active_theorem_audit_json(payload),
        AUDIT_CSV: render_active_theorem_audit_csv(payload),
        AUDIT_MD: render_active_theorem_audit_md(payload),
        REMEDIATION_MD: render_active_theorem_remediation_md(payload),
    }
    for path, expected in expected_outputs.items():
        if not path.exists():
            findings.append(f"Missing generated audit artifact: {path}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            findings.append(
                f"{path}: generated theorem audit artifact is out of sync. Run scripts/build_active_theorem_audit.py."
            )

    if findings:
        print("[validate_theorem_surface] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("[validate_theorem_surface] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
