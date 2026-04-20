#!/usr/bin/env python3
"""Validate the YAML-canonical ORIUS theorem audit surface."""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import yaml

from _active_theorem_program import (
    ASSUMPTION_MAP_CSV,
    ASSUMPTION_MAP_MD,
    AUDIT_CSV,
    AUDIT_JSON,
    AUDIT_MD,
    BATTERY_CLAIM_EVIDENCE_REGISTER,
    DEFENDED_CORE_CSV,
    DEFENDED_CORE_JSON,
    DEFENDED_CORE_MD,
    EXTERNAL_AUDIT_PACKET_MD,
    LINEAR_READY_JSON,
    REGISTRY_YAML,
    REMEDIATION_MD,
    THEOREM_REGISTER_CSV,
    THEOREM_REGISTER_TEX,
    THEOREM_SURFACE_SUMMARY_CSV,
    THEOREM_SURFACE_SUMMARY_TEX,
    build_active_theorem_audit_payload,
    render_active_theorem_audit_csv,
    render_active_theorem_audit_json,
    render_active_theorem_audit_md,
    render_active_theorem_remediation_md,
    render_assumption_map_csv,
    render_assumption_map_md,
    render_battery_claim_evidence_register,
    render_defended_core_csv,
    render_defended_core_json,
    render_defended_core_md,
    render_external_audit_packet_md,
    render_linear_ready_json,
    render_theorem_surface_register_csv,
    render_theorem_surface_register_tex,
    render_theorem_surface_summary_csv,
    render_theorem_surface_summary_tex,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSUMPTION_REGISTER = REPO_ROOT / "appendices" / "app_b_assumptions.tex"
PROOF_SURFACES = [
    REPO_ROOT / "chapters_merged/ch04_theoretical_foundations.tex",
    REPO_ROOT / "chapters/ch37_universality_completeness.tex",
    REPO_ROOT / "appendices/app_c_full_proofs.tex",
]

GENERAL_TARGET_FILES = [
    REPO_ROOT / "chapters_merged/ch04_theoretical_foundations.tex",
    REPO_ROOT / "chapters/ch37_universality_completeness.tex",
    REPO_ROOT / "appendices/app_b_assumptions.tex",
    REPO_ROOT / "appendices/app_c_full_proofs.tex",
    REPO_ROOT / "appendices/app_m_verified_theorems_and_gap_audit.tex",
    REPO_ROOT / "appendices/app_s_claim_evidence_registers.tex",
    REPO_ROOT / "src/orius/dc3s/theoretical_guarantees.py",
    REPO_ROOT / "src/orius/universal_theory/battery_instantiation.py",
    REPO_ROOT / "src/orius/dc3s/safety_filter_theory.py",
    REPO_ROOT / "src/orius/dc3s/guarantee_checks.py",
    REPO_ROOT / "src/orius/certos/runtime.py",
    REPO_ROOT / "scripts/verify_phase_346_closure.py",
    REGISTRY_YAML,
    THEOREM_REGISTER_CSV,
]

EXPECTED_THEOREM_IDS = [
    "T1",
    "T2",
    "T3a",
    "T3b",
    "T4",
    "T5",
    "T6",
    "T7",
    "T8",
    "T9",
    "T10",
    "T11",
    "L1",
    "L2",
    "L3",
    "L4",
    "T11_Byzantine",
    "T_stale_decay",
    "T_minimax",
    "T_sensor_converse",
    "T_trajectory_PAC",
]
EXPECTED_FLAGSHIP = ["T1", "T2", "T3a", "T4", "T11", "T_trajectory_PAC"]
EXPECTED_SUPPORTING = ["T3b", "T6", "T8", "T11_Byzantine", "T_stale_decay"]
EXPECTED_DRAFT = ["T5", "T7", "T9", "T10", "L1", "L2", "L3", "L4", "T_minimax", "T_sensor_converse"]

SYNC_EXPECTATIONS = {
    ASSUMPTION_REGISTER: [
        "A5 --- Absorbed monotone tightening",
        "A10a --- Polynomial mixing telemetry",
        "A10b --- Geometric mixing telemetry",
        "A11 --- Arbitrage boundary reachability",
        "A12 --- Controller-fault independence",
        "A13 --- TV bridge",
    ],
    REPO_ROOT / "appendices/app_c_full_proofs.tex": [
        "C.11\\quad Universal Impossibility (T9)",
        "C.12\\quad Boundary-Indistinguishability Lower Bound (T10)",
        "C.13\\quad Typed Structural Transfer and Failure-Mode Converse (T11)",
        "L1 — Rate-Distortion Safety Law",
    ],
    REPO_ROOT / "src/orius/universal_theory/battery_instantiation.py": [
        "floor(delta_bnd^2 / (2 * sigma_d^2 * log(2 / delta)))",
    ],
    REPO_ROOT / "appendices/app_m_verified_theorems_and_gap_audit.tex": [
        "Theorem 3a: ORIUS Core Envelope Derivation",
        "Corollary 3b: ORIUS Core Aggregation Corollary",
        "Definition 5: Certificate Validity Horizon",
        "T11 Typed Structural Transfer & Forward-only one-step transfer",
    ],
    REPO_ROOT / "appendices/app_s_claim_evidence_registers.tex": [
        "\\texttt{theorem\\_t3a}",
        "\\texttt{theorem\\_t3b}",
        "\\texttt{theorem\\_t11}",
        "draft / non-defended",
    ],
}

EXPECTED_REGISTER = {
    "T3": ("alias", "compute_expected_violation_bound"),
    "T3a": ("risk_envelope_derivation", "compute_expected_violation_bound"),
    "T3b": ("risk_envelope_aggregation", "compute_episode_risk_bound"),
    "T5": ("definition", "certificate_validity_horizon"),
    "T6": ("expiration_bound", "certificate_expiration_bound"),
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


ASSUMPTION_BEGIN_RE = re.compile(r"\\begin\{assumption\}\[(A\d+[a-z]?)\s+[-\u2014]+")
ASSUMPTION_REF_RE = re.compile(r"\bA\d+[a-z]?\b")


def _known_assumption_ids() -> set[str]:
    known: set[str] = set()
    for line in ASSUMPTION_REGISTER.read_text(encoding="utf-8").splitlines():
        match = ASSUMPTION_BEGIN_RE.search(line)
        if match:
            known.add(match.group(1))
    return known


def _check_assumption_references(known_ids: set[str]) -> list[str]:
    findings: list[str] = []
    for path in PROOF_SURFACES:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in ASSUMPTION_REF_RE.finditer(line):
                token = match.group(0)
                if token not in known_ids:
                    findings.append(f"{path}:{lineno}: unknown assumption reference '{token}'")
    registry = yaml.safe_load(REGISTRY_YAML.read_text(encoding="utf-8")) or {}
    theorem_ids = set()
    for theorem in registry.get("theorems", []):
        theorem_id = theorem.get("id")
        if theorem_id in theorem_ids:
            findings.append(f"{REGISTRY_YAML}: duplicate theorem id '{theorem_id}'")
        theorem_ids.add(theorem_id)
        for item in theorem.get("assumptions", []):
            if isinstance(item, str) and item.startswith("A") and item not in known_ids:
                findings.append(f"{REGISTRY_YAML}: theorem '{theorem_id}' cites unknown assumption '{item}'")
    return findings


def main() -> int:
    findings: list[str] = []

    for path in GENERAL_TARGET_FILES:
        if not path.exists():
            findings.append(f"Missing target file: {path}")

    for path, required_phrases in SYNC_EXPECTATIONS.items():
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for phrase in required_phrases:
            if phrase not in text:
                findings.append(f"{path}: missing synced phrase '{phrase}'")

    known_assumptions = _known_assumption_ids()
    if known_assumptions != {
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
    }:
        findings.append(f"Appendix B assumption IDs out of sync: found {sorted(known_assumptions)}")
    findings.extend(_check_assumption_references(known_assumptions))

    theorem_register = _load_theorem_register()
    for theorem_id, (expected_type, expected_witness) in EXPECTED_REGISTER.items():
        entry = theorem_register.get(theorem_id)
        if entry is None:
            findings.append(f"Theorem register missing entry {theorem_id}")
            continue
        if entry.get("type") != expected_type:
            findings.append(f"{theorem_id}: expected type '{expected_type}', found '{entry.get('type')}'")
        if entry.get("code_witness") != expected_witness:
            findings.append(
                f"{theorem_id}: expected witness '{expected_witness}', found '{entry.get('code_witness')}'"
            )

    payload = build_active_theorem_audit_payload()
    theorem_ids = [row["theorem_id"] for row in payload["theorems"]]
    if theorem_ids != EXPECTED_THEOREM_IDS:
        findings.append(f"Active theorem audit IDs out of sync: expected {EXPECTED_THEOREM_IDS}, found {theorem_ids}")

    flagship_ids = [row["theorem_id"] for row in payload["theorems"] if row["defense_tier"] == "flagship_defended"]
    supporting_ids = [row["theorem_id"] for row in payload["theorems"] if row["defense_tier"] == "supporting_defended"]
    draft_ids = [row["theorem_id"] for row in payload["theorems"] if row["defense_tier"] == "draft_non_defended"]
    if flagship_ids != EXPECTED_FLAGSHIP:
        findings.append(f"Flagship defended core drifted: expected {EXPECTED_FLAGSHIP}, found {flagship_ids}")
    if supporting_ids != EXPECTED_SUPPORTING:
        findings.append(f"Supporting defended core drifted: expected {EXPECTED_SUPPORTING}, found {supporting_ids}")
    if draft_ids != EXPECTED_DRAFT:
        findings.append(f"Draft theorem surface drifted: expected {EXPECTED_DRAFT}, found {draft_ids}")

    t5 = next(row for row in payload["theorems"] if row["theorem_id"] == "T5")
    if t5["surface_kind"] != "definition":
        findings.append("T5 must remain retiered as a definition.")
    t6 = next(row for row in payload["theorems"] if row["theorem_id"] == "T6")
    if t6["proof_tier"] != "V2_linked":
        findings.append("T6 must remain the V2-linked auxiliary theorem surface.")
    t11 = next(row for row in payload["theorems"] if row["theorem_id"] == "T11")
    if "forward four-obligation" not in t11["scope_note"]:
        findings.append("T11 scope note must preserve the forward-only four-obligation narrowing.")

    required_drift_surfaces = {
        "src/orius/dc3s/coverage_theorem.py and tests/test_conditional_coverage.py",
        "src/orius/universal/contract.py, tests/test_universal_contract.py, and tests/test_unification.py",
        "reports/publication/theorem_surface_register.csv",
    }
    drift_surfaces = {entry["surface"] for entry in payload["namespace_drift"]}
    if drift_surfaces != required_drift_surfaces:
        findings.append(f"Namespace drift register mismatch: found {sorted(drift_surfaces)}")

    registry = yaml.safe_load(REGISTRY_YAML.read_text(encoding="utf-8")) or {}
    expected_outputs = {
        AUDIT_JSON: render_active_theorem_audit_json(payload),
        AUDIT_CSV: render_active_theorem_audit_csv(payload),
        AUDIT_MD: render_active_theorem_audit_md(payload),
        REMEDIATION_MD: render_active_theorem_remediation_md(payload),
        DEFENDED_CORE_JSON: render_defended_core_json(payload),
        DEFENDED_CORE_CSV: render_defended_core_csv(payload),
        DEFENDED_CORE_MD: render_defended_core_md(payload),
        ASSUMPTION_MAP_CSV: render_assumption_map_csv(payload),
        ASSUMPTION_MAP_MD: render_assumption_map_md(payload),
        THEOREM_REGISTER_CSV: render_theorem_surface_register_csv(registry),
        THEOREM_REGISTER_TEX: render_theorem_surface_register_tex(registry),
        THEOREM_SURFACE_SUMMARY_CSV: render_theorem_surface_summary_csv(registry),
        THEOREM_SURFACE_SUMMARY_TEX: render_theorem_surface_summary_tex(registry),
        BATTERY_CLAIM_EVIDENCE_REGISTER: render_battery_claim_evidence_register(payload),
        LINEAR_READY_JSON: render_linear_ready_json(payload),
        EXTERNAL_AUDIT_PACKET_MD: render_external_audit_packet_md(payload),
    }
    for path, expected in expected_outputs.items():
        if not path.exists():
            findings.append(f"Missing generated theorem artifact: {path}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            findings.append(f"{path}: generated theorem artifact is out of sync. Run scripts/build_active_theorem_audit.py.")

    if findings:
        print("[validate_theorem_surface] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1

    print("[validate_theorem_surface] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
