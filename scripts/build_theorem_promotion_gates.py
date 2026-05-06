#!/usr/bin/env python3
"""Build the T9/T10 theorem-promotion gate package.

This package never retieres T9/T10 by hand. It makes the promotion path
executable: all required gates are explicit, blockers are tracked when present,
and the generated status changes only when the registry plus Battery, AV, and
Healthcare evidence satisfy every gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
AUDIT_JSON = PUBLICATION_DIR / "active_theorem_audit.json"
CANDIDATE_THEOREMS = ("T9", "T10")
DOMAINS = ("battery", "av", "healthcare")
BLOCKED_CANDIDATE_TIER = "draft_non_defended"
BLOCKED_CANDIDATE_STATUS = "draft_tracked"
READY_CANDIDATE_STATUS = "promotion_ready"
REQUIRED_GATES = (
    "formal_theorem_statement",
    "explicit_assumptions",
    "mathematical_proof",
    "mechanized_proof",
    "research_package",
    "code_anchor",
    "tests",
    "artifact_evidence:battery",
    "artifact_evidence:av",
    "artifact_evidence:healthcare",
    "domain_applicability_matrix",
    "universal_constants_and_assumptions",
)
PROMOTABLE_PROOF_TIERS = {
    "V1",
    "V1_linked",
    "V1_runtime_linked",
    "V2",
    "V2_linked",
    "V3",
    "formal",
    "mechanized",
    "publication_ready",
}
NON_PROMOTABLE_RIGOR_RATINGS = {
    "",
    "inventory",
    "draft",
    "scoped_extension",
    "stylized_surrogate",
    "proxy_bridge",
    "formula_only",
}

GATE_COLUMNS = (
    "theorem_id",
    "current_tier",
    "candidate_status",
    "gate",
    "gate_pass",
    "evidence",
    "blocker",
    "required_for_flagship",
)

DOMAIN_COLUMNS = (
    "theorem_id",
    "domain",
    "applicability_status",
    "artifact_source",
    "constants_status",
    "assumptions_status",
    "promotion_ready",
    "blocker",
)


@dataclass(frozen=True)
class DomainDischarge:
    theorem_id: str
    domain: str
    applicability_status: str
    artifact_source: str
    constants_status: str
    assumptions_status: str
    promotion_ready: bool
    blocker: str

    def as_row(self) -> dict[str, str]:
        return {
            "theorem_id": self.theorem_id,
            "domain": self.domain,
            "applicability_status": self.applicability_status,
            "artifact_source": self.artifact_source,
            "constants_status": self.constants_status,
            "assumptions_status": self.assumptions_status,
            "promotion_ready": str(self.promotion_ready),
            "blocker": self.blocker,
        }


def _load_audit(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "reports" / "publication" / "active_theorem_audit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _source_sha256(repo_root: Path) -> str:
    path = repo_root / "reports" / "publication" / "active_theorem_audit.json"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_rows_by_id(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["theorem_id"]): dict(row) for row in payload.get("theorems", [])}


def _default_domain_discharge_rows(theorem_id: str) -> list[DomainDischarge]:
    if theorem_id == "T9":
        blocker = "witness constant and geometric-mixing bridge are not discharged across Battery, AV, and Healthcare"
        return [
            DomainDischarge(
                theorem_id,
                "battery",
                "domain_scoped_not_universal",
                "",
                "domain_scoped_witness_constant",
                "A10b_A11_not_three_domain_discharged",
                False,
                blocker,
            ),
            DomainDischarge(theorem_id, "av", "not_discharged", "", "missing", "missing", False, blocker),
            DomainDischarge(
                theorem_id, "healthcare", "not_discharged", "", "missing", "missing", False, blocker
            ),
        ]
    if theorem_id == "T10":
        blocker = "unsafe-side boundary-mass bridge is explicit and not discharged as a universal three-domain constant"
        return [
            DomainDischarge(
                theorem_id,
                "battery",
                "boundary_model_scoped_not_universal",
                "",
                "boundary_mass_supplied_explicitly",
                "A13_boundary_mass_bridge_not_universal",
                False,
                blocker,
            ),
            DomainDischarge(theorem_id, "av", "not_discharged", "", "missing", "missing", False, blocker),
            DomainDischarge(
                theorem_id, "healthcare", "not_discharged", "", "missing", "missing", False, blocker
            ),
        ]
    raise KeyError(theorem_id)


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _path_exists(reference: str, publication_dir: Path) -> bool:
    if not reference:
        return False
    path = Path(reference)
    if path.is_absolute():
        return path.exists()
    if reference.startswith("reports/publication/"):
        candidate = publication_dir / reference.removeprefix("reports/publication/")
        if candidate.exists():
            return True
    return (REPO_ROOT / path).exists()


def _positive(value: Any, minimum: float) -> bool:
    try:
        return float(value) > float(minimum)
    except (TypeError, ValueError):
        return False


def _payload_supports_promotion(payload: dict[str, Any], publication_dir: Path) -> bool:
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    min_rows = int(thresholds.get("min_rows", 1000) or 1000)
    minimum = float(thresholds.get("min_positive_rate", 1e-6) or 1e-6)
    if int(payload.get("n_usable_rows", 0) or 0) < min_rows:
        return False
    if not _path_exists(str(payload.get("artifact_source", "")), publication_dir):
        return False
    if not _path_exists(str(payload.get("source_trace_path", "")), publication_dir):
        return False
    if payload.get("theorem_id") == "T9":
        mixing = payload.get("mixing_proxy") if isinstance(payload.get("mixing_proxy"), dict) else {}
        return (
            _positive(payload.get("witness_constant"), minimum)
            and _positive(payload.get("degradation_rate"), minimum)
            and _positive(payload.get("boundary_reachability_rate"), minimum)
            and bool(mixing.get("finite_mixing_proxy"))
        )
    if payload.get("theorem_id") == "T10":
        tv_bridge = payload.get("tv_bridge") if isinstance(payload.get("tv_bridge"), dict) else {}
        return (
            _positive(payload.get("unsafe_boundary_mass"), minimum)
            and _positive(payload.get("safe_side_mass"), minimum)
            and _positive(payload.get("boundary_mass"), minimum)
            and bool(tv_bridge.get("passed"))
            and payload.get("le_cam_lower_bound") is not None
        )
    return False


def _domain_discharge_from_evidence(
    *,
    theorem_id: str,
    domain: str,
    evidence_path: Path,
    fallback: DomainDischarge,
    publication_dir: Path,
) -> DomainDischarge:
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    artifact_source = str(
        payload.get("artifact_source")
        or f"reports/publication/theorem_promotion_evidence/{evidence_path.name}"
    )
    promotion_ready = (
        _is_true(payload.get("promotion_ready"))
        and bool(artifact_source)
        and _payload_supports_promotion(payload, publication_dir)
    )
    blocker = str(payload.get("blocker") or fallback.blocker)
    if _is_true(payload.get("promotion_ready")) and not promotion_ready:
        blocker = "promotion_ready true is not supported by numeric discharge evidence"
    return DomainDischarge(
        theorem_id=theorem_id,
        domain=domain,
        applicability_status=str(payload.get("applicability_status") or fallback.applicability_status),
        artifact_source=artifact_source,
        constants_status=str(payload.get("constants_status") or fallback.constants_status),
        assumptions_status=str(payload.get("assumptions_status") or fallback.assumptions_status),
        promotion_ready=promotion_ready,
        blocker="" if promotion_ready else blocker,
    )


def _domain_discharge_rows(theorem_id: str, publication_dir: Path) -> list[DomainDischarge]:
    rows = _default_domain_discharge_rows(theorem_id)
    evidence_dir = publication_dir / "theorem_promotion_evidence"
    resolved: list[DomainDischarge] = []
    for row in rows:
        evidence_path = evidence_dir / f"{theorem_id}_{row.domain}.json"
        if evidence_path.exists():
            resolved.append(
                _domain_discharge_from_evidence(
                    theorem_id=theorem_id,
                    domain=row.domain,
                    evidence_path=evidence_path,
                    fallback=row,
                    publication_dir=publication_dir,
                )
            )
        else:
            resolved.append(row)
    return resolved


def _has_assumptions(row: dict[str, Any]) -> bool:
    return bool(
        row.get("assumptions_used") or row.get("typed_obligations") or row.get("unresolved_assumptions")
    )


def _gate_row(
    *,
    theorem_id: str,
    current_tier: str,
    gate_name: str,
    gate_pass: bool,
    evidence: str,
    blocker: str = "",
) -> dict[str, str]:
    return {
        "theorem_id": theorem_id,
        "current_tier": current_tier,
        "candidate_status": "draft_tracked",
        "gate": gate_name,
        "gate_pass": "1" if gate_pass else "0",
        "evidence": evidence,
        "blocker": "" if gate_pass else blocker,
        "required_for_flagship": "True",
    }


def _gate_passes(row: dict[str, str]) -> bool:
    return row.get("gate_pass", "").strip().lower() in {"1", "true", "yes"}


def _package_status_for_candidate(
    audit_row: dict[str, Any],
    blocking_gates: list[str],
) -> tuple[str, str]:
    if blocking_gates:
        return BLOCKED_CANDIDATE_TIER, BLOCKED_CANDIDATE_STATUS
    return str(audit_row.get("defense_tier", BLOCKED_CANDIDATE_TIER)), READY_CANDIDATE_STATUS


def _mathematical_proof_gate(row: dict[str, Any]) -> tuple[bool, str]:
    blockers: list[str] = []
    proof_tier = str(row.get("proof_tier", "")).strip()
    rigor_rating = str(row.get("rigor_rating", "")).strip()
    code_correspondence = str(row.get("code_correspondence", "")).strip()

    if not row.get("proof_location"):
        blockers.append("missing mathematical proof anchor")
    if proof_tier not in PROMOTABLE_PROOF_TIERS:
        blockers.append(f"proof_tier must be flagship-grade, observed {proof_tier or 'missing'}")
    if row.get("unresolved_assumptions"):
        blockers.append("unresolved assumptions remain")
    if code_correspondence != "matches":
        blockers.append("code_correspondence must be matches")
    if rigor_rating in NON_PROMOTABLE_RIGOR_RATINGS:
        blockers.append(f"rigor_rating is not flagship-grade, observed {rigor_rating or 'missing'}")

    return not blockers, "; ".join(blockers)


def _mechanized_proof_gate(theorem_id: str, publication_dir: Path) -> tuple[bool, str, str]:
    path = publication_dir / "t9_t10_mechanized_proof_status.json"
    payload = _load_optional_json(path)
    theorem = dict(payload.get("theorems", {}).get(theorem_id, {}))
    evidence = "reports/publication/t9_t10_mechanized_proof_status.json"
    blockers: list[str] = []
    if not payload:
        blockers.append("missing mechanized proof status artifact")
    if payload.get("forbidden_tokens"):
        blockers.append(f"formal proof contains forbidden placeholders: {payload.get('forbidden_tokens')}")
    if payload.get("lean_status") != "passed":
        blockers.append(f"Lean build must pass, observed {payload.get('lean_status', 'missing')}")
    if theorem.get("status") not in {"verified", "kernel_verified"}:
        blockers.append(f"{theorem_id} mechanized proof status must be verified")
    if theorem.get("mechanization_scope") != "theorem_kernel_not_domain_discharge":
        blockers.append(
            f"{theorem_id} mechanization scope must separate theorem kernel from domain discharge"
        )
    return not blockers, evidence, "; ".join(blockers)


def _research_package_gate(publication_dir: Path) -> tuple[bool, str, str]:
    path = publication_dir / "t9_t10_research_scorecard.json"
    payload = _load_optional_json(path)
    evidence = "reports/publication/t9_t10_research_scorecard.json"
    blockers: list[str] = []
    if not payload:
        blockers.append("missing T9/T10 research scorecard")
    if int(payload.get("source_count", 0) or 0) < 500:
        blockers.append(f"source_count must be at least 500, observed {payload.get('source_count', 0)}")
    if not payload.get("proof_dependency_complete"):
        blockers.append("proof dependency matrix is incomplete")
    if not payload.get("pass"):
        blockers.append("research package validator has not passed")
    return not blockers, evidence, "; ".join(blockers)


def _gate_rows_for_candidate(
    row: dict[str, Any],
    domain_rows: list[DomainDischarge],
    publication_dir: Path,
) -> list[dict[str, str]]:
    theorem_id = str(row["theorem_id"])
    current_tier = str(row.get("defense_tier", "unknown"))
    code_anchors = row.get("code_anchors") or []
    test_anchors = row.get("test_anchors") or []
    proof_tier = str(row.get("proof_tier", ""))
    all_domains_ready = all(item.promotion_ready for item in domain_rows)
    all_domains_present = {item.domain for item in domain_rows} == set(DOMAINS)
    proof_promotable, proof_blocker = _mathematical_proof_gate(row)
    mechanized_promotable, mechanized_evidence, mechanized_blocker = _mechanized_proof_gate(
        theorem_id,
        publication_dir,
    )
    research_promotable, research_evidence, research_blocker = _research_package_gate(publication_dir)

    rows = [
        _gate_row(
            theorem_id=theorem_id,
            current_tier=current_tier,
            gate_name="formal_theorem_statement",
            gate_pass=bool(row.get("statement_location")),
            evidence=str(row.get("statement_location", "")),
            blocker="missing formal theorem statement anchor",
        ),
        _gate_row(
            theorem_id=theorem_id,
            current_tier=current_tier,
            gate_name="explicit_assumptions",
            gate_pass=_has_assumptions(row),
            evidence=" | ".join(
                str(item)
                for item in [
                    *(row.get("assumptions_used") or []),
                    *(row.get("typed_obligations") or []),
                    *(row.get("unresolved_assumptions") or []),
                ]
            ),
            blocker="missing explicit assumptions or typed obligations",
        ),
        _gate_row(
            theorem_id=theorem_id,
            current_tier=current_tier,
            gate_name="mathematical_proof",
            gate_pass=proof_promotable,
            evidence=f"{row.get('proof_location', '')}; proof_tier={proof_tier}",
            blocker=proof_blocker,
        ),
        _gate_row(
            theorem_id=theorem_id,
            current_tier=current_tier,
            gate_name="mechanized_proof",
            gate_pass=mechanized_promotable,
            evidence=mechanized_evidence,
            blocker=mechanized_blocker,
        ),
        _gate_row(
            theorem_id=theorem_id,
            current_tier=current_tier,
            gate_name="research_package",
            gate_pass=research_promotable,
            evidence=research_evidence,
            blocker=research_blocker,
        ),
        _gate_row(
            theorem_id=theorem_id,
            current_tier=current_tier,
            gate_name="code_anchor",
            gate_pass=bool(code_anchors),
            evidence=" | ".join(
                str(anchor.get("location") or anchor.get("path", "")) for anchor in code_anchors
            ),
            blocker="missing code anchor implementing the bound/check",
        ),
        _gate_row(
            theorem_id=theorem_id,
            current_tier=current_tier,
            gate_name="tests",
            gate_pass=bool(test_anchors),
            evidence=" | ".join(
                str(anchor.get("location") or anchor.get("path", "")) for anchor in test_anchors
            ),
            blocker="missing tests for the theorem helper and assumption violations",
        ),
        _gate_row(
            theorem_id=theorem_id,
            current_tier=current_tier,
            gate_name="domain_applicability_matrix",
            gate_pass=all_domains_present,
            evidence="reports/publication/theorem_promotion_domain_matrix.csv",
            blocker="missing Battery/AV/Healthcare applicability rows",
        ),
        _gate_row(
            theorem_id=theorem_id,
            current_tier=current_tier,
            gate_name="universal_constants_and_assumptions",
            gate_pass=all_domains_ready,
            evidence="three-domain constants and assumptions discharge matrix",
            blocker="universal constants and assumptions are not discharged across Battery, AV, and Healthcare",
        ),
    ]

    domain_by_name = {item.domain: item for item in domain_rows}
    for domain in DOMAINS:
        item = domain_by_name[domain]
        rows.append(
            _gate_row(
                theorem_id=theorem_id,
                current_tier=current_tier,
                gate_name=f"artifact_evidence:{domain}",
                gate_pass=item.promotion_ready and bool(item.artifact_source),
                evidence=item.artifact_source,
                blocker=item.blocker or f"missing {domain} artifact evidence",
            )
        )

    return sorted(rows, key=lambda item: REQUIRED_GATES.index(item["gate"]))


def _write_csv(path: Path, rows: list[dict[str, str]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def build_promotion_package(
    repo_root: Path = REPO_ROOT, publication_dir: Path = PUBLICATION_DIR
) -> dict[str, Any]:
    payload = _load_audit(repo_root)
    rows_by_id = _audit_rows_by_id(payload)
    gate_rows: list[dict[str, str]] = []
    domain_rows: list[dict[str, str]] = []
    candidates: dict[str, dict[str, Any]] = {}

    for theorem_id in CANDIDATE_THEOREMS:
        audit_row = rows_by_id[theorem_id]
        discharge_rows = _domain_discharge_rows(theorem_id, publication_dir)
        gate_rows_for_theorem = _gate_rows_for_candidate(audit_row, discharge_rows, publication_dir)
        domain_rows.extend(item.as_row() for item in discharge_rows)
        blocking_gates = [row["gate"] for row in gate_rows_for_theorem if not _gate_passes(row)]
        package_tier, candidate_status = _package_status_for_candidate(audit_row, blocking_gates)
        for gate_row in gate_rows_for_theorem:
            gate_row["current_tier"] = package_tier
            gate_row["candidate_status"] = candidate_status
        gate_rows.extend(gate_rows_for_theorem)
        candidates[theorem_id] = {
            "title": audit_row["title"],
            "current_tier": package_tier,
            "candidate_status": candidate_status,
            "promotion_ready": not blocking_gates,
            "blocking_gates": blocking_gates,
            "weakest_step": audit_row.get("weakest_step", ""),
            "remediation_detail": audit_row.get("remediation_detail", ""),
        }

    promotion_ready = all(item["promotion_ready"] for item in candidates.values())
    scorecard = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": "reports/publication/active_theorem_audit.json",
        "source_sha256": _source_sha256(repo_root),
        "promotion_ready": promotion_ready,
        "required_domains": list(DOMAINS),
        "required_gates": list(REQUIRED_GATES),
        "policy": (
            "T9/T10 can become flagship only when formal statement, explicit assumptions, mathematical proof, "
            "code anchor, tests, artifact evidence for Battery/AV/Healthcare, domain applicability matrix, "
            "and domain constants/assumptions all pass."
        ),
        "candidates": candidates,
    }

    _write_csv(publication_dir / "theorem_promotion_gates.csv", gate_rows, GATE_COLUMNS)
    _write_csv(publication_dir / "theorem_promotion_domain_matrix.csv", domain_rows, DOMAIN_COLUMNS)
    (publication_dir / "theorem_promotion_scorecard.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return scorecard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--publication-dir", type=Path, default=PUBLICATION_DIR)
    args = parser.parse_args()
    result = build_promotion_package(args.repo_root.resolve(), args.publication_dir.resolve())
    print(
        "[build_theorem_promotion_gates] "
        f"promotion_ready={result['promotion_ready']} candidates={','.join(result['candidates'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
