from __future__ import annotations

import csv
import json
from pathlib import Path

import scripts.build_theorem_promotion_gates as builder
import scripts.validate_theorem_promotion_gates as validator

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_GATES = {
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
}
REQUIRED_DOMAINS = {"battery", "av", "healthcare"}
CSV_TRUE = "1"
CSV_FALSE = "0"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_tmp_audit(repo_root: Path, transform) -> None:
    source = json.loads(
        (REPO_ROOT / "reports" / "publication" / "active_theorem_audit.json").read_text(encoding="utf-8")
    )
    transformed = transform(source)
    audit_path = repo_root / "reports" / "publication" / "active_theorem_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(transformed, indent=2), encoding="utf-8")


def test_builder_emits_complete_t9_t10_promotion_gate_package(tmp_path: Path) -> None:
    publication_dir = tmp_path / "publication"

    result = builder.build_promotion_package(REPO_ROOT, publication_dir)

    assert result["promotion_ready"] is False
    gate_rows = _read_csv(publication_dir / "theorem_promotion_gates.csv")
    domain_rows = _read_csv(publication_dir / "theorem_promotion_domain_matrix.csv")
    scorecard = json.loads((publication_dir / "theorem_promotion_scorecard.json").read_text(encoding="utf-8"))

    for theorem_id in ("T9", "T10"):
        theorem_gates = {row["gate"] for row in gate_rows if row["theorem_id"] == theorem_id}
        theorem_domains = {row["domain"] for row in domain_rows if row["theorem_id"] == theorem_id}
        assert theorem_gates == REQUIRED_GATES
        assert theorem_domains == REQUIRED_DOMAINS
        assert scorecard["candidates"][theorem_id]["current_tier"] == "draft_non_defended"
        assert scorecard["candidates"][theorem_id]["promotion_ready"] is False
        assert scorecard["candidates"][theorem_id]["blocking_gates"]
        assert "mechanized_proof" in scorecard["candidates"][theorem_id]["blocking_gates"]
        assert "research_package" in scorecard["candidates"][theorem_id]["blocking_gates"]


def test_builder_keeps_t9_t10_non_promoted_when_audit_tier_drifts_up(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"

    def transform(payload: dict) -> dict:
        for row in payload["theorems"]:
            if row["theorem_id"] in {"T9", "T10"}:
                row["defense_tier"] = "flagship_defended"
                row["proof_tier"] = "V3"
                row["rigor_rating"] = "formal"
                row["code_correspondence"] = "matches"
                row["unresolved_assumptions"] = []
        return payload

    _write_tmp_audit(repo_root, transform)
    publication_dir = tmp_path / "publication"

    builder.build_promotion_package(repo_root, publication_dir)

    gate_rows = _read_csv(publication_dir / "theorem_promotion_gates.csv")
    scorecard = json.loads((publication_dir / "theorem_promotion_scorecard.json").read_text(encoding="utf-8"))
    for theorem_id in ("T9", "T10"):
        assert scorecard["candidates"][theorem_id]["promotion_ready"] is False
        assert scorecard["candidates"][theorem_id]["current_tier"] == "draft_non_defended"
        assert scorecard["candidates"][theorem_id]["candidate_status"] == "draft_tracked"
        assert {row["current_tier"] for row in gate_rows if row["theorem_id"] == theorem_id} == {
            "draft_non_defended"
        }


def test_validator_accepts_draft_package_with_explicit_blockers(tmp_path: Path) -> None:
    publication_dir = tmp_path / "publication"
    builder.build_promotion_package(REPO_ROOT, publication_dir)

    result = validator.validate_promotion_package(publication_dir=publication_dir)

    assert result["pass"] is True
    assert result["promotion_ready"] is False
    assert any("T9" in blocker for blocker in result["blockers"])
    assert any("T10" in blocker for blocker in result["blockers"])


def test_validator_blocks_promote_requested_when_any_gate_is_missing(tmp_path: Path) -> None:
    publication_dir = tmp_path / "publication"
    builder.build_promotion_package(REPO_ROOT, publication_dir)
    gate_path = publication_dir / "theorem_promotion_gates.csv"
    rows = _read_csv(gate_path)
    for row in rows:
        if row["theorem_id"] == "T9":
            row["candidate_status"] = "promote_requested"
    _write_csv(gate_path, rows)

    result = validator.validate_promotion_package(publication_dir=publication_dir)

    assert result["pass"] is False
    assert any("T9" in finding and "promote_requested" in finding for finding in result["findings"])


def test_validator_recomputes_gates_and_rejects_hand_edited_false_promotion(tmp_path: Path) -> None:
    publication_dir = tmp_path / "publication"
    builder.build_promotion_package(REPO_ROOT, publication_dir)
    gate_path = publication_dir / "theorem_promotion_gates.csv"
    rows = _read_csv(gate_path)
    for row in rows:
        if row["theorem_id"] == "T9":
            row["candidate_status"] = "promoted"
            row["gate_pass"] = CSV_TRUE
            row["blocker"] = ""
            if row["gate"].startswith("artifact_evidence"):
                row["evidence"] = "reports/publication/fake_t9_evidence.json"
    _write_csv(gate_path, rows)
    scorecard_path = publication_dir / "theorem_promotion_scorecard.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    scorecard["promotion_ready"] = True
    scorecard["candidates"]["T9"]["promotion_ready"] = True
    scorecard["candidates"]["T9"]["blocking_gates"] = []
    scorecard_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8")

    result = validator.validate_promotion_package(publication_dir=publication_dir, repo_root=REPO_ROOT)

    assert result["pass"] is False
    assert any(
        "T9" in finding and "does not match recomputed gate" in finding for finding in result["findings"]
    )


def test_validator_rejects_stale_source_hash(tmp_path: Path) -> None:
    publication_dir = tmp_path / "publication"
    builder.build_promotion_package(REPO_ROOT, publication_dir)
    scorecard_path = publication_dir / "theorem_promotion_scorecard.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    scorecard["source_sha256"] = "0" * 64
    scorecard_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8")

    result = validator.validate_promotion_package(publication_dir=publication_dir, repo_root=REPO_ROOT)

    assert result["pass"] is False
    assert any("source_sha256" in finding for finding in result["findings"])


def test_builder_discovers_domain_discharge_evidence_files(tmp_path: Path) -> None:
    publication_dir = tmp_path / "publication"
    evidence_dir = publication_dir / "theorem_promotion_evidence"
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "T9_battery.json").write_text(
        json.dumps(
            {
                "theorem_id": "T9",
                "domain": "battery",
                "applicability_status": "discharged",
                "artifact_source": "reports/publication/theorem_promotion_evidence/T9_battery.json",
                "source_trace_path": "reports/publication/theorem_promotion_evidence/T9_battery.json",
                "n_usable_rows": 1000,
                "thresholds": {"min_rows": 1000, "min_positive_rate": 1e-06},
                "witness_constant": 0.01,
                "degradation_rate": 0.2,
                "boundary_reachability_rate": 0.1,
                "mixing_proxy": {"finite_mixing_proxy": True},
                "constants_status": "witness_constant_discharged",
                "assumptions_status": "A10b_A11_discharged",
                "promotion_ready": True,
            }
        ),
        encoding="utf-8",
    )

    builder.build_promotion_package(REPO_ROOT, publication_dir)

    rows = _read_csv(publication_dir / "theorem_promotion_domain_matrix.csv")
    battery = next(row for row in rows if row["theorem_id"] == "T9" and row["domain"] == "battery")
    assert battery["promotion_ready"] == "True"
    assert battery["artifact_source"] == "reports/publication/theorem_promotion_evidence/T9_battery.json"


def test_mathematical_proof_gate_requires_matching_code_and_no_unresolved_assumptions(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"

    def transform(payload: dict) -> dict:
        for row in payload["theorems"]:
            if row["theorem_id"] == "T9":
                row["proof_tier"] = "V1"
                row["code_correspondence"] = "partial"
                row["unresolved_assumptions"] = []
            if row["theorem_id"] == "T10":
                row["proof_tier"] = "V1"
                row["code_correspondence"] = "matches"
                row["unresolved_assumptions"] = ["synthetic unresolved bridge"]
        return payload

    _write_tmp_audit(repo_root, transform)
    publication_dir = tmp_path / "publication"
    builder.build_promotion_package(repo_root, publication_dir)

    rows = _read_csv(publication_dir / "theorem_promotion_gates.csv")
    proof_rows = {
        row["theorem_id"]: row
        for row in rows
        if row["gate"] == "mathematical_proof" and row["theorem_id"] in {"T9", "T10"}
    }
    assert proof_rows["T9"]["gate_pass"] == CSV_FALSE
    assert "code_correspondence must be matches" in proof_rows["T9"]["blocker"]
    assert proof_rows["T10"]["gate_pass"] == CSV_FALSE
    assert "unresolved assumptions" in proof_rows["T10"]["blocker"]


def test_validator_requires_three_domain_applicability_matrix(tmp_path: Path) -> None:
    publication_dir = tmp_path / "publication"
    builder.build_promotion_package(REPO_ROOT, publication_dir)
    matrix_path = publication_dir / "theorem_promotion_domain_matrix.csv"
    rows = [
        row
        for row in _read_csv(matrix_path)
        if not (row["theorem_id"] == "T10" and row["domain"] == "healthcare")
    ]
    _write_csv(matrix_path, rows)

    result = validator.validate_promotion_package(publication_dir=publication_dir)

    assert result["pass"] is False
    assert any("T10" in finding and "healthcare" in finding for finding in result["findings"])


def test_require_promoted_fails_until_t9_t10_pass_every_gate(tmp_path: Path) -> None:
    publication_dir = tmp_path / "publication"
    builder.build_promotion_package(REPO_ROOT, publication_dir)

    result = validator.validate_promotion_package(
        publication_dir=publication_dir,
        require_promoted={"T9", "T10"},
    )

    assert result["pass"] is False
    assert any("T9" in finding and "not promotion-ready" in finding for finding in result["findings"])
    assert any("T10" in finding and "not promotion-ready" in finding for finding in result["findings"])


def test_make_quality_rebuilds_promotion_package_before_validation() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    quality_block = makefile.split("\nclean:", 1)[0]

    assert "$(MAKE) theorem-promotion-verify" in quality_block
    assert "scripts/validate_theorem_promotion_gates.py" not in quality_block
