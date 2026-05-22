from __future__ import annotations

import json
from pathlib import Path

import scripts.validate_theorem_defensibility_10 as defensibility

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_theorem_defensibility_10_gate_passes_for_current_package() -> None:
    result = defensibility.validate(repo_root=REPO_ROOT, run_lean=False)

    assert result["pass"] is True
    assert result["score"] == 10.0
    assert result["blockers"] == []
    assert result["checks"]["active_code_correspondence_all_match"] is True
    assert result["checks"]["no_draft_defended_rows"] is True
    assert result["checks"]["no_unresolved_assumptions"] is True
    assert result["checks"]["formal_core_no_sorry_or_admit"] is True
    assert result["checks"]["flagship_promotion_cards_complete"] is True


def test_theorem_defensibility_10_gate_rejects_partial_code_correspondence(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    publication = repo / "reports" / "publication"
    publication.mkdir(parents=True)
    formal = repo / "formal" / "Orius"
    formal.mkdir(parents=True)
    (formal / "Core.lean").write_text("namespace Orius\n theorem ok : True := by trivial\n", encoding="utf-8")

    active = json.loads((REPO_ROOT / "reports" / "publication" / "active_theorem_audit.json").read_text())
    active["theorems"][0]["code_correspondence"] = "partial"
    (publication / "active_theorem_audit.json").write_text(json.dumps(active), encoding="utf-8")
    (publication / "theorem_promotion_matrix.json").write_text(
        (REPO_ROOT / "reports" / "publication" / "theorem_promotion_matrix.json").read_text(),
        encoding="utf-8",
    )
    cards_src = REPO_ROOT / "reports" / "publication" / "theorem_result_cards"
    cards_dst = publication / "theorem_result_cards"
    cards_dst.mkdir()
    for card in cards_src.glob("*.json"):
        (cards_dst / card.name).write_text(card.read_text(), encoding="utf-8")

    result = defensibility.validate(repo_root=repo, run_lean=False)

    assert result["pass"] is False
    assert result["score"] < 10.0
    assert any("code_correspondence" in blocker for blocker in result["blockers"])


def test_formal_core_scan_rejects_sorry_in_executable_lean(tmp_path: Path) -> None:
    formal = tmp_path / "formal" / "Orius"
    formal.mkdir(parents=True)
    (formal / "Bad.lean").write_text("namespace Orius\n theorem bad : True := by sorry\n", encoding="utf-8")

    result = defensibility.validate_formal_core(tmp_path, run_lean=False)

    assert result["pass"] is False
    assert any("sorry" in blocker for blocker in result["blockers"])
