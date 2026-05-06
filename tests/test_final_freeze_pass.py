from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CENTRAL_CLAIM = (
    "ORIUS provides a reliability-aware runtime safety layer for physical AI under "
    "degraded observation, enforcing certificate-backed action release through "
    "uncertainty coverage, repair, and fallback."
)


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_canonical_claim_sentence_is_identical_across_headline_surfaces() -> None:
    surfaces = [
        "README.md",
        "reports/publication/README.md",
        "paper/review/orius_review_dossier.tex",
        "paper/monograph/ch01_introduction_and_thesis_claims.tex",
        "paper/monograph/ch02_related_work_and_novelty_gap.tex",
    ]
    for rel_path in surfaces:
        assert CENTRAL_CLAIM in _read(rel_path), rel_path

    manifest = json.loads(_read("paper/orius_program_manifest.json"))
    assert manifest["monograph_posture"]["book_level_claim"] == CENTRAL_CLAIM
    assert manifest["canonical_review_build"] == "make review-compile"


def test_final_freeze_release_note_exists_and_is_actionable() -> None:
    text = _read("reports/publication/final_freeze_release_note.md")
    assert "make orius-book" in text
    assert "make review-compile" in text
    assert "orius_book.tex" in text
    assert "reports/battery_av_healthcare/overall" in text
    assert "Battery: witness row" in text
    assert "AV: runtime-closed narrowed contract row" in text
    assert "Healthcare: runtime-closed narrowed contract row" in text
    assert "No equal-depth empirical closure beyond the active three-domain program." in text


def test_release_manifest_keeps_only_canonical_pdf_surfaces() -> None:
    payload = json.loads(_read("reports/publication/release_manifest.json"))
    assert "frozen_pdfs" not in payload
    assert payload["archive_policy"]["active_manifest_policy"].startswith("canonical pdfs only")
    assert set(payload["paper_assets"]) == {
        "CANONICAL_MONOGRAPH_PDF",
        "CANONICAL_ROOT_MONOGRAPH_PDF",
        "CANONICAL_REVIEW_DOSSIER_PDF",
        "PUBLICATION_REVIEW_DOSSIER_PDF",
        "IEEE_MAIN_PDF",
        "IEEE_APPENDIX_PDF",
        "IEEE_PROFESSOR_MAIN_PDF",
        "IEEE_PROFESSOR_APPENDIX_A_PDF",
        "IEEE_PROFESSOR_APPENDIX_B_PDF",
    }


def test_legacy_longform_surfaces_are_fenced_as_noncanonical() -> None:
    paper_readme = _read("paper/README.md").lower()
    legacy_controller = " ".join(_read("orius_battery_409page_figures_upgraded_main.tex").lower().split())

    assert "non-canonical internal archive" in paper_readme
    assert "not the submission authority" in paper_readme
    assert "internal archive notice" in legacy_controller
    assert "excluded from the defended submission workflow" in legacy_controller
    assert "battery + av + healthcare only" in legacy_controller


def test_existing_pdf_inventory_is_limited_to_canonical_and_diagnostic_surfaces() -> None:
    pdfs = {
        str(path.relative_to(REPO_ROOT))
        for root in (REPO_ROOT / "paper", REPO_ROOT / "reports", REPO_ROOT / "papers")
        for path in root.rglob("*.pdf")
        if not path.name.startswith("._")
    }
    if (REPO_ROOT / "paper.pdf").exists():
        pdfs.add("paper.pdf")
    canonical_surfaces = {
        "paper.pdf",
        "paper/paper.pdf",
        "paper/review/orius_review_dossier.pdf",
        "reports/publication/orius_review_dossier.pdf",
        "paper/ieee/orius_ieee_main.pdf",
        "paper/ieee/orius_ieee_detailed_main.pdf",
        "paper/ieee/orius_ieee_appendix.pdf",
        "paper/ieee/orius_ieee_professor_main.pdf",
        "paper/ieee/orius_ieee_professor_appendix_a.pdf",
        "paper/ieee/orius_ieee_professor_appendix_b.pdf",
        # Diagnostic surface: Phase 3 flagship closure proof book (committed evidence artifact).
        "reports/publication/phase3_flagship_v1_proof_book.pdf",
    }
    camera_ready_figure_exports = {
        rel_path
        for rel_path in pdfs
        if rel_path.startswith("paper/assets/figures/fig") or rel_path.startswith("reports/publication/fig_")
    }
    assert pdfs == canonical_surfaces | camera_ready_figure_exports
