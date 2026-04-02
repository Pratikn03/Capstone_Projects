from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_final_project_artifacts_exist() -> None:
    required = [
        REPO_ROOT / "docs" / "ORIUS_THESIS_TERMINOLOGY_GUIDE.md",
        REPO_ROOT / "docs" / "UNIVERSAL_KERNEL_ARCHITECTURE.md",
        REPO_ROOT / "docs" / "UNIVERSAL_BENCHMARK_SPEC.md",
        REPO_ROOT / "docs" / "UNIVERSAL_GOVERNANCE_SPEC.md",
        REPO_ROOT / "reports" / "publication" / "orius_literature_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_framework_gap_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_maturity_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_artifact_appendix.md",
        REPO_ROOT / "reports" / "final_thesis_submission_checklist.md",
        REPO_ROOT / "reports" / "final_thesis_submission_audit.md",
        REPO_ROOT / "reports" / "final_submission_reproducibility_note.md",
        REPO_ROOT / "reports" / "final_code_data_availability_statement.md",
    ]

    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    assert missing == []


def test_docs_index_points_to_canonical_manuscript_and_universal_specs() -> None:
    text = (REPO_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "paper/paper.tex" in text
    assert "UNIVERSAL_KERNEL_ARCHITECTURE.md" in text
    assert "UNIVERSAL_BENCHMARK_SPEC.md" in text
    assert "UNIVERSAL_GOVERNANCE_SPEC.md" in text


def test_submission_governance_points_to_canonical_manuscript() -> None:
    metrics_manifest = (REPO_ROOT / "paper" / "metrics_manifest.json").read_text(encoding="utf-8")
    release_manifest = (REPO_ROOT / "reports" / "publication" / "release_manifest.json").read_text(encoding="utf-8")
    assert '"master_manuscript": "paper/paper.tex"' in metrics_manifest
    assert '"master_manuscript": "paper/paper.tex"' in release_manifest


def test_submission_frontmatter_has_no_draft_or_committee_markers() -> None:
    titlepage = (REPO_ROOT / "frontmatter" / "titlepage.tex").read_text(encoding="utf-8").lower()
    acknowledgments = (REPO_ROOT / "frontmatter" / "acknowledgments.tex").read_text(encoding="utf-8").lower()
    abstract = (REPO_ROOT / "frontmatter" / "abstract.tex").read_text(encoding="utf-8").lower()

    assert "draft" not in titlepage
    assert "committee" not in titlepage
    assert "advisor" not in titlepage
    assert "draft" not in acknowledgments
    assert "draft" not in abstract


def test_canonical_manuscript_uses_ieee_bibliography_style() -> None:
    paper_tex = (REPO_ROOT / "paper" / "paper.tex").read_text(encoding="utf-8")
    assert "\\bibliographystyle{IEEEtran}" in paper_tex
