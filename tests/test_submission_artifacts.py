from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_surviving_papers_use_main_tex_only() -> None:
    for directory in ("oasg_signature", "no_free_safety_principle", "reliability_weighted_cp"):
        assert (PAPERS / directory / "main.tex").exists()
        assert not (PAPERS / directory / "paper.tex").exists()


def test_removed_submission_dirs_are_absent() -> None:
    for directory in ("certificate_half_life", "inflation_law_derived", "reliability_constraints"):
        assert not (PAPERS / directory).exists()


def test_rwcp_false_theorem_language_removed() -> None:
    text = _read(PAPERS / "reliability_weighted_cp" / "main.tex").lower()
    assert "conditional marginal coverage" not in text
    assert "theorem rwcp" not in text
    assert "weighted exchangeability" in text
    assert "does not" in text


def test_oasg_formula_text_matches_canonical_definition() -> None:
    text = _read(PAPERS / "oasg_signature" / "main.tex")
    assert r"\Sigma_{\mathrm{OASG}} = \rho_{\mathrm{exp}} \cdot s_{\mathrm{sev}}" in text
    assert r"d(x_t,\mathcal{C})^+ \cdot \mathbf{1}[\mathrm{gap}_t]" in text


def test_no_fabricated_out_of_scope_numeric_rows() -> None:
    forbidden_domains = ("Healthcare", "Industrial", "Navigation", "Aerospace")
    for paper in PAPERS.glob("*/main.tex"):
        text = _read(paper)
        assert "illustrative values" not in text.lower()
        for line in text.splitlines():
            if any(domain.lower() in line.lower() for domain in forbidden_domains):
                assert re.search(r"\d", line) is None
