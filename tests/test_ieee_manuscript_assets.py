from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CLAIM_STATUSES = {
    "current_repo_supported",
    "supported_by_primary_literature_only",
    "visionary_requires_closure_work",
}


def _pdf_page_count(path: Path) -> int:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "pdf_page_count.py"),
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip())


def test_ieee_family_uses_ieeetran_and_is_separated_from_legacy_short_paper() -> None:
    main_tex = (REPO_ROOT / "paper" / "ieee" / "orius_ieee_main.tex").read_text(encoding="utf-8")
    appendix_tex = (REPO_ROOT / "paper" / "ieee" / "orius_ieee_appendix.tex").read_text(encoding="utf-8")
    legacy_tex = (REPO_ROOT / "paper" / "paper_r1.tex").read_text(encoding="utf-8")

    assert r"\documentclass[10pt,journal]{IEEEtran}" in main_tex
    assert r"\documentclass[10pt,journal,onecolumn]{IEEEtran}" in appendix_tex
    assert "Legacy non-canonical battery-centric short draft." in legacy_tex
    assert "Superseded by paper/ieee/orius_ieee_main.tex" in legacy_tex


def test_ieee_main_keeps_conclusion_before_bibliography_without_appendix_afterward() -> None:
    main_tex = (REPO_ROOT / "paper" / "ieee" / "orius_ieee_main.tex").read_text(encoding="utf-8")
    conclusion_marker = r"\input{sections/ieee_conclusion.tex}"
    bibliography_marker = r"\bibliography{../bibliography/orius_monograph}"

    assert conclusion_marker in main_tex
    assert bibliography_marker in main_tex
    assert main_tex.index(conclusion_marker) < main_tex.index(bibliography_marker)
    assert "appendices/" not in main_tex.split(conclusion_marker, 1)[1]


def test_ieee_main_includes_all_six_domain_rows() -> None:
    main_tex = (REPO_ROOT / "paper" / "ieee" / "orius_ieee_main.tex").read_text(encoding="utf-8")
    domain_section = (REPO_ROOT / "paper" / "ieee" / "sections" / "ieee_domain_instantiations.tex").read_text(encoding="utf-8")

    assert r"\input{sections/ieee_battery_witness.tex}" in main_tex
    assert r"\input{sections/ieee_domain_instantiations.tex}" in main_tex
    for phrase in [
        "Battery energy storage",
        "Autonomous vehicles",
        "Industrial process control",
        "Healthcare monitoring",
        "Navigation and guidance",
        "Aerospace control",
    ]:
        assert phrase in domain_section


def test_shared_bibliography_depth_and_ieee_benchmark_corpus_floor() -> None:
    bib_text = (REPO_ROOT / "paper" / "bibliography" / "orius_monograph.bib").read_text(encoding="utf-8")
    benchmark_path = REPO_ROOT / "reports" / "publication" / "orius_top_tier_benchmark_corpus.csv"
    with benchmark_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    bib_entries = sum(1 for line in bib_text.splitlines() if line.startswith("@"))
    assert bib_entries >= 220
    assert len(rows) >= 100


def test_claim_delta_ledgers_exist_and_use_only_allowed_status_values() -> None:
    csv_path = REPO_ROOT / "reports" / "editorial" / "orius_claim_delta_ledger.csv"
    md_path = REPO_ROOT / "reports" / "editorial" / "orius_claim_delta_ledger.md"
    revision_path = REPO_ROOT / "reports" / "editorial" / "orius_flagship_revision_ledger.csv"

    assert csv_path.exists()
    assert md_path.exists()
    assert revision_path.exists()

    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert rows
    assert {row["support_status"] for row in rows} <= ALLOWED_CLAIM_STATUSES


def test_makefile_exposes_ieee_targets_and_long_draft_floor() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "ieee-assets" in makefile
    assert "ieee-main-compile" in makefile
    assert "ieee-appendix-compile" in makefile
    assert "ieee-pack" in makefile
    assert "ieee-prof-assets" in makefile
    assert "ieee-prof-main-compile" in makefile
    assert "ieee-prof-appa-compile" in makefile
    assert "ieee-prof-appb-compile" in makefile
    assert "ieee-prof-pack" in makefile
    assert "orius-flagship-manuscripts" in makefile
    assert "IEEE_MIN_PAGES ?= 8" in makefile
    assert "IEEE_PROF_MAIN_MIN_PAGES ?= 20" in makefile
    assert "IEEE_PROF_APP_MIN_PAGES ?= 20" in makefile


def test_compiled_ieee_outputs_exist_and_main_is_review_ready_split_draft() -> None:
    main_pdf = REPO_ROOT / "paper" / "ieee" / "orius_ieee_main.pdf"
    appendix_pdf = REPO_ROOT / "paper" / "ieee" / "orius_ieee_appendix.pdf"

    assert main_pdf.exists()
    assert appendix_pdf.exists()
    assert _pdf_page_count(main_pdf) >= 8
    assert _pdf_page_count(appendix_pdf) >= 20


def test_ieee_log_records_review_ready_double_column_page_count() -> None:
    log_path = REPO_ROOT / "paper" / "ieee" / "orius_ieee_main.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="latin-1")
        match = re.search(r"Output written on paper/ieee/orius_ieee_main\.pdf \((\d+) pages", log_text)

        assert match is not None
        assert int(match.group(1)) >= 8
        return

    assert _pdf_page_count(REPO_ROOT / "paper" / "ieee" / "orius_ieee_main.pdf") >= 8


def test_professor_ieee_family_exists_and_uses_separate_main_and_appendices() -> None:
    main_tex = (REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_main.tex").read_text(encoding="utf-8")
    app_a_tex = (REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_appendix_a.tex").read_text(encoding="utf-8")
    app_b_tex = (REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_appendix_b.tex").read_text(encoding="utf-8")

    assert r"\documentclass[10pt,journal]{IEEEtran}" in main_tex
    assert r"\documentclass[10pt,journal,onecolumn]{IEEEtran}" in app_a_tex
    assert r"\documentclass[10pt,journal,onecolumn]{IEEEtran}" in app_b_tex
    assert r"\input{professor_sections/prof_intro.tex}" in main_tex
    assert r"\input{professor_sections/prof_problem_oasg.tex}" in main_tex
    assert r"\input{professor_sections/prof_conclusion.tex}" in main_tex
    assert r"\input{professor_sections/prof_appendix_a_formal.tex}" in app_a_tex
    assert r"\input{professor_sections/prof_appendix_b_runtime.tex}" in app_b_tex
    assert r"\input{professor_sections/prof_appendix_b_empirical.tex}" in app_b_tex


def test_professor_main_keeps_conclusion_before_bibliography_without_appendix_afterward() -> None:
    main_tex = (REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_main.tex").read_text(encoding="utf-8")
    conclusion_marker = r"\input{professor_sections/prof_conclusion.tex}"
    bibliography_marker = r"\bibliography{../bibliography/orius_monograph}"

    assert conclusion_marker in main_tex
    assert bibliography_marker in main_tex
    assert main_tex.index(conclusion_marker) < main_tex.index(bibliography_marker)
    assert "appendix" not in main_tex.split(conclusion_marker, 1)[1].lower()


def test_professor_sections_and_generated_support_assets_exist() -> None:
    for relative in [
        "paper/ieee/professor_sections/prof_intro.tex",
        "paper/ieee/professor_sections/prof_problem_oasg.tex",
        "paper/ieee/professor_sections/prof_related_work.tex",
        "paper/ieee/professor_sections/prof_runtime_architecture.tex",
        "paper/ieee/professor_sections/prof_runtime_contracts.tex",
        "paper/ieee/professor_sections/prof_theory_bridge.tex",
        "paper/ieee/professor_sections/prof_battery_results.tex",
        "paper/ieee/professor_sections/prof_results.tex",
        "paper/ieee/professor_sections/prof_cross_domain.tex",
        "paper/ieee/professor_sections/prof_evidence_boundary.tex",
        "paper/ieee/professor_sections/prof_conclusion.tex",
        "paper/ieee/generated/orius_professor_parity_table.tex",
        "paper/ieee/generated/orius_professor_runtime_governance_table.tex",
        "paper/ieee/generated/orius_professor_battery_witness_table.tex",
    ]:
        assert (REPO_ROOT / relative).exists(), relative


def test_compiled_professor_outputs_exist_and_meet_page_floors() -> None:
    main_pdf = REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_main.pdf"
    app_a_pdf = REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_appendix_a.pdf"
    app_b_pdf = REPO_ROOT / "paper" / "ieee" / "orius_ieee_professor_appendix_b.pdf"

    assert main_pdf.exists()
    assert app_a_pdf.exists()
    assert app_b_pdf.exists()
    assert _pdf_page_count(main_pdf) >= 20
    assert _pdf_page_count(app_a_pdf) >= 20
    assert _pdf_page_count(app_b_pdf) >= 20
