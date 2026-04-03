from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SCAN_DIRS = [
    REPO_ROOT / "docs",
    REPO_ROOT / "frontmatter",
    REPO_ROOT / "paper",
    REPO_ROOT / "chapters",
    REPO_ROOT / "reports" / "publication",
    REPO_ROOT / "scripts",
]
ACTIVE_SCAN_SUFFIXES = {".md", ".tex", ".csv", ".json", ".py", ".yaml", ".yml", ".txt"}
LEGACY_PATTERN = re.compile(
    r"Paper~?[1-6]\b|Papers~?[1-6]|Papers~?2--6|program-first|battery-first"
)
MERGE_MARKER_PATTERN = re.compile(r"^(<<<<<<< .+|=======|>>>>>>> .+)$", re.MULTILINE)


def _iter_active_text_surfaces() -> list[Path]:
    files: list[Path] = [REPO_ROOT / "README.md", REPO_ROOT / "Makefile"]
    excluded_prefixes = {
        REPO_ROOT / "reports" / "legacy_archive",
        REPO_ROOT / "reports" / "publication" / "final_package_FINAL_20260317T024839Z",
        REPO_ROOT / "reports" / "publication" / "final_package_FINAL_20260317T030715Z",
        REPO_ROOT / "reports" / "publication" / "final_package_FINAL_BAT12_20260317T040740Z",
        REPO_ROOT / "reports" / "publication" / "final_package_FINAL_BATTERY_FULL_20260317T052750Z",
        REPO_ROOT / "reports" / "publication" / "final_package_FINAL_DEUS_20260317T034523Z",
        REPO_ROOT / "paper" / "bibliography",
    }
    excluded_suffixes = {
        ".aux",
        ".bbl",
        ".blg",
        ".log",
        ".lof",
        ".lot",
        ".out",
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
    }
    for directory in ACTIVE_SCAN_DIRS:
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if any(path.is_relative_to(prefix) for prefix in excluded_prefixes):
                continue
            if path.suffix.lower() in excluded_suffixes:
                continue
            if path.name in {"paper.bib", "orius_monograph.bib", "test_thesis_package_assets.py"}:
                continue
            if path.suffix.lower() not in ACTIVE_SCAN_SUFFIXES and path.name != "Makefile":
                continue
            files.append(path)
    return sorted(set(files))


def test_final_project_artifacts_exist() -> None:
    required = [
        REPO_ROOT / "docs" / "ORIUS_THESIS_TERMINOLOGY_GUIDE.md",
        REPO_ROOT / "docs" / "UNIVERSAL_KERNEL_ARCHITECTURE.md",
        REPO_ROOT / "docs" / "UNIVERSAL_BENCHMARK_SPEC.md",
        REPO_ROOT / "docs" / "UNIVERSAL_GOVERNANCE_SPEC.md",
        REPO_ROOT / "docs" / "BOUNDED_UNIVERSAL_CLOSURE_PROGRAM.md",
        REPO_ROOT / "reports" / "publication" / "orius_literature_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_framework_gap_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_maturity_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_monograph_chapter_map.csv",
        REPO_ROOT / "reports" / "publication" / "orius_equal_domain_parity_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_domain_closure_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_submission_scorecard.csv",
        REPO_ROOT / "reports" / "publication" / "orius_submission_scorecard.json",
        REPO_ROOT / "reports" / "publication" / "orius_submission_scorecard.md",
        REPO_ROOT / "reports" / "publication" / "orius_calibration_diagnostics_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_runtime_budget_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_governance_lifecycle_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_deployment_validation_scope.csv",
        REPO_ROOT / "reports" / "publication" / "orius_93plus_gap_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "orius_93plus_reviewer_rerun.csv",
        REPO_ROOT / "reports" / "publication" / "orius_93plus_closure_program.md",
        REPO_ROOT / "reports" / "publication" / "orius_refresh_lane_status.csv",
        REPO_ROOT / "reports" / "publication" / "orius_supplemental_hf_evidence.csv",
        REPO_ROOT / "reports" / "publication" / "orius_supplemental_hf_evidence.md",
        REPO_ROOT / "reports" / "publication" / "orius_fresh_results_package.md",
        REPO_ROOT / "reports" / "publication" / "orius_refresh_execution.json",
        REPO_ROOT / "reports" / "publication" / "orius_refresh_execution.md",
        REPO_ROOT / "reports" / "publication" / "orius_artifact_appendix.md",
        REPO_ROOT / "reports" / "publication" / "battery_deep_oqe_summary.csv",
        REPO_ROOT / "reports" / "publication" / "battery_deep_oqe_summary.json",
        REPO_ROOT / "reports" / "publication" / "battery_deep_oqe_summary.md",
        REPO_ROOT / "reports" / "publication" / "battery_deep_oqe_buckets.csv",
        REPO_ROOT / "reports" / "publication" / "battery_deep_oqe_safety_metrics.csv",
        REPO_ROOT / "reports" / "publication" / "battery_deep_oqe_safety_metrics.md",
        REPO_ROOT / "reports" / "publication" / "battery_raw_sequence_track_benchmark.csv",
        REPO_ROOT / "reports" / "publication" / "battery_raw_sequence_track_benchmark.md",
        REPO_ROOT / "reports" / "publication" / "battery_raw_sequence_track_slices.csv",
        REPO_ROOT / "reports" / "publication" / "battery_deep_learning_novelty_register.json",
        REPO_ROOT / "reports" / "publication" / "battery_deep_learning_novelty_register.md",
        REPO_ROOT / "reports" / "publication" / "fig_battery_deep_oqe_summary.png",
        REPO_ROOT / "reports" / "publication" / "fig_battery_deep_oqe_safety_metrics.png",
        REPO_ROOT / "reports" / "publication" / "fig_battery_raw_sequence_track_benchmark.png",
        REPO_ROOT / "reports" / "publication" / "fig_orius_equal_domain_gate_timeline.png",
        REPO_ROOT / "reports" / "publication" / "fig_orius_calibration_coverage_matrix.png",
        REPO_ROOT / "reports" / "publication" / "fig_orius_runtime_governance_matrix.png",
        REPO_ROOT / "reports" / "publication" / "aerospace_public_flight_runtime_summary.json",
        REPO_ROOT / "reports" / "publication" / "aerospace_public_flight_runtime_summary.csv",
        REPO_ROOT / "reports" / "publication" / "aerospace_public_flight_runtime_summary.md",
        REPO_ROOT / "reports" / "publication" / "aerospace_public_flight_governance_matrix.csv",
        REPO_ROOT / "reports" / "publication" / "aerospace_public_flight_calibration_diagnostics.csv",
        REPO_ROOT / "reports" / "publication" / "aerospace_public_flight_candidate_parity.csv",
        REPO_ROOT / "reports" / "final_thesis_submission_checklist.md",
        REPO_ROOT / "reports" / "final_thesis_submission_audit.md",
        REPO_ROOT / "reports" / "final_submission_reproducibility_note.md",
        REPO_ROOT / "reports" / "final_code_data_availability_statement.md",
        REPO_ROOT / "scripts" / "hf_jobs" / "README.md",
        REPO_ROOT / "scripts" / "hf_jobs" / "canonical_closure_refresh_job.py",
        REPO_ROOT / "scripts" / "hf_jobs" / "navigation_realdata_closure_job.py",
        REPO_ROOT / "scripts" / "hf_jobs" / "aerospace_flight_closure_job.py",
        REPO_ROOT / "scripts" / "hf_jobs" / "aerospace_public_adsb_runtime_job.py",
        REPO_ROOT / "scripts" / "hf_jobs" / "deep_learning_novelty_job.py",
        REPO_ROOT / "scripts" / "hf_jobs" / "calibration_diagnostics_job.py",
        REPO_ROOT / "scripts" / "hf_jobs" / "runtime_governance_trace_job.py",
        REPO_ROOT / "scripts" / "build_aerospace_public_adsb_runtime.py",
        REPO_ROOT / "scripts" / "build_aerospace_real_flight_dataset.py",
        REPO_ROOT / "scripts" / "run_orius_canonical_closure_refresh.py",
        REPO_ROOT / "scripts" / "run_battery_deep_novelty.py",
        REPO_ROOT / "src" / "orius" / "dc3s" / "deep_oqe.py",
        REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "tbl_battery_deep_oqe_summary.tex",
        REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "tbl_battery_deep_oqe_safety_metrics.tex",
        REPO_ROOT / "paper" / "assets" / "tables" / "generated" / "tbl_battery_raw_sequence_track.tex",
        REPO_ROOT / "paper" / "assets" / "figures" / "fig_battery_deep_oqe_summary.png",
        REPO_ROOT / "paper" / "assets" / "figures" / "fig_battery_deep_oqe_safety_metrics.png",
        REPO_ROOT / "paper" / "assets" / "figures" / "fig_battery_raw_sequence_track_benchmark.png",
        REPO_ROOT / "paper" / "assets" / "figures" / "fig_orius_equal_domain_gate_timeline.png",
        REPO_ROOT / "paper" / "assets" / "figures" / "fig_orius_calibration_coverage_matrix.png",
        REPO_ROOT / "paper" / "assets" / "figures" / "fig_orius_runtime_governance_matrix.png",
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


def test_active_surfaces_are_monograph_native() -> None:
    violations: list[str] = []
    for path in _iter_active_text_surfaces():
        text = path.read_text(encoding="utf-8")
        if MERGE_MARKER_PATTERN.search(text):
            violations.append(f"merge markers in {path.relative_to(REPO_ROOT)}")
        if LEGACY_PATTERN.search(text):
            violations.append(f"legacy narrative in {path.relative_to(REPO_ROOT)}")
    assert violations == []


def test_orius_expansion_is_consistent_on_core_surfaces() -> None:
    expected_ascii = "Observation--Reality Integrity for Universal Safety"
    expected_unicode = "Observation–Reality Integrity for Universal Safety"
    titlepage = (REPO_ROOT / "frontmatter" / "titlepage.tex").read_text(encoding="utf-8")
    abstract = (REPO_ROOT / "frontmatter" / "abstract.tex").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert expected_ascii in titlepage
    assert expected_ascii in abstract
    assert expected_unicode in readme


def test_monograph_depth_floor_is_preserved() -> None:
    bib_entries = sum(
        1 for line in (REPO_ROOT / "paper" / "bibliography" / "orius_monograph.bib").read_text(encoding="utf-8").splitlines() if line.startswith("@")
    )
    paper_log_path = REPO_ROOT / "paper" / "paper.log"
    match = None
    if paper_log_path.exists():
        match = re.search(r"Output written on paper/paper\.pdf \((\d+) pages", paper_log_path.read_text(encoding="utf-8"))
    pdf_pages = None
    if match is None:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "pdf_page_count.py"),
                str(REPO_ROOT / "paper" / "paper.pdf"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        pdf_pages = int(completed.stdout.strip())

    assert bib_entries >= 150
    if match is not None:
        assert int(match.group(1)) >= 450
    if pdf_pages is not None:
        assert pdf_pages >= 450


def test_compiled_chapters_40_to_44_are_evidence_complete() -> None:
    chapters = {
        "ch09_av_domain.tex": "av",
        "ch10_industrial_domain.tex": "industrial",
        "ch11_healthcare_domain.tex": "healthcare",
        "ch12_navigation_domain.tex": "navigation",
        "ch13_aerospace_domain.tex": "aerospace",
    }

    for filename, domain_id in chapters.items():
        text = (REPO_ROOT / "paper" / "monograph" / filename).read_text(encoding="utf-8")
        assert "Dataset and training surface" in text
        assert "Feature, split, and training protocol" in text
        assert "Replay and evidence surface" in text
        assert "Fallback and runtime behavior" in text
        assert "Evidence tier and promotion blocker" in text
        assert "Limitations and exact non-claims" in text
        assert f"tbl_{domain_id}_training_surface" in text
        assert f"tbl_{domain_id}_replay_surface" in text
        assert f"fig_{domain_id}_chapter_snapshot.png" in text

    synthesis = (REPO_ROOT / "paper" / "monograph" / "ch14_cross_domain_synthesis.tex").read_text(encoding="utf-8")
    roadmap = (REPO_ROOT / "paper" / "monograph" / "ch15_societal_impact_and_roadmap.tex").read_text(encoding="utf-8")
    assert "fig_orius_equal_domain_parity_matrix.png" in synthesis
    assert "fig_orius_equal_domain_gate_timeline.png" in synthesis
    assert "fig_orius_calibration_coverage_matrix.png" in synthesis
    assert "tbl_ch40_44_cross_domain_support" in synthesis
    assert "tbl_orius_submission_readiness" in synthesis
    assert "tbl_orius_calibration_diagnostics" in synthesis
    assert "fig_orius_runtime_governance_matrix.png" in roadmap
    aerospace = (REPO_ROOT / "paper" / "monograph" / "ch13_aerospace_domain.tex").read_text(encoding="utf-8")
    assert "tbl_aerospace_public_flight_support" in aerospace
    assert "Bounded public-flight support lane" in aerospace


def test_93plus_support_tables_are_referenced_in_manuscript_and_scripts() -> None:
    synthesis = (REPO_ROOT / "paper" / "monograph" / "ch14_cross_domain_synthesis.tex").read_text(encoding="utf-8")
    roadmap = (REPO_ROOT / "paper" / "monograph" / "ch15_societal_impact_and_roadmap.tex").read_text(encoding="utf-8")
    scorecard = (REPO_ROOT / "reports" / "publication" / "orius_submission_scorecard.md").read_text(encoding="utf-8")
    hf_jobs_readme = (REPO_ROOT / "scripts" / "hf_jobs" / "README.md").read_text(encoding="utf-8")

    assert "tbl_orius_submission_readiness" in synthesis
    assert "tbl_orius_refresh_lane_status" in synthesis
    assert "tbl_orius_calibration_diagnostics" in synthesis
    assert "tbl_orius_runtime_budget_matrix" in roadmap
    assert "tbl_orius_governance_lifecycle_matrix" in roadmap
    assert "tbl_orius_deployment_validation_scope" in roadmap
    assert "bounded_93_candidate" in scorecard
    assert "public_flight_93_candidate" in scorecard
    assert "equal_domain_93" in scorecard
    assert "official canonical lane" in (REPO_ROOT / "reports" / "publication" / "orius_fresh_results_package.md").read_text(encoding="utf-8").lower()
    assert "navigation_realdata_closure_job.py" in hf_jobs_readme
    assert "aerospace_flight_closure_job.py" in hf_jobs_readme
    assert "aerospace_public_adsb_runtime_job.py" in hf_jobs_readme
    assert "canonical_closure_refresh_job.py" in hf_jobs_readme


def test_canonical_closure_refresh_script_help_runs() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_orius_canonical_closure_refresh.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--mode" in completed.stdout
    assert "canonical_plus_hf_support" in completed.stdout
    assert "equal_domain_gate" in completed.stdout


def test_canonical_refresh_execution_manifest_has_no_user_path_leaks() -> None:
    payload = json.loads(
        (REPO_ROOT / "reports" / "publication" / "orius_refresh_execution.json").read_text(encoding="utf-8")
    )
    found: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                _walk(child)
        elif isinstance(value, list):
            for child in value:
                _walk(child)
        elif isinstance(value, str):
            if "/Users/" in value:
                found.append(value)

    _walk(payload)

    assert found == []


def test_battery_deep_learning_novelty_surfaces_are_referenced() -> None:
    chapter = (REPO_ROOT / "chapters" / "ch08_forecasting_calibration.tex").read_text(encoding="utf-8")
    hf_jobs_readme = (REPO_ROOT / "scripts" / "hf_jobs" / "README.md").read_text(encoding="utf-8")

    assert "Why Deep Models Do Not Win on the Engineered Track" in chapter
    assert "Battery-Scoped Deep-Learning Novelty Track" in chapter
    assert "DeepOQE: Learned Telemetry Reliability" in chapter
    assert "Raw-Sequence Probabilistic Forecasting" in chapter
    assert "Safety-Metric Evaluation" in chapter
    assert "tbl_battery_deep_oqe_summary.tex" in chapter
    assert "tbl_battery_deep_oqe_safety_metrics.tex" in chapter
    assert "tbl_battery_raw_sequence_track.tex" in chapter
    assert "fig_battery_deep_oqe_summary.png" in chapter
    assert "fig_battery_deep_oqe_safety_metrics.png" in chapter
    assert "fig_battery_raw_sequence_track_benchmark.png" in chapter
    assert "deep_learning_novelty_job.py" in hf_jobs_readme


def test_run_battery_deep_novelty_help_runs() -> None:
    completed = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_battery_deep_novelty.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--deep-oqe-epochs" in completed.stdout
    assert "--forecast-epochs" in completed.stdout
