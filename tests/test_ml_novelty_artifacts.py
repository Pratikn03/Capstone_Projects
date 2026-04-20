from __future__ import annotations

import csv
import duckdb
import json
import re
from pathlib import Path

from scripts.build_three_domain_ml_artifacts import CENTRAL_NOVELTY_SENTENCE


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = REPO_ROOT / "reports" / "publication"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalized(path: Path) -> str:
    return re.sub(r"\s+", " ", _read(path)).strip()


def _parquet_count(path: Path) -> int:
    con = duckdb.connect()
    try:
        return int(con.execute("select count(*) from read_parquet(?)", [str(path)]).fetchone()[0] or 0)
    finally:
        con.close()


def test_three_domain_ml_bundle_exists() -> None:
    required = [
        PUBLICATION / "three_domain_ml_benchmark.csv",
        PUBLICATION / "three_domain_ml_benchmark_summary.json",
        PUBLICATION / "three_domain_reliability_calibration.csv",
        PUBLICATION / "three_domain_grouped_coverage.csv",
        PUBLICATION / "three_domain_grouped_width.csv",
        PUBLICATION / "three_domain_nonvacuity_checks.json",
        PUBLICATION / "three_domain_runtime_safety_tradeoff.csv",
        PUBLICATION / "three_domain_baseline_suite.csv",
        PUBLICATION / "three_domain_ablation_matrix.csv",
        PUBLICATION / "three_domain_ablation_stats.json",
        PUBLICATION / "three_domain_negative_controls.csv",
        PUBLICATION / "novelty_separation_matrix.csv",
        PUBLICATION / "novelty_separation_matrix.json",
        PUBLICATION / "novelty_separation_matrix.md",
        PUBLICATION / "what_orius_is_not_matrix.csv",
        PUBLICATION / "what_orius_is_not_matrix.json",
        PUBLICATION / "what_orius_is_not_matrix.md",
        PUBLICATION / "three_domain_calibration_figures" / "grouped_coverage.png",
        PUBLICATION / "three_domain_calibration_figures" / "grouped_width.png",
        PUBLICATION / "three_domain_ablation_plots.png",
    ]
    missing = [str(path.relative_to(REPO_ROOT)) for path in required if not path.exists()]
    assert missing == []


def test_three_domain_benchmark_has_exact_domains_and_ci_fields() -> None:
    rows = list(csv.DictReader((PUBLICATION / "three_domain_ml_benchmark.csv").open()))
    assert {row["domain"] for row in rows} == {
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    }
    for row in rows:
        for field in (
            "baseline_tsvr_ci_low",
            "baseline_tsvr_ci_high",
            "orius_tsvr_ci_low",
            "orius_tsvr_ci_high",
            "absolute_delta",
            "relative_delta",
            "intervention_rate",
            "fallback_activation_rate",
            "certificate_valid_release_rate",
            "runtime_latency_p95_ms",
        ):
            assert row[field] != ""


def test_baseline_suite_has_required_families_for_each_domain() -> None:
    rows = list(csv.DictReader((PUBLICATION / "three_domain_baseline_suite.csv").open()))
    required = {
        "nominal_deterministic_controller",
        "fixed_threshold_or_fixed_inflation_runtime",
        "standard_conformal_nonreliability_runtime",
        "no_quality_signal_runtime",
        "no_adaptive_response_runtime",
        "no_temporal_guard_or_no_certificate_refresh_runtime",
        "orius_full_stack",
    }
    by_domain: dict[str, set[str]] = {}
    battery_rows = []
    av_healthcare_rows = []
    for row in rows:
        by_domain.setdefault(row["domain"], set()).add(row["baseline_family"])
        assert row["evidence_status"] != ""
        if row["domain"] == "Battery Energy Storage":
            battery_rows.append(row)
        else:
            av_healthcare_rows.append(row)
    for domain in (
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    ):
        assert by_domain[domain] == required
    assert battery_rows
    assert all(row["surface_role"] == "witness_row_comparator" for row in battery_rows)
    assert all(row["metric_surface"] == "locked_publication_witness" for row in battery_rows)
    assert all(row["evidence_status"] == "witness_grade_locked_surface" for row in battery_rows)
    assert av_healthcare_rows
    assert all(row["surface_role"] == "diagnostic_cross_domain_proxy" for row in av_healthcare_rows)


def test_ablation_matrix_has_required_rows_for_each_domain() -> None:
    rows = list(csv.DictReader((PUBLICATION / "three_domain_ablation_matrix.csv").open()))
    required = {
        "no_quality_signal",
        "no_reliability_conditioned_widening",
        "no_repair_release_without_repair",
        "no_fallback_or_no_temporal_guard",
        "no_certificate_refresh_stale_certificate_policy",
    }
    by_domain: dict[str, set[str]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], set()).add(row["ablation_name"])
        assert row["evidence_status"] != ""
    for domain in (
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    ):
        assert by_domain[domain] == required


def test_negative_controls_have_required_rows_for_each_domain() -> None:
    rows = list(csv.DictReader((PUBLICATION / "three_domain_negative_controls.csv").open()))
    required = {
        "actual_reliability",
        "shuffled_reliability_score",
        "delayed_reliability_score",
        "constant_low_reliability_conservative_policy",
        "stronger_predictor_without_runtime_adaptation",
    }
    by_domain: dict[str, set[str]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], set()).add(row["control_name"])
    for domain in (
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    ):
        assert by_domain[domain] == required


def test_grouped_coverage_has_nonempty_low_mid_high_buckets() -> None:
    rows = list(csv.DictReader((PUBLICATION / "three_domain_grouped_coverage.csv").open()))
    by_domain: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_domain.setdefault(row["domain"], {})[row["bucket_label"]] = row
    for domain in (
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    ):
        assert set(by_domain[domain].keys()) == {"low", "mid", "high"}
        assert all(int(by_domain[domain][bucket]["n"]) > 0 for bucket in ("low", "mid", "high"))


def test_novelty_and_nonclaim_matrices_cover_required_families() -> None:
    novelty_rows = list(csv.DictReader((PUBLICATION / "novelty_separation_matrix.csv").open()))
    novelty_families = {row["prior_work_family"] for row in novelty_rows}
    assert novelty_families == {
        "standard_conformal_prediction",
        "adaptive_conformal_prediction",
        "runtime_assurance_simplex",
        "safety_filters_barrier_methods_robust_mpc",
        "anomaly_detection_and_drift_detection",
        "generic_uncertainty_estimation",
    }

    nonclaim_rows = list(csv.DictReader((PUBLICATION / "what_orius_is_not_matrix.csv").open()))
    nonclaims = {row["boundary"] for row in nonclaim_rows}
    assert nonclaims == {
        "not_a_new_universal_controller",
        "not_a_new_conditional_coverage_theorem",
        "not_better_forecasting_by_default",
        "not_full_autonomous_driving_closure",
        "not_clinical_deployment_readiness",
    }


def test_central_novelty_sentence_is_consistent_across_flagship_surfaces() -> None:
    surfaces = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "executive_summary.md",
        REPO_ROOT / "docs" / "claim_ledger.md",
        REPO_ROOT / "paper" / "review" / "orius_review_dossier.tex",
        REPO_ROOT / "paper" / "monograph" / "ch01_introduction_and_thesis_claims.tex",
        REPO_ROOT / "paper" / "monograph" / "ch02_related_work_and_novelty_gap.tex",
        REPO_ROOT / "reports" / "publication" / "README.md",
    ]
    normalized_sentence = re.sub(r"\s+", " ", CENTRAL_NOVELTY_SENTENCE).strip()
    for path in surfaces:
        assert normalized_sentence in _normalized(path), path


def test_flagship_ml_surfaces_avoid_stronger_than_implemented_calibration_language() -> None:
    surfaces = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "executive_summary.md",
        REPO_ROOT / "docs" / "claim_ledger.md",
        REPO_ROOT / "paper" / "review" / "orius_review_dossier.tex",
        REPO_ROOT / "paper" / "monograph" / "ch01_introduction_and_thesis_claims.tex",
        REPO_ROOT / "paper" / "monograph" / "ch02_related_work_and_novelty_gap.tex",
        REPO_ROOT / "paper" / "monograph" / "ch08_witness_results_and_failure_analysis.tex",
        REPO_ROOT / "reports" / "publication" / "README.md",
    ]
    forbidden = ("conditional coverage", "group-conditional coverage", "new conformal theorem")
    for path in surfaces:
        lowered = _read(path).lower()
        for token in forbidden:
            assert token not in lowered, f"{token} leaked in {path.relative_to(REPO_ROOT)}"


def test_benchmark_summary_repeats_central_novelty_sentence() -> None:
    summary = json.loads((PUBLICATION / "three_domain_ml_benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["central_novelty_sentence"] == CENTRAL_NOVELTY_SENTENCE


def test_battery_deep_oqe_summary_is_diagnostic_only_and_repo_relative() -> None:
    payload = json.loads((PUBLICATION / "battery_deep_oqe_summary.json").read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["artifact_status"] == "diagnostic_only"
    assert summary["diagnostic_only"] is True
    assert summary["preferred_backend"] == "heuristic"
    assert "diagnostic-only" in summary["preference_reason"].lower()
    assert "lower held-out reliability mae" in summary["preference_reason"].lower()
    assert not summary["model_path"].startswith("/")
    assert summary["model_path"].startswith("artifacts/")


def test_healthcare_nonvacuity_split_matches_live_split_parquets() -> None:
    payload = json.loads((PUBLICATION / "three_domain_nonvacuity_checks.json").read_text(encoding="utf-8"))
    split_meta = payload["healthcare_calibration_split"]
    splits_dir = REPO_ROOT / "data" / "healthcare" / "processed" / "splits"

    assert split_meta["calibration_rows"] == _parquet_count(splits_dir / "calibration.parquet")
    assert split_meta["evaluation_rows"] == (
        _parquet_count(splits_dir / "val.parquet") + _parquet_count(splits_dir / "test.parquet")
    )
