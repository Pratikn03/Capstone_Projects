from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import duckdb

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
        PUBLICATION / "three_domain_forecast_calibration_runtime_evidence.csv",
        PUBLICATION / "three_domain_forecast_calibration_runtime_evidence.json",
        PUBLICATION / "three_domain_reliability_calibration.csv",
        PUBLICATION / "three_domain_grouped_coverage.csv",
        PUBLICATION / "three_domain_grouped_width.csv",
        PUBLICATION / "three_domain_nonvacuity_checks.json",
        PUBLICATION / "three_domain_runtime_safety_tradeoff.csv",
        PUBLICATION / "three_domain_baseline_suite.csv",
        PUBLICATION / "three_domain_ablation_matrix.csv",
        PUBLICATION / "three_domain_ablation_stats.json",
        PUBLICATION / "three_domain_negative_controls.csv",
        PUBLICATION / "equal_domain_artifact_discipline.csv",
        PUBLICATION / "equal_domain_artifact_discipline.json",
        PUBLICATION / "equal_domain_artifact_discipline.md",
        PUBLICATION / "equal_domain_reproducibility_manifest.json",
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


def _csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open()))


def _csv_record_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _uniform_evidence_rows() -> dict[str, dict[str, str]]:
    rows = _csv_rows(PUBLICATION / "three_domain_forecast_calibration_runtime_evidence.csv")
    return {row["domain"]: row for row in rows}


def test_uniform_evidence_table_has_exact_domains_and_numeric_schema() -> None:
    rows = _uniform_evidence_rows()
    assert set(rows) == {
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    }
    numeric_fields = (
        "forecast_coverage",
        "forecast_interval_width",
        "calibration_bucket_count",
        "calibration_min_coverage",
        "calibration_min_bucket_n",
        "runtime_trace_rows",
        "orius_runtime_rows",
        "baseline_tsvr",
        "orius_tsvr",
        "oasg",
        "intervention_rate",
        "fallback_activation_rate",
        "certificate_valid_rate",
        "t11_pass_rate",
        "domain_postcondition_pass_rate",
        "runtime_witness_pass_rate",
    )
    for row in rows.values():
        for field in numeric_fields:
            assert row[field] != "", field
            float(row[field])
        assert row["strict_runtime_gate"] in {"True", "False"}
        assert row["claim_boundary"]

    payload = json.loads(
        (PUBLICATION / "three_domain_forecast_calibration_runtime_evidence.json").read_text()
    )
    assert payload["row_count"] == 3
    assert {row["domain"] for row in payload["rows"]} == set(rows)


def test_uniform_evidence_av_is_nuplan_only() -> None:
    av = _uniform_evidence_rows()["Autonomous Vehicles"]
    assert av["active_dataset"] == "nuPlan all-zip grouped replay"
    assert av["forecast_source"] == "reports/orius_av/nuplan_allzip_grouped/training_summary.csv"
    assert av["forecast_target"] == "ego_speed_mps_1s"
    forbidden = ("Waymo Motion", "reports/av/week2_metrics.json", "waymo")
    joined = " ".join(av.values())
    for token in forbidden:
        assert token not in joined


def test_uniform_evidence_postcondition_rates_match_runtime_authorities() -> None:
    rows = _uniform_evidence_rows()
    battery_trace_rows = [
        row
        for row in _csv_rows(REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_traces.csv")
        if row["controller_label"] == "heuristic:dc3s_ftit"
    ]
    battery_pass = sum(row["true_constraint_violated"] == "False" for row in battery_trace_rows) / len(
        battery_trace_rows
    )
    assert rows["Battery Energy Storage"]["postcondition_basis"] == "battery_true_state_witness_postcondition"
    assert float(rows["Battery Energy Storage"]["domain_postcondition_pass_rate"]) == round(battery_pass, 6)

    witness = json.loads((PUBLICATION / "domain_runtime_contract_summary.json").read_text())["domains"]
    for domain, witness_key in {
        "Autonomous Vehicles": "av",
        "Medical and Healthcare Monitoring": "healthcare",
    }.items():
        assert float(rows[domain]["domain_postcondition_pass_rate"]) == round(
            float(witness[witness_key]["postcondition_pass_rate"]),
            6,
        )
        assert float(rows[domain]["runtime_witness_pass_rate"]) == round(
            float(witness[witness_key]["witness_pass_rate"]),
            6,
        )


def test_uniform_evidence_runtime_rows_match_promoted_sources() -> None:
    rows = _uniform_evidence_rows()
    battery_summary = {
        row["controller"]: row
        for row in _csv_rows(REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_summary.csv")
    }
    av_summary = {
        row["controller"]: row
        for row in _csv_rows(
            REPO_ROOT
            / "reports"
            / "orius_av"
            / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
            / "runtime_summary.csv"
        )
    }
    healthcare_summary = {
        row["controller"]: row
        for row in _csv_rows(REPO_ROOT / "reports" / "healthcare" / "runtime_summary.csv")
    }

    assert int(rows["Battery Energy Storage"]["runtime_trace_rows"]) == _csv_record_count(
        REPO_ROOT / "reports" / "battery_av" / "battery" / "runtime_traces.csv"
    )
    assert int(rows["Battery Energy Storage"]["orius_runtime_rows"]) == int(
        float(battery_summary["heuristic:dc3s_ftit"]["n_steps"])
    )
    assert int(rows["Autonomous Vehicles"]["runtime_trace_rows"]) == sum(
        int(float(row["n_steps"])) for row in av_summary.values()
    )
    assert int(rows["Autonomous Vehicles"]["orius_runtime_rows"]) == int(
        float(av_summary["orius"]["n_steps"])
    )
    assert int(rows["Medical and Healthcare Monitoring"]["runtime_trace_rows"]) == _csv_record_count(
        REPO_ROOT / "reports" / "healthcare" / "runtime_traces.csv"
    )
    assert int(rows["Medical and Healthcare Monitoring"]["orius_runtime_rows"]) == int(
        float(healthcare_summary["orius"]["n_steps"])
    )


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
    assert all(
        row["metric_surface"] in {"locked_publication_witness", "runtime_denominator"} for row in battery_rows
    )
    assert all("proxy" not in row["evidence_status"] for row in battery_rows)
    assert av_healthcare_rows
    assert all(row["surface_role"] == "runtime_native_domain_comparator" for row in av_healthcare_rows)
    assert all(row["metric_surface"] == "runtime_denominator" for row in av_healthcare_rows)
    assert all("proxy" not in row["evidence_status"] for row in av_healthcare_rows)
    assert all(row["evidence_status"] != "missing" for row in av_healthcare_rows)


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
        assert row["metric_surface"] == "runtime_denominator"
        assert "proxy" not in row["evidence_status"]
        assert row["evidence_status"] != "missing"
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
        assert row["surface"] == "runtime_denominator"
        assert row["status"] == "runtime_native_available"
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
        "runtime_monitoring_and_supervisory_veto",
        "runtime_assurance_simplex",
        "safety_filters_barrier_methods_robust_mpc",
        "anomaly_detection_and_drift_detection",
        "generic_uncertainty_estimation",
    }

    nonclaim_rows = list(csv.DictReader((PUBLICATION / "what_orius_is_not_matrix.csv").open()))
    nonclaims = {row["boundary"] for row in nonclaim_rows}
    assert nonclaims == {
        "not_a_new_conformal_method",
        "not_a_new_robust_optimization_primitive",
        "not_a_runtime_monitor_or_simplex_clone",
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
