from __future__ import annotations

import csv
from pathlib import Path

from scripts import build_utility_preserving_safety_scorecard as builder
from scripts.validate_utility_preserving_safety import (
    validate_scorecard,
    validate_utility_safety_outputs,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_scorecard_builds_three_domain_utility_preserving_rows(tmp_path: Path, monkeypatch) -> None:
    battery = tmp_path / "graceful_four_policy_metrics.csv"
    dominance = tmp_path / "three_domain_utility_safety_dominance.csv"
    _write_csv(
        battery,
        [
            {
                "policy": "blind_persistence",
                "useful_work_mwh_mean": 10.0,
                "violation_rate_mean": 0.2,
            },
            {
                "policy": "immediate_shutdown",
                "useful_work_mwh_mean": 0.0,
                "violation_rate_mean": 0.0,
            },
            {
                "policy": "optimized_graceful",
                "useful_work_mwh_mean": 6.0,
                "violation_rate_mean": 0.0,
            },
        ],
    )
    _write_csv(
        dominance,
        [
            {
                "domain": "Autonomous Vehicles",
                "runtime_surface": "bounded_closed_loop_planner",
                "source_surface": "av.csv",
                "safety_reference_controller": "always_brake",
                "orius_tsvr": 0.1,
                "safety_reference_tsvr": 0.1,
                "baseline_tsvr": 0.3,
                "excess_tsvr_over_safety_reference": 0.0,
                "orius_fallback_activation_rate": 0.3,
                "safety_reference_fallback_activation_rate": 1.0,
                "fallback_reduction_vs_safety_reference": 0.7,
                "orius_intervention_rate": 0.5,
                "safety_reference_intervention_rate": 1.0,
                "intervention_reduction_vs_safety_reference": 0.5,
                "orius_useful_work_total": 100.0,
                "safety_reference_useful_work_total": 10.0,
                "utility_gain_over_safety_reference": 10.0,
                "utility_delta_over_safety_reference": 90.0,
                "nonvacuous_utility_gate": "True",
                "claim_boundary": "bounded, not road deployment",
            },
            {
                "domain": "Medical and Healthcare Monitoring",
                "runtime_surface": "retrospective_fail_safe_release",
                "source_surface": "hc.csv",
                "safety_reference_controller": "always_alert",
                "orius_tsvr": 0.0,
                "safety_reference_tsvr": 0.0,
                "baseline_tsvr": 0.2,
                "excess_tsvr_over_safety_reference": 0.0,
                "orius_fallback_activation_rate": 0.4,
                "safety_reference_fallback_activation_rate": 1.0,
                "fallback_reduction_vs_safety_reference": 0.6,
                "orius_intervention_rate": 0.4,
                "safety_reference_intervention_rate": 1.0,
                "intervention_reduction_vs_safety_reference": 0.6,
                "orius_useful_work_total": 50.0,
                "safety_reference_useful_work_total": 0.0,
                "utility_gain_over_safety_reference": "inf",
                "utility_delta_over_safety_reference": 50.0,
                "nonvacuous_utility_gate": "True",
                "claim_boundary": "retrospective, not live clinical deployment",
            },
        ],
    )
    monkeypatch.setattr(builder, "BATTERY_T8", battery)
    monkeypatch.setattr(builder, "THREE_DOMAIN_UTILITY", dominance)

    rows = builder.build_scorecard()

    by_domain = {row["domain"]: row for row in rows}
    assert set(by_domain) == {
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    }
    assert by_domain["Battery Energy Storage"]["utility_preserving_safety_gate"] == "True"
    assert by_domain["Autonomous Vehicles"]["utility_preserving_safety_gate"] == "True"
    assert by_domain["Medical and Healthcare Monitoring"]["utility_gain_over_safety_reference"] == "inf"


def test_claim_tables_separate_predictor_safety_from_failsafe_conservatism(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scorecard_rows = [
        {
            "domain": "Battery Energy Storage",
            "source_artifacts": "battery.csv",
            "safety_reference_controller": "immediate_shutdown",
            "orius_controller": "optimized_graceful",
            "orius_tsvr": "0.000000",
            "safety_reference_tsvr": "0.000000",
            "orius_utility": "10.000000",
            "safety_reference_utility": "0.000000",
            "utility_delta_over_safety_reference": "10.000000",
            "orius_intervention_rate": "",
            "safety_reference_intervention_rate": "",
            "orius_fallback_rate": "",
            "safety_reference_fallback_rate": "",
        },
        {
            "domain": "Autonomous Vehicles",
            "source_artifacts": "av.csv",
            "safety_reference_controller": "always_brake",
            "orius_controller": "orius",
            "orius_tsvr": "0.000163",
            "safety_reference_tsvr": "0.000000",
            "orius_utility": "116191.503707",
            "safety_reference_utility": "69226.115673",
            "utility_delta_over_safety_reference": "46965.388034",
            "orius_intervention_rate": "0.500301",
            "safety_reference_intervention_rate": "1.000000",
            "orius_fallback_rate": "0.173716",
            "safety_reference_fallback_rate": "1.000000",
        },
        {
            "domain": "Medical and Healthcare Monitoring",
            "source_artifacts": "hc.csv",
            "safety_reference_controller": "always_alert",
            "orius_controller": "orius",
            "orius_tsvr": "0.000000",
            "safety_reference_tsvr": "0.000000",
            "orius_utility": "142767.000000",
            "safety_reference_utility": "0.000000",
            "utility_delta_over_safety_reference": "142767.000000",
            "orius_intervention_rate": "0.479907",
            "safety_reference_intervention_rate": "1.000000",
            "orius_fallback_rate": "0.479907",
            "safety_reference_fallback_rate": "1.000000",
        },
    ]
    baseline_suite = tmp_path / "three_domain_baseline_suite.csv"
    negative_controls = tmp_path / "three_domain_negative_controls.csv"
    _write_csv(
        baseline_suite,
        [
            {
                "domain": "Battery Energy Storage",
                "baseline_family": "orius_full_stack",
                "implemented_controller": "deep:dc3s_ftit",
                "tsvr": "0.000000",
                "intervention_rate": "0.020833",
                "fallback_activation_rate": "0.020833",
                "useful_work_total": "572.539327",
            },
            {
                "domain": "Battery Energy Storage",
                "baseline_family": "no_quality_signal_runtime",
                "implemented_controller": "deep:dc3s_wrapped",
                "tsvr": "0.000000",
                "intervention_rate": "0.000000",
                "fallback_activation_rate": "0.000000",
                "useful_work_total": "576.000000",
            },
            {
                "domain": "Autonomous Vehicles",
                "baseline_family": "orius_full_stack",
                "implemented_controller": "orius",
                "tsvr": "0.000163",
                "intervention_rate": "0.500301",
                "fallback_activation_rate": "0.173716",
                "useful_work_total": "116191.503707",
            },
            {
                "domain": "Autonomous Vehicles",
                "baseline_family": "no_quality_signal_runtime",
                "implemented_controller": "predictor_only_no_runtime",
                "tsvr": "0.289309",
                "intervention_rate": "0.000000",
                "fallback_activation_rate": "0.000000",
                "useful_work_total": "232083.171893",
            },
            {
                "domain": "Medical and Healthcare Monitoring",
                "baseline_family": "orius_full_stack",
                "implemented_controller": "orius",
                "tsvr": "0.000000",
                "intervention_rate": "0.479907",
                "fallback_activation_rate": "0.479907",
                "useful_work_total": "142767.000000",
            },
            {
                "domain": "Medical and Healthcare Monitoring",
                "baseline_family": "no_quality_signal_runtime",
                "implemented_controller": "predictor_only_no_runtime",
                "tsvr": "0.200420",
                "intervention_rate": "0.000000",
                "fallback_activation_rate": "0.000000",
                "useful_work_total": "109752.000000",
            },
        ],
    )
    _write_csv(
        negative_controls,
        [
            {
                "domain": "Battery Energy Storage",
                "control_name": "stronger_predictor_without_runtime_adaptation",
                "coverage_gap_abs_mean": "0.000000",
                "mean_interval_width": "0.000000",
            },
            {
                "domain": "Autonomous Vehicles",
                "control_name": "stronger_predictor_without_runtime_adaptation",
                "coverage_gap_abs_mean": "0.289250",
                "mean_interval_width": "0.000000",
            },
            {
                "domain": "Medical and Healthcare Monitoring",
                "control_name": "stronger_predictor_without_runtime_adaptation",
                "coverage_gap_abs_mean": "0.194489",
                "mean_interval_width": "0.000000",
            },
        ],
    )
    monkeypatch.setattr(builder, "THREE_DOMAIN_BASELINE_SUITE", baseline_suite)
    monkeypatch.setattr(builder, "THREE_DOMAIN_NEGATIVE_CONTROLS", negative_controls)

    rows = builder.build_claim_comparison_rows(scorecard_rows)

    by_key = {(row["domain"], row["comparison"]): row for row in rows}
    av_predictor = by_key["Autonomous Vehicles", "predictor_only_safety"]
    battery_predictor = by_key["Battery Energy Storage", "predictor_only_safety"]
    hc_failsafe = by_key[
        "Medical and Healthcare Monitoring",
        "shutdown_or_fallback_only_conservatism",
    ]
    assert av_predictor["claim_relation"] == "safer_than_predictor_only"
    assert av_predictor["absolute_tsvr_reduction"] == "0.289087"
    assert battery_predictor["comparability"] == "non_comparable_no_observed_safety_separation"
    assert hc_failsafe["claim_relation"] == "less_conservative_than_shutdown_or_fallback_only"


def test_ablation_surfaces_cover_required_slots_without_inventing_missing_uncertainty(
    tmp_path: Path,
    monkeypatch,
) -> None:
    baseline_suite = tmp_path / "three_domain_baseline_suite.csv"
    ablation_matrix = tmp_path / "three_domain_ablation_matrix.csv"
    security_matrix = tmp_path / "security_governance_ablation_matrix.csv"
    _write_csv(
        baseline_suite,
        [
            {
                "domain": "Autonomous Vehicles",
                "baseline_family": "fixed_threshold_or_fixed_inflation_runtime",
                "implemented_controller": "robust_fixed_deceleration",
            },
            {
                "domain": "Autonomous Vehicles",
                "baseline_family": "nominal_deterministic_controller",
                "implemented_controller": "baseline",
                "tsvr": "0.289250",
                "intervention_rate": "0.000000",
            },
        ],
    )
    _write_csv(
        ablation_matrix,
        [
            {
                "domain": "Autonomous Vehicles",
                "ablation_name": "no_reliability_conditioned_widening",
                "baseline_family": "fixed_threshold_or_fixed_inflation_runtime",
                "evidence_status": "runtime_native_ablation",
                "baseline_tsvr": "0.252319",
                "orius_tsvr": "0.000163",
                "absolute_delta": "0.252156",
                "relative_delta": "0.999354",
                "baseline_intervention_rate": "0.896897",
                "orius_intervention_rate": "0.500301",
                "metric_surface": "runtime_denominator",
                "note": "runtime-native",
            },
            {
                "domain": "Autonomous Vehicles",
                "ablation_name": "no_repair_release_without_repair",
                "baseline_family": "nominal_deterministic_controller",
                "evidence_status": "runtime_native_ablation",
                "baseline_tsvr": "0.289250",
                "orius_tsvr": "0.000163",
                "absolute_delta": "0.289087",
                "relative_delta": "0.999436",
                "baseline_intervention_rate": "0.000000",
                "orius_intervention_rate": "0.500301",
                "metric_surface": "runtime_denominator",
                "note": "runtime-native",
            },
        ],
    )
    _write_csv(
        security_matrix,
        [
            {
                "ablation": "missing_model_hash",
                "removed_or_corrupted_component": "Model artifact hash manifest is absent in strict mode",
                "expected_runtime_response": "refuse load before deserialization",
                "evidence_surface": "tests/test_model_artifact_hash_verification.py",
                "paper_interpretation": "Strict release evidence cannot rely on unverifiable binaries.",
            },
            {
                "ablation": "bad_certificate_signature",
                "removed_or_corrupted_component": "Certificate payload or signature is tampered",
                "expected_runtime_response": "verification fails closed",
                "evidence_surface": "tests/test_dc3s_certificate_full.py",
                "paper_interpretation": "Invalid signatures deny release.",
            },
        ],
    )
    monkeypatch.setattr(builder, "THREE_DOMAIN_BASELINE_SUITE", baseline_suite)
    monkeypatch.setattr(builder, "THREE_DOMAIN_ABLATION_MATRIX", ablation_matrix)
    monkeypatch.setattr(builder, "SECURITY_GOVERNANCE_ABLATION", security_matrix)

    rows = builder.build_ablation_surface_rows()

    assert {
        "no_reliability",
        "no_uncertainty",
        "no_repair",
        "no_fallback",
        "no_certificate_gate",
        "no_signature_hash_gate",
    } <= {row["requested_surface"] for row in rows}
    no_reliability = next(row for row in rows if row["requested_surface"] == "no_reliability")
    no_uncertainty = next(row for row in rows if row["requested_surface"] == "no_uncertainty")
    signature_hash = next(row for row in rows if row["requested_surface"] == "no_signature_hash_gate")
    assert no_reliability["baseline_controller"] == "robust_fixed_deceleration"
    assert no_reliability["absolute_tsvr_reduction"] == "0.252156"
    assert no_uncertainty["comparability"].startswith("non_comparable")
    assert signature_hash["evidence_surface"] == "governance_fail_closed"
    assert signature_hash["baseline_tsvr"] == ""


def test_publication_materialization_replaces_blank_cells() -> None:
    rows = [
        {
            "domain": "Battery Energy Storage",
            "orius_intervention_rate": "",
            "fallback_reduction_vs_safety_reference": "",
            "claim_relation": "bounded_predeployment",
        }
    ]

    materialized = builder._materialize_no_blank_cells(rows)

    assert materialized[0]["orius_intervention_rate"] != ""
    assert materialized[0]["fallback_reduction_vs_safety_reference"] == "not_defined_for_battery_t8"
    assert materialized[0]["claim_relation"] == "bounded_predeployment"


def test_validator_rejects_safety_only_without_utility(tmp_path: Path) -> None:
    scorecard = tmp_path / "utility_preserving_safety_scorecard.csv"
    _write_csv(
        scorecard,
        [
            {
                "domain": "Battery Energy Storage",
                "utility_preserving_safety_gate": "True",
                "excess_tsvr_over_safety_reference": "0.0",
                "utility_delta_over_safety_reference": "1.0",
                "claim_boundary": "bounded predeployment",
            },
            {
                "domain": "Autonomous Vehicles",
                "utility_preserving_safety_gate": "False",
                "excess_tsvr_over_safety_reference": "0.0",
                "utility_delta_over_safety_reference": "0.0",
                "fallback_reduction_vs_safety_reference": "0.0",
                "intervention_reduction_vs_safety_reference": "0.0",
                "claim_boundary": "bounded predeployment",
            },
            {
                "domain": "Medical and Healthcare Monitoring",
                "utility_preserving_safety_gate": "True",
                "excess_tsvr_over_safety_reference": "0.0",
                "utility_delta_over_safety_reference": "1.0",
                "fallback_reduction_vs_safety_reference": "0.5",
                "intervention_reduction_vs_safety_reference": "0.5",
                "claim_boundary": "retrospective, not live clinical deployment",
            },
        ],
    )

    findings = validate_scorecard(scorecard)

    assert any("Autonomous Vehicles" in finding for finding in findings)
    assert any("utility_preserving_safety_gate is not True" in finding for finding in findings)


def test_validator_rejects_missing_claim_and_ablation_surfaces(tmp_path: Path) -> None:
    scorecard = tmp_path / "utility_preserving_safety_scorecard.csv"
    claim_table = tmp_path / "utility_preserving_safety_claim_table.csv"
    ablation_surfaces = tmp_path / "utility_preserving_safety_ablation_surfaces.csv"
    _write_csv(
        scorecard,
        [
            {
                "domain": domain,
                "utility_preserving_safety_gate": "True",
                "excess_tsvr_over_safety_reference": "0.0",
                "utility_delta_over_safety_reference": "1.0",
                "fallback_reduction_vs_safety_reference": "0.5",
                "intervention_reduction_vs_safety_reference": "0.5",
                "claim_boundary": "bounded predeployment",
            }
            for domain in (
                "Battery Energy Storage",
                "Autonomous Vehicles",
                "Medical and Healthcare Monitoring",
            )
        ],
    )
    _write_csv(
        claim_table,
        [
            {
                "domain": "Autonomous Vehicles",
                "comparison": "predictor_only_safety",
                "claim_relation": "safer_than_predictor_only",
                "comparability": "comparable_runtime_native",
            }
        ],
    )
    _write_csv(
        ablation_surfaces,
        [
            {
                "requested_surface": "no_reliability",
                "domain": "Autonomous Vehicles",
                "comparability": "comparable_runtime_native",
            }
        ],
    )

    findings = validate_utility_safety_outputs(scorecard, claim_table, ablation_surfaces)

    assert any("missing claim comparison rows" in finding for finding in findings)
    assert any("missing ablation surfaces" in finding for finding in findings)
