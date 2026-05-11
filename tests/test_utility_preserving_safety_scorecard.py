from __future__ import annotations

import csv
from pathlib import Path

from scripts import build_utility_preserving_safety_scorecard as builder
from scripts.validate_utility_preserving_safety import validate_scorecard


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
