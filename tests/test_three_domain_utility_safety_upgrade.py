from __future__ import annotations

import csv
from pathlib import Path

from orius.universal_framework.healthcare_adapter import HealthcareDomainAdapter
from scripts import build_three_domain_ml_artifacts as builder
from scripts import run_predeployment_external_validation as external


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_healthcare_interval_rows_use_calibration_bins_and_finite_sample_quantile() -> None:
    calibration_rows = []
    for idx in range(10):
        calibration_rows.append(
            {
                "reliability": 0.1,
                "spo2_pct": 95.0 if idx < 9 else 85.0,
                "forecast_spo2_pct": 95.0,
            }
        )
    calibration_rows.extend(
        {"reliability": 0.5, "spo2_pct": 96.0, "forecast_spo2_pct": 96.0} for _ in range(10)
    )
    calibration_rows.extend(
        {"reliability": 1.0, "spo2_pct": 97.0, "forecast_spo2_pct": 97.0} for _ in range(10)
    )
    eval_rows = [
        {"reliability": 0.1, "spo2_pct": 85.0, "forecast_spo2_pct": 95.0},
        {"reliability": 0.5, "spo2_pct": 96.0, "forecast_spo2_pct": 96.0},
        {"reliability": 1.0, "spo2_pct": 97.0, "forecast_spo2_pct": 97.0},
    ]

    rows, meta = builder._healthcare_interval_rows_from_records(calibration_rows, eval_rows)

    by_bucket = {row.bucket_label: row for row in rows}
    assert by_bucket["low"].coverage == 1.0
    assert by_bucket["low"].mean_interval_width == 20.0
    assert meta["bucket_qhat"]["low"] == 10.0
    assert by_bucket["low"].reliability_lower == 0.1
    assert by_bucket["mid"].reliability_lower == 0.5
    assert by_bucket["high"].reliability_lower == 1.0


def test_utility_safety_dominance_rows_capture_nonvacuous_av_and_healthcare(
    tmp_path: Path,
    monkeypatch,
) -> None:
    av_frontier = tmp_path / "av_utility_safety_frontier.csv"
    healthcare_summary = tmp_path / "healthcare_runtime_summary.csv"
    _write_csv(
        av_frontier,
        [
            {
                "controller": "baseline",
                "tsvr": 0.30,
                "fallback_activation_rate": 0.0,
                "intervention_rate": 0.0,
                "useful_work_total": 100.0,
                "n_steps": 1000,
            },
            {
                "controller": "always_brake",
                "tsvr": 0.05,
                "fallback_activation_rate": 1.0,
                "intervention_rate": 1.0,
                "useful_work_total": 10.0,
                "n_steps": 1000,
            },
            {
                "controller": "orius",
                "tsvr": 0.05,
                "fallback_activation_rate": 0.25,
                "intervention_rate": 0.40,
                "useful_work_total": 50.0,
                "n_steps": 1000,
            },
        ],
    )
    _write_csv(
        healthcare_summary,
        [
            {
                "controller": "baseline",
                "tsvr": 0.20,
                "fallback_activation_rate": 0.0,
                "intervention_rate": 0.0,
                "max_alert_rate": 0.0,
                "useful_work_total": 100.0,
                "n_steps": 1000,
            },
            {
                "controller": "always_alert",
                "tsvr": 0.0,
                "fallback_activation_rate": 1.0,
                "intervention_rate": 1.0,
                "max_alert_rate": 1.0,
                "useful_work_total": 0.0,
                "n_steps": 1000,
            },
            {
                "controller": "orius",
                "tsvr": 0.0,
                "fallback_activation_rate": 0.45,
                "intervention_rate": 0.80,
                "max_alert_rate": 0.45,
                "useful_work_total": 300.0,
                "n_steps": 1000,
            },
        ],
    )
    monkeypatch.setattr(builder, "AV_PLANNER_FRONTIER", av_frontier)
    monkeypatch.setattr(builder, "HEALTHCARE_RUNTIME_SUMMARY", healthcare_summary)

    rows = builder._utility_safety_dominance_rows()

    by_domain = {row["domain"]: row for row in rows}
    assert by_domain["Autonomous Vehicles"]["nonvacuous_utility_gate"] == "True"
    assert by_domain["Autonomous Vehicles"]["excess_tsvr_over_safety_reference"] == "0.000000"
    assert by_domain["Autonomous Vehicles"]["utility_gain_over_safety_reference"] == "5.000000"
    assert by_domain["Medical and Healthcare Monitoring"]["nonvacuous_utility_gate"] == "True"
    assert by_domain["Medical and Healthcare Monitoring"]["safety_reference_controller"] == "always_alert"


def test_healthcare_projected_release_can_preserve_hold_action_when_configured() -> None:
    adapter = HealthcareDomainAdapter(
        {
            "graded_alert_projection_margin": 1.0,
            "graded_alert_floor": 0.0,
            "graded_alert_base": 0.0,
            "graded_alert_reliability_weight": 0.0,
            "graded_alert_margin_weight": 0.0,
        }
    )
    uncertainty = {
        "spo2_lower_pct": 89.0,
        "spo2_upper_pct": 97.0,
        "forecast_spo2_lower_pct": 89.0,
        "forecast_spo2_upper_pct": 97.0,
        "hr_lower_bpm": 60.0,
        "hr_upper_bpm": 90.0,
        "rr_lower": 12.0,
        "rr_upper": 18.0,
        "spo2_pct": 94.0,
        "forecast_spo2_pct": 94.0,
        "hr_bpm": 75.0,
        "respiratory_rate": 16.0,
        "meta": {"validity_status": "degraded", "w_t": 0.5},
    }

    tightened = adapter.tighten_action_set(
        uncertainty,
        {"spo2_min_pct": 90.0, "hr_min_bpm": 40.0, "hr_max_bpm": 120.0, "rr_min": 8.0, "rr_max": 30.0},
        cfg={"validity_delta": 0.05, "validity_sigma_d": 1.0},
    )
    safe_action, repair_meta = adapter.repair_action(
        {"alert_level": 0.0},
        tightened,
        state={"spo2_pct": 94.0, "forecast_spo2_pct": 94.0, "hr_bpm": 75.0, "respiratory_rate": 16.0},
        uncertainty=uncertainty,
        constraints={},
        cfg={},
    )

    assert tightened["projected_release"] is True
    assert tightened["fallback_required"] is False
    assert safe_action == {"alert_level": 0.0}
    assert repair_meta["mode"] == "projection"
    assert repair_meta["repaired"] is False


def test_av_closed_loop_gate_uses_excess_tsvr_over_fail_safe_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    planner_dir = tmp_path / "av_closed_loop_planner"
    planner_dir.mkdir(parents=True)
    _write_csv(
        planner_dir / "av_planner_closed_loop_summary.csv",
        [
            {
                "validation_surface": "nuplan_bounded_kinematic_closed_loop_planner",
                "status": "bounded_closed_loop_planner_pass",
                "simulation_semantics": "ego_action_updates_future_state",
                "closed_loop_state_feedback": "True",
                "n_steps": 420000,
                "scenario_count": 732,
                "baseline_tsvr": 0.31,
                "orius_tsvr": 0.2412,
                "always_brake_tsvr": 0.2413,
                "orius_fallback_activation_rate": 0.28,
                "orius_intervention_rate": 0.57,
                "orius_useful_work_total": 7800.0,
                "always_brake_useful_work_total": 1900.0,
                "orius_collision_proxy_rate": 0.0073,
                "claim_boundary": "bounded planner only",
            }
        ],
    )
    _write_csv(
        planner_dir / "av_utility_safety_frontier.csv",
        [
            {
                "controller": "always_brake",
                "tsvr": 0.2413,
                "fallback_activation_rate": 1.0,
                "intervention_rate": 1.0,
                "useful_work_total": 1900.0,
                "collision_proxy_rate": 0.0074,
                "mean_abs_jerk": 0.3,
                "progress_total": 3500.0,
                "near_miss_rate": 0.02,
                "n_steps": 35000,
            },
            {
                "controller": "orius",
                "tsvr": 0.2412,
                "fallback_activation_rate": 0.28,
                "intervention_rate": 0.57,
                "useful_work_total": 7800.0,
                "collision_proxy_rate": 0.0073,
                "mean_abs_jerk": 3.4,
                "progress_total": 8500.0,
                "near_miss_rate": 0.02,
                "n_steps": 35000,
            },
        ],
    )
    monkeypatch.setattr(external, "AV_PLANNER_SUMMARY", planner_dir / "av_planner_closed_loop_summary.csv")
    monkeypatch.setattr(external, "AV_PLANNER_FRONTIER", planner_dir / "av_utility_safety_frontier.csv")
    monkeypatch.setattr(external, "AV_PLANNER_DIR", planner_dir)

    row, details = external._av_closed_loop_gate(min_steps=300000, max_fallback_rate=0.5)

    assert row["pass"] is True
    assert row["validation_surface"] == "nuplan_bounded_kinematic_closed_loop_planner"
    assert row["excess_tsvr_over_safety_reference"] == 0.0
    assert row["collision_proxy_excess_over_safety_reference"] == 0.0
    assert any(detail.get("detail_surface") == "utility_safety_frontier" for detail in details)
