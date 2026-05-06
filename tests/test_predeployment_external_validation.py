from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import run_predeployment_external_validation as external
from scripts.run_predeployment_external_validation import build_predeployment_external_validation


def test_predeployment_external_validation_requires_three_external_surfaces(tmp_path: Path) -> None:
    report = build_predeployment_external_validation(out_dir=tmp_path)

    assert report["all_passed"] is True
    rows = {row["domain"]: row for row in report["domains"]}
    assert set(rows) == {
        "Battery Energy Storage",
        "Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    }

    assert rows["Battery Energy Storage"]["validation_surface"] == "battery_hil_or_simulator"
    assert rows["Battery Energy Storage"]["safety_violations"] == 0
    assert rows["Battery Energy Storage"]["certificate_valid_rate"] == 1.0

    av = rows["Autonomous Vehicles"]
    assert av["validation_surface"] in {
        "nuplan_allzip_grouped_runtime_replay_surrogate",
        "nuplan_bounded_kinematic_closed_loop_planner",
    }
    if av["validation_surface"] == "nuplan_bounded_kinematic_closed_loop_planner":
        assert av["closed_loop_state_feedback"] is True
        assert av["closed_loop_simulation_semantics"] == "ego_action_updates_future_state"
        assert av["safety_reference_controller"] == "always_brake"
        assert av["excess_tsvr_over_safety_reference"] <= 1e-3
    else:
        assert av["orius_tsvr"] <= 1e-3
    assert av["fallback_or_intervention_rate"] <= 0.50
    assert av["certificate_valid_rate"] >= 0.999
    assert "not CARLA" in av["claim_boundary"] or "not completed CARLA simulation" in av["claim_boundary"]
    assert "road deployment" in av["claim_boundary"]

    healthcare = rows["Medical and Healthcare Monitoring"]
    assert healthcare["validation_surface"] == "healthcare_retrospective_time_forward_source_holdout"
    assert healthcare["orius_tsvr"] == 0.0
    assert healthcare["fallback_or_intervention_rate"] <= 0.50
    assert healthcare["patient_disjoint"] is True
    assert healthcare["time_forward"] is True
    assert healthcare["site_holdout"] is True
    assert healthcare["development_source"] != healthcare["holdout_source"]
    assert "not live clinical deployment" in healthcare["claim_boundary"]


def test_predeployment_healthcare_site_split_manifest_is_source_holdout(tmp_path: Path) -> None:
    report = build_predeployment_external_validation(out_dir=tmp_path)
    manifest_path = tmp_path / "healthcare_site_splits" / "manifest.json"
    details_path = tmp_path / "healthcare_retrospective_holdout_split_details.csv"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    details = pd.read_csv(details_path)
    split_sources = {
        row["split"]: set(str(row["sources"]).split("|"))
        for _, row in details.iterrows()
        if not str(row["split"]).startswith("source:")
    }

    assert manifest["split_strategy"] == "development_site_patient_blocks_plus_later_source_holdout"
    assert set(manifest["source_datasets"]) >= {"bidmc", "mimic3"}
    assert split_sources["test"].isdisjoint(
        split_sources["train"] | split_sources["calibration"] | split_sources["val"]
    )
    assert all(Path(report["detail_artifacts"][name]).exists() for name in ("battery", "av", "healthcare"))


def test_predeployment_av_gate_prefers_bounded_planner_closed_loop_when_present(
    tmp_path: Path, monkeypatch
) -> None:
    planner_dir = tmp_path / "planner"
    planner_dir.mkdir()
    pd.DataFrame(
        [
            {
                "validation_surface": "nuplan_bounded_kinematic_closed_loop_planner",
                "status": "bounded_closed_loop_planner_pass",
                "simulation_semantics": "ego_action_updates_future_state",
                "closed_loop_state_feedback": True,
                "n_steps": 128,
                "scenario_count": 7,
                "baseline_tsvr": 0.25,
                "orius_tsvr": 0.0,
                "always_brake_tsvr": 0.0,
                "orius_fallback_activation_rate": 0.2,
                "orius_intervention_rate": 0.3,
                "orius_useful_work_total": 80.0,
                "always_brake_useful_work_total": 20.0,
                "orius_mean_abs_jerk": 0.4,
                "orius_progress_total": 200.0,
                "orius_near_miss_rate": 0.01,
                "orius_collision_proxy_rate": 0.0,
                "pass": True,
                "claim_boundary": "Bounded kinematic nuPlan closed-loop planner evaluation; not CARLA and not road deployment.",
            }
        ]
    ).to_csv(planner_dir / "av_planner_closed_loop_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "controller": "orius",
                "tsvr": 0.0,
                "fallback_activation_rate": 0.2,
                "intervention_rate": 0.3,
                "useful_work_total": 80.0,
                "mean_abs_jerk": 0.4,
                "progress_total": 200.0,
                "near_miss_rate": 0.01,
            }
        ]
    ).to_csv(planner_dir / "av_utility_safety_frontier.csv", index=False)
    monkeypatch.setattr(external, "AV_PLANNER_SUMMARY", planner_dir / "av_planner_closed_loop_summary.csv")
    monkeypatch.setattr(external, "AV_PLANNER_FRONTIER", planner_dir / "av_utility_safety_frontier.csv")

    row, details = external._av_closed_loop_gate(min_steps=128, max_fallback_rate=0.5)

    assert row["pass"] is True
    assert row["validation_surface"] == "nuplan_bounded_kinematic_closed_loop_planner"
    assert row["evidence_level"] == "bounded_kinematic_closed_loop_planner"
    assert row["closed_loop_state_feedback"] is True
    assert row["closed_loop_simulation_semantics"] == "ego_action_updates_future_state"
    assert any(detail.get("detail_surface") == "utility_safety_frontier" for detail in details)
