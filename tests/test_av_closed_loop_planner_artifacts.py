from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_av_closed_loop_planner_artifacts import build_av_closed_loop_planner_artifacts


def _write_unit_runtime(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    controllers = [
        "baseline",
        "rss_cbf_filter",
        "robust_fixed_deceleration",
        "nonreliability_conformal_runtime",
        "stale_certificate_no_temporal_guard",
        "always_brake",
        "orius",
    ]
    fault_families = ["dropout", "stale", "delay_jitter", "spikes", "drift_combo", "blackout", "out_of_order"]
    for scenario_index, fault_family in enumerate(fault_families):
        for controller in controllers:
            for step_index in range(8):
                candidate_accel = 0.8 if controller in {"baseline", "nonreliability_conformal_runtime"} else 0.1
                if controller == "robust_fixed_deceleration":
                    candidate_accel = -0.8
                if controller == "rss_cbf_filter":
                    candidate_accel = -0.3
                if controller == "always_brake":
                    candidate_accel = -2.0
                if controller == "orius":
                    candidate_accel = -1.2 if step_index >= 3 else 0.0
                safe_accel = min(candidate_accel, -1.0) if controller == "orius" and step_index >= 3 else candidate_accel
                rows.append(
                    {
                        "trace_id": f"s{scenario_index}-{step_index}-{controller}",
                        "scenario_id": f"s{scenario_index}",
                        "step_index": step_index,
                        "fault_family": fault_family,
                        "controller": controller,
                        "candidate_acceleration_mps2": candidate_accel,
                        "safe_acceleration_mps2": safe_accel,
                        "intervened": controller in {"orius", "always_brake", "robust_fixed_deceleration"},
                        "fallback_used": controller in {"orius", "always_brake"} and step_index >= 3,
                        "certificate_valid": controller == "orius",
                        "true_constraint_violated": controller in {"baseline", "nonreliability_conformal_runtime"} and step_index >= 6,
                        "domain_postcondition_passed": not (
                            controller in {"baseline", "nonreliability_conformal_runtime"} and step_index >= 6
                        ),
                        "min_gap_m": 14.0 - step_index * 1.1,
                        "ttc_s": 6.0 - step_index * 0.4,
                        "ego_speed_mps": 7.0 + 0.1 * step_index,
                        "target_relative_gap_1s": 13.0 - step_index * 1.1,
                        "reliability_w": 0.35 if step_index >= 3 else 0.95,
                        "true_margin": 6.0 - step_index * 1.1,
                        "observed_margin": 8.0 - step_index * 0.5,
                    }
                )
    pd.DataFrame(rows).to_csv(runtime_dir / "runtime_traces.csv", index=False)
    pd.DataFrame(
        [
            {
                "controller": controller,
                "tsvr": 0.25 if controller in {"baseline", "nonreliability_conformal_runtime"} else 0.0,
                "intervention_rate": 1.0 if controller == "always_brake" else 0.4 if controller == "orius" else 0.1,
                "fallback_activation_rate": 1.0 if controller == "always_brake" else 0.25 if controller == "orius" else 0.0,
                "useful_work_total": 100.0 if controller == "baseline" else 60.0 if controller == "orius" else 20.0,
                "n_steps": 56,
            }
            for controller in controllers
        ]
    ).to_csv(runtime_dir / "runtime_summary.csv", index=False)


def test_bounded_closed_loop_planner_artifacts_include_frontier_baselines_stress_and_ablations(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    out_dir = tmp_path / "closed_loop"
    _write_unit_runtime(runtime_dir)

    manifest = build_av_closed_loop_planner_artifacts(
        runtime_dir=runtime_dir,
        out_dir=out_dir,
        max_rows=10_000,
    )

    summary = pd.read_csv(out_dir / "av_planner_closed_loop_summary.csv")
    traces = pd.read_csv(out_dir / "av_planner_closed_loop_traces.csv")
    frontier = pd.read_csv(out_dir / "av_utility_safety_frontier.csv")
    stress = pd.read_csv(out_dir / "av_closed_loop_stress_family_summary.csv")
    ablations = pd.read_csv(out_dir / "av_closed_loop_ablation_summary.csv")

    assert manifest["status"] == "bounded_closed_loop_planner_pass"
    assert manifest["simulation_semantics"] == "ego_action_updates_future_state"
    assert manifest["pass"] is True
    assert summary.loc[0, "validation_surface"] == "nuplan_bounded_kinematic_closed_loop_planner"
    assert summary.loc[0, "closed_loop_state_feedback"] is True or str(summary.loc[0, "closed_loop_state_feedback"]) == "True"
    assert summary.loc[0, "orius_tsvr"] <= summary.loc[0, "baseline_tsvr"]
    assert summary.loc[0, "orius_useful_work_total"] > summary.loc[0, "always_brake_useful_work_total"]

    required_trace_columns = {
        "closed_loop_gap_m",
        "closed_loop_speed_mps",
        "closed_loop_acceleration_mps2",
        "closed_loop_true_constraint_violated",
        "comfort_jerk_mps3",
        "progress_m",
    }
    assert required_trace_columns <= set(traces.columns)

    assert {
        "baseline",
        "rss_cbf_filter",
        "robust_fixed_deceleration",
        "nonreliability_conformal_runtime",
        "stale_certificate_no_temporal_guard",
        "always_brake",
        "orius",
    } <= set(frontier["controller"])
    assert {
        "orius_profile_aggressive",
        "orius_profile_balanced",
        "orius_profile_conservative",
        "orius_profile_fail_closed",
    } <= set(frontier["controller"])
    assert "frontier_policy" in frontier.columns
    assert {"tsvr", "fallback_activation_rate", "intervention_rate", "useful_work_total", "mean_abs_jerk", "progress_total", "near_miss_rate"} <= set(frontier.columns)
    assert {"dropout", "stale", "delay_jitter", "spikes", "drift_combo", "blackout", "out_of_order"} <= set(stress["stress_family"])
    assert {
        "cp_only",
        "reliability_only",
        "certificate_only",
        "no_reliability",
        "no_certificate_half_life",
        "no_fallback_projection",
        "orius_full",
    } <= set(ablations["method"])
