"""Smoke tests for the Waymo AV dry-run pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from orius.av_waymo import (
    WaymoReplayTrackAdapter,
    build_feature_tables,
    build_replay_surface,
    build_subset_manifest,
    build_validation_surface,
    run_runtime_dry_run,
    serialize_example_features,
    train_dry_run_models,
    write_tfrecord_records,
)
from orius.av_waymo.runtime import (
    _independent_certificate_validity,
    _predict_certificate_validity,
    _runtime_aligned_longitudinal_controller,
    deterministic_longitudinal_controller,
)
from orius.certos.verification import load_certificates_from_duckdb, verify_certificates
from orius.dc3s.certificate import make_certificate


def _make_scenario_features(*, scenario_id: str, speed_bias: float, gap_bias: float) -> dict[str, list[float | int | bytes]]:
    actor_count = 3
    total_steps = 91
    timestamps = [step * 100_000 for step in range(total_steps)]

    ego_x = [float(step) * (1.0 + 0.02 * speed_bias) for step in range(total_steps)]
    nbr1_x = [ego_x[step] + gap_bias + 15.0 + 0.2 * step for step in range(total_steps)]
    nbr2_x = [ego_x[step] - 8.0 - 0.1 * step for step in range(total_steps)]

    def split(series):
        return series[:10], [series[10]], series[11:]

    features: dict[str, list[float | int | bytes]] = {
        "scenario/id": [scenario_id.encode("utf-8")],
        "state/id": [101.0, 202.0, 303.0],
        "state/type": [1.0, 1.0, 2.0],
        "state/is_sdc": [1, 0, 0],
        "state/tracks_to_predict": [1, 1, 1],
        "state/objects_of_interest": [1, 1, 0],
    }

    def add_series(field: str, actor_series: list[list[float]], *, is_int: bool = False) -> None:
        phase_values = {"past": [], "current": [], "future": []}
        for series in actor_series:
            past, current, future = split(series)
            phase_values["past"].extend(past)
            phase_values["current"].extend(current)
            phase_values["future"].extend(future)
        for phase, values in phase_values.items():
            features[f"state/{phase}/{field}"] = [int(v) for v in values] if is_int else values

    actors_x = [ego_x, nbr1_x, nbr2_x]
    actors_y = [[0.0] * total_steps, [1.0] * total_steps, [-1.5] * total_steps]
    actors_speed = [
        [9.0 + speed_bias] * total_steps,
        [7.0 + 0.5 * speed_bias] * total_steps,
        [2.0 + 0.1 * speed_bias] * total_steps,
    ]
    actors_vx = actors_speed
    actors_vy = [[0.0] * total_steps for _ in range(actor_count)]
    actors_length = [[4.5] * total_steps, [4.2] * total_steps, [1.2] * total_steps]
    actors_width = [[2.0] * total_steps, [1.9] * total_steps, [0.6] * total_steps]
    actors_height = [[1.5] * total_steps, [1.5] * total_steps, [1.7] * total_steps]
    actors_yaw = [[0.0] * total_steps for _ in range(actor_count)]
    actors_valid = [[1] * total_steps for _ in range(actor_count)]

    add_series("x", actors_x)
    add_series("y", actors_y)
    add_series("z", [[0.0] * total_steps for _ in range(actor_count)])
    add_series("velocity_x", actors_vx)
    add_series("velocity_y", actors_vy)
    add_series("speed", actors_speed)
    add_series("bbox_yaw", actors_yaw)
    add_series("vel_yaw", actors_yaw)
    add_series("length", actors_length)
    add_series("width", actors_width)
    add_series("height", actors_height)
    add_series("valid", actors_valid, is_int=True)

    past_ts, current_ts, future_ts = split(timestamps)
    features["state/past/timestamp_micros"] = past_ts * actor_count
    features["state/current/timestamp_micros"] = current_ts * actor_count
    features["state/future/timestamp_micros"] = future_ts * actor_count
    return features


def _write_synthetic_waymo_raw(raw_dir: Path, scenario_count: int = 12) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    shard_payloads: dict[str, list[bytes]] = {"validation_tfexample.tfrecord-00000-of-00002": [], "validation_tfexample.tfrecord-00001-of-00002": []}
    for idx in range(scenario_count):
        payload = serialize_example_features(
            _make_scenario_features(
                scenario_id=f"scenario-{idx:04d}",
                speed_bias=float(idx % 5),
                gap_bias=float(idx % 7),
            )
        )
        shard_name = "validation_tfexample.tfrecord-00000-of-00002" if idx % 2 == 0 else "validation_tfexample.tfrecord-00001-of-00002"
        shard_payloads[shard_name].append(payload)
    for shard_name, payloads in shard_payloads.items():
        write_tfrecord_records(raw_dir / shard_name, payloads)


def _force_balanced_splits(processed_dir: Path) -> None:
    anchor_path = processed_dir / "anchor_features.parquet"
    step_path = processed_dir / "step_features.parquet"
    anchor_df = pd.read_parquet(anchor_path)
    step_df = pd.read_parquet(step_path)
    scenario_ids = sorted(anchor_df["scenario_id"].unique().tolist())
    split_cycle = ["train", "train", "train", "train", "train", "train", "calibration", "calibration", "val", "val", "test", "test"]
    split_map = {scenario_id: split_cycle[idx % len(split_cycle)] for idx, scenario_id in enumerate(scenario_ids)}
    anchor_df["split"] = anchor_df["scenario_id"].map(split_map)
    step_df["split"] = step_df["scenario_id"].map(split_map)
    anchor_df.to_parquet(anchor_path, index=False)
    step_df.to_parquet(step_path, index=False)
    splits_dir = processed_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in anchor_df.groupby("split"):
        split_df.to_parquet(splits_dir / f"{split_name}.parquet", index=False)


def test_waymo_dry_run_smoke(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    uncertainty_dir = tmp_path / "uncertainty"
    reports_dir = tmp_path / "reports"
    _write_synthetic_waymo_raw(raw_dir)

    validation = build_validation_surface(raw_dir=raw_dir, out_dir=processed_dir)
    assert validation["scenario_count"] == 12
    subset = build_subset_manifest(raw_dir=raw_dir, processed_dir=processed_dir, target_count=12)
    assert subset["selected_count"] == 12

    replay = build_replay_surface(
        raw_dir=raw_dir,
        subset_manifest_path=processed_dir / "dry_run_subset_manifest.parquet",
        out_dir=processed_dir,
    )
    assert replay["row_count"] == 12 * 91

    feature_report = build_feature_tables(
        replay_windows_path=processed_dir / "replay_windows.parquet",
        out_dir=processed_dir,
    )
    assert feature_report["anchor_row_count"] == 12
    _force_balanced_splits(processed_dir)

    train_report = train_dry_run_models(
        anchor_features_path=processed_dir / "anchor_features.parquet",
        step_features_path=processed_dir / "step_features.parquet",
        models_dir=models_dir,
        uncertainty_dir=uncertainty_dir,
        reports_dir=reports_dir,
    )
    training_summary_df = pd.read_csv(reports_dir / "training_summary.csv")
    assert (reports_dir / "training_summary.csv").exists()
    assert (reports_dir / "subgroup_coverage.csv").exists()
    assert Path(train_report["shift_aware_config_json"]).exists()
    assert "ego_speed_mps_1s" in train_report["artifact_registry"]
    assert training_summary_df["widened_mean_width"].notna().all()
    assert training_summary_df["mean_widening_factor"].notna().all()

    runtime_report = run_runtime_dry_run(
        replay_windows_path=processed_dir / "replay_windows.parquet",
        step_features_path=processed_dir / "step_features.parquet",
        models_dir=models_dir,
        out_dir=reports_dir,
        max_scenarios=2,
    )
    summary_df = pd.read_csv(runtime_report["runtime_summary_csv"])
    coverage_df = pd.read_csv(runtime_report["fault_family_coverage_csv"])
    trace_df = pd.read_csv(runtime_report["runtime_traces_csv"])
    comparator_df = pd.read_csv(runtime_report["runtime_comparator_summary_csv"])
    ablation_df = pd.read_csv(runtime_report["runtime_ablation_summary_csv"])
    negative_df = pd.read_csv(runtime_report["runtime_negative_controls_csv"])
    assert {
        "baseline",
        "rss_cbf_filter",
        "robust_fixed_deceleration",
        "predictor_only_no_runtime",
        "nonreliability_conformal_runtime",
        "stale_certificate_no_temporal_guard",
        "always_brake",
        "orius",
    } <= set(summary_df["controller"])
    assert "controller" in coverage_df.columns
    assert {"baseline", "always_brake", "orius"} <= set(coverage_df["controller"])
    assert {
        "trace_id",
        "shard_id",
        "reliability_w",
        "certificate_predicted_valid",
        "base_pred_ego_speed_lower_mps",
        "base_pred_relative_gap_lower_m",
        "repair_mode",
        "fallback_region",
        "theorem_contract",
        "contract_id",
        "source_theorem",
        "t11_status",
        "t11_failed_obligations",
        "domain_postcondition_passed",
        "domain_postcondition_failure",
        "validity_scope",
        "validity_theorem_id",
        "validity_theorem_contract",
        "geometric_constraint_violated",
        "projected_release",
        "projected_release_margin",
        "runtime_policy_family",
    }.issubset(trace_df.columns)
    orius_trace_df = trace_df[trace_df["controller"] == "orius"]
    assert set(orius_trace_df["contract_id"]) == {"AV.T11.brake_hold_runtime_lemma"}
    assert set(orius_trace_df["source_theorem"]) == {"T11"}
    assert set(orius_trace_df["t11_status"]).issubset({"runtime_linked", "contract_violation"})
    assert float(summary_df.loc[summary_df["controller"] == "orius", "tsvr"].iloc[0]) == 0.0
    assert float(summary_df.loc[summary_df["controller"] == "orius", "fallback_activation_rate"].iloc[0]) <= 0.50
    assert float(summary_df.loc[summary_df["controller"] == "always_brake", "tsvr"].iloc[0]) == 0.0
    assert float(summary_df.loc[summary_df["controller"] == "orius", "useful_work_total"].iloc[0]) >= float(
        summary_df.loc[summary_df["controller"] == "always_brake", "useful_work_total"].iloc[0]
    )
    assert {
        "nominal_deterministic_controller",
        "fixed_threshold_or_fixed_inflation_runtime",
        "standard_conformal_nonreliability_runtime",
        "no_quality_signal_runtime",
        "no_adaptive_response_runtime",
        "no_temporal_guard_or_no_certificate_refresh_runtime",
        "orius_full_stack",
        "degenerate_fallback_runtime",
    } == set(comparator_df["baseline_family"])
    assert set(comparator_df["metric_surface"]) == {"runtime_denominator"}
    assert "proxy_current_shared_harness" not in set(comparator_df["evidence_status"])
    assert set(ablation_df["metric_surface"]) == {"runtime_denominator"}
    assert set(negative_df["surface"]) == {"runtime_denominator"}
    orius_comparator = comparator_df[comparator_df["baseline_family"] == "orius_full_stack"].iloc[0]
    fallback_comparator = comparator_df[comparator_df["baseline_family"] == "degenerate_fallback_runtime"].iloc[0]
    assert float(orius_comparator["runtime_witness_pass_rate"]) == 1.0
    assert float(orius_comparator["fallback_activation_rate"]) <= 0.50
    assert float(orius_comparator["useful_work_total"]) >= float(fallback_comparator["useful_work_total"])
    independent_rows = comparator_df[
        ~comparator_df["baseline_family"].isin(["orius_full_stack", "degenerate_fallback_runtime"])
    ]
    assert {str(value) for value in independent_rows["independent_baseline"]} == {"True"}
    assert independent_rows["controller"].nunique() == len(independent_rows)
    assert Path(runtime_report["audit_db_path"]).exists()
    cert_summary, _, _, _ = verify_certificates(load_certificates_from_duckdb(runtime_report["audit_db_path"]))
    assert cert_summary["chain_valid"] is True
    shift_artifacts = runtime_report["shift_aware_artifacts"]
    assert Path(shift_artifacts["summary_csv"]).exists()
    assert Path(shift_artifacts["targets"]["ego_speed_mps"]["validity_trace_csv"]).exists()
    assert Path(shift_artifacts["targets"]["relative_gap_m"]["adaptive_trace_csv"]).exists()


def test_runtime_aligned_nominal_controller_tracks_projection_region() -> None:
    state = {
        "ego_speed_mps": 12.0,
        "speed_limit_mps": 22.0,
        "min_gap_m": 14.0,
        "lead_speed_mps": 7.0,
        "lead_rel_speed_mps": 5.0,
        "ttc_s": 2.8,
        "pred_relative_gap_lower_m": 4.75,
    }
    constraints = {
        "accel_min_mps2": -6.0,
        "accel_max_mps2": 3.0,
        "hard_headway_m": 5.0,
        "hard_ttc_s": 2.0,
        "entry_headway_m": 10.0,
        "entry_ttc_s": 4.0,
    }
    base = deterministic_longitudinal_controller(state)
    aligned = _runtime_aligned_longitudinal_controller(
        state,
        constraints=constraints,
        policy_config={"align_nominal_with_runtime_projection": True},
    )

    assert base["acceleration_mps2"] > 0.0
    assert abs(aligned["acceleration_mps2"] - -5.9) < 1e-9


def test_waymo_replay_track_faults_only_corrupt_observation(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_synthetic_waymo_raw(raw_dir, scenario_count=2)
    build_validation_surface(raw_dir=raw_dir, out_dir=processed_dir)
    build_subset_manifest(raw_dir=raw_dir, processed_dir=processed_dir, target_count=2)
    build_replay_surface(
        raw_dir=raw_dir,
        subset_manifest_path=processed_dir / "dry_run_subset_manifest.parquet",
        out_dir=processed_dir,
    )
    track = WaymoReplayTrackAdapter(processed_dir / "replay_windows.parquet")
    state = track.reset(seed=0)
    observed = track.observe(state, {"kind": "spikes"})
    assert observed["ego_speed_mps"] != state["ego_speed_mps"]
    assert track.true_state()["ego_speed_mps"] == state["ego_speed_mps"]


def test_av_certificate_prediction_and_post_emit_validation_can_diverge() -> None:
    certificate = make_certificate(
        command_id="waymo-cmd-1",
        device_id="ego-1",
        zone_id="0",
        controller="waymo_orius_dry_run",
        proposed_action={"acceleration_mps2": 0.2},
        safe_action={"acceleration_mps2": -0.4},
        uncertainty={"gap_lower_m": 10.0, "gap_upper_m": 15.0},
        reliability={"w_t": 0.82},
        drift={"drift": False},
        model_hash="model",
        config_hash="cfg",
        validity_horizon_H_t=5,
        validity_status="nominal",
        runtime_surface="waymo_motion_replay_dry_run",
        closure_tier="defended_bounded_row",
        reliability_feature_basis={"cadence_ok": True},
        reliability_w=0.82,
        intervened=True,
        intervention_reason="ttc_clamp",
    )

    assert _predict_certificate_validity(certificate, None) is True
    certificate["safe_action"] = {"acceleration_mps2": 99.0}
    assert _independent_certificate_validity(certificate, None) is False
