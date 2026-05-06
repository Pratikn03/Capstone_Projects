"""Validation tests for the scenario-native Waymo AV surface."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from orius.av_waymo import (
    build_validation_surface,
    decode_motion_scenario,
    parse_example_bytes,
    select_anchor_neighbors,
    serialize_example_features,
    validate_scenario,
    write_tfrecord_records,
)


def _repeat_per_actor(actor_values: list[list[float | int]], steps: int) -> list[float | int]:
    flat: list[float | int] = []
    for row in actor_values:
        assert len(row) == steps
        flat.extend(row)
    return flat


def _make_scenario_features(
    *, break_timestamps: bool = False, pad_actor0_tail_steps: int = 0
) -> dict[str, list[float | int | bytes]]:
    actor_count = 3
    total_steps = 91
    timestamps = [step * 100_000 for step in range(total_steps)]
    if break_timestamps:
        timestamps[12] = timestamps[11]

    ego_x = [float(step) for step in range(total_steps)]
    nbr1_x = [float(step) + 15.0 for step in range(total_steps)]
    nbr2_x = [float(step) - 10.0 for step in range(total_steps)]

    def split(series: list[float]) -> tuple[list[float], list[float], list[float]]:
        return series[:10], [series[10]], series[11:]

    def split_timestamps(series: list[int]) -> tuple[list[int], list[int], list[int]]:
        return series[:10], [series[10]], series[11:]

    actor_timestamp_series = [list(timestamps) for _ in range(actor_count)]
    if pad_actor0_tail_steps > 0:
        for offset in range(1, pad_actor0_tail_steps + 1):
            actor_timestamp_series[0][-offset] = -1

    actors_x = [ego_x, nbr1_x, nbr2_x]
    actors_y = [[0.0] * total_steps, [1.0] * total_steps, [-2.0] * total_steps]
    actors_speed = [[10.0] * total_steps, [8.0] * total_steps, [3.0] * total_steps]
    actors_vx = [[10.0] * total_steps, [8.0] * total_steps, [3.0] * total_steps]
    actors_vy = [[0.0] * total_steps, [0.0] * total_steps, [0.0] * total_steps]
    actors_length = [[4.5] * total_steps, [4.2] * total_steps, [1.2] * total_steps]
    actors_width = [[2.0] * total_steps, [1.9] * total_steps, [0.6] * total_steps]
    actors_height = [[1.5] * total_steps, [1.5] * total_steps, [1.7] * total_steps]
    actors_yaw = [[0.0] * total_steps, [0.0] * total_steps, [0.0] * total_steps]
    actors_valid = [[1] * total_steps, [1] * total_steps, [1] * total_steps]
    if pad_actor0_tail_steps > 0:
        for offset in range(1, pad_actor0_tail_steps + 1):
            actors_valid[0][-offset] = 0

    features: dict[str, list[float | int | bytes]] = {
        "scenario/id": [b"scenario-test-001"],
        "state/id": [101.0, 202.0, 303.0],
        "state/type": [1.0, 1.0, 2.0],
        "state/is_sdc": [1, 0, 0],
        "state/tracks_to_predict": [1, 1, 1],
        "state/objects_of_interest": [1, 1, 0],
    }

    def add_series(field: str, actor_series: list[list[float]], *, is_int: bool = False) -> None:
        phase_values: dict[str, list[float | int]] = {"past": [], "current": [], "future": []}
        for series in actor_series:
            past, current, future = split(series)
            phase_values["past"].extend(past)
            phase_values["current"].extend(current)
            phase_values["future"].extend(future)
        for phase, values in phase_values.items():
            features[f"state/{phase}/{field}"] = [int(v) for v in values] if is_int else values

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
    timestamp_phases: dict[str, list[int]] = {"past": [], "current": [], "future": []}
    for actor_timestamps in actor_timestamp_series:
        past_ts, current_ts, future_ts = split_timestamps(actor_timestamps)
        timestamp_phases["past"].extend(past_ts)
        timestamp_phases["current"].extend(current_ts)
        timestamp_phases["future"].extend(future_ts)
    features["state/past/timestamp_micros"] = timestamp_phases["past"]
    features["state/current/timestamp_micros"] = timestamp_phases["current"]
    features["state/future/timestamp_micros"] = timestamp_phases["future"]
    return features


def test_decode_and_neighbor_selection_is_deterministic() -> None:
    payload = serialize_example_features(_make_scenario_features())
    parsed = parse_example_bytes(payload)
    scenario = decode_motion_scenario(parsed, shard_id="validation-00000", record_index=0)

    validation = validate_scenario(scenario)
    neighbors = select_anchor_neighbors(scenario)

    assert scenario["scenario_id"] == "scenario-test-001"
    assert validation.timestamps_monotone is True
    assert validation.cadence_close_to_10hz is True
    assert validation.ego_track_id == 101
    assert validation.neighbor_count == 2
    assert neighbors == [2, 1]


def test_build_validation_surface_materializes_stage2_artifacts(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)

    payload = serialize_example_features(_make_scenario_features())
    tfrecord_path = raw_dir / "validation_tfexample.tfrecord-00000-of-00001"
    write_tfrecord_records(tfrecord_path, [payload])

    report = build_validation_surface(raw_dir=raw_dir, out_dir=out_dir)

    assert report["all_shards_opened"] is True
    assert report["scenario_count"] == 1
    assert report["usable_scenario_count"] == 1
    assert (out_dir / "scenario_index.parquet").exists()
    assert (out_dir / "actor_tracks.parquet").exists()
    assert (out_dir / "ego_neighborhood_anchors.parquet").exists()
    assert (out_dir / "parse_validation_report.json").exists()

    scenario_index = pd.read_parquet(out_dir / "scenario_index.parquet")
    anchors = pd.read_parquet(out_dir / "ego_neighborhood_anchors.parquet")
    actor_tracks = pd.read_parquet(out_dir / "actor_tracks.parquet")

    assert len(scenario_index) == 1
    assert int(scenario_index.loc[0, "ego_track_id"]) == 101
    assert int(scenario_index.loc[0, "neighbor_count"]) == 2
    assert scenario_index.loc[0, "neighbor_ids_csv"] == "303,202"

    assert len(anchors) == 1
    assert anchors.loc[0, "neighbor_slot_0_track_id"] == 303
    assert anchors.loc[0, "neighbor_slot_1_track_id"] == 202
    assert int(anchors.loc[0, "neighbor_count"]) == 2

    assert len(actor_tracks) == 3 * 91
    assert set(actor_tracks["role"]) == {"ego", "neighbor"}

    report_payload = json.loads((out_dir / "parse_validation_report.json").read_text(encoding="utf-8"))
    assert report_payload["timestamps_monotone_count"] == 1
    assert report_payload["timestamps_10hz_count"] == 1
    assert report_payload["ego_resolved_count"] == 1


def test_validation_flags_broken_timestamps() -> None:
    payload = serialize_example_features(_make_scenario_features(break_timestamps=True))
    scenario = decode_motion_scenario(
        parse_example_bytes(payload), shard_id="validation-00000", record_index=0
    )
    validation = validate_scenario(scenario)

    assert validation.timestamps_monotone is False
    assert "timestamps_not_monotone" in validation.issues


def test_validation_ignores_padded_negative_actor_timestamps() -> None:
    payload = serialize_example_features(_make_scenario_features(pad_actor0_tail_steps=5))
    scenario = decode_motion_scenario(
        parse_example_bytes(payload), shard_id="validation-00000", record_index=0
    )
    validation = validate_scenario(scenario)

    assert scenario["timestamps_us"][-1] == 9_000_000
    assert validation.timestamps_monotone is True
    assert validation.cadence_close_to_10hz is True
