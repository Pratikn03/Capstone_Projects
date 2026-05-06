"""Scenario-native Waymo Motion parsing and validation builders."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .tfrecord import iter_tfrecord_records, parse_example_bytes

PAST_STEPS = 10
CURRENT_STEPS = 1
FUTURE_STEPS = 80
TOTAL_SCENARIO_STEPS = PAST_STEPS + CURRENT_STEPS + FUTURE_STEPS
ANCHOR_CURRENT_INDEX = PAST_STEPS
EXPECTED_CADENCE_US = 100_000
NEIGHBOR_RADIUS_M = 60.0
MAX_NEIGHBORS = 8

STATE_FLOAT_FIELDS = (
    "x",
    "y",
    "z",
    "velocity_x",
    "velocity_y",
    "speed",
    "bbox_yaw",
    "vel_yaw",
    "length",
    "width",
    "height",
)
STATE_PHASES = (("past", PAST_STEPS), ("current", CURRENT_STEPS), ("future", FUTURE_STEPS))
TRACK_TYPE_LABELS = {
    -1: "unknown",
    0: "unset",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
    4: "other",
}


@dataclass(slots=True)
class ScenarioValidation:
    """Validation summary for one scenario record."""

    scenario_id: str
    shard_id: str
    record_index: int
    actor_count: int
    timestamp_count: int
    timestamps_monotone: bool
    cadence_close_to_10hz: bool
    cadence_mean_us: float
    cadence_max_abs_error_us: float
    ego_resolved: bool
    ego_track_id: int | None
    neighbor_count: int
    neighbor_ids: list[int]
    usable: bool
    issues: list[str]


class ParquetRowWriter:
    """Incremental parquet writer for large row-oriented artifacts."""

    def __init__(self, path: Path, *, batch_size: int = 8_192):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = int(batch_size)
        self._rows: list[dict[str, Any]] = []
        self._schema: pa.Schema | None = None
        self._writer: pq.ParquetWriter | None = None
        self._sink: pa.NativeFile | None = None

    def append(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pylist(self._rows)
        if self._schema is None:
            self._schema = table.schema
            if self.path.exists():
                self.path.unlink()
            self._sink = pa.OSFile(str(self.path), "wb")
            self._writer = pq.ParquetWriter(self._sink, self._schema)
        else:
            table = table.cast(self._schema)
        assert self._writer is not None
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._sink is not None:
            self._sink.close()
            self._sink = None


def _decode_scenario_id(features: dict[str, list[Any]]) -> str:
    raw = features.get("scenario/id", [])
    if not raw:
        return ""
    value = raw[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _reshape_actor_feature(
    values: list[Any],
    *,
    actor_count: int,
    steps: int,
    dtype: type[float] | type[int],
    default: float | int,
) -> np.ndarray:
    total = actor_count * steps
    if len(values) == total:
        return np.asarray(values, dtype=dtype).reshape(actor_count, steps)
    if len(values) == steps:
        base = np.asarray(values, dtype=dtype).reshape(1, steps)
        return np.repeat(base, actor_count, axis=0)
    if len(values) == actor_count and steps == 1:
        return np.asarray(values, dtype=dtype).reshape(actor_count, 1)
    if not values:
        return np.full((actor_count, steps), default, dtype=dtype)
    raise ValueError(f"Unexpected feature length {len(values)} for actor_count={actor_count}, steps={steps}")


def _concat_state_feature(
    features: dict[str, list[Any]],
    *,
    actor_count: int,
    field: str,
    dtype: type[float] | type[int],
    default: float | int,
) -> np.ndarray:
    parts = []
    for phase, steps in STATE_PHASES:
        name = f"state/{phase}/{field}"
        values = features.get(name, [])
        parts.append(
            _reshape_actor_feature(values, actor_count=actor_count, steps=steps, dtype=dtype, default=default)
        )
    return np.concatenate(parts, axis=1)


def _safe_int_array(values: list[Any]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.int64)
    return np.asarray([int(round(float(item))) for item in values], dtype=np.int64)


def _boolish(values: np.ndarray) -> np.ndarray:
    return np.asarray(values > 0, dtype=bool)


def _canonicalize_scenario_timestamps(timestamp_matrix: np.ndarray) -> np.ndarray:
    """Collapse actor-level timestamps into one per-step scenario timestamp."""
    if timestamp_matrix.ndim != 2:
        raise ValueError("Timestamp matrix must be 2D.")
    actor_count, step_count = timestamp_matrix.shape
    if actor_count <= 0:
        raise ValueError("Timestamp matrix must contain at least one actor.")

    canonical = np.full(step_count, -1, dtype=np.int64)
    for step_index in range(step_count):
        column = timestamp_matrix[:, step_index].astype(np.int64)
        non_negative = column[column >= 0]
        if non_negative.size > 0:
            canonical[step_index] = int(np.median(non_negative))

    if np.all(canonical < 0):
        raise ValueError("Scenario timestamps are missing for all steps.")

    for step_index in range(step_count):
        if canonical[step_index] >= 0:
            continue
        prev_index = step_index - 1
        while prev_index >= 0 and canonical[prev_index] < 0:
            prev_index -= 1
        next_index = step_index + 1
        while next_index < step_count and canonical[next_index] < 0:
            next_index += 1
        if prev_index >= 0 and next_index < step_count:
            span = next_index - prev_index
            step_delta = (canonical[next_index] - canonical[prev_index]) / float(span)
            canonical[step_index] = int(round(canonical[prev_index] + step_delta * (step_index - prev_index)))
        elif prev_index >= 0:
            canonical[step_index] = int(
                canonical[prev_index] + EXPECTED_CADENCE_US * (step_index - prev_index)
            )
        elif next_index < step_count:
            canonical[step_index] = int(
                canonical[next_index] - EXPECTED_CADENCE_US * (next_index - step_index)
            )

    return canonical


def decode_motion_scenario(
    features: dict[str, list[Any]],
    *,
    shard_id: str,
    record_index: int,
) -> dict[str, Any]:
    """Decode one Waymo Motion Example into a scenario-native dict."""
    actor_ids = _safe_int_array(features.get("state/id", []))
    actor_count = int(actor_ids.shape[0])
    if actor_count <= 0:
        raise ValueError("Scenario has no actors")

    timestamps = _concat_state_feature(
        features,
        actor_count=actor_count,
        field="timestamp_micros",
        dtype=int,
        default=0,
    )
    valid = _boolish(
        _concat_state_feature(
            features,
            actor_count=actor_count,
            field="valid",
            dtype=int,
            default=0,
        )
    )
    state_arrays: dict[str, np.ndarray] = {"valid": valid}
    for field in STATE_FLOAT_FIELDS:
        state_arrays[field] = _concat_state_feature(
            features,
            actor_count=actor_count,
            field=field,
            dtype=float,
            default=float("nan"),
        )

    track_types = _safe_int_array(features.get("state/type", []))
    is_sdc = _boolish(np.asarray(features.get("state/is_sdc", []), dtype=np.int64))
    tracks_to_predict = _boolish(np.asarray(features.get("state/tracks_to_predict", []), dtype=np.int64))
    objects_of_interest = _boolish(np.asarray(features.get("state/objects_of_interest", []), dtype=np.int64))

    if track_types.shape[0] != actor_count:
        raise ValueError("Track type vector does not match actor count")

    scenario_id = _decode_scenario_id(features)
    scenario_timestamps = _canonicalize_scenario_timestamps(timestamps.astype(np.int64))

    return {
        "scenario_id": scenario_id,
        "shard_id": str(shard_id),
        "record_index": int(record_index),
        "actor_count": actor_count,
        "timestamps_us": scenario_timestamps,
        "timestamp_matrix_us": timestamps.astype(np.int64),
        "track_id": actor_ids,
        "track_type": track_types,
        "track_type_label": np.asarray(
            [TRACK_TYPE_LABELS.get(int(code), "unknown") for code in track_types],
            dtype=object,
        ),
        "is_sdc": is_sdc,
        "tracks_to_predict": tracks_to_predict,
        "objects_of_interest": objects_of_interest,
        **state_arrays,
    }


def _resolve_ego_index(scenario: dict[str, Any]) -> int | None:
    matches = np.flatnonzero(np.asarray(scenario["is_sdc"], dtype=bool))
    if matches.size != 1:
        return None
    return int(matches[0])


def _timestamp_summary(timestamps_us: np.ndarray) -> tuple[bool, bool, float, float]:
    if timestamps_us.shape[0] != TOTAL_SCENARIO_STEPS:
        return False, False, float("nan"), float("inf")
    deltas = np.diff(timestamps_us.astype(np.int64))
    monotone = bool(np.all(deltas > 0))
    cadence_mean = float(np.mean(deltas)) if deltas.size else float("nan")
    cadence_max_error = float(np.max(np.abs(deltas - EXPECTED_CADENCE_US))) if deltas.size else float("inf")
    cadence_close = bool(monotone and cadence_max_error <= 20_000.0)
    return monotone, cadence_close, cadence_mean, cadence_max_error


def _history_valid_mask(scenario: dict[str, Any]) -> np.ndarray:
    valid = np.asarray(scenario["valid"], dtype=bool)
    return np.all(valid[:, :ANCHOR_CURRENT_INDEX], axis=1) & valid[:, ANCHOR_CURRENT_INDEX]


def _dynamic_actor_mask(scenario: dict[str, Any], ego_index: int) -> np.ndarray:
    history_valid = _history_valid_mask(scenario)
    speed = np.asarray(scenario["speed"], dtype=float)
    moving = np.nanmax(np.abs(speed[:, : ANCHOR_CURRENT_INDEX + 1]), axis=1) > 0.5
    type_code = np.asarray(scenario["track_type"], dtype=np.int64)
    valid_type = type_code > 0
    mask = history_valid & moving & valid_type
    mask[int(ego_index)] = False
    return mask


def _relative_xy(
    scenario: dict[str, Any], ego_index: int, actor_index: int, step_index: int
) -> tuple[float, float]:
    ego_x = float(scenario["x"][ego_index, step_index])
    ego_y = float(scenario["y"][ego_index, step_index])
    actor_x = float(scenario["x"][actor_index, step_index])
    actor_y = float(scenario["y"][actor_index, step_index])
    return actor_x - ego_x, actor_y - ego_y


def _distance_at_anchor(scenario: dict[str, Any], ego_index: int, actor_index: int) -> float:
    rel_x, rel_y = _relative_xy(scenario, ego_index, actor_index, ANCHOR_CURRENT_INDEX)
    return float(math.hypot(rel_x, rel_y))


def select_anchor_neighbors(
    scenario: dict[str, Any],
    *,
    max_neighbors: int = MAX_NEIGHBORS,
    radius_m: float = NEIGHBOR_RADIUS_M,
) -> list[int]:
    """Select the closest valid dynamic actors around the SDC at current time."""
    ego_index = _resolve_ego_index(scenario)
    if ego_index is None:
        return []
    mask = _dynamic_actor_mask(scenario, ego_index)
    candidate_indices = np.flatnonzero(mask)
    ranked = []
    for actor_index in candidate_indices:
        distance = _distance_at_anchor(scenario, ego_index, int(actor_index))
        if distance <= float(radius_m):
            ranked.append((distance, int(scenario["track_id"][actor_index]), int(actor_index)))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return [actor_index for _, _, actor_index in ranked[:max_neighbors]]


def validate_scenario(scenario: dict[str, Any]) -> ScenarioValidation:
    """Run the Stage 2 validation checks for one decoded scenario."""
    issues: list[str] = []
    timestamps = np.asarray(scenario["timestamps_us"], dtype=np.int64)
    monotone, cadence_close, cadence_mean, cadence_max_error = _timestamp_summary(timestamps)
    if timestamps.shape[0] != TOTAL_SCENARIO_STEPS:
        issues.append(f"timestamp_count={timestamps.shape[0]}")
    if not monotone:
        issues.append("timestamps_not_monotone")
    if not cadence_close:
        issues.append("timestamps_not_10hz")

    ego_index = _resolve_ego_index(scenario)
    ego_resolved = ego_index is not None
    if not ego_resolved:
        issues.append("ego_resolution_failed")

    neighbor_indices = select_anchor_neighbors(scenario) if ego_resolved else []
    if ego_resolved and not neighbor_indices:
        issues.append("no_usable_neighbors")

    return ScenarioValidation(
        scenario_id=str(scenario["scenario_id"]),
        shard_id=str(scenario["shard_id"]),
        record_index=int(scenario["record_index"]),
        actor_count=int(scenario["actor_count"]),
        timestamp_count=int(timestamps.shape[0]),
        timestamps_monotone=bool(monotone),
        cadence_close_to_10hz=bool(cadence_close),
        cadence_mean_us=float(cadence_mean),
        cadence_max_abs_error_us=float(cadence_max_error),
        ego_resolved=bool(ego_resolved),
        ego_track_id=int(scenario["track_id"][ego_index]) if ego_resolved else None,
        neighbor_count=len(neighbor_indices),
        neighbor_ids=[int(scenario["track_id"][idx]) for idx in neighbor_indices],
        usable=bool(monotone and cadence_close and ego_resolved and bool(neighbor_indices)),
        issues=issues,
    )


def _slot_fields(prefix: str, slot_count: int = MAX_NEIGHBORS) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for slot in range(slot_count):
        base = f"{prefix}{slot}"
        fields[f"{base}_track_id"] = None
        fields[f"{base}_type_code"] = None
        fields[f"{base}_type_label"] = None
        fields[f"{base}_distance_m"] = None
        fields[f"{base}_rel_x_m"] = None
        fields[f"{base}_rel_y_m"] = None
        fields[f"{base}_speed_mps"] = None
        fields[f"{base}_history_valid"] = False
        fields[f"{base}_current_valid"] = False
    return fields


def _object_mix_bin(type_codes: list[int]) -> str:
    vehicles = sum(1 for code in type_codes if int(code) == 1)
    pedestrians = sum(1 for code in type_codes if int(code) == 2)
    cyclists = sum(1 for code in type_codes if int(code) == 3)
    others = sum(1 for code in type_codes if int(code) not in {1, 2, 3})
    return f"veh{vehicles}_ped{pedestrians}_cyc{cyclists}_oth{others}"


def _scenario_index_row(scenario: dict[str, Any], validation: ScenarioValidation) -> dict[str, Any]:
    ego_index = _resolve_ego_index(scenario)
    neighbor_indices = select_anchor_neighbors(scenario) if ego_index is not None else []
    return {
        "scenario_id": validation.scenario_id,
        "shard_id": validation.shard_id,
        "record_index": validation.record_index,
        "actor_count": validation.actor_count,
        "dynamic_actor_count": int(_dynamic_actor_mask(scenario, ego_index).sum())
        if ego_index is not None
        else 0,
        "timestamp_count": validation.timestamp_count,
        "timestamp_start_us": int(scenario["timestamps_us"][0]),
        "timestamp_end_us": int(scenario["timestamps_us"][-1]),
        "cadence_mean_us": validation.cadence_mean_us,
        "cadence_max_abs_error_us": validation.cadence_max_abs_error_us,
        "timestamps_monotone": validation.timestamps_monotone,
        "cadence_close_to_10hz": validation.cadence_close_to_10hz,
        "ego_track_id": validation.ego_track_id,
        "ego_resolved": validation.ego_resolved,
        "neighbor_count": validation.neighbor_count,
        "neighbor_ids_csv": ",".join(str(track_id) for track_id in validation.neighbor_ids),
        "usable": validation.usable,
        "issues_csv": ",".join(validation.issues),
        "object_mix_bin": _object_mix_bin([int(scenario["track_type"][idx]) for idx in neighbor_indices]),
    }


def _anchor_row(scenario: dict[str, Any], validation: ScenarioValidation) -> dict[str, Any]:
    ego_index = _resolve_ego_index(scenario)
    if ego_index is None:
        raise ValueError("Cannot build anchor row without a unique ego track")
    neighbor_indices = select_anchor_neighbors(scenario)
    current_step = ANCHOR_CURRENT_INDEX
    row = {
        "scenario_id": validation.scenario_id,
        "shard_id": validation.shard_id,
        "record_index": validation.record_index,
        "anchor_step_index": current_step,
        "anchor_timestamp_us": int(scenario["timestamps_us"][current_step]),
        "ego_track_id": int(scenario["track_id"][ego_index]),
        "ego_type_code": int(scenario["track_type"][ego_index]),
        "ego_type_label": str(scenario["track_type_label"][ego_index]),
        "ego_x_m": float(scenario["x"][ego_index, current_step]),
        "ego_y_m": float(scenario["y"][ego_index, current_step]),
        "ego_speed_mps": float(scenario["speed"][ego_index, current_step]),
        "ego_velocity_x_mps": float(scenario["velocity_x"][ego_index, current_step]),
        "ego_velocity_y_mps": float(scenario["velocity_y"][ego_index, current_step]),
        "ego_heading_rad": float(scenario["bbox_yaw"][ego_index, current_step]),
        "ego_length_m": float(scenario["length"][ego_index, current_step]),
        "ego_width_m": float(scenario["width"][ego_index, current_step]),
        "ego_history_valid": bool(_history_valid_mask(scenario)[ego_index]),
        "neighbor_count": len(neighbor_indices),
        "neighbor_ids_csv": ",".join(str(int(scenario["track_id"][idx])) for idx in neighbor_indices),
        "object_mix_bin": _object_mix_bin([int(scenario["track_type"][idx]) for idx in neighbor_indices]),
        **_slot_fields("neighbor_slot_"),
    }
    history_valid_mask = _history_valid_mask(scenario)
    for slot, actor_index in enumerate(neighbor_indices):
        prefix = f"neighbor_slot_{slot}"
        rel_x, rel_y = _relative_xy(scenario, ego_index, actor_index, current_step)
        row[f"{prefix}_track_id"] = int(scenario["track_id"][actor_index])
        row[f"{prefix}_type_code"] = int(scenario["track_type"][actor_index])
        row[f"{prefix}_type_label"] = str(scenario["track_type_label"][actor_index])
        row[f"{prefix}_distance_m"] = float(math.hypot(rel_x, rel_y))
        row[f"{prefix}_rel_x_m"] = float(rel_x)
        row[f"{prefix}_rel_y_m"] = float(rel_y)
        row[f"{prefix}_speed_mps"] = float(scenario["speed"][actor_index, current_step])
        row[f"{prefix}_history_valid"] = bool(history_valid_mask[actor_index])
        row[f"{prefix}_current_valid"] = bool(scenario["valid"][actor_index, current_step])
    return row


def _phase_name(step_index: int) -> str:
    if step_index < PAST_STEPS:
        return "past"
    if step_index == ANCHOR_CURRENT_INDEX:
        return "current"
    return "future"


def _actor_track_rows(scenario: dict[str, Any], validation: ScenarioValidation) -> Iterable[dict[str, Any]]:
    ego_index = _resolve_ego_index(scenario)
    neighbor_indices = set(select_anchor_neighbors(scenario)) if ego_index is not None else set()
    for actor_index in range(int(scenario["actor_count"])):
        role = "other"
        if ego_index is not None and actor_index == ego_index:
            role = "ego"
        elif actor_index in neighbor_indices:
            role = "neighbor"
        for step_index in range(TOTAL_SCENARIO_STEPS):
            yield {
                "scenario_id": validation.scenario_id,
                "shard_id": validation.shard_id,
                "record_index": validation.record_index,
                "track_id": int(scenario["track_id"][actor_index]),
                "track_type_code": int(scenario["track_type"][actor_index]),
                "track_type_label": str(scenario["track_type_label"][actor_index]),
                "role": role,
                "is_ego": bool(role == "ego"),
                "is_selected_neighbor": bool(role == "neighbor"),
                "step_index": step_index,
                "phase": _phase_name(step_index),
                "timestamp_us": int(scenario["timestamps_us"][step_index]),
                "valid": bool(scenario["valid"][actor_index, step_index]),
                "x_m": float(scenario["x"][actor_index, step_index]),
                "y_m": float(scenario["y"][actor_index, step_index]),
                "z_m": float(scenario["z"][actor_index, step_index]),
                "speed_mps": float(scenario["speed"][actor_index, step_index]),
                "velocity_x_mps": float(scenario["velocity_x"][actor_index, step_index]),
                "velocity_y_mps": float(scenario["velocity_y"][actor_index, step_index]),
                "heading_rad": float(scenario["bbox_yaw"][actor_index, step_index]),
                "vel_yaw_rad": float(scenario["vel_yaw"][actor_index, step_index]),
                "length_m": float(scenario["length"][actor_index, step_index]),
                "width_m": float(scenario["width"][actor_index, step_index]),
                "height_m": float(scenario["height"][actor_index, step_index]),
            }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def build_validation_surface(
    *,
    raw_dir: str | Path,
    out_dir: str | Path,
    max_shards: int | None = None,
    max_scenarios: int | None = None,
    verify_crc: bool = False,
    write_actor_tracks: bool = True,
) -> dict[str, Any]:
    """Parse Waymo validation shards into Stage 2 ORIUS AV artifacts."""
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    shard_paths = sorted(raw_path.glob("validation_tfexample.tfrecord-*"))
    if max_shards is not None:
        shard_paths = shard_paths[: int(max_shards)]
    if not shard_paths:
        raise FileNotFoundError(f"No validation TFRecord shards found under {raw_path}")

    scenario_rows: list[dict[str, Any]] = []
    anchor_rows: list[dict[str, Any]] = []
    actor_writer = ParquetRowWriter(out_path / "actor_tracks.parquet") if write_actor_tracks else None

    issues: list[dict[str, Any]] = []
    shard_reports: list[dict[str, Any]] = []
    scenario_counter = 0
    usable_counter = 0
    monotone_counter = 0
    cadence_counter = 0
    ego_counter = 0
    neighbor_counter = 0

    for shard_path in shard_paths:
        shard_record_count = 0
        shard_success_count = 0
        shard_id = shard_path.name
        for record_index, payload in enumerate(iter_tfrecord_records(shard_path, verify_crc=verify_crc)):
            if max_scenarios is not None and scenario_counter >= int(max_scenarios):
                break
            shard_record_count += 1
            scenario_counter += 1
            try:
                features = parse_example_bytes(payload)
                scenario = decode_motion_scenario(features, shard_id=shard_id, record_index=record_index)
                validation = validate_scenario(scenario)
                scenario_rows.append(_scenario_index_row(scenario, validation))
                if validation.ego_resolved:
                    ego_counter += 1
                if validation.timestamps_monotone:
                    monotone_counter += 1
                if validation.cadence_close_to_10hz:
                    cadence_counter += 1
                if validation.neighbor_count > 0:
                    neighbor_counter += 1
                if validation.usable:
                    usable_counter += 1
                anchor_rows.append(_anchor_row(scenario, validation))
                if actor_writer is not None:
                    for row in _actor_track_rows(scenario, validation):
                        actor_writer.append(row)
                shard_success_count += 1
                if validation.issues:
                    issues.append(
                        {
                            "scenario_id": validation.scenario_id,
                            "shard_id": validation.shard_id,
                            "record_index": validation.record_index,
                            "issues": list(validation.issues),
                        }
                    )
            except Exception as exc:
                issues.append(
                    {
                        "scenario_id": None,
                        "shard_id": shard_id,
                        "record_index": record_index,
                        "issues": [f"exception:{type(exc).__name__}:{exc}"],
                    }
                )
        shard_reports.append(
            {
                "shard_id": shard_id,
                "record_count": shard_record_count,
                "success_count": shard_success_count,
                "readable": bool(shard_success_count > 0),
                "path": str(shard_path),
                "size_bytes": int(shard_path.stat().st_size),
            }
        )
        if max_scenarios is not None and scenario_counter >= int(max_scenarios):
            break

    if actor_writer is not None:
        actor_writer.close()

    scenario_index_path = out_path / "scenario_index.parquet"
    anchor_path = out_path / "ego_neighborhood_anchors.parquet"
    pd.DataFrame(scenario_rows).to_parquet(scenario_index_path, index=False)
    pd.DataFrame(anchor_rows).to_parquet(anchor_path, index=False)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_dir": str(raw_path),
        "out_dir": str(out_path),
        "shard_count": len(shard_reports),
        "scenario_count": scenario_counter,
        "usable_scenario_count": usable_counter,
        "timestamps_monotone_count": monotone_counter,
        "timestamps_10hz_count": cadence_counter,
        "ego_resolved_count": ego_counter,
        "neighbor_usable_count": neighbor_counter,
        "all_shards_opened": all(item["readable"] for item in shard_reports),
        "shards": shard_reports,
        "issues": issues[:128],
        "issue_count": len(issues),
        "artifacts": {
            "scenario_index": str(scenario_index_path),
            "actor_tracks": str(out_path / "actor_tracks.parquet") if write_actor_tracks else None,
            "ego_neighborhood_anchors": str(anchor_path),
        },
    }
    report_path = out_path / "parse_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return report
