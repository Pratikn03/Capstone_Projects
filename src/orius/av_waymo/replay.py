"""Replay-surface builders and Waymo-specific benchmark track."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from orius.orius_bench.adapter import BenchmarkAdapter

from .dataset import (
    ANCHOR_CURRENT_INDEX,
    MAX_NEIGHBORS,
    TOTAL_SCENARIO_STEPS,
    _resolve_ego_index,
    decode_motion_scenario,
    parse_example_bytes,
    select_anchor_neighbors,
    validate_scenario,
)
from .tfrecord import iter_tfrecord_records


FAULT_FAMILIES = ("dropout", "stale", "delay_jitter", "out_of_order", "spikes", "drift_combo")
BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _slot_fields(prefix: str, slot_count: int = MAX_NEIGHBORS) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for slot in range(slot_count):
        base = f"{prefix}{slot}"
        fields[f"{base}_track_id"] = None
        fields[f"{base}_x_m"] = None
        fields[f"{base}_y_m"] = None
        fields[f"{base}_speed_mps"] = None
        fields[f"{base}_length_m"] = None
        fields[f"{base}_width_m"] = None
        fields[f"{base}_rel_x_m"] = None
        fields[f"{base}_rel_y_m"] = None
        fields[f"{base}_rel_longitudinal_gap_m"] = None
        fields[f"{base}_rel_lateral_offset_m"] = None
        fields[f"{base}_valid"] = False
    return fields


def _timestamp_to_iso(timestamp_us: int) -> str:
    ts = BASE_TS + timedelta(microseconds=int(timestamp_us))
    return ts.isoformat().replace("+00:00", "Z")


def _project_to_ego_frame(
    *,
    ego_x: float,
    ego_y: float,
    ego_heading_rad: float,
    actor_x: float,
    actor_y: float,
) -> tuple[float, float]:
    rel_x = float(actor_x - ego_x)
    rel_y = float(actor_y - ego_y)
    cos_h = math.cos(float(ego_heading_rad))
    sin_h = math.sin(float(ego_heading_rad))
    longitudinal = rel_x * cos_h + rel_y * sin_h
    lateral = -rel_x * sin_h + rel_y * cos_h
    return longitudinal, lateral


def infer_speed_limit_mps(scenario: Mapping[str, Any], ego_index: int) -> float:
    valid = np.asarray(scenario["valid"][ego_index], dtype=bool)
    speeds = np.asarray(scenario["speed"][ego_index], dtype=float)
    if not valid.any():
        return 15.0
    peak = float(np.nanmax(speeds[valid]))
    return float(np.clip(peak + 2.0, 5.0, 40.0))


def compute_state_safety_metrics(state: Mapping[str, Any]) -> dict[str, float | bool | int | None]:
    """Compute overlap, gap, TTC, and a scalar longitudinal margin."""
    neighbor_count = int(state.get("neighbor_count", 0) or 0)
    ego_speed = float(state.get("ego_speed_mps", 0.0) or 0.0)
    ego_length = float(state.get("ego_length_m", 4.5) or 4.5)
    ego_width = float(state.get("ego_width_m", 2.0) or 2.0)

    front_gap = float("inf")
    front_track_id: int | None = None
    front_speed = 0.0
    front_closing_speed = 0.0
    overlap = False
    for slot in range(neighbor_count):
        prefix = f"neighbor_slot_{slot}"
        valid = bool(state.get(f"{prefix}_valid", False))
        if not valid:
            continue
        longitudinal = state.get(f"{prefix}_rel_longitudinal_gap_m")
        lateral = state.get(f"{prefix}_rel_lateral_offset_m")
        neighbor_length = float(state.get(f"{prefix}_length_m", 4.0) or 4.0)
        neighbor_width = float(state.get(f"{prefix}_width_m", 2.0) or 2.0)
        neighbor_speed = float(state.get(f"{prefix}_speed_mps", 0.0) or 0.0)
        if longitudinal is None or lateral is None:
            continue
        longitudinal_val = float(longitudinal)
        lateral_val = abs(float(lateral))
        bumper_gap = longitudinal_val - 0.5 * (ego_length + neighbor_length)
        same_lane = lateral_val <= 0.5 * (ego_width + neighbor_width) + 1.5
        if same_lane and longitudinal_val >= 0.0 and bumper_gap <= 0.0:
            overlap = True
        if same_lane and longitudinal_val >= 0.0 and bumper_gap < front_gap:
            front_gap = bumper_gap
            front_track_id = int(state.get(f"{prefix}_track_id") or 0)
            front_speed = neighbor_speed
            front_closing_speed = max(0.0, ego_speed - neighbor_speed)

    if front_track_id is None:
        front_gap = 100.0
        ttc = float("inf")
    else:
        ttc = front_gap / max(front_closing_speed, 1e-9) if front_closing_speed > 0.0 else float("inf")
    margin = float(front_gap - max(5.0, 2.0 * front_closing_speed))
    violated = bool(overlap or front_gap < 5.0 or ttc < 2.0)
    return {
        "lead_track_id": front_track_id,
        "min_gap_m": float(front_gap),
        "lead_speed_mps": float(front_speed),
        "lead_rel_speed_mps": float(front_closing_speed),
        "ttc_s": float(ttc),
        "overlap": bool(overlap),
        "true_constraint_violated": violated,
        "true_margin": float(margin),
    }


def _row_from_scenario(
    scenario: Mapping[str, Any],
    *,
    validation: Any,
    step_index: int,
    ego_index: int,
    neighbor_indices: list[int],
    speed_limit_mps: float,
) -> dict[str, Any]:
    ego_x = float(scenario["x"][ego_index, step_index])
    ego_y = float(scenario["y"][ego_index, step_index])
    ego_heading = float(scenario["bbox_yaw"][ego_index, step_index])
    row = {
        "scenario_id": str(validation.scenario_id),
        "shard_id": str(validation.shard_id),
        "record_index": int(validation.record_index),
        "step_index": int(step_index),
        "timestamp_us": int(scenario["timestamps_us"][step_index]),
        "ts_utc": _timestamp_to_iso(int(scenario["timestamps_us"][step_index])),
        "ego_track_id": int(scenario["track_id"][ego_index]),
        "ego_x_m": ego_x,
        "ego_y_m": ego_y,
        "ego_speed_mps": float(scenario["speed"][ego_index, step_index]),
        "ego_velocity_x_mps": float(scenario["velocity_x"][ego_index, step_index]),
        "ego_velocity_y_mps": float(scenario["velocity_y"][ego_index, step_index]),
        "ego_heading_rad": ego_heading,
        "ego_length_m": float(scenario["length"][ego_index, step_index]),
        "ego_width_m": float(scenario["width"][ego_index, step_index]),
        "ego_valid": bool(scenario["valid"][ego_index, step_index]),
        "speed_limit_mps": float(speed_limit_mps),
        "neighbor_count": len(neighbor_indices),
        "object_mix_bin": validation.neighbor_count and (
            f"n{validation.neighbor_count}_{','.join(str(track_id) for track_id in validation.neighbor_ids)}"
        ) or "n0",
        **_slot_fields("neighbor_slot_"),
    }
    for slot, actor_index in enumerate(neighbor_indices):
        prefix = f"neighbor_slot_{slot}"
        actor_x = float(scenario["x"][actor_index, step_index])
        actor_y = float(scenario["y"][actor_index, step_index])
        longitudinal, lateral = _project_to_ego_frame(
            ego_x=ego_x,
            ego_y=ego_y,
            ego_heading_rad=ego_heading,
            actor_x=actor_x,
            actor_y=actor_y,
        )
        row[f"{prefix}_track_id"] = int(scenario["track_id"][actor_index])
        row[f"{prefix}_x_m"] = actor_x
        row[f"{prefix}_y_m"] = actor_y
        row[f"{prefix}_speed_mps"] = float(scenario["speed"][actor_index, step_index])
        row[f"{prefix}_length_m"] = float(scenario["length"][actor_index, step_index])
        row[f"{prefix}_width_m"] = float(scenario["width"][actor_index, step_index])
        row[f"{prefix}_rel_x_m"] = actor_x - ego_x
        row[f"{prefix}_rel_y_m"] = actor_y - ego_y
        row[f"{prefix}_rel_longitudinal_gap_m"] = float(longitudinal)
        row[f"{prefix}_rel_lateral_offset_m"] = float(lateral)
        row[f"{prefix}_valid"] = bool(scenario["valid"][actor_index, step_index])
    row.update(compute_state_safety_metrics(row))
    return row


def _selected_record_map(subset_manifest: pd.DataFrame) -> dict[str, dict[int, dict[str, Any]]]:
    selected: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in subset_manifest.to_dict(orient="records"):
        selected[str(row["shard_id"])][int(row["record_index"])] = row
    return selected


def _decode_selected_scenarios(raw_dir: Path, subset_manifest: pd.DataFrame) -> Iterable[tuple[dict[str, Any], Any]]:
    selected = _selected_record_map(subset_manifest)
    for shard_id, record_map in selected.items():
        shard_path = raw_dir / shard_id
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard referenced by subset manifest: {shard_path}")
        for record_index, payload in enumerate(iter_tfrecord_records(shard_path)):
            if record_index not in record_map:
                continue
            features = parse_example_bytes(payload)
            scenario = decode_motion_scenario(features, shard_id=shard_id, record_index=record_index)
            validation = validate_scenario(scenario)
            yield scenario, validation


def build_replay_surface(
    *,
    raw_dir: str | Path,
    subset_manifest_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Build the 91-step replay windows for the selected dry-run subset."""
    raw_path = Path(raw_dir)
    subset_manifest = pd.read_parquet(subset_manifest_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scenario_counter = 0
    for scenario, validation in _decode_selected_scenarios(raw_path, subset_manifest):
        if not validation.usable:
            continue
        ego_index = _resolve_ego_index(scenario)
        if ego_index is None:
            continue
        neighbors = select_anchor_neighbors(scenario)
        speed_limit = infer_speed_limit_mps(scenario, ego_index)
        for step_index in range(TOTAL_SCENARIO_STEPS):
            rows.append(
                _row_from_scenario(
                    scenario,
                    validation=validation,
                    step_index=step_index,
                    ego_index=ego_index,
                    neighbor_indices=neighbors,
                    speed_limit_mps=speed_limit,
                )
            )
        scenario_counter += 1

    replay_df = pd.DataFrame(rows).sort_values(["scenario_id", "step_index"]).reset_index(drop=True)
    replay_path = out_path / "replay_windows.parquet"
    replay_df.to_parquet(replay_path, index=False)
    report = {
        "raw_dir": str(raw_path),
        "subset_manifest_path": str(subset_manifest_path),
        "out_dir": str(out_path),
        "scenario_count": int(scenario_counter),
        "row_count": int(len(replay_df)),
        "artifacts": {
            "replay_windows": str(replay_path),
        },
    }
    report_path = out_path / "replay_surface_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def apply_observation_fault(
    true_state: Mapping[str, Any],
    *,
    fault_kind: str,
    step_index: int,
    rng: np.random.Generator,
    memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one deterministic fault family to a replay state."""
    observed = dict(true_state)
    state = memory if memory is not None else {}
    dynamic_keys = [
        "ego_speed_mps",
        "ego_x_m",
        "ego_y_m",
        "min_gap_m",
        "lead_speed_mps",
        "lead_rel_speed_mps",
        "ttc_s",
    ]
    for slot in range(int(observed.get("neighbor_count", 0) or 0)):
        dynamic_keys.extend(
            [
                f"neighbor_slot_{slot}_speed_mps",
                f"neighbor_slot_{slot}_rel_longitudinal_gap_m",
                f"neighbor_slot_{slot}_rel_lateral_offset_m",
            ]
        )

    if fault_kind == "dropout":
        for key in dynamic_keys:
            if key in observed:
                observed[key] = float("nan")
        observed["neighbor_count"] = 0
    elif fault_kind == "stale":
        previous = state.get("previous_observed")
        if isinstance(previous, Mapping):
            for key in dynamic_keys:
                if key in previous:
                    observed[key] = previous[key]
            observed["timestamp_us"] = previous.get("timestamp_us", observed.get("timestamp_us"))
            observed["ts_utc"] = previous.get("ts_utc", observed.get("ts_utc"))
    elif fault_kind == "delay_jitter":
        jitter_us = int(rng.integers(-180_000, 180_001))
        observed["timestamp_us"] = int(observed.get("timestamp_us", 0) or 0) + jitter_us
        observed["ts_utc"] = _timestamp_to_iso(int(observed["timestamp_us"]))
    elif fault_kind == "out_of_order":
        previous = state.get("previous_true")
        if isinstance(previous, Mapping):
            for key in dynamic_keys:
                if key in previous:
                    observed[key] = previous[key]
            observed["timestamp_us"] = previous.get("timestamp_us", observed.get("timestamp_us"))
            observed["ts_utc"] = previous.get("ts_utc", observed.get("ts_utc"))
    elif fault_kind == "spikes":
        observed["ego_speed_mps"] = float(observed.get("ego_speed_mps", 0.0) or 0.0) + float(rng.normal(0.0, 6.0))
        if "min_gap_m" in observed and observed["min_gap_m"] is not None:
            observed["min_gap_m"] = float(observed["min_gap_m"]) + float(rng.normal(0.0, 8.0))
    elif fault_kind == "drift_combo":
        drift = float(state.get("drift_bias", 0.0)) + 0.15
        state["drift_bias"] = drift
        observed["ego_speed_mps"] = float(observed.get("ego_speed_mps", 0.0) or 0.0) + drift
        if "min_gap_m" in observed and observed["min_gap_m"] is not None:
            observed["min_gap_m"] = float(observed["min_gap_m"]) - 1.5 * drift
    state["previous_observed"] = dict(observed)
    state["previous_true"] = dict(true_state)
    return observed


class WaymoReplayTrackAdapter(BenchmarkAdapter):
    """Action-sensitive replay track over Waymo AV dry-run windows."""

    def __init__(
        self,
        replay_windows_path: str | Path,
        *,
        start_step_index: int = ANCHOR_CURRENT_INDEX,
        scenario_ids: Iterable[str] | None = None,
    ):
        self._replay_path = Path(replay_windows_path)
        scenario_id_filter = sorted({str(scenario_id) for scenario_id in scenario_ids or []})
        if scenario_id_filter:
            try:
                self._df = pd.read_parquet(self._replay_path, filters=[("scenario_id", "in", scenario_id_filter)])
            except Exception:
                scenario_id_set = set(scenario_id_filter)
                frames: list[pd.DataFrame] = []
                for batch in pq.ParquetFile(self._replay_path).iter_batches(batch_size=200_000):
                    frame = batch.to_pandas()
                    frame = frame[frame["scenario_id"].astype(str).isin(scenario_id_set)]
                    if not frame.empty:
                        frames.append(frame)
                self._df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        else:
            self._df = pd.read_parquet(self._replay_path)
        self._df = self._df.sort_values(["scenario_id", "step_index"]).reset_index(drop=True)
        self._scenarios = {
            str(scenario_id): group.reset_index(drop=True)
            for scenario_id, group in self._df.groupby("scenario_id", sort=True)
        }
        self._scenario_order = sorted(self._scenarios)
        self._rng = np.random.default_rng(42)
        self._start_step_index = int(start_step_index)
        self._episode_rows: pd.DataFrame | None = None
        self._episode_idx = 0
        self._state: dict[str, Any] = {}
        self._fault_state: dict[str, Any] = {}

    def load_scenario(self, scenario_id: str, *, start_step_index: int | None = None) -> Mapping[str, Any]:
        if scenario_id not in self._scenarios:
            raise KeyError(f"Unknown replay scenario_id: {scenario_id}")
        self._episode_rows = self._scenarios[scenario_id]
        start = self._start_step_index if start_step_index is None else int(start_step_index)
        available = self._episode_rows[self._episode_rows["step_index"] >= start]
        if available.empty:
            raise ValueError(f"No replay rows available for scenario {scenario_id} at step >= {start}")
        self._episode_idx = int(available.index.min())
        self._state = dict(self._episode_rows.iloc[self._episode_idx].to_dict())
        self._fault_state = {}
        return self.true_state()

    def reset(self, seed: int = 42) -> Mapping[str, Any]:
        self._rng = np.random.default_rng(seed)
        if not self._scenario_order:
            raise RuntimeError("WaymoReplayTrackAdapter has no scenarios loaded.")
        scenario_id = self._scenario_order[int(self._rng.integers(0, len(self._scenario_order)))]
        return self.load_scenario(scenario_id)

    def true_state(self) -> Mapping[str, Any]:
        return dict(self._state)

    def observe(
        self,
        true_state: Mapping[str, Any],
        fault: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if not fault:
            observed = dict(true_state)
            self._fault_state["previous_observed"] = dict(observed)
            self._fault_state["previous_true"] = dict(true_state)
            return observed
        kind = str(fault.get("kind", "")).strip()
        return apply_observation_fault(
            true_state,
            fault_kind=kind,
            step_index=int(true_state.get("step_index", 0) or 0),
            rng=self._rng,
            memory=self._fault_state,
        )

    def safe_action_set(
        self,
        state: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del uncertainty
        speed_limit = float(state.get("speed_limit_mps", 25.0) or 25.0)
        return {"acceleration_mps2_lower": -6.0, "acceleration_mps2_upper": 3.0, "speed_limit_mps": speed_limit}

    def _recompute_state(self, reference_row: Mapping[str, Any], *, ego_x_m: float, ego_y_m: float, ego_speed_mps: float) -> dict[str, Any]:
        state = dict(reference_row)
        state["ego_x_m"] = float(ego_x_m)
        state["ego_y_m"] = float(ego_y_m)
        state["ego_speed_mps"] = float(ego_speed_mps)
        ego_heading = float(state.get("ego_heading_rad", 0.0) or 0.0)
        for slot in range(int(state.get("neighbor_count", 0) or 0)):
            prefix = f"neighbor_slot_{slot}"
            if not state.get(f"{prefix}_valid", False):
                continue
            actor_x = float(state.get(f"{prefix}_x_m", 0.0) or 0.0)
            actor_y = float(state.get(f"{prefix}_y_m", 0.0) or 0.0)
            longitudinal, lateral = _project_to_ego_frame(
                ego_x=ego_x_m,
                ego_y=ego_y_m,
                ego_heading_rad=ego_heading,
                actor_x=actor_x,
                actor_y=actor_y,
            )
            state[f"{prefix}_rel_x_m"] = actor_x - ego_x_m
            state[f"{prefix}_rel_y_m"] = actor_y - ego_y_m
            state[f"{prefix}_rel_longitudinal_gap_m"] = float(longitudinal)
            state[f"{prefix}_rel_lateral_offset_m"] = float(lateral)
        state.update(compute_state_safety_metrics(state))
        return state

    def step(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._episode_rows is None:
            raise RuntimeError("WaymoReplayTrackAdapter.reset() or load_scenario() must be called first.")
        current = dict(self._state)
        next_index = min(self._episode_idx + 1, len(self._episode_rows) - 1)
        reference_next = dict(self._episode_rows.iloc[next_index].to_dict())
        dt_s = max(
            0.1,
            (int(reference_next.get("timestamp_us", current.get("timestamp_us", 0))) - int(current.get("timestamp_us", 0))) / 1_000_000.0,
        )
        accel = float(action.get("acceleration_mps2", 0.0) or 0.0)
        replay_speed = float(reference_next.get("ego_speed_mps", current.get("ego_speed_mps", 0.0)) or 0.0)
        current_speed = float(current.get("ego_speed_mps", 0.0) or 0.0)
        speed_limit = float(current.get("speed_limit_mps", 25.0) or 25.0)
        next_speed = np.clip(current_speed + accel * dt_s + 0.10 * (replay_speed - current_speed), 0.0, speed_limit)
        heading = float(current.get("ego_heading_rad", 0.0) or 0.0)
        next_x = float(current.get("ego_x_m", 0.0) or 0.0) + float(next_speed) * dt_s * math.cos(heading)
        next_y = float(current.get("ego_y_m", 0.0) or 0.0) + float(next_speed) * dt_s * math.sin(heading)
        self._episode_idx = next_index
        self._state = self._recompute_state(reference_next, ego_x_m=next_x, ego_y_m=next_y, ego_speed_mps=float(next_speed))
        return self.true_state()

    def compute_useful_work(self, trajectory: Iterable[Mapping[str, Any]]) -> float:
        total = 0.0
        previous = None
        for state in trajectory:
            if previous is not None and not self.check_violation(state)["violated"]:
                dx = float(state.get("ego_x_m", 0.0) or 0.0) - float(previous.get("ego_x_m", 0.0) or 0.0)
                if dx > 0.0:
                    total += dx
            previous = state
        return float(total)

    def check_violation(self, state: Mapping[str, Any]) -> dict[str, Any]:
        metrics = compute_state_safety_metrics(state)
        severity = 0.0
        if bool(metrics["overlap"]):
            severity = max(severity, 1.0)
        if float(metrics["min_gap_m"]) < 5.0:
            severity = max(severity, 5.0 - float(metrics["min_gap_m"]))
        if math.isfinite(float(metrics["ttc_s"])) and float(metrics["ttc_s"]) < 2.0:
            severity = max(severity, 2.0 - float(metrics["ttc_s"]))
        return {"violated": bool(metrics["true_constraint_violated"]), "severity": float(severity)}

    def observed_constraint_satisfied(self, observed_state: Mapping[str, Any]) -> bool | None:
        if any(math.isnan(float(observed_state.get(key, 0.0))) for key in ("ego_speed_mps", "min_gap_m") if observed_state.get(key) is not None):
            return None
        metrics = compute_state_safety_metrics(observed_state)
        return not bool(metrics["true_constraint_violated"])

    def constraint_margin(self, state: Mapping[str, Any]) -> float | None:
        metrics = compute_state_safety_metrics(state)
        return float(metrics["true_margin"])

    @property
    def domain_name(self) -> str:
        return "av_waymo"
