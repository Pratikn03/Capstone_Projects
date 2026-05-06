#!/usr/bin/env python3
"""Build bounded AV closed-loop planner artifacts from nuPlan runtime traces.

This is a kinematic closed-loop layer over the locked runtime trace.  It is not
CARLA and not road deployment: ego actions update future ego speed and gap inside
a deterministic longitudinal model, while lead-vehicle motion is estimated from
the nuPlan trace.  The artifact exists to separate true action-feedback evidence
from pure replay/runtime-denominator evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = (
    REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
)
DEFAULT_OUT = REPO_ROOT / "reports" / "predeployment_external_validation" / "av_closed_loop_planner"

CLOSED_LOOP_SURFACE = "nuplan_bounded_kinematic_closed_loop_planner"
CLAIM_BOUNDARY = (
    "Bounded kinematic nuPlan closed-loop planner evaluation: ego actions update "
    "future ego speed and gap, but this is not CARLA, road deployment, or full "
    "autonomous-driving field closure."
)

TRACE_COLUMNS = {
    "trace_id",
    "scenario_id",
    "step_index",
    "fault_family",
    "controller",
    "candidate_acceleration_mps2",
    "safe_acceleration_mps2",
    "intervened",
    "fallback_used",
    "certificate_valid",
    "true_constraint_violated",
    "domain_postcondition_passed",
    "min_gap_m",
    "ttc_s",
    "ego_speed_mps",
    "target_relative_gap_1s",
    "reliability_w",
    "true_margin",
    "observed_margin",
}

SERIOUS_BASELINE_CONTROLLERS = [
    "baseline",
    "rss_cbf_filter",
    "robust_fixed_deceleration",
    "nonreliability_conformal_runtime",
    "stale_certificate_no_temporal_guard",
    "always_brake",
    "orius",
]

SOURCE_REQUIRED_FAULTS = {
    "dropout",
    "stale",
    "delay_jitter",
    "spikes",
    "drift_combo",
    "out_of_order",
}

REQUIRED_PLANNER_STRESS_FAMILIES = SOURCE_REQUIRED_FAULTS | {"blackout"}

ABLATION_PROXIES = [
    {
        "method": "planner_only",
        "controller": "baseline",
        "component_removed": "all_runtime_safety_contracts",
    },
    {
        "method": "cp_only",
        "controller": "nonreliability_conformal_runtime",
        "component_removed": "reliability_conditioning",
    },
    {
        "method": "reliability_only",
        "controller": "robust_fixed_deceleration",
        "component_removed": "conformal_certificate_and_temporal_guard",
    },
    {
        "method": "certificate_only",
        "controller": "stale_certificate_no_temporal_guard",
        "component_removed": "certificate_half_life_refresh",
    },
    {
        "method": "no_reliability",
        "controller": "nonreliability_conformal_runtime",
        "component_removed": "runtime_reliability_score",
    },
    {
        "method": "no_certificate_half_life",
        "controller": "stale_certificate_no_temporal_guard",
        "component_removed": "certificate_expiry_half_life",
    },
    {
        "method": "no_fallback_projection",
        "controller": "predictor_only_no_runtime",
        "fallback_controller": "baseline",
        "component_removed": "safe_projection_and_fallback",
    },
    {
        "method": "orius_full",
        "controller": "orius",
        "component_removed": "none",
    },
]

ORIUS_FRONTIER_PROFILES = [
    {
        "controller": "orius_profile_aggressive",
        "threshold": 0.25,
        "fallback_threshold": 0.15,
        "brake_cap": None,
    },
    {
        "controller": "orius_profile_balanced",
        "threshold": 0.45,
        "fallback_threshold": 0.30,
        "brake_cap": None,
    },
    {
        "controller": "orius_profile_conservative",
        "threshold": 0.65,
        "fallback_threshold": 0.50,
        "brake_cap": -1.5,
    },
    {
        "controller": "orius_profile_fail_closed",
        "threshold": 1.01,
        "fallback_threshold": 1.01,
        "brake_cap": -2.0,
    },
]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _remove_appledouble_files(root: Path) -> None:
    for path in root.rglob("._*"):
        if path.is_file():
            with suppress(OSError):
                path.unlink()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _boolish(value: Any) -> bool:
    if isinstance(value, bool | np.bool_):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _floatish(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return result


def _read_selected_trace_rows(
    runtime_traces: Path,
    *,
    max_rows: int | None,
    rows_per_controller_fault_family: int | None,
) -> pd.DataFrame:
    if not runtime_traces.exists():
        raise FileNotFoundError(runtime_traces)
    if max_rows is not None:
        frame = pd.read_csv(
            runtime_traces,
            usecols=lambda column: column in TRACE_COLUMNS,
            nrows=max_rows,
            low_memory=False,
        )
        frame.attrs["sampling_policy"] = f"prefix_max_rows_{max_rows}"
        frame.attrs["source_rows_scanned"] = int(len(frame))
        return frame

    if rows_per_controller_fault_family is None:
        frame = pd.read_csv(
            runtime_traces,
            usecols=lambda column: column in TRACE_COLUMNS,
            low_memory=False,
        )
        frame.attrs["sampling_policy"] = "full_trace"
        frame.attrs["source_rows_scanned"] = int(len(frame))
        return frame

    selected_chunks: list[pd.DataFrame] = []
    counts: dict[tuple[str, str], int] = {}
    rows_scanned = 0
    for chunk in pd.read_csv(
        runtime_traces,
        usecols=lambda column: column in TRACE_COLUMNS,
        chunksize=200_000,
        low_memory=False,
    ):
        rows_scanned += int(len(chunk))
        if "fault_family" not in chunk:
            chunk["fault_family"] = "observed_runtime"
        keep_parts: list[pd.DataFrame] = []
        for key, group in chunk.groupby(["controller", "fault_family"], sort=False, dropna=False):
            controller, fault_family = str(key[0]), str(key[1])
            current = counts.get((controller, fault_family), 0)
            remaining = int(rows_per_controller_fault_family) - current
            if remaining <= 0:
                continue
            sample = group.head(remaining)
            counts[(controller, fault_family)] = current + int(len(sample))
            keep_parts.append(sample)
        if keep_parts:
            selected_chunks.append(pd.concat(keep_parts, ignore_index=True))
        if all(
            counts.get((controller, fault), 0) >= int(rows_per_controller_fault_family)
            for controller in SERIOUS_BASELINE_CONTROLLERS
            for fault in SOURCE_REQUIRED_FAULTS
        ):
            break
    frame = pd.concat(selected_chunks, ignore_index=True) if selected_chunks else pd.DataFrame()
    frame.attrs["sampling_policy"] = (
        f"stratified_first_{rows_per_controller_fault_family}_rows_per_controller_fault_family"
    )
    frame.attrs["source_rows_scanned"] = rows_scanned
    return frame


def _load_trace_frame(
    runtime_traces: Path,
    *,
    max_rows: int | None,
    rows_per_controller_fault_family: int | None,
) -> pd.DataFrame:
    frame = _read_selected_trace_rows(
        runtime_traces,
        max_rows=max_rows,
        rows_per_controller_fault_family=rows_per_controller_fault_family,
    )
    sampling_policy = frame.attrs.get("sampling_policy", "unknown")
    source_rows_scanned = int(frame.attrs.get("source_rows_scanned", len(frame)))
    if frame.empty:
        raise ValueError(f"{runtime_traces} has no trace rows")
    missing = {"scenario_id", "step_index", "controller", "min_gap_m", "ego_speed_mps"} - set(frame.columns)
    if missing:
        raise ValueError(f"{runtime_traces} is missing required columns: {sorted(missing)}")
    if "fault_family" not in frame:
        frame["fault_family"] = "observed_runtime"
    for column in (
        "step_index",
        "candidate_acceleration_mps2",
        "safe_acceleration_mps2",
        "min_gap_m",
        "ego_speed_mps",
        "target_relative_gap_1s",
        "reliability_w",
        "true_margin",
        "observed_margin",
    ):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("intervened", "fallback_used", "certificate_valid", "true_constraint_violated"):
        if column in frame:
            frame[column] = frame[column].map(_boolish)
        else:
            frame[column] = False
    frame = frame.sort_values(["controller", "scenario_id", "step_index"], kind="stable").reset_index(
        drop=True
    )
    frame.attrs["sampling_policy"] = sampling_policy
    frame.attrs["source_rows_scanned"] = source_rows_scanned
    return frame


def _controller_acceleration(row: pd.Series) -> float:
    controller = str(row.get("controller", ""))
    candidate = _floatish(row.get("candidate_acceleration_mps2"), 0.0)
    safe = _floatish(row.get("safe_acceleration_mps2"), candidate)
    if controller == "always_brake":
        return min(safe, -2.0)
    if controller == "baseline":
        return candidate
    if controller == "predictor_only_no_runtime":
        return candidate
    return safe


def _with_orius_frontier_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    if "controller" not in frame:
        return frame
    orius = frame[frame["controller"].astype(str) == "orius"].copy()
    if orius.empty:
        frame = frame.copy()
        frame["frontier_policy"] = "baseline_comparator"
        return frame
    expanded = [frame.copy()]
    expanded[0]["frontier_policy"] = np.where(
        expanded[0]["controller"].astype(str) == "orius", "orius_full_stack", "baseline_comparator"
    )
    reliability_raw = (
        orius["reliability_w"] if "reliability_w" in orius else pd.Series(1.0, index=orius.index)
    )
    candidate_raw = (
        orius["candidate_acceleration_mps2"]
        if "candidate_acceleration_mps2" in orius
        else pd.Series(0.0, index=orius.index)
    )
    reliability = pd.to_numeric(reliability_raw, errors="coerce").fillna(1.0)
    candidate = pd.to_numeric(candidate_raw, errors="coerce").fillna(0.0)
    safe_raw = orius.get("safe_acceleration_mps2", candidate)
    safe = pd.to_numeric(safe_raw, errors="coerce").fillna(candidate)
    base_fallback = orius.get("fallback_used", False)
    if not isinstance(base_fallback, pd.Series):
        base_fallback = pd.Series(bool(base_fallback), index=orius.index)
    base_fallback = base_fallback.map(_boolish)
    for profile in ORIUS_FRONTIER_PROFILES:
        profiled = orius.copy()
        intervene = (reliability <= float(profile["threshold"])) | base_fallback
        fallback = reliability <= float(profile["fallback_threshold"])
        action = np.where(intervene, safe, candidate)
        brake_cap = profile["brake_cap"]
        if brake_cap is not None:
            action = np.where(intervene, np.minimum(action, float(brake_cap)), action)
        profiled["controller"] = profile["controller"]
        profiled["frontier_policy"] = "orius_policy_sweep"
        profiled["safe_acceleration_mps2"] = action
        profiled["intervened"] = intervene
        profiled["fallback_used"] = fallback
        profiled["certificate_valid"] = True
        expanded.append(profiled)
    return pd.concat(expanded, ignore_index=True)


def _with_controlled_blackout_stress(frame: pd.DataFrame) -> pd.DataFrame:
    if "fault_family" not in frame:
        return frame
    fault_families = set(frame["fault_family"].astype(str).unique())
    if "blackout" in fault_families or "dropout" not in fault_families:
        frame = frame.copy()
        frame["stress_source"] = "locked_runtime_trace"
        return frame
    blackout = frame[frame["fault_family"].astype(str) == "dropout"].copy()
    blackout["fault_family"] = "blackout"
    blackout["stress_source"] = "controlled_blackout_injection_from_dropout_seed"
    blackout["trace_id"] = blackout.get("trace_id", "").astype(str) + ":blackout"
    blackout["reliability_w"] = 0.0
    if "observed_margin" in blackout:
        blackout["observed_margin"] = np.minimum(
            pd.to_numeric(blackout["observed_margin"], errors="coerce").fillna(0.0), 0.0
        )
    orius_like = blackout["controller"].astype(str).str.startswith("orius")
    safe = pd.to_numeric(blackout.get("safe_acceleration_mps2", 0.0), errors="coerce").fillna(0.0)
    blackout.loc[orius_like, "safe_acceleration_mps2"] = np.minimum(safe.loc[orius_like], -2.0)
    blackout.loc[orius_like, "fallback_used"] = True
    blackout.loc[orius_like, "intervened"] = True
    frame = frame.copy()
    frame["stress_source"] = "locked_runtime_trace"
    return pd.concat([frame, blackout], ignore_index=True)


def _simulate_closed_loop(
    frame: pd.DataFrame, *, dt_sec: float, min_gap_floor_m: float, headway_sec: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["controller", "scenario_id"]
    for (_, _), group in frame.groupby(group_cols, sort=False):
        group = group.sort_values("step_index", kind="stable")
        first = group.iloc[0]
        speed = max(0.0, _floatish(first.get("ego_speed_mps"), 0.0))
        gap = max(0.0, _floatish(first.get("min_gap_m"), 0.0))
        previous_accel = _controller_acceleration(first)
        for _, row in group.iterrows():
            observed_speed = max(0.0, _floatish(row.get("ego_speed_mps"), speed))
            observed_gap = max(0.0, _floatish(row.get("min_gap_m"), gap))
            target_gap = _floatish(row.get("target_relative_gap_1s"), observed_gap)
            lead_speed_estimate = max(0.0, observed_speed + (target_gap - observed_gap))
            accel = float(np.clip(_controller_acceleration(row), -4.5, 2.5))
            next_speed = max(0.0, speed + accel * dt_sec)
            next_gap = gap + (lead_speed_estimate - speed) * dt_sec - 0.5 * accel * dt_sec * dt_sec
            next_gap = float(np.clip(next_gap, -25.0, 250.0))
            safe_gap = max(min_gap_floor_m, headway_sec * max(next_speed, 0.0))
            closed_loop_margin = next_gap - safe_gap
            jerk = abs(accel - previous_accel) / max(dt_sec, 1e-9)
            progress = max(0.0, speed * dt_sec + 0.5 * accel * dt_sec * dt_sec)
            fallback = _boolish(row.get("fallback_used"))
            intervened = _boolish(row.get("intervened")) or fallback
            useful_work = progress * (1.0 - 0.35 * float(fallback) - 0.10 * float(intervened))
            useful_work = max(0.0, useful_work)
            controller = str(row.get("controller", ""))
            trace_id = str(row.get("trace_id", ""))
            if not trace_id:
                trace_id = f"{row.get('scenario_id')}-{row.get('step_index')}-{controller}"
            violation = bool(next_gap <= 0.0 or closed_loop_margin < 0.0)
            rows.append(
                {
                    "trace_id": trace_id,
                    "scenario_id": row.get("scenario_id"),
                    "step_index": int(_floatish(row.get("step_index"), 0)),
                    "fault_family": str(row.get("fault_family", "observed_runtime")),
                    "controller": controller,
                    "closed_loop_state_feedback": True,
                    "closed_loop_gap_m": next_gap,
                    "closed_loop_speed_mps": next_speed,
                    "closed_loop_acceleration_mps2": accel,
                    "closed_loop_margin_m": closed_loop_margin,
                    "closed_loop_true_constraint_violated": violation,
                    "near_miss": bool(0.0 <= closed_loop_margin < 2.0),
                    "collision_proxy": bool(next_gap <= 0.0),
                    "comfort_jerk_mps3": jerk,
                    "progress_m": progress,
                    "useful_work": useful_work,
                    "intervened": intervened,
                    "fallback_used": fallback,
                    "certificate_valid": _boolish(row.get("certificate_valid")),
                    "reliability_w": _floatish(row.get("reliability_w"), 1.0),
                    "frontier_policy": str(row.get("frontier_policy", "baseline_comparator")),
                    "stress_source": str(row.get("stress_source", "locked_runtime_trace")),
                    "validation_surface": CLOSED_LOOP_SURFACE,
                }
            )
            speed = next_speed
            gap = max(0.0, next_gap)
            previous_accel = accel
    return pd.DataFrame(rows)


def _metric_rows(traces: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for controller, group in traces.groupby("controller", sort=False):
        rows.append(
            {
                "controller": str(controller),
                "frontier_policy": str(group["frontier_policy"].iloc[0])
                if "frontier_policy" in group
                else "baseline_comparator",
                "tsvr": float(group["closed_loop_true_constraint_violated"].astype(bool).mean()),
                "fallback_activation_rate": float(group["fallback_used"].astype(bool).mean()),
                "intervention_rate": float(group["intervened"].astype(bool).mean()),
                "useful_work_total": float(group["useful_work"].sum()),
                "useful_work_mean": float(group["useful_work"].mean()),
                "mean_abs_jerk": float(group["comfort_jerk_mps3"].mean()),
                "progress_total": float(group["progress_m"].sum()),
                "near_miss_rate": float(group["near_miss"].astype(bool).mean()),
                "collision_proxy_rate": float(group["collision_proxy"].astype(bool).mean()),
                "certificate_valid_rate": float(group["certificate_valid"].astype(bool).mean()),
                "n_steps": int(len(group)),
            }
        )
    return rows


def _row_by_controller(rows: list[dict[str, Any]], controller: str) -> dict[str, Any]:
    for row in rows:
        if row.get("controller") == controller:
            return row
    return {}


def _stress_rows(traces: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (fault_family, controller), group in traces.groupby(["fault_family", "controller"], sort=True):
        rows.append(
            {
                "stress_family": str(fault_family),
                "controller": str(controller),
                "tsvr": float(group["closed_loop_true_constraint_violated"].astype(bool).mean()),
                "fallback_activation_rate": float(group["fallback_used"].astype(bool).mean()),
                "intervention_rate": float(group["intervened"].astype(bool).mean()),
                "useful_work_total": float(group["useful_work"].sum()),
                "mean_abs_jerk": float(group["comfort_jerk_mps3"].mean()),
                "near_miss_rate": float(group["near_miss"].astype(bool).mean()),
                "collision_proxy_rate": float(group["collision_proxy"].astype(bool).mean()),
                "n_steps": int(len(group)),
                "stress_source": "|".join(sorted(group["stress_source"].astype(str).unique()))
                if "stress_source" in group
                else "locked_runtime_trace",
                "validation_surface": CLOSED_LOOP_SURFACE,
            }
        )
    return rows


def _ablation_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    orius = _row_by_controller(metric_rows, "orius")
    rows: list[dict[str, Any]] = []
    for spec in ABLATION_PROXIES:
        controller = spec["controller"]
        row = _row_by_controller(metric_rows, controller)
        if not row and spec.get("fallback_controller"):
            controller = str(spec["fallback_controller"])
            row = _row_by_controller(metric_rows, controller)
        if not row:
            continue
        rows.append(
            {
                "method": spec["method"],
                "controller_proxy": controller,
                "component_removed": spec["component_removed"],
                "tsvr": row.get("tsvr", 0.0),
                "fallback_activation_rate": row.get("fallback_activation_rate", 0.0),
                "intervention_rate": row.get("intervention_rate", 0.0),
                "useful_work_total": row.get("useful_work_total", 0.0),
                "near_miss_rate": row.get("near_miss_rate", 0.0),
                "delta_tsvr_vs_orius": float(row.get("tsvr", 0.0)) - float(orius.get("tsvr", 0.0)),
                "validation_surface": CLOSED_LOOP_SURFACE,
            }
        )
    return rows


def _summary_row(
    metric_rows: list[dict[str, Any]],
    stress_rows: list[dict[str, Any]],
    traces: pd.DataFrame,
    *,
    sampling_policy: str,
    source_rows_scanned: int,
) -> dict[str, Any]:
    baseline = _row_by_controller(metric_rows, "baseline")
    orius = _row_by_controller(metric_rows, "orius")
    always_brake = _row_by_controller(metric_rows, "always_brake")
    observed_stress = {str(row["stress_family"]) for row in stress_rows}
    pass_gate = bool(
        orius
        and baseline
        and float(orius["tsvr"]) <= float(baseline["tsvr"])
        and float(orius["useful_work_total"]) > float(always_brake.get("useful_work_total", -1.0))
        and REQUIRED_PLANNER_STRESS_FAMILIES.issubset(observed_stress)
    )
    return {
        "validation_surface": CLOSED_LOOP_SURFACE,
        "status": "bounded_closed_loop_planner_pass" if pass_gate else "bounded_closed_loop_planner_failed",
        "simulation_semantics": "ego_action_updates_future_state",
        "closed_loop_state_feedback": True,
        "source_dataset": "nuplan_multi_city",
        "evidence_level": "bounded_kinematic_closed_loop_planner",
        "sampling_policy": sampling_policy,
        "source_rows_scanned": int(source_rows_scanned),
        "controllers": "|".join(sorted(str(row["controller"]) for row in metric_rows)),
        "stress_families": "|".join(sorted(observed_stress)),
        "n_steps": int(len(traces)),
        "scenario_count": int(traces["scenario_id"].astype(str).nunique()) if "scenario_id" in traces else 0,
        "baseline_tsvr": float(baseline.get("tsvr", 0.0)),
        "orius_tsvr": float(orius.get("tsvr", 1.0)),
        "always_brake_tsvr": float(always_brake.get("tsvr", 0.0)),
        "orius_fallback_activation_rate": float(orius.get("fallback_activation_rate", 0.0)),
        "orius_intervention_rate": float(orius.get("intervention_rate", 0.0)),
        "orius_useful_work_total": float(orius.get("useful_work_total", 0.0)),
        "always_brake_useful_work_total": float(always_brake.get("useful_work_total", 0.0)),
        "orius_mean_abs_jerk": float(orius.get("mean_abs_jerk", 0.0)),
        "orius_progress_total": float(orius.get("progress_total", 0.0)),
        "orius_near_miss_rate": float(orius.get("near_miss_rate", 0.0)),
        "orius_collision_proxy_rate": float(orius.get("collision_proxy_rate", 0.0)),
        "pass": pass_gate,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def build_av_closed_loop_planner_artifacts(
    *,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
    out_dir: Path = DEFAULT_OUT,
    max_rows: int | None = None,
    rows_per_controller_fault_family: int | None = 5_000,
    dt_sec: float = 0.1,
    min_gap_floor_m: float = 8.0,
    headway_sec: float = 1.2,
) -> dict[str, Any]:
    runtime_dir = Path(runtime_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_traces = runtime_dir / "runtime_traces.csv"

    source = _load_trace_frame(
        runtime_traces,
        max_rows=max_rows,
        rows_per_controller_fault_family=rows_per_controller_fault_family,
    )
    sampling_policy = str(source.attrs.get("sampling_policy", "unknown"))
    source_rows_scanned = int(source.attrs.get("source_rows_scanned", len(source)))
    supported_controllers = {*SERIOUS_BASELINE_CONTROLLERS, "predictor_only_no_runtime"}
    source = source[source["controller"].astype(str).isin(supported_controllers)]
    if source.empty:
        raise ValueError("no supported AV controllers found for closed-loop planner evaluation")
    source = _with_orius_frontier_profiles(source)
    source = _with_controlled_blackout_stress(source)
    traces = _simulate_closed_loop(
        source,
        dt_sec=dt_sec,
        min_gap_floor_m=min_gap_floor_m,
        headway_sec=headway_sec,
    )
    metric_rows = _metric_rows(traces)
    stress_rows = _stress_rows(traces)
    ablation_rows = _ablation_rows(metric_rows)
    summary_row = _summary_row(
        metric_rows,
        stress_rows,
        traces,
        sampling_policy=sampling_policy,
        source_rows_scanned=source_rows_scanned,
    )

    summary_path = out_dir / "av_planner_closed_loop_summary.csv"
    traces_path = out_dir / "av_planner_closed_loop_traces.csv"
    frontier_path = out_dir / "av_utility_safety_frontier.csv"
    stress_path = out_dir / "av_closed_loop_stress_family_summary.csv"
    ablation_path = out_dir / "av_closed_loop_ablation_summary.csv"
    manifest_path = out_dir / "av_closed_loop_manifest.json"

    _write_csv(summary_path, [summary_row])
    traces.to_csv(traces_path, index=False)
    _write_csv(frontier_path, metric_rows)
    _write_csv(stress_path, stress_rows)
    _write_csv(ablation_path, ablation_rows)

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": summary_row["status"],
        "pass": bool(summary_row["pass"]),
        "validation_surface": CLOSED_LOOP_SURFACE,
        "simulation_semantics": "ego_action_updates_future_state",
        "claim_boundary": CLAIM_BOUNDARY,
        "runtime_source": _display_path(runtime_traces),
        "summary_csv": _display_path(summary_path),
        "traces_csv": _display_path(traces_path),
        "frontier_csv": _display_path(frontier_path),
        "stress_family_csv": _display_path(stress_path),
        "ablation_csv": _display_path(ablation_path),
        "n_steps": int(summary_row["n_steps"]),
        "scenario_count": int(summary_row["scenario_count"]),
        "sampling_policy": sampling_policy,
        "source_rows_scanned": source_rows_scanned,
        "controllers": summary_row["controllers"].split("|") if summary_row["controllers"] else [],
        "stress_families": summary_row["stress_families"].split("|")
        if summary_row["stress_families"]
        else [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _remove_appledouble_files(out_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--rows-per-controller-fault-family", type=int, default=5_000)
    parser.add_argument("--dt-sec", type=float, default=0.1)
    parser.add_argument("--min-gap-floor-m", type=float, default=8.0)
    parser.add_argument("--headway-sec", type=float, default=1.2)
    args = parser.parse_args()
    manifest = build_av_closed_loop_planner_artifacts(
        runtime_dir=args.runtime_dir,
        out_dir=args.out,
        max_rows=args.max_rows,
        rows_per_controller_fault_family=args.rows_per_controller_fault_family,
        dt_sec=args.dt_sec,
        min_gap_floor_m=args.min_gap_floor_m,
        headway_sec=args.headway_sec,
    )
    print(f"[av-closed-loop-planner] pass={manifest['pass']} summary={manifest['summary_csv']}")
    return 0 if manifest["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
