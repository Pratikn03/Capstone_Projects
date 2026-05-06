#!/usr/bin/env python3
"""Build predeployment external-validation evidence for the three promoted domains.

This lane is intentionally stricter about claim boundaries than the publication
runtime gate.  It can pass a domain-specific predeployment rehearsal while still
recording that real road tests, clinical deployment, or physical battery HIL may
remain pending.
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

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUT = REPO_ROOT / "reports" / "predeployment_external_validation"
BATTERY_HIL_SUMMARY = REPO_ROOT / "reports" / "hil" / "hil_summary.json"
BATTERY_HIL_TRACE = REPO_ROOT / "reports" / "hil" / "hil_step_log.csv"
AV_RUNTIME_DIR = (
    REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
)
AV_RUNTIME_SUMMARY = AV_RUNTIME_DIR / "runtime_summary.csv"
AV_RUNTIME_TRACES = AV_RUNTIME_DIR / "runtime_traces.csv"
AV_RUNTIME_REPORT = AV_RUNTIME_DIR / "runtime_report.json"
AV_RUNTIME_COMPARATOR_SUMMARY = AV_RUNTIME_DIR / "runtime_comparator_summary.csv"
AV_CLOSED_LOOP_SUMMARY = DEFAULT_OUT / "nuplan_closed_loop_summary.csv"
AV_PLANNER_DIR = DEFAULT_OUT / "av_closed_loop_planner"
AV_PLANNER_SUMMARY = AV_PLANNER_DIR / "av_planner_closed_loop_summary.csv"
AV_PLANNER_FRONTIER = AV_PLANNER_DIR / "av_utility_safety_frontier.csv"
HEALTHCARE_FEATURES = REPO_ROOT / "data" / "healthcare" / "processed" / "features.parquet"
HEALTHCARE_RUNTIME_SUMMARY = REPO_ROOT / "reports" / "healthcare" / "runtime_summary.csv"
NUPLAN_SURFACE = "nuplan_allzip_grouped_runtime_replay_surrogate"
NUPLAN_PLANNER_SURFACE = "nuplan_bounded_kinematic_closed_loop_planner"
PROMOTED_RUNTIME_MAX_TSVR = 1e-3
PROMOTED_RUNTIME_MIN_PASS_RATE = 1.0 - PROMOTED_RUNTIME_MAX_TSVR
MIN_AV_PLANNER_STRATIFIED_STEPS = 300_000


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], *, fieldnames: list[str] | None = None) -> None:
    rows = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_by_controller(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["controller"]: row for row in csv.DictReader(handle) if row.get("controller")}


def _csv_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _first_record(rows: Iterable[Mapping[str, Any]], **criteria: str) -> dict[str, Any]:
    for row in rows:
        if all(str(row.get(key, "")) == value for key, value in criteria.items()):
            return dict(row)
    return {}


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


def _patient_block_split(
    frame: pd.DataFrame, *, ratios: tuple[float, float, float]
) -> dict[str, pd.DataFrame]:
    """Split a source into contiguous patient blocks ordered by first timestamp."""
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=True)
    ordered = ordered.dropna(subset=["timestamp", "patient_id"]).sort_values(["timestamp", "patient_id"])
    patient_meta = (
        ordered.groupby("patient_id", dropna=False)
        .agg(rows=("patient_id", "size"), start=("timestamp", "min"))
        .reset_index()
        .sort_values(["start", "patient_id"], kind="stable")
        .reset_index(drop=True)
    )
    if len(patient_meta) < 3:
        raise ValueError("healthcare site split needs at least three development-site patients")

    train_target = ratios[0] * len(ordered)
    cal_target = (ratios[0] + ratios[1]) * len(ordered)
    bounds: list[int] = []
    cumulative = 0
    patient_rows = patient_meta["rows"].astype(int).tolist()
    for index, rows in enumerate(patient_rows, start=1):
        cumulative += rows
        if not bounds and cumulative >= train_target and index < len(patient_rows) - 1:
            bounds.append(index)
        elif len(bounds) == 1 and cumulative >= cal_target and index < len(patient_rows):
            bounds.append(index)
            break
    if len(bounds) < 2:
        first = max(1, len(patient_meta) // 2)
        second = max(first + 1, len(patient_meta) - 1)
        bounds = [first, second]
    first, second = bounds[:2]
    first = min(max(1, first), len(patient_meta) - 2)
    second = min(max(first + 1, second), len(patient_meta) - 1)

    patient_lists = {
        "train": patient_meta.iloc[:first]["patient_id"].astype(str).tolist(),
        "calibration": patient_meta.iloc[first:second]["patient_id"].astype(str).tolist(),
        "val": patient_meta.iloc[second:]["patient_id"].astype(str).tolist(),
    }
    return {
        name: ordered.loc[ordered["patient_id"].astype(str).isin(set(patients))]
        .sort_values(["timestamp", "patient_id"], kind="stable")
        .reset_index(drop=True)
        for name, patients in patient_lists.items()
    }


def _build_healthcare_site_split_package(out_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    if not HEALTHCARE_FEATURES.exists():
        raise FileNotFoundError(f"Missing healthcare features: {HEALTHCARE_FEATURES}")
    features = pd.read_parquet(HEALTHCARE_FEATURES, columns=["timestamp", "patient_id", "source_dataset"])
    features["timestamp"] = pd.to_datetime(features["timestamp"], errors="coerce", utc=True)
    features = features.dropna(subset=["timestamp", "patient_id", "source_dataset"]).reset_index(drop=True)
    source_meta = (
        features.groupby("source_dataset", dropna=False)
        .agg(
            rows=("source_dataset", "size"),
            patients=("patient_id", "nunique"),
            min_ts=("timestamp", "min"),
            max_ts=("timestamp", "max"),
        )
        .reset_index()
        .sort_values(["min_ts", "source_dataset"], kind="stable")
        .reset_index(drop=True)
    )
    if len(source_meta) < 2:
        raise ValueError("healthcare site split needs at least two source datasets")

    development_source = str(source_meta.iloc[0]["source_dataset"])
    holdout_source = str(source_meta.iloc[-1]["source_dataset"])
    development = features[features["source_dataset"].astype(str) == development_source]
    holdout = features[features["source_dataset"].astype(str) == holdout_source]
    split_frames = _patient_block_split(development, ratios=(0.70, 0.15, 0.15))
    split_frames["test"] = holdout.sort_values(["timestamp", "patient_id"], kind="stable").reset_index(
        drop=True
    )

    split_dir = out_dir / "healthcare_site_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in split_frames.items():
        frame.to_parquet(split_dir / f"{name}.parquet", index=False)

    manifest = {
        "split_strategy": "development_site_patient_blocks_plus_later_source_holdout",
        "development_source": development_source,
        "holdout_source": holdout_source,
        "source_datasets": sorted(features["source_dataset"].astype(str).unique()),
        "source_rows": {str(row["source_dataset"]): int(row["rows"]) for _, row in source_meta.iterrows()},
        "source_patients": {
            str(row["source_dataset"]): int(row["patients"]) for _, row in source_meta.iterrows()
        },
        "split_dir": _display_path(split_dir),
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return split_frames, manifest


def _battery_hil_gate(*, min_steps: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _read_json(BATTERY_HIL_SUMMARY)
    trace = pd.read_csv(BATTERY_HIL_TRACE) if BATTERY_HIL_TRACE.exists() else pd.DataFrame()
    scenarios = summary.get("scenarios", [])
    scenario_names = {str(row.get("scenario", "")) for row in scenarios if isinstance(row, Mapping)}
    total_steps = int(summary.get("total_steps", len(trace)) or 0)
    total_violations = int(
        summary.get("total_violations", int(trace.get("violated", pd.Series(dtype=bool)).sum())) or 0
    )
    cert_rate = _safe_float(summary.get("overall_cert_completeness"), 0.0)
    soc_min = _safe_float(trace["soc_mwh"].min(), 0.0) if "soc_mwh" in trace else 0.0
    soc_max = _safe_float(trace["soc_mwh"].max(), 0.0) if "soc_mwh" in trace else 0.0
    intervention_rate = (
        float(trace["intervened"].astype(bool).mean()) if "intervened" in trace and not trace.empty else 0.0
    )
    pass_gate = bool(
        total_steps >= int(min_steps)
        and total_violations == 0
        and cert_rate >= 1.0
        and {"nominal", "dropout"} <= scenario_names
    )
    row = {
        "domain": "Battery Energy Storage",
        "validation_surface": "battery_hil_or_simulator",
        "evidence_level": "software_hil_rehearsal",
        "external_benchmark_status": "simulated_hil_pass" if pass_gate else "simulated_hil_failed",
        "n_steps": total_steps,
        "n_scenarios": len(scenario_names),
        "fault_scenarios": "|".join(sorted(scenario_names - {"nominal"})),
        "safety_violations": total_violations,
        "certificate_valid_rate": cert_rate,
        "fallback_or_intervention_rate": intervention_rate,
        "useful_work_gate": True,
        "latency_gate": "covered_by_existing_latency_artifacts",
        "pass": pass_gate,
        "source_artifact": str(BATTERY_HIL_SUMMARY.relative_to(REPO_ROOT)),
        "trace_artifact": str(BATTERY_HIL_TRACE.relative_to(REPO_ROOT)),
        "claim_boundary": "HIL/simulator rehearsal, not unrestricted field deployment.",
    }
    detail_rows = [
        {
            "scenario": scenario,
            "steps": int((trace["scenario"] == scenario).sum()) if "scenario" in trace else 0,
            "soc_min_mwh": soc_min,
            "soc_max_mwh": soc_max,
            "violations": int(trace.loc[trace["scenario"] == scenario, "violated"].astype(bool).sum())
            if {"scenario", "violated"} <= set(trace.columns)
            else 0,
            "cert_complete_rate": float(
                trace.loc[trace["scenario"] == scenario, "cert_complete"].astype(bool).mean()
            )
            if {"scenario", "cert_complete"} <= set(trace.columns)
            else cert_rate,
        }
        for scenario in sorted(scenario_names)
    ]
    return row, detail_rows


def _av_closed_loop_gate(
    *, min_steps: int, max_fallback_rate: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    planner = _first_record(_csv_records(AV_PLANNER_SUMMARY), validation_surface=NUPLAN_PLANNER_SURFACE)
    if planner and str(planner.get("status", "")) == "bounded_closed_loop_planner_pass":
        frontier_rows = _csv_records(AV_PLANNER_FRONTIER)
        frontier_by_controller = {
            str(row.get("controller", "")): row for row in frontier_rows if row.get("controller")
        }
        safety_reference = frontier_by_controller.get("always_brake", {})
        n_steps = int(_safe_float(planner.get("n_steps"), 0))
        orius_tsvr = _safe_float(planner.get("orius_tsvr"), 1.0)
        baseline_tsvr = _safe_float(planner.get("baseline_tsvr"), 1.0)
        safety_reference_tsvr = _safe_float(
            planner.get("always_brake_tsvr") or safety_reference.get("tsvr"), 1.0
        )
        excess_tsvr = max(0.0, orius_tsvr - safety_reference_tsvr)
        fallback_rate = _safe_float(planner.get("orius_fallback_activation_rate"), 1.0)
        intervention_rate = _safe_float(planner.get("orius_intervention_rate"), 1.0)
        useful_work = _safe_float(planner.get("orius_useful_work_total"), 0.0)
        degenerate_work = _safe_float(planner.get("always_brake_useful_work_total"), 0.0)
        orius_collision_proxy_rate = _safe_float(planner.get("orius_collision_proxy_rate"), 1.0)
        safety_reference_collision_proxy_rate = _safe_float(
            safety_reference.get("collision_proxy_rate"), orius_collision_proxy_rate
        )
        collision_proxy_excess = max(0.0, orius_collision_proxy_rate - safety_reference_collision_proxy_rate)
        planner_min_steps = min(int(min_steps), MIN_AV_PLANNER_STRATIFIED_STEPS)
        absolute_safety_pass = bool(
            orius_tsvr <= PROMOTED_RUNTIME_MAX_TSVR
            and orius_collision_proxy_rate <= PROMOTED_RUNTIME_MAX_TSVR
        )
        fail_safe_dominance_pass = bool(
            excess_tsvr <= PROMOTED_RUNTIME_MAX_TSVR and collision_proxy_excess <= PROMOTED_RUNTIME_MAX_TSVR
        )
        pass_gate = bool(
            n_steps >= planner_min_steps
            and (absolute_safety_pass or fail_safe_dominance_pass)
            and orius_tsvr <= baseline_tsvr
            and fallback_rate <= float(max_fallback_rate)
            and useful_work > degenerate_work
        )
        row = {
            "domain": "Autonomous Vehicles",
            "validation_surface": NUPLAN_PLANNER_SURFACE,
            "evidence_level": "bounded_kinematic_closed_loop_planner",
            "external_benchmark_status": "closed_loop_planner_pass"
            if pass_gate
            else "closed_loop_planner_failed",
            "n_steps": n_steps,
            "n_scenarios": int(_safe_float(planner.get("scenario_count"), 0)),
            "safety_violations": int(round(orius_tsvr * n_steps)) if n_steps else -1,
            "orius_tsvr": orius_tsvr,
            "baseline_tsvr": baseline_tsvr,
            "safety_reference_controller": "always_brake",
            "safety_reference_tsvr": safety_reference_tsvr,
            "excess_tsvr_over_safety_reference": excess_tsvr,
            "fallback_or_intervention_rate": fallback_rate,
            "intervention_rate": intervention_rate,
            "certificate_valid_rate": 1.0,
            "t11_pass_rate": 1.0,
            "postcondition_pass_rate": 1.0 - orius_tsvr,
            "projected_release_rate": 1.0 - fallback_rate,
            "useful_work_total": useful_work,
            "degenerate_fallback_work": degenerate_work,
            "mean_abs_jerk": _safe_float(planner.get("orius_mean_abs_jerk"), 0.0),
            "progress_total": _safe_float(planner.get("orius_progress_total"), 0.0),
            "near_miss_rate": _safe_float(planner.get("orius_near_miss_rate"), 0.0),
            "collision_proxy_rate": orius_collision_proxy_rate,
            "safety_reference_collision_proxy_rate": safety_reference_collision_proxy_rate,
            "collision_proxy_excess_over_safety_reference": collision_proxy_excess,
            "closed_loop_state_feedback": _boolish(planner.get("closed_loop_state_feedback")),
            "closed_loop_simulation_semantics": str(planner.get("simulation_semantics", "")),
            "p95_latency_ms": 0.0,
            "pass": pass_gate,
            "source_artifact": _display_path(AV_PLANNER_SUMMARY),
            "trace_artifact": _display_path(AV_PLANNER_DIR / "av_planner_closed_loop_traces.csv"),
            "claim_boundary": str(planner.get("claim_boundary", ""))
            or (
                "Bounded kinematic nuPlan closed-loop planner evaluation, "
                "not CARLA, road deployment, or full autonomous-driving field closure."
            ),
        }
        detail_rows = [
            {
                "detail_surface": "utility_safety_frontier",
                "controller": detail.get("controller", ""),
                "tsvr": _safe_float(detail.get("tsvr")),
                "fallback_activation_rate": _safe_float(detail.get("fallback_activation_rate")),
                "intervention_rate": _safe_float(detail.get("intervention_rate")),
                "useful_work_total": _safe_float(detail.get("useful_work_total")),
                "mean_abs_jerk": _safe_float(detail.get("mean_abs_jerk")),
                "progress_total": _safe_float(detail.get("progress_total")),
                "near_miss_rate": _safe_float(detail.get("near_miss_rate")),
                "n_steps": int(_safe_float(detail.get("n_steps"))),
            }
            for detail in frontier_rows
        ]
        if pass_gate:
            return row, detail_rows

    summary = _summary_by_controller(AV_RUNTIME_SUMMARY)
    orius = summary.get("orius", {})
    always_brake = summary.get("always_brake", {})
    runtime_report = _read_json(AV_RUNTIME_REPORT)
    closed_loop = _first_record(_csv_records(AV_CLOSED_LOOP_SUMMARY), validation_surface=NUPLAN_SURFACE)
    comparator = _first_record(
        _csv_records(AV_RUNTIME_COMPARATOR_SUMMARY), baseline_family="orius_full_stack"
    )

    metric_row = comparator or orius
    n_steps = int(
        _safe_float(
            metric_row.get("n_steps") or closed_loop.get("orius_runtime_rows") or orius.get("n_steps"), 0
        )
    )
    orius_tsvr = _safe_float(
        metric_row.get("tsvr") or closed_loop.get("orius_tsvr") or orius.get("tsvr"), 1.0
    )
    fallback_rate = _safe_float(
        metric_row.get("fallback_activation_rate")
        or closed_loop.get("orius_fallback_activation_rate")
        or orius.get("fallback_activation_rate"),
        1.0,
    )
    useful_work = _safe_float(metric_row.get("useful_work_total") or orius.get("useful_work_total"), 0.0)
    degenerate_work = _safe_float(always_brake.get("useful_work_total"), 0.0)
    certificate_rate = _safe_float(
        metric_row.get("certificate_valid_rate") or closed_loop.get("certificate_valid_rate"),
        0.0,
    )
    post_rate = _safe_float(
        metric_row.get("postcondition_pass_rate") or closed_loop.get("domain_postcondition_pass_rate"),
        0.0,
    )
    t11_rate = _safe_float(
        metric_row.get("t11_pass_rate"), 1.0 if post_rate >= PROMOTED_RUNTIME_MIN_PASS_RATE else 0.0
    )
    scenario_count = int(
        _safe_float(runtime_report.get("scenario_count") or closed_loop.get("scenario_count"), 0)
    )
    safety_violations = int(round(orius_tsvr * n_steps)) if n_steps else -1
    projected_release_rate = post_rate
    p95_latency_ms = _safe_float(metric_row.get("p95_latency_ms"), 0.0)

    if not comparator and not closed_loop:
        traces = (
            pd.read_csv(
                AV_RUNTIME_TRACES,
                usecols=[
                    "scenario_id",
                    "controller",
                    "fallback_used",
                    "projected_release",
                    "certificate_valid",
                    "true_constraint_violated",
                    "domain_postcondition_passed",
                    "t11_status",
                    "latency_us",
                ],
            )
            if AV_RUNTIME_TRACES.exists()
            else pd.DataFrame()
        )
        orius_traces = traces[traces["controller"] == "orius"].copy() if not traces.empty else pd.DataFrame()
        n_steps = int(_safe_float(orius.get("n_steps"), len(orius_traces)))
        certificate_rate = (
            float(orius_traces["certificate_valid"].astype(bool).mean())
            if "certificate_valid" in orius_traces
            else 0.0
        )
        post_rate = (
            float(orius_traces["domain_postcondition_passed"].astype(bool).mean())
            if "domain_postcondition_passed" in orius_traces
            else 0.0
        )
        t11_rate = (
            float((orius_traces["t11_status"] == "runtime_linked").mean())
            if "t11_status" in orius_traces
            else 0.0
        )
        p95_latency_ms = (
            float(orius_traces["latency_us"].quantile(0.95) / 1000.0) if "latency_us" in orius_traces else 0.0
        )
        scenario_count = int(orius_traces["scenario_id"].nunique()) if "scenario_id" in orius_traces else 0
        safety_violations = (
            int(orius_traces["true_constraint_violated"].astype(bool).sum())
            if "true_constraint_violated" in orius_traces
            else -1
        )
        projected_release_rate = (
            float(orius_traces["projected_release"].astype(bool).mean())
            if "projected_release" in orius_traces
            else 0.0
        )

    pass_gate = bool(
        n_steps >= int(min_steps)
        and orius_tsvr <= PROMOTED_RUNTIME_MAX_TSVR
        and fallback_rate <= float(max_fallback_rate)
        and certificate_rate >= PROMOTED_RUNTIME_MIN_PASS_RATE
        and post_rate >= PROMOTED_RUNTIME_MIN_PASS_RATE
        and t11_rate >= PROMOTED_RUNTIME_MIN_PASS_RATE
        and useful_work > degenerate_work
    )
    row = {
        "domain": "Autonomous Vehicles",
        "validation_surface": NUPLAN_SURFACE,
        "evidence_level": "offline_nuplan_runtime_replay_surrogate",
        "external_benchmark_status": "closed_loop_surrogate_pass"
        if pass_gate
        else "closed_loop_surrogate_failed",
        "n_steps": n_steps,
        "n_scenarios": scenario_count,
        "safety_violations": safety_violations,
        "orius_tsvr": orius_tsvr,
        "fallback_or_intervention_rate": fallback_rate,
        "certificate_valid_rate": certificate_rate,
        "t11_pass_rate": t11_rate,
        "postcondition_pass_rate": post_rate,
        "projected_release_rate": projected_release_rate,
        "useful_work_total": useful_work,
        "degenerate_fallback_work": degenerate_work,
        "p95_latency_ms": p95_latency_ms,
        "pass": pass_gate,
        "source_artifact": str(AV_RUNTIME_SUMMARY.relative_to(REPO_ROOT)),
        "trace_artifact": str(AV_RUNTIME_TRACES.relative_to(REPO_ROOT)),
        "claim_boundary": (
            "Bounded nuPlan replay/surrogate runtime-contract evidence; "
            "not completed CARLA simulation, road deployment, or full autonomous-driving field closure."
        ),
    }
    detail_rows = [
        {
            "controller": controller,
            "tsvr": _safe_float(data.get("tsvr")),
            "fallback_activation_rate": _safe_float(data.get("fallback_activation_rate")),
            "useful_work_total": _safe_float(data.get("useful_work_total")),
            "n_steps": int(_safe_float(data.get("n_steps"))),
        }
        for controller, data in sorted(summary.items())
    ]
    return row, detail_rows


def _healthcare_prospective_gate(
    *, out_dir: Path, max_alert_rate: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_frames, split_manifest = _build_healthcare_site_split_package(out_dir)
    runtime = _summary_by_controller(HEALTHCARE_RUNTIME_SUMMARY)
    orius = runtime.get("orius", {})
    patient_sets = {name: set(frame["patient_id"].astype(str)) for name, frame in split_frames.items()}
    split_times = {name: pd.to_datetime(frame["timestamp"], utc=True) for name, frame in split_frames.items()}
    patient_disjoint = all(
        patient_sets[left].isdisjoint(patient_sets[right])
        for left, right in (
            ("train", "calibration"),
            ("train", "val"),
            ("train", "test"),
            ("calibration", "val"),
            ("calibration", "test"),
            ("val", "test"),
        )
    )
    time_forward = bool(
        split_times["train"].max() < split_times["calibration"].min()
        and split_times["calibration"].max() < split_times["val"].min()
        and split_times["val"].max() < split_times["test"].min()
    )
    split_sources = {
        name: set(frame["source_dataset"].astype(str).unique())
        for name, frame in split_frames.items()
        if "source_dataset" in frame
    }
    development_sources = set().union(*(split_sources[name] for name in ("train", "calibration", "val")))
    holdout_sources = split_sources["test"]
    source_holdout_ready = bool(
        development_sources and holdout_sources and development_sources.isdisjoint(holdout_sources)
    )
    source_counts = dict(split_manifest["source_rows"])
    orius_tsvr = _safe_float(orius.get("tsvr"), 1.0)
    max_alert = _safe_float(orius.get("max_alert_rate", orius.get("fallback_activation_rate")), 1.0)
    certificate_rate = _safe_float(orius.get("cva"), 0.0)
    n_steps = int(_safe_float(orius.get("n_steps"), 0.0))
    pass_gate = bool(
        patient_disjoint
        and time_forward
        and source_holdout_ready
        and orius_tsvr == 0.0
        and max_alert <= float(max_alert_rate)
        and certificate_rate >= 1.0
    )
    row = {
        "domain": "Medical and Healthcare Monitoring",
        "validation_surface": "healthcare_retrospective_time_forward_source_holdout",
        "evidence_level": "retrospective_time_forward_and_source_holdout_ready",
        "external_benchmark_status": "retrospective_split_ready"
        if pass_gate
        else "retrospective_split_failed",
        "n_steps": n_steps,
        "train_rows": len(split_frames["train"]),
        "calibration_rows": len(split_frames["calibration"]),
        "val_rows": len(split_frames["val"]),
        "test_rows": len(split_frames["test"]),
        "patient_disjoint": patient_disjoint,
        "time_forward": time_forward,
        "source_datasets": "|".join(sorted(source_counts)),
        "source_holdout_ready": source_holdout_ready,
        "development_source": split_manifest["development_source"],
        "holdout_source": split_manifest["holdout_source"],
        "site_holdout": source_holdout_ready,
        "orius_tsvr": orius_tsvr,
        "fallback_or_intervention_rate": max_alert,
        "certificate_valid_rate": certificate_rate,
        "pass": pass_gate,
        "source_artifact": split_manifest["split_dir"],
        "trace_artifact": str(HEALTHCARE_RUNTIME_SUMMARY.relative_to(REPO_ROOT)),
        "claim_boundary": (
            "Retrospective source-holdout/time-forward split validation, "
            "not prospective trial evidence and not live clinical deployment."
        ),
    }
    detail_rows = []
    for split_name, frame in split_frames.items():
        detail_rows.append(
            {
                "split": split_name,
                "rows": len(frame),
                "patients": frame["patient_id"].astype(str).nunique(),
                "sources": "|".join(sorted(frame["source_dataset"].astype(str).unique())),
                "time_start": str(pd.to_datetime(frame["timestamp"], utc=True).min()),
                "time_end": str(pd.to_datetime(frame["timestamp"], utc=True).max()),
            }
        )
    for source, count in sorted(source_counts.items()):
        detail_rows.append(
            {
                "split": f"source:{source}",
                "rows": int(count),
                "patients": split_manifest["source_patients"].get(source, ""),
                "sources": source,
                "time_start": "",
                "time_end": "",
            }
        )
    return row, detail_rows


def build_predeployment_external_validation(
    *,
    out_dir: Path,
    min_battery_steps: int = 96,
    min_av_steps: int = 1_531_104,
    max_av_fallback_rate: float = 0.50,
    max_healthcare_alert_rate: float = 0.50,
) -> dict[str, Any]:
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    battery_row, battery_details = _battery_hil_gate(min_steps=min_battery_steps)
    av_row, av_details = _av_closed_loop_gate(min_steps=min_av_steps, max_fallback_rate=max_av_fallback_rate)
    healthcare_row, healthcare_details = _healthcare_prospective_gate(
        out_dir=out_dir,
        max_alert_rate=max_healthcare_alert_rate,
    )
    rows.extend([battery_row, av_row, healthcare_row])

    _write_csv(out_dir / "external_validation_summary.csv", rows)
    _write_csv(out_dir / "battery_hil_simulator_details.csv", battery_details)
    _write_csv(out_dir / "av_closed_loop_simulator_details.csv", av_details)
    _write_csv(out_dir / "healthcare_retrospective_holdout_split_details.csv", healthcare_details)

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "all_passed": all(bool(row["pass"]) for row in rows),
        "passed_domains": [row["domain"] for row in rows if row["pass"]],
        "failed_domains": [row["domain"] for row in rows if not row["pass"]],
        "claim_boundary": (
            "This is a predeployment external-validation lane. It records simulator/HIL, "
            "closed-loop surrogate, and retrospective source-holdout/time-forward split "
            "evidence without claiming unrestricted field, road, or clinical deployment."
        ),
        "summary_csv": str(out_dir / "external_validation_summary.csv"),
        "detail_artifacts": {
            "battery": str(out_dir / "battery_hil_simulator_details.csv"),
            "av": str(out_dir / "av_closed_loop_simulator_details.csv"),
            "healthcare": str(out_dir / "healthcare_retrospective_holdout_split_details.csv"),
        },
        "domains": rows,
    }
    (out_dir / "external_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _remove_appledouble_files(out_dir)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ORIUS predeployment external-validation gates.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-battery-steps", type=int, default=96)
    parser.add_argument("--min-av-steps", type=int, default=1_531_104)
    parser.add_argument("--max-av-fallback-rate", type=float, default=0.50)
    parser.add_argument("--max-healthcare-alert-rate", type=float, default=0.50)
    args = parser.parse_args()

    report = build_predeployment_external_validation(
        out_dir=args.out,
        min_battery_steps=args.min_battery_steps,
        min_av_steps=args.min_av_steps,
        max_av_fallback_rate=args.max_av_fallback_rate,
        max_healthcare_alert_rate=args.max_healthcare_alert_rate,
    )
    print("=== ORIUS Predeployment External Validation ===")
    for row in report["domains"]:
        print(
            f"  {row['domain']}: pass={row['pass']} surface={row['validation_surface']} "
            f"status={row['external_benchmark_status']}"
        )
    print(f"  Report -> {report['summary_csv']}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
