"""Waymo AV runtime adapter and dry-run evaluation."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from orius.dc3s.certificate import recompute_certificate_hash, store_certificate
from orius.dc3s.domain_adapter import DomainAdapter
from orius.dc3s.certificate import make_certificate
from orius.orius_bench.metrics_engine import StepRecord, compute_all_metrics
from orius.universal_framework.pipeline import run_universal_step
from orius.universal_framework.reliability_runtime import assess_domain_reliability

from .replay import FAULT_FAMILIES, WaymoReplayTrackAdapter, compute_state_safety_metrics
from .training import estimate_shift_score, load_model_bundle, predict_interval_from_bundle, widen_bounds


def _f(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _scenario_fault_family(scenario_id: str) -> str:
    digest = hashlib.sha256(str(scenario_id).encode("utf-8")).digest()
    return FAULT_FAMILIES[digest[0] % len(FAULT_FAMILIES)]


def deterministic_longitudinal_controller(state: dict[str, Any]) -> dict[str, float]:
    """Baseline longitudinal controller shared by baseline and ORIUS."""
    speed = _f(state.get("ego_speed_mps"), 0.0)
    speed_limit = _f(state.get("speed_limit_mps"), max(speed + 2.0, 10.0))
    gap = _f(state.get("min_gap_m"), 30.0)
    lead_rel_speed = _f(state.get("lead_rel_speed_mps"), 0.0)
    speed_error = speed_limit - speed
    gap_error = gap - 12.0
    accel = 0.15 * speed_error + 0.08 * gap_error - 0.25 * lead_rel_speed
    return {"acceleration_mps2": float(np.clip(accel, -4.0, 2.5))}


class WaymoAVDomainAdapter(DomainAdapter):
    """Waymo-native AV runtime adapter for replay-based dry runs."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self._cfg = dict(cfg or {})
        self.domain_id = "av_waymo"
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 0.1)
        self._accel_min = _f(self._cfg.get("accel_min_mps2"), -6.0)
        self._accel_max = _f(self._cfg.get("accel_max_mps2"), 3.0)
        self._min_headway_m = _f(self._cfg.get("min_headway_m"), 5.0)
        self._ttc_min_s = _f(self._cfg.get("ttc_min_s"), 2.0)

    def capability_profile(self) -> dict[str, Any]:
        return {
            "safety_surface_type": "waymo_longitudinal_headway_barrier",
            "repair_mode": "acceleration_projection",
            "fallback_mode": "full_brake",
            "supports_multi_agent_eval": True,
            "supports_certos_eval": True,
        }

    def ingest_telemetry(self, raw_packet: dict[str, Any]) -> dict[str, Any]:
        hold = dict(raw_packet)
        fields = (
            "ego_speed_mps",
            "speed_limit_mps",
            "min_gap_m",
            "lead_speed_mps",
            "lead_rel_speed_mps",
            "neighbor_count",
            "reliability_proxy",
            "shift_score",
            "ego_accel_mps2",
            "ego_jerk_mps3",
            "neighbor_instability",
            "pred_ego_speed_center_mps",
            "pred_ego_speed_lower_mps",
            "pred_ego_speed_upper_mps",
            "pred_relative_gap_center_m",
            "pred_relative_gap_lower_m",
            "pred_relative_gap_upper_m",
        )
        state = {}
        for field in fields:
            value = raw_packet.get(field)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                state[field] = raw_packet.get(f"_hold_{field}", 0.0)
            else:
                state[field] = value
        state["ts_utc"] = raw_packet.get("ts_utc", "")
        state["scenario_id"] = raw_packet.get("scenario_id")
        state["shard_id"] = raw_packet.get("shard_id")
        state["ego_track_id"] = raw_packet.get("ego_track_id")
        state["neighbor_ids_csv"] = raw_packet.get("neighbor_ids_csv", "")
        return state

    def compute_oqe(self, state: dict[str, Any], history: list[dict[str, Any]] | None = None) -> tuple[float, dict[str, Any]]:
        base_w, flags = assess_domain_reliability(
            domain_id=self.domain_id,
            state=state,
            history=history,
            feature_sources={
                "ego_speed_mps": "ego_speed_mps",
                "lead_gap_m": "min_gap_m",
                "lead_rel_speed_mps": "lead_rel_speed_mps",
                "neighbor_count": "neighbor_count",
            },
            expected_cadence_s=self._expected_cadence_s,
            reliability_cfg=self._cfg.get("reliability", {}),
            ftit_cfg=self._cfg.get("ftit", {}),
            runtime_surface="waymo_motion_replay_dry_run",
            closure_tier="defended_bounded_row",
        )
        valid_ratio = float(np.clip(_f(state.get("reliability_proxy"), 1.0), 0.05, 1.0))
        instability = float(np.clip(_f(state.get("neighbor_instability"), 0.0), 0.0, 1.0))
        accel_penalty = 0.5 if abs(_f(state.get("ego_accel_mps2"), 0.0)) > 8.0 else 1.0
        jerk_penalty = 0.6 if abs(_f(state.get("ego_jerk_mps3"), 0.0)) > 15.0 else 1.0
        w_t = float(np.clip(base_w * valid_ratio * (1.0 - 0.5 * instability) * accel_penalty * jerk_penalty, 0.05, 1.0))
        flags["actor_validity_ratio"] = valid_ratio
        flags["neighbor_instability"] = instability
        flags["impossible_acceleration"] = abs(_f(state.get("ego_accel_mps2"), 0.0)) > 8.0
        flags["impossible_jerk"] = abs(_f(state.get("ego_jerk_mps3"), 0.0)) > 15.0
        flags["w_t"] = w_t
        return w_t, flags

    def build_uncertainty_set(
        self,
        state: dict[str, Any],
        reliability_w: float,
        quantile: float,
        *,
        cfg: dict[str, Any],
        drift_flag: bool | None = None,
        prev_meta: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del drift_flag, prev_meta
        speed_center = _f(state.get("pred_ego_speed_center_mps"), _f(state.get("ego_speed_mps"), 0.0))
        speed_lower = _f(state.get("pred_ego_speed_lower_mps"), speed_center - quantile)
        speed_upper = _f(state.get("pred_ego_speed_upper_mps"), speed_center + quantile)
        gap_center = _f(state.get("pred_relative_gap_center_m"), _f(state.get("min_gap_m"), 30.0))
        gap_lower = _f(state.get("pred_relative_gap_lower_m"), gap_center - quantile)
        gap_upper = _f(state.get("pred_relative_gap_upper_m"), gap_center + quantile)
        shift_score = float(np.clip(_f(state.get("shift_score"), 0.0), 0.0, 1.0))
        factor = float(np.clip(1.0 + 0.7 * (1.0 - float(reliability_w)) + 0.5 * shift_score, 1.0, 3.0))
        speed_radius = 0.5 * (speed_upper - speed_lower) * factor
        gap_radius = 0.5 * (gap_upper - gap_lower) * factor
        uncertainty = {
            "speed_lower_mps": max(0.0, speed_center - speed_radius),
            "speed_upper_mps": speed_center + speed_radius,
            "gap_lower_m": gap_center - gap_radius,
            "gap_upper_m": gap_center + gap_radius,
            "speed_center_mps": speed_center,
            "gap_center_m": gap_center,
            "lead_speed_mps": _f(state.get("lead_speed_mps"), speed_center),
            "current_speed_mps": _f(state.get("ego_speed_mps"), speed_center),
            "speed_limit_mps": _f(state.get("speed_limit_mps"), max(speed_center + 2.0, 10.0)),
            "meta": {
                "inflation": factor,
                "widening_factor": factor,
                "shift_score": shift_score,
                "base_gap_width_m": gap_upper - gap_lower,
                "base_speed_width_mps": speed_upper - speed_lower,
                "drift_flag": bool(cfg.get("drift_flag", False)),
            },
        }
        return uncertainty, dict(uncertainty["meta"])

    def tighten_action_set(self, uncertainty: dict[str, Any], constraints: dict[str, Any], *, cfg: dict[str, Any]) -> dict[str, Any]:
        del cfg
        current_speed = _f(uncertainty.get("current_speed_mps"), 0.0)
        speed_limit = _f(constraints.get("speed_limit_mps"), uncertainty.get("speed_limit_mps", 25.0))
        lead_speed = _f(uncertainty.get("lead_speed_mps"), current_speed)
        gap_lower = _f(uncertainty.get("gap_lower_m"), 30.0)
        dt_s = _f(constraints.get("dt_s"), 0.1)
        min_headway_m = _f(constraints.get("min_headway_m"), self._min_headway_m)
        ttc_min_s = _f(constraints.get("ttc_min_s"), self._ttc_min_s)
        a_lower = _f(constraints.get("accel_min_mps2"), self._accel_min)
        a_upper = _f(constraints.get("accel_max_mps2"), self._accel_max)
        active: list[str] = []

        if current_speed + a_upper * dt_s > speed_limit:
            a_upper = min(a_upper, (speed_limit - current_speed) / max(dt_s, 1e-9))
            active.append("speed_limit")

        if gap_lower <= min_headway_m:
            a_lower = a_upper = _f(constraints.get("accel_min_mps2"), self._accel_min)
            active.append("headway_predictive_entry_barrier")
        else:
            max_next_speed = lead_speed + gap_lower / max(ttc_min_s + dt_s, 1e-9)
            ttc_upper = (max_next_speed - current_speed) / max(dt_s, 1e-9)
            if ttc_upper < a_upper:
                a_upper = ttc_upper
                active.append("ttc_clamp")

        if a_lower > a_upper:
            a_lower = a_upper = _f(constraints.get("accel_min_mps2"), self._accel_min)
            active.append("fallback_collapse")

        return {
            "acceleration_mps2_lower": float(a_lower),
            "acceleration_mps2_upper": float(a_upper),
            "fallback_action": {"acceleration_mps2": float(_f(constraints.get("accel_min_mps2"), self._accel_min))},
            "active_constraints": active,
            "projection_surface": "waymo_longitudinal_barrier",
            "viable": True,
            "entry_barrier_triggered": "headway_predictive_entry_barrier" in active,
            "widening_factor": float(uncertainty.get("meta", {}).get("widening_factor", 1.0)),
        }

    def repair_action(
        self,
        candidate_action: dict[str, Any],
        tightened_set: dict[str, Any],
        *,
        state: dict[str, Any],
        uncertainty: dict[str, Any],
        constraints: dict[str, Any],
        cfg: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        del state, uncertainty, constraints, cfg
        proposal = _f(candidate_action.get("acceleration_mps2"), 0.0)
        lower = _f(tightened_set.get("acceleration_mps2_lower"), self._accel_min)
        upper = _f(tightened_set.get("acceleration_mps2_upper"), self._accel_max)
        safe = float(np.clip(proposal, lower, upper))
        repaired = abs(safe - proposal) > 1e-9
        reason = None
        if repaired:
            if "headway_predictive_entry_barrier" in tightened_set.get("active_constraints", []):
                reason = "headway_predictive_entry_barrier"
            elif "ttc_clamp" in tightened_set.get("active_constraints", []):
                reason = "ttc_clamp"
            elif "speed_limit" in tightened_set.get("active_constraints", []):
                reason = "speed_limit_clamp"
            else:
                reason = "acceleration_clamp"
        return {"acceleration_mps2": safe}, {
            "repaired": repaired,
            "mode": "projection",
            "intervention_reason": reason,
            "widening_factor": float(tightened_set.get("widening_factor", 1.0)),
            "entry_barrier_triggered": bool(tightened_set.get("entry_barrier_triggered", False)),
        }

    def emit_certificate(
        self,
        *,
        command_id: str,
        device_id: str,
        zone_id: str,
        controller: str,
        proposed_action: Mapping[str, Any],
        safe_action: Mapping[str, Any],
        uncertainty: Mapping[str, Any],
        reliability: Mapping[str, Any],
        drift: Mapping[str, Any],
        cfg: Mapping[str, Any],
        prev_hash: str | None = None,
        dispatch_plan: Mapping[str, Any] | None = None,
        repair_meta: Mapping[str, Any] | None = None,
        guarantee_meta: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        cert = make_certificate(
            command_id=command_id,
            device_id=device_id,
            zone_id=zone_id,
            controller=controller,
            proposed_action=dict(proposed_action),
            safe_action=dict(safe_action),
            uncertainty=dict(uncertainty),
            reliability=dict(reliability),
            drift=dict(drift),
            model_hash=str(cfg.get("model_hash", "")),
            config_hash=str(cfg.get("config_hash", "")),
            prev_hash=prev_hash,
            dispatch_plan=dict(dispatch_plan or {}),
            intervened=bool(repair_meta.get("repaired")) if isinstance(repair_meta, Mapping) else None,
            intervention_reason=repair_meta.get("intervention_reason") if isinstance(repair_meta, Mapping) else None,
            reliability_w=_f(reliability.get("w_t", reliability.get("w")), 1.0),
            drift_flag=bool(drift.get("drift", False)),
            inflation=_f(uncertainty.get("meta", {}).get("widening_factor"), 1.0),
            runtime_surface="waymo_motion_replay_dry_run",
            closure_tier="defended_bounded_row",
            reliability_feature_basis=reliability.get("reliability_feature_basis"),
        )
        cert = dict(cert)
        cert["scenario_id"] = cfg.get("scenario_id")
        cert["shard_id"] = cfg.get("shard_id")
        cert["ego_track_id"] = cfg.get("ego_track_id")
        cert["neighbor_ids"] = list(cfg.get("neighbor_ids", []))
        cert["widening_factor"] = _f(uncertainty.get("meta", {}).get("widening_factor"), 1.0)
        cert["shift_score"] = _f(cfg.get("shift_score"), 0.0)
        cert["true_margin"] = cfg.get("true_margin")
        cert["observed_margin"] = cfg.get("observed_margin")
        cert["intervention_trace_id"] = cfg.get("intervention_trace_id", command_id)
        cert["certificate_hash"] = recompute_certificate_hash(cert)
        return cert


def load_runtime_bundles(models_dir: str | Path) -> dict[str, Any]:
    models_path = Path(models_dir)
    speed_bundle = load_model_bundle(models_path / "waymo_av_ego_speed_mps_1s_bundle.pkl")
    gap_bundle = load_model_bundle(models_path / "waymo_av_relative_gap_m_1s_bundle.pkl")
    return {"ego_speed_mps_1s": speed_bundle, "relative_gap_m_1s": gap_bundle}


def _build_runtime_lookup(step_features: pd.DataFrame) -> dict[tuple[str, int], pd.Series]:
    return {
        (str(row["scenario_id"]), int(row["step_index"])): row
        for _, row in step_features.iterrows()
    }


def _predict_runtime_telemetry(feature_row: pd.Series, bundles: dict[str, Any]) -> dict[str, Any]:
    feature_frame = pd.DataFrame([feature_row.to_dict()])
    speed_bundle = bundles["ego_speed_mps_1s"]
    gap_bundle = bundles["relative_gap_m_1s"]
    speed_center, speed_lower, speed_upper = predict_interval_from_bundle(speed_bundle, feature_frame)
    gap_center, gap_lower, gap_upper = predict_interval_from_bundle(gap_bundle, feature_frame)
    shift_score = float(max(estimate_shift_score(speed_bundle, feature_frame)[0], estimate_shift_score(gap_bundle, feature_frame)[0]))
    return {
        "pred_ego_speed_center_mps": float(speed_center[0]),
        "pred_ego_speed_lower_mps": float(speed_lower[0]),
        "pred_ego_speed_upper_mps": float(speed_upper[0]),
        "pred_relative_gap_center_m": float(gap_center[0]),
        "pred_relative_gap_lower_m": float(gap_lower[0]),
        "pred_relative_gap_upper_m": float(gap_upper[0]),
        "shift_score": shift_score,
    }


def _observed_margin_from_state(state: Mapping[str, Any]) -> float:
    gap = _f(state.get("min_gap_m"), 30.0)
    closing = _f(state.get("lead_rel_speed_mps"), 0.0)
    return float(gap - max(5.0, 2.0 * closing))


def _run_episode(
    *,
    track: WaymoReplayTrackAdapter,
    scenario_id: str,
    step_features: pd.DataFrame,
    bundles: dict[str, Any],
    fault_family: str,
    use_orius: bool,
    audit_db_path: Path,
    table_name: str,
) -> tuple[list[StepRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    adapter = WaymoAVDomainAdapter({"expected_cadence_s": 0.1})
    feature_lookup = _build_runtime_lookup(step_features)
    scenario_steps = sorted(step_features[step_features["scenario_id"] == scenario_id]["step_index"].astype(int).tolist())
    if not scenario_steps:
        return [], [], []
    track.load_scenario(scenario_id, start_step_index=min(scenario_steps))
    prev_cert_hash: str | None = None
    history: list[dict[str, Any]] = []
    records: list[StepRecord] = []
    trace_rows: list[dict[str, Any]] = []
    cert_rows: list[dict[str, Any]] = []

    previous_speed = None
    previous_accel = None
    for step_index in scenario_steps:
        true_state = dict(track.true_state())
        observed = dict(track.observe(true_state, {"kind": fault_family}))
        feature_row = feature_lookup[(scenario_id, step_index)]
        observed.update(_predict_runtime_telemetry(feature_row, bundles))
        observed["reliability_proxy"] = float(feature_row["reliability_proxy"])
        observed["scenario_id"] = scenario_id
        observed["shard_id"] = true_state.get("shard_id")
        observed["ego_track_id"] = true_state.get("ego_track_id")
        observed["neighbor_ids_csv"] = ",".join(
            str(true_state.get(f"neighbor_slot_{slot}_track_id"))
            for slot in range(int(true_state.get("neighbor_count", 0) or 0))
            if true_state.get(f"neighbor_slot_{slot}_track_id") is not None
        )
        current_speed = _f(observed.get("ego_speed_mps"), 0.0)
        accel = 0.0 if previous_speed is None else (current_speed - previous_speed) / 0.1
        jerk = 0.0 if previous_accel is None else (accel - previous_accel) / 0.1
        observed["ego_accel_mps2"] = accel
        observed["ego_jerk_mps3"] = jerk
        observed["neighbor_instability"] = float(max(0.0, abs(int(feature_row["neighbor_count"]) - int(true_state.get("neighbor_count", 0)))) / 8.0)
        observed["observed_margin"] = _observed_margin_from_state(observed)
        candidate_action = deterministic_longitudinal_controller(observed)
        latency_start = time.perf_counter_ns()
        certificate_valid = False
        intervened = False
        fallback_used = False
        certificate = None

        if use_orius:
            constraints = {
                "speed_limit_mps": _f(true_state.get("speed_limit_mps"), 25.0),
                "accel_min_mps2": -6.0,
                "accel_max_mps2": 3.0,
                "dt_s": 0.1,
                "min_headway_m": 5.0,
                "ttc_min_s": 2.0,
            }
            result = run_universal_step(
                domain_adapter=adapter,
                raw_telemetry=observed,
                history=history[-10:] if history else None,
                candidate_action=candidate_action,
                constraints=constraints,
                quantile=1.0,
                cfg={
                    "scenario_id": scenario_id,
                    "shard_id": true_state.get("shard_id"),
                    "ego_track_id": true_state.get("ego_track_id"),
                    "neighbor_ids": [
                        int(true_state.get(f"neighbor_slot_{slot}_track_id"))
                        for slot in range(int(true_state.get("neighbor_count", 0) or 0))
                        if true_state.get(f"neighbor_slot_{slot}_track_id") is not None
                    ],
                    "shift_score": observed["shift_score"],
                    "true_margin": true_state.get("true_margin"),
                    "observed_margin": observed["observed_margin"],
                    "intervention_trace_id": f"{scenario_id}-{step_index}",
                },
                prev_cert_hash=prev_cert_hash,
                device_id=f"ego-{true_state.get('ego_track_id')}",
                zone_id=str(true_state.get("shard_id")),
                controller="waymo_orius_dry_run",
            )
            safe_action = dict(result["safe_action"])
            certificate = dict(result["certificate"])
            prev_cert_hash = certificate.get("certificate_hash")
            store_certificate(certificate, duckdb_path=str(audit_db_path), table_name=table_name)
            certificate_valid = True
            intervened = bool(result["repair_meta"]["repaired"])
        else:
            safe_action = dict(candidate_action)
        latency_us = (time.perf_counter_ns() - latency_start) / 1_000.0
        history.append(dict(observed))
        next_true = dict(track.step(safe_action))

        true_metrics = compute_state_safety_metrics(true_state)
        observed_metrics = compute_state_safety_metrics(observed) if observed.get("min_gap_m") is not None and not (
            isinstance(observed.get("min_gap_m"), float) and math.isnan(observed.get("min_gap_m"))
        ) else {"true_constraint_violated": None}
        records.append(
            StepRecord(
                step=int(step_index),
                true_state=true_state,
                observed_state=observed,
                action=safe_action,
                true_constraint_violated=bool(true_metrics["true_constraint_violated"]),
                observed_constraint_satisfied=(
                    None if observed_metrics.get("true_constraint_violated") is None else not bool(observed_metrics["true_constraint_violated"])
                ),
                true_margin=float(true_metrics["true_margin"]),
                observed_margin=float(observed.get("observed_margin", np.nan)),
                intervened=intervened,
                fallback_used=fallback_used,
                certificate_valid=certificate_valid,
                certificate_predicted_valid=certificate_valid,
                useful_work=float(max(0.0, next_true.get("ego_x_m", 0.0) - true_state.get("ego_x_m", 0.0))),
                audit_fields_present=4 if certificate is not None else 0,
                audit_fields_required=4,
                latency_us=float(latency_us),
                domain_metrics={
                    "min_gap_m": float(true_metrics["min_gap_m"]),
                    "ttc_s": float(true_metrics["ttc_s"]),
                    "neighbor_count": float(true_state.get("neighbor_count", 0)),
                },
            )
        )
        trace_rows.append(
            {
                "scenario_id": scenario_id,
                "step_index": int(step_index),
                "fault_family": fault_family,
                "controller": "orius" if use_orius else "baseline",
                "candidate_acceleration_mps2": float(candidate_action["acceleration_mps2"]),
                "safe_acceleration_mps2": float(safe_action["acceleration_mps2"]),
                "intervened": bool(intervened),
                "certificate_valid": bool(certificate_valid),
                "true_margin": float(true_metrics["true_margin"]),
                "observed_margin": float(observed.get("observed_margin", np.nan)),
                "min_gap_m": float(true_metrics["min_gap_m"]),
                "ttc_s": float(true_metrics["ttc_s"]),
                "latency_us": float(latency_us),
                "target_ego_speed_1s": float(feature_row["target_ego_speed_mps__1s"]),
                "target_relative_gap_1s": float(feature_row["target_relative_gap_m__1s"]),
                "pred_ego_speed_lower_mps": float(observed["pred_ego_speed_lower_mps"]),
                "pred_ego_speed_upper_mps": float(observed["pred_ego_speed_upper_mps"]),
                "pred_relative_gap_lower_m": float(observed["pred_relative_gap_lower_m"]),
                "pred_relative_gap_upper_m": float(observed["pred_relative_gap_upper_m"]),
                "shift_score": float(observed["shift_score"]),
            }
        )
        if certificate is not None:
            cert_rows.append(certificate)
        previous_speed = current_speed
        previous_accel = accel
    return records, trace_rows, cert_rows


def run_runtime_dry_run(
    *,
    replay_windows_path: str | Path,
    step_features_path: str | Path,
    models_dir: str | Path,
    out_dir: str | Path,
    max_scenarios: int | None = None,
) -> dict[str, Any]:
    replay_path = Path(replay_windows_path)
    step_features = pd.read_parquet(step_features_path)
    test_scenarios = sorted(step_features[step_features["split"] == "test"]["scenario_id"].unique().tolist())
    if max_scenarios is not None:
        test_scenarios = test_scenarios[: int(max_scenarios)]
    bundles = load_runtime_bundles(models_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    audit_db_path = out_path / "dc3s_av_waymo_dryrun.duckdb"
    table_name = "dispatch_certificates"

    all_records: dict[str, list[StepRecord]] = {"baseline": [], "orius": []}
    all_traces: list[dict[str, Any]] = []
    all_certs: list[dict[str, Any]] = []
    for scenario_id in test_scenarios:
        fault_family = _scenario_fault_family(str(scenario_id))
        baseline_track = WaymoReplayTrackAdapter(replay_path)
        orius_track = WaymoReplayTrackAdapter(replay_path)
        baseline_records, baseline_traces, _ = _run_episode(
            track=baseline_track,
            scenario_id=str(scenario_id),
            step_features=step_features,
            bundles=bundles,
            fault_family=fault_family,
            use_orius=False,
            audit_db_path=audit_db_path,
            table_name=table_name,
        )
        orius_records, orius_traces, orius_certs = _run_episode(
            track=orius_track,
            scenario_id=str(scenario_id),
            step_features=step_features,
            bundles=bundles,
            fault_family=fault_family,
            use_orius=True,
            audit_db_path=audit_db_path,
            table_name=table_name,
        )
        all_records["baseline"].extend(baseline_records)
        all_records["orius"].extend(orius_records)
        all_traces.extend(baseline_traces)
        all_traces.extend(orius_traces)
        all_certs.extend(orius_certs)

    trace_df = pd.DataFrame(all_traces)
    trace_path = out_path / "runtime_traces.csv"
    trace_df.to_csv(trace_path, index=False)

    summary_rows = []
    for controller_name, records in all_records.items():
        metrics = compute_all_metrics(records)
        summary_rows.append(
            {
                "controller": controller_name,
                "tsvr": metrics.tsvr,
                "oasg": metrics.oasg,
                "cva": metrics.cva,
                "gdq": metrics.gdq,
                "intervention_rate": metrics.intervention_rate,
                "audit_completeness": metrics.audit_completeness,
                "recovery_latency": metrics.recovery_latency,
                "n_steps": metrics.n_steps,
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_path / "runtime_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    coverage_rows = []
    if not trace_df.empty:
        for fault_family, group in trace_df.groupby("fault_family"):
            for target_name, lower_col, upper_col, true_col in (
                ("ego_speed_mps", "pred_ego_speed_lower_mps", "pred_ego_speed_upper_mps", "target_ego_speed_1s"),
                ("relative_gap_m", "pred_relative_gap_lower_m", "pred_relative_gap_upper_m", "target_relative_gap_1s"),
            ):
                y_true = group[true_col].to_numpy(dtype=float)
                lower = group[lower_col].to_numpy(dtype=float)
                upper = group[upper_col].to_numpy(dtype=float)
                coverage_rows.append(
                    {
                        "fault_family": fault_family,
                        "target": target_name,
                        "coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
                        "mean_width": float(np.mean(upper - lower)),
                    }
                )
    coverage_path = out_path / "fault_family_coverage.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False)

    report = {
        "runtime_summary_csv": str(summary_path),
        "runtime_traces_csv": str(trace_path),
        "fault_family_coverage_csv": str(coverage_path),
        "audit_db_path": str(audit_db_path),
        "certificate_count": int(len(all_certs)),
        "scenario_count": int(len(test_scenarios)),
    }
    (out_path / "runtime_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
