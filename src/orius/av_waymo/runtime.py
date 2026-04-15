"""Waymo AV runtime adapter and dry-run evaluation."""
from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from orius.dc3s.certificate import recompute_certificate_hash, store_certificates_batch
from orius.dc3s.domain_adapter import DomainAdapter
from orius.dc3s.certificate import make_certificate
from orius.forecasting.uncertainty.shift_aware import (
    ShiftAwareConfig,
    ShiftAwareRuntimeEngine,
    apply_interval_policy,
    compute_validity_score,
    write_shift_aware_artifacts,
)
from orius.forecasting.uncertainty.shift_aware.state import GroupCoverageStats
from orius.orius_bench.metrics_engine import StepRecord, compute_all_metrics
from orius.universal_framework.pipeline import run_universal_step
from orius.universal_framework.reliability_runtime import assess_domain_reliability

from .replay import FAULT_FAMILIES, WaymoReplayTrackAdapter, compute_state_safety_metrics
from .training import default_shift_aware_config, estimate_shift_score, load_model_bundle, predict_interval_from_bundle, widen_bounds


def _f(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(numeric):
        return float(default)
    return numeric


def _finite_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _scenario_fault_family(scenario_id: str) -> str:
    digest = hashlib.sha256(str(scenario_id).encode("utf-8")).digest()
    return FAULT_FAMILIES[digest[0] % len(FAULT_FAMILIES)]


def _validity_rank(status: str) -> int:
    order = {"nominal": 0, "watch": 1, "degraded": 2, "invalid": 3}
    return int(order.get(str(status), 0))


def _remap_path_prefix(value: Any, source_prefix: Path, target_prefix: Path) -> Any:
    if isinstance(value, dict):
        return {key: _remap_path_prefix(nested, source_prefix, target_prefix) for key, nested in value.items()}
    if isinstance(value, list):
        return [_remap_path_prefix(item, source_prefix, target_prefix) for item in value]
    if not isinstance(value, str):
        return value
    try:
        source_text = str(source_prefix.resolve())
        candidate_text = str(Path(value).resolve())
    except OSError:
        source_text = str(source_prefix)
        candidate_text = str(value)
    if candidate_text == source_text or candidate_text.startswith(source_text + "/"):
        suffix = candidate_text[len(source_text):].lstrip("/")
        return str(target_prefix / suffix) if suffix else str(target_prefix)
    return value


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


def _compact_step_state(state: Mapping[str, Any], *, include_runtime: bool = False) -> dict[str, float | str]:
    compact: dict[str, float | str] = {
        "scenario_id": str(state.get("scenario_id", "")),
        "shard_id": str(state.get("shard_id", "")),
        "ego_track_id": str(state.get("ego_track_id", "")),
        "ego_speed_mps": _f(state.get("ego_speed_mps"), 0.0),
        "min_gap_m": _f(state.get("min_gap_m"), 0.0),
        "neighbor_count": _f(state.get("neighbor_count"), 0.0),
    }
    if include_runtime:
        compact["shift_score"] = _f(state.get("shift_score"), 0.0)
        compact["reliability_proxy"] = _f(state.get("reliability_proxy"), 1.0)
    return compact


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
        shift_cfg = self._cfg.get("shift_aware_uncertainty")
        if shift_cfg:
            self._shift_cfg = ShiftAwareConfig.from_mapping(dict(shift_cfg))
        else:
            self._shift_cfg = default_shift_aware_config()
        shift_dir = self._cfg.get("shift_output_dir")
        self._shift_engines: dict[str, ShiftAwareRuntimeEngine] = {}
        if self._shift_cfg.enabled:
            for target_name in ("ego_speed_mps", "relative_gap_m"):
                state_path = None
                if shift_dir:
                    state_path = str(Path(str(shift_dir)) / f"{target_name}_runtime_state.json")
                self._shift_engines[target_name] = ShiftAwareRuntimeEngine(cfg=self._shift_cfg, state_path=state_path)

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
            "target_ego_speed_mps_1s",
            "target_relative_gap_m_1s",
            "fault_family",
        )
        state = {}
        for field in fields:
            value = raw_packet.get(field)
            numeric = _finite_float_or_none(value) if isinstance(value, (float, int, np.floating, np.integer)) else value
            if value is None or (isinstance(value, (float, int, np.floating, np.integer)) and numeric is None):
                state[field] = raw_packet.get(f"_hold_{field}", 0.0)
            else:
                state[field] = value
        state["ts_utc"] = raw_packet.get("ts_utc", "")
        state["scenario_id"] = raw_packet.get("scenario_id")
        state["shard_id"] = raw_packet.get("shard_id")
        state["ego_track_id"] = raw_packet.get("ego_track_id")
        state["neighbor_ids_csv"] = raw_packet.get("neighbor_ids_csv", "")
        return state

    def _shift_aware_interval(
        self,
        *,
        target_name: str,
        center: float,
        lower: float,
        upper: float,
        reliability_w: float,
        shift_score: float,
        state: Mapping[str, Any],
        drift_flag: bool,
        update_runtime_state: bool,
    ) -> tuple[tuple[float, float], dict[str, Any]]:
        base_half_width = max(0.5 * max(upper - lower, 0.0), 1e-6)
        max_inflation = float(max(self._shift_cfg.max_inflation_multiplier, 1.0))
        planned_factor = float(np.clip(1.0 + 0.7 * (1.0 - float(reliability_w)) + 0.5 * shift_score, 1.0, max_inflation))
        widened_half_width = base_half_width * planned_factor
        if not self._shift_cfg.enabled:
            return (center - widened_half_width, center + widened_half_width), {
                "target_name": target_name,
                "planned_factor": planned_factor,
                "final_factor": planned_factor,
                "validity_score": 1.0,
                "validity_status": "nominal",
                "under_coverage_gap": 0.0,
                "coverage_group_key": "global",
                "shift_alert_flag": False,
                "adaptive_quantile": self._shift_cfg.alpha,
                "runtime_interval_policy": "planned_linear",
            }

        engine = self._shift_engines[target_name]
        target_key = "target_ego_speed_mps_1s" if target_name == "ego_speed_mps" else "target_relative_gap_m_1s"
        target_true_val = _finite_float_or_none(state.get(target_key))
        if target_true_val is None:
            abs_residual = 0.0
        else:
            abs_residual = abs(target_true_val - center)
        volatility_raw = abs(_f(state.get("ego_accel_mps2"), 0.0)) if target_name == "ego_speed_mps" else abs(_f(state.get("lead_rel_speed_mps"), 0.0))
        volatility = float(np.clip(volatility_raw / (8.0 if target_name == "ego_speed_mps" else 10.0), 0.0, 1.0))
        group_key = engine.state.tracker.build_group_key(
            reliability_score=float(reliability_w),
            volatility=volatility,
            fault_type=str(state.get("fault_family", "none")),
            ts=str(state.get("ts_utc", "")),
            custom_key=target_name,
            reliability_bins=int(self._shift_cfg.reliability_bins),
            volatility_bins=int(self._shift_cfg.volatility_bins),
        )
        engine.state.adaptive.mode = self._shift_cfg.aci_mode
        engine.state.adaptive.base_alpha = float(self._shift_cfg.alpha)
        engine.state.adaptive.learning_rate = float(self._shift_cfg.adaptation_step)
        engine.state.adaptive.alpha_min = float(self._shift_cfg.alpha_min)
        engine.state.adaptive.alpha_max = float(self._shift_cfg.alpha_max)
        stats = engine.state.tracker.get_group_stats(group_key)
        adaptive_quantile = float(engine.state.adaptive.effective_alpha)
        if stats is None:
            stats = GroupCoverageStats(group_key=group_key, target_coverage=self._shift_cfg.coverage_target)
        validity = compute_validity_score(
            reliability_score=float(reliability_w),
            drift_magnitude=float(shift_score if drift_flag else 0.5 * shift_score),
            normalized_residual=float(min(abs_residual / max(widened_half_width, 1e-6), 1.0)),
            subgroup_under_coverage_gap=float(stats.under_coverage_gap),
            adaptation_instability=float(abs(adaptive_quantile - engine.state.adaptive.base_alpha)),
            cfg=self._shift_cfg,
        )
        decision = apply_interval_policy(
            y_hat=float(center),
            base_half_width=float(widened_half_width),
            reliability_score=float(reliability_w),
            drift_signal=float(shift_score),
            adaptive_quantile=adaptive_quantile,
            subgroup_under_coverage_gap=float(stats.under_coverage_gap),
            validity=validity,
            cfg=self._shift_cfg,
            coverage_group_key=group_key,
        )
        final_half_width = min(base_half_width * max_inflation, float(decision.adjusted_half_width))
        if target_true_val is None:
            covered = True
        else:
            covered = (target_true_val >= center - final_half_width) and (target_true_val <= center + final_half_width)
        if update_runtime_state:
            stats = engine.state.tracker.update(
                group_key=group_key,
                covered=covered,
                interval_width=float(2.0 * final_half_width),
                abs_residual=float(abs_residual),
            )
            engine.record_adaptive_step(miss=not covered)
            engine.record_validity(validity)
        return (float(center - final_half_width), float(center + final_half_width)), {
            "target_name": target_name,
            "planned_factor": planned_factor,
            "final_factor": float(final_half_width / max(base_half_width, 1e-6)),
            "validity_score": float(validity.validity_score),
            "validity_status": str(validity.validity_status),
            "under_coverage_gap": float(stats.under_coverage_gap),
            "coverage_group_key": str(group_key),
            "shift_alert_flag": bool(validity.shift_alert_flag),
            "adaptive_quantile": adaptive_quantile,
            "runtime_interval_policy": str(decision.applied_policy),
        }

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
        speed_center = _f(state.get("pred_ego_speed_center_mps"), _f(state.get("ego_speed_mps"), 0.0))
        speed_lower = _f(state.get("pred_ego_speed_lower_mps"), speed_center - quantile)
        speed_upper = _f(state.get("pred_ego_speed_upper_mps"), speed_center + quantile)
        gap_center = _f(state.get("pred_relative_gap_center_m"), _f(state.get("min_gap_m"), 30.0))
        gap_lower = _f(state.get("pred_relative_gap_lower_m"), gap_center - quantile)
        gap_upper = _f(state.get("pred_relative_gap_upper_m"), gap_center + quantile)
        shift_score = float(np.clip(_f(state.get("shift_score"), 0.0), 0.0, 1.0))
        drift_enabled = bool(drift_flag)
        planned_contract_factor = float(np.clip(1.0 + 0.7 * (1.0 - float(reliability_w)) + 0.5 * shift_score, 1.0, 3.0))
        update_runtime_state = bool(cfg.get("record_shift_runtime", False))
        (speed_lower_final, speed_upper_final), speed_meta = self._shift_aware_interval(
            target_name="ego_speed_mps",
            center=speed_center,
            lower=speed_lower,
            upper=speed_upper,
            reliability_w=float(reliability_w),
            shift_score=shift_score,
            state=state,
            drift_flag=drift_enabled,
            update_runtime_state=update_runtime_state,
        )
        (gap_lower_final, gap_upper_final), gap_meta = self._shift_aware_interval(
            target_name="relative_gap_m",
            center=gap_center,
            lower=gap_lower,
            upper=gap_upper,
            reliability_w=float(reliability_w),
            shift_score=shift_score,
            state=state,
            drift_flag=drift_enabled,
            update_runtime_state=update_runtime_state,
        )
        worst_meta = min((speed_meta, gap_meta), key=lambda meta: (meta["validity_score"], -_validity_rank(meta["validity_status"])))
        actual_final_factor = float(max(speed_meta["final_factor"], gap_meta["final_factor"]))

        # Monotone uncertainty floor: when prev_meta carries the current
        # (higher-reliability) uncertainty bounds, the degraded (lower-reliability)
        # probe must produce intervals at least as wide.  This ensures the
        # tightened_set_monotonicity contract holds even when the shift-aware
        # engine returns non-monotone widths across reliability bins.
        if prev_meta is not None and isinstance(prev_meta, dict):
            ref_speed_lo = prev_meta.get("speed_lower_mps")
            ref_speed_hi = prev_meta.get("speed_upper_mps")
            ref_gap_lo = prev_meta.get("gap_lower_m")
            ref_gap_hi = prev_meta.get("gap_upper_m")
            if ref_speed_lo is not None:
                speed_lower_final = min(speed_lower_final, float(ref_speed_lo))
            if ref_speed_hi is not None:
                speed_upper_final = max(speed_upper_final, float(ref_speed_hi))
            if ref_gap_lo is not None:
                gap_lower_final = min(gap_lower_final, float(ref_gap_lo))
            if ref_gap_hi is not None:
                gap_upper_final = max(gap_upper_final, float(ref_gap_hi))

        uncertainty = {
            "speed_lower_mps": max(0.0, speed_lower_final),
            "speed_upper_mps": speed_upper_final,
            "gap_lower_m": gap_lower_final,
            "gap_upper_m": gap_upper_final,
            "speed_center_mps": speed_center,
            "gap_center_m": gap_center,
            "lead_speed_mps": _f(state.get("lead_speed_mps"), speed_center),
            "current_speed_mps": _f(state.get("ego_speed_mps"), speed_center),
            "speed_limit_mps": _f(state.get("speed_limit_mps"), max(speed_center + 2.0, 10.0)),
            "meta": {
                "inflation": planned_contract_factor,
                "widening_factor": actual_final_factor,
                "planned_contract_inflation": planned_contract_factor,
                "shift_aware_final_widening_factor": actual_final_factor,
                "shift_score": shift_score,
                "base_gap_width_m": gap_upper - gap_lower,
                "base_speed_width_mps": speed_upper - speed_lower,
                "drift_flag": bool(drift_enabled or cfg.get("drift_flag", False)),
                "validity_score": float(worst_meta["validity_score"]),
                "validity_status": str(worst_meta["validity_status"]),
                "conditional_coverage_gap": float(max(speed_meta["under_coverage_gap"], gap_meta["under_coverage_gap"])),
                "coverage_group_key": str(worst_meta["coverage_group_key"]),
                "adaptive_quantile": float(max(speed_meta["adaptive_quantile"], gap_meta["adaptive_quantile"])),
                "runtime_interval_policy": str(worst_meta["runtime_interval_policy"]),
                "shift_alert_flag": bool(speed_meta["shift_alert_flag"] or gap_meta["shift_alert_flag"]),
                "target_interval_meta": {
                    "ego_speed_mps": speed_meta,
                    "relative_gap_m": gap_meta,
                },
            },
        }
        return uncertainty, dict(uncertainty["meta"])

    def write_shift_aware_outputs(self, out_dir: str | Path) -> dict[str, Any]:
        if not self._shift_cfg.enabled:
            return {}
        base_dir = Path(out_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        summary_rows: list[dict[str, Any]] = []
        target_artifacts: dict[str, Any] = {}
        for target_name, engine in self._shift_engines.items():
            target_dir = base_dir / target_name
            write_shift_aware_artifacts(
                tracker=engine.state.tracker,
                validity_trace=engine.state.validity_trace,
                adaptive_trace=engine.state.adaptive_trace,
                publication_dir=str(target_dir),
            )
            rows = engine.state.tracker.group_rows()
            target_artifacts[target_name] = {
                "group_coverage_csv": str(target_dir / "fault_group_coverage.csv"),
                "validity_trace_csv": str(target_dir / "shift_validity_trace.csv"),
                "adaptive_trace_csv": str(target_dir / "adaptive_quantile_trace.csv"),
            }
            engine.save()
            summary_rows.append(
                {
                    "target": target_name,
                    "steps": int(engine.state.step),
                    "num_groups": int(len(rows)),
                    "max_under_coverage_gap": float(max((row.get("under_coverage_gap", 0.0) for row in rows), default=0.0)),
                    "final_validity_score": float(engine.state.last_validity.validity_score if engine.state.last_validity else 1.0),
                    "final_validity_status": str(engine.state.last_validity.validity_status if engine.state.last_validity else "nominal"),
                    "shift_alert_rate": float(np.mean([row.get("shift_alert_flag", False) for row in engine.state.validity_trace])) if engine.state.validity_trace else 0.0,
                }
            )
        summary_path = base_dir / "shift_aware_runtime_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        return {
            "summary_csv": str(summary_path),
            "targets": target_artifacts,
        }

    def record_shift_aware_step(
        self,
        *,
        state: Mapping[str, Any],
        reliability_w: float,
        drift_flag: bool,
    ) -> dict[str, Any]:
        if not self._shift_cfg.enabled:
            return {}
        speed_center = _f(state.get("pred_ego_speed_center_mps"), _f(state.get("ego_speed_mps"), 0.0))
        speed_lower = _f(state.get("pred_ego_speed_lower_mps"), speed_center)
        speed_upper = _f(state.get("pred_ego_speed_upper_mps"), speed_center)
        gap_center = _f(state.get("pred_relative_gap_center_m"), _f(state.get("min_gap_m"), 30.0))
        gap_lower = _f(state.get("pred_relative_gap_lower_m"), gap_center)
        gap_upper = _f(state.get("pred_relative_gap_upper_m"), gap_center)
        shift_score = float(np.clip(_f(state.get("shift_score"), 0.0), 0.0, 1.0))
        _, speed_meta = self._shift_aware_interval(
            target_name="ego_speed_mps",
            center=speed_center,
            lower=speed_lower,
            upper=speed_upper,
            reliability_w=float(reliability_w),
            shift_score=shift_score,
            state=state,
            drift_flag=bool(drift_flag),
            update_runtime_state=True,
        )
        _, gap_meta = self._shift_aware_interval(
            target_name="relative_gap_m",
            center=gap_center,
            lower=gap_lower,
            upper=gap_upper,
            reliability_w=float(reliability_w),
            shift_score=shift_score,
            state=state,
            drift_flag=bool(drift_flag),
            update_runtime_state=True,
        )
        return {"ego_speed_mps": speed_meta, "relative_gap_m": gap_meta}

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
            validity_score=uncertainty.get("meta", {}).get("validity_score"),
            adaptive_quantile=uncertainty.get("meta", {}).get("adaptive_quantile"),
            conditional_coverage_gap=uncertainty.get("meta", {}).get("conditional_coverage_gap"),
            runtime_interval_policy=uncertainty.get("meta", {}).get("runtime_interval_policy"),
            coverage_group_key=uncertainty.get("meta", {}).get("coverage_group_key"),
            shift_alert_flag=uncertainty.get("meta", {}).get("shift_alert_flag"),
            assumptions_version="waymo-av-shift-aware-v1",
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
        cert["validity_score"] = uncertainty.get("meta", {}).get("validity_score")
        cert["validity_status"] = uncertainty.get("meta", {}).get("validity_status")
        cert["coverage_group_key"] = uncertainty.get("meta", {}).get("coverage_group_key")
        cert["shift_alert_flag"] = uncertainty.get("meta", {}).get("shift_alert_flag")
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
    adapter: WaymoAVDomainAdapter | None = None,
) -> tuple[list[StepRecord], list[dict[str, Any]], list[dict[str, Any]]]:
    adapter = adapter or WaymoAVDomainAdapter({"expected_cadence_s": 0.1})
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
        observed["fault_family"] = fault_family
        observed["target_ego_speed_mps_1s"] = float(feature_row["target_ego_speed_mps__1s"])
        observed["target_relative_gap_m_1s"] = float(feature_row["target_relative_gap_m__1s"])
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
            # The universal layer augments the certificate after the adapter emits
            # its initial payload, so re-hash the final mapping before chaining
            # and persistence.
            certificate["certificate_hash"] = recompute_certificate_hash(certificate)
            prev_cert_hash = certificate.get("certificate_hash")
            certificate_valid = True
            intervened = bool(result["repair_meta"]["repaired"])
            adapter.record_shift_aware_step(
                state=observed,
                reliability_w=float(result["reliability_w"]),
                drift_flag=bool(result["drift_flag"]),
            )
            uncertainty_meta = dict(result["uncertainty_set"].get("meta", {}))
            base_speed_lower = float(observed["pred_ego_speed_lower_mps"])
            base_speed_upper = float(observed["pred_ego_speed_upper_mps"])
            base_gap_lower = float(observed["pred_relative_gap_lower_m"])
            base_gap_upper = float(observed["pred_relative_gap_upper_m"])
            trace_speed_lower = float(result["uncertainty_set"]["speed_lower_mps"])
            trace_speed_upper = float(result["uncertainty_set"]["speed_upper_mps"])
            trace_gap_lower = float(result["uncertainty_set"]["gap_lower_m"])
            trace_gap_upper = float(result["uncertainty_set"]["gap_upper_m"])
        else:
            safe_action = dict(candidate_action)
            uncertainty_meta = {
                "widening_factor": 1.0,
                "validity_score": 1.0,
                "validity_status": "nominal",
                "coverage_group_key": "baseline",
                "shift_alert_flag": False,
            }
            base_speed_lower = float(observed["pred_ego_speed_lower_mps"])
            base_speed_upper = float(observed["pred_ego_speed_upper_mps"])
            base_gap_lower = float(observed["pred_relative_gap_lower_m"])
            base_gap_upper = float(observed["pred_relative_gap_upper_m"])
            trace_speed_lower = float(observed["pred_ego_speed_lower_mps"])
            trace_speed_upper = float(observed["pred_ego_speed_upper_mps"])
            trace_gap_lower = float(observed["pred_relative_gap_lower_m"])
            trace_gap_upper = float(observed["pred_relative_gap_upper_m"])
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
                true_state=_compact_step_state(true_state),
                observed_state=_compact_step_state(observed, include_runtime=True),
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
                "trace_id": f"{scenario_id}-{step_index}-{'orius' if use_orius else 'baseline'}",
                "scenario_id": scenario_id,
                "shard_id": str(true_state.get("shard_id", "")),
                "ego_track_id": int(true_state.get("ego_track_id", 0) or 0),
                "step_index": int(step_index),
                "fault_family": fault_family,
                "controller": "orius" if use_orius else "baseline",
                "candidate_acceleration_mps2": float(candidate_action["acceleration_mps2"]),
                "safe_acceleration_mps2": float(safe_action["acceleration_mps2"]),
                "intervened": bool(intervened),
                "fallback_used": bool(fallback_used),
                "certificate_valid": bool(certificate_valid),
                "true_constraint_violated": bool(true_metrics["true_constraint_violated"]),
                "observed_constraint_satisfied": (
                    None if observed_metrics.get("true_constraint_violated") is None else not bool(observed_metrics["true_constraint_violated"])
                ),
                "true_margin": float(true_metrics["true_margin"]),
                "observed_margin": float(observed.get("observed_margin", np.nan)),
                "min_gap_m": float(true_metrics["min_gap_m"]),
                "ttc_s": float(true_metrics["ttc_s"]),
                "latency_us": float(latency_us),
                "reliability_w": float(result["reliability_w"]) if use_orius else float(np.clip(observed["reliability_proxy"], 0.05, 1.0)),
                "neighbor_count": int(true_state.get("neighbor_count", 0) or 0),
                "ego_speed_mps": float(true_state.get("ego_speed_mps", 0.0) or 0.0),
                "target_ego_speed_1s": float(feature_row["target_ego_speed_mps__1s"]),
                "target_relative_gap_1s": float(feature_row["target_relative_gap_m__1s"]),
                "base_pred_ego_speed_lower_mps": base_speed_lower,
                "base_pred_ego_speed_upper_mps": base_speed_upper,
                "pred_ego_speed_lower_mps": trace_speed_lower,
                "pred_ego_speed_upper_mps": trace_speed_upper,
                "base_pred_relative_gap_lower_m": base_gap_lower,
                "base_pred_relative_gap_upper_m": base_gap_upper,
                "pred_relative_gap_lower_m": trace_gap_lower,
                "pred_relative_gap_upper_m": trace_gap_upper,
                "shift_score": float(observed["shift_score"]),
                "widening_factor": float(uncertainty_meta.get("widening_factor", 1.0)),
                "validity_score": float(uncertainty_meta.get("validity_score", 1.0)),
                "validity_status": str(uncertainty_meta.get("validity_status", "nominal")),
                "coverage_group_key": str(uncertainty_meta.get("coverage_group_key", "global")),
                "shift_alert_flag": bool(uncertainty_meta.get("shift_alert_flag", False)),
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
    stage_dir = out_path / f".runtime_stage_{time.time_ns()}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    final_audit_db_path = out_path / "dc3s_av_waymo_dryrun.duckdb"
    audit_db_path = stage_dir / "dc3s_av_waymo_dryrun.duckdb"
    table_name = "dispatch_certificates"
    shift_out_dir = stage_dir / "shift_aware"
    shift_cfg_path = out_path / "shift_aware_config.json"
    if shift_cfg_path.exists():
        shift_cfg = json.loads(shift_cfg_path.read_text(encoding="utf-8"))
    else:
        shift_cfg = default_shift_aware_config(publication_dir=str(shift_out_dir)).to_dict()
    shift_cfg["publication_dir"] = str(shift_out_dir)
    orius_adapter = WaymoAVDomainAdapter(
        {
            "expected_cadence_s": 0.1,
            "shift_aware_uncertainty": shift_cfg,
            "shift_output_dir": str(shift_out_dir),
        }
    )

    all_records: dict[str, list[StepRecord]] = {"baseline": [], "orius": []}
    all_traces: list[dict[str, Any]] = []
    certificate_count = 0
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
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
            )
            orius_records, orius_traces, orius_certs = _run_episode(
                track=orius_track,
                scenario_id=str(scenario_id),
                step_features=step_features,
                bundles=bundles,
                fault_family=fault_family,
                use_orius=True,
                adapter=orius_adapter,
            )
            all_records["baseline"].extend(baseline_records)
            all_records["orius"].extend(orius_records)
            all_traces.extend(baseline_traces)
            all_traces.extend(orius_traces)
            if orius_certs:
                store_certificates_batch(orius_certs, duckdb_path=str(audit_db_path), table_name=table_name)
                certificate_count += int(len(orius_certs))
    finally:
        if gc_was_enabled:
            gc.enable()
            gc.collect()

    trace_df = pd.DataFrame(all_traces)
    trace_path = stage_dir / "runtime_traces.csv"
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
    summary_path = stage_dir / "runtime_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    coverage_rows = []
    if not trace_df.empty:
        for (controller_name, fault_family), group in trace_df.groupby(["controller", "fault_family"], dropna=False):
            for target_name, lower_col, upper_col, true_col in (
                ("ego_speed_mps", "pred_ego_speed_lower_mps", "pred_ego_speed_upper_mps", "target_ego_speed_1s"),
                ("relative_gap_m", "pred_relative_gap_lower_m", "pred_relative_gap_upper_m", "target_relative_gap_1s"),
            ):
                y_true = group[true_col].to_numpy(dtype=float)
                lower = group[lower_col].to_numpy(dtype=float)
                upper = group[upper_col].to_numpy(dtype=float)
                coverage_rows.append(
                    {
                        "controller": str(controller_name),
                        "fault_family": fault_family,
                        "target": target_name,
                        "coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
                        "mean_width": float(np.mean(upper - lower)),
                    }
                )
    coverage_path = stage_dir / "fault_family_coverage.csv"
    pd.DataFrame(coverage_rows, columns=["controller", "fault_family", "target", "coverage", "mean_width"]).to_csv(coverage_path, index=False)
    shift_artifacts = orius_adapter.write_shift_aware_outputs(shift_out_dir)

    final_trace_path = out_path / "runtime_traces.csv"
    final_summary_path = out_path / "runtime_summary.csv"
    final_coverage_path = out_path / "fault_family_coverage.csv"
    final_report_path = out_path / "runtime_report.json"
    final_shift_out_dir = out_path / "shift_aware"
    for source_path, target_path in (
        (trace_path, final_trace_path),
        (summary_path, final_summary_path),
        (coverage_path, final_coverage_path),
        (audit_db_path, final_audit_db_path),
    ):
        source_path.replace(target_path)
    if final_shift_out_dir.exists():
        shutil.rmtree(final_shift_out_dir)
    if shift_out_dir.exists():
        shift_out_dir.replace(final_shift_out_dir)
    shift_artifacts = _remap_path_prefix(shift_artifacts, shift_out_dir, final_shift_out_dir)

    report = {
        "runtime_summary_csv": str(final_summary_path),
        "runtime_traces_csv": str(final_trace_path),
        "fault_family_coverage_csv": str(final_coverage_path),
        "audit_db_path": str(final_audit_db_path),
        "certificate_count": int(certificate_count),
        "scenario_count": int(len(test_scenarios)),
        "shift_aware_artifacts": shift_artifacts,
    }
    final_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    return report
