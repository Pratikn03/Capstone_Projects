"""Waymo AV runtime adapter and dry-run evaluation."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from orius.certos.verification import (
    REQUIRED_CERTIFICATE_FIELDS,
    certificate_intervention_semantics_valid,
    count_present_required_certificate_fields,
    extract_certificate_validity_horizon,
    formal_validity_predicate,
    missing_required_certificate_fields,
)
from orius.dc3s.certificate import make_certificate, recompute_certificate_hash, store_certificates_batch
from orius.dc3s.domain_adapter import DomainAdapter
from orius.forecasting.uncertainty.shift_aware import (
    ShiftAwareConfig,
    ShiftAwareRuntimeEngine,
    apply_interval_policy,
    compute_validity_score,
    write_shift_aware_artifacts,
)
from orius.forecasting.uncertainty.shift_aware.state import GroupCoverageStats
from orius.orius_bench.metrics_engine import StepRecord
from orius.universal_framework.pipeline import run_universal_step
from orius.universal_framework.reliability_runtime import assess_domain_reliability
from orius.universal_theory.domain_runtime_contracts import (
    AV_BRAKE_HOLD_CONTRACT_ID,
    witness_trace_fields_from_result,
)
from orius.universal_theory.domain_validity import domain_certificate_validity_semantics

from .replay import FAULT_FAMILIES, WaymoReplayTrackAdapter, compute_state_safety_metrics
from .training import (
    default_shift_aware_config,
    estimate_shift_score,
    load_model_bundle,
    predict_interval_from_bundle,
)

RUNTIME_PREDICTION_BATCH_ROWS = 50_000
RUNTIME_PREDICTION_COLUMNS = (
    "pred_ego_speed_center_mps",
    "pred_ego_speed_lower_mps",
    "pred_ego_speed_upper_mps",
    "pred_relative_gap_center_m",
    "pred_relative_gap_lower_m",
    "pred_relative_gap_upper_m",
    "shift_score",
)


def _runtime_test_scenarios_from_anchors(
    step_features_path: Path, max_scenarios: int | None
) -> list[str] | None:
    anchor_path = step_features_path.parent / "anchor_features.parquet"
    if not anchor_path.exists():
        return None
    try:
        anchors = pd.read_parquet(
            anchor_path, columns=["scenario_id", "split"], filters=[("split", "==", "test")]
        )
    except Exception:
        anchors = pd.read_parquet(anchor_path, columns=["scenario_id", "split"])
        anchors = anchors[anchors["split"] == "test"].copy()
    scenario_ids = sorted(anchors["scenario_id"].astype(str).unique().tolist())
    if max_scenarios is not None:
        scenario_ids = scenario_ids[: int(max_scenarios)]
    return scenario_ids


def _load_runtime_test_step_features(
    step_features_path: str | Path,
    *,
    max_scenarios: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    step_path = Path(step_features_path)
    selected_scenarios = _runtime_test_scenarios_from_anchors(step_path, max_scenarios)
    if selected_scenarios is None:
        try:
            step_features = pd.read_parquet(step_path, filters=[("split", "==", "test")])
        except Exception:
            # Filter pushdown is an optimization; older parquet engines may not support it.
            step_features = pd.read_parquet(step_path)
            step_features = step_features[step_features["split"] == "test"].copy()
        step_features["scenario_id"] = step_features["scenario_id"].astype(str)
        test_scenarios = sorted(step_features["scenario_id"].unique().tolist())
        if max_scenarios is not None:
            test_scenarios = test_scenarios[: int(max_scenarios)]
            step_features = step_features[step_features["scenario_id"].isin(test_scenarios)].copy()
        return step_features, test_scenarios

    selected_set = set(selected_scenarios)
    chunks: list[pd.DataFrame] = []
    for batch in pq.ParquetFile(step_path).iter_batches(batch_size=200_000):
        frame = batch.to_pandas()
        frame["scenario_id"] = frame["scenario_id"].astype(str)
        frame = frame[frame["scenario_id"].isin(selected_set)]
        if not frame.empty:
            chunks.append(frame)
    step_features = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    return step_features, selected_scenarios


def _write_equal_domain_artifacts(domain_key: str, out_dir: Path) -> dict[str, str]:
    script_path = (
        Path(__file__).resolve().parents[3] / "scripts" / "build_equal_domain_artifact_discipline.py"
    )
    spec = importlib.util.spec_from_file_location("build_equal_domain_artifact_discipline", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load equal-domain artifact builder from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.write_runtime_comparator_artifacts_for_domain(domain_key, out_dir=out_dir)


def _f(value: Any, default: float) -> float:
    try:
        v = float(value)
        if not math.isfinite(v):
            return float(default)
        return v
    except (TypeError, ValueError):
        return float(default)


def _finite_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _bool_cfg(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _certificate_reliability_weight(certificate: Mapping[str, Any]) -> float:
    for key in ("reliability_w", "w_t"):
        numeric = _finite_float_or_none(certificate.get(key))
        if numeric is not None:
            return float(numeric)
    reliability = certificate.get("reliability")
    if isinstance(reliability, Mapping):
        for key in ("w_t", "w", "reliability_w"):
            numeric = _finite_float_or_none(reliability.get(key))
            if numeric is not None:
                return float(numeric)
        return 0.0
    numeric = _finite_float_or_none(reliability)
    return float(numeric) if numeric is not None else 0.0


def _certificate_required_field_counts(certificate: Mapping[str, Any] | None) -> tuple[int, int]:
    return count_present_required_certificate_fields(certificate), len(REQUIRED_CERTIFICATE_FIELDS)


def _predict_certificate_validity(
    certificate: Mapping[str, Any] | None,
    previous_certificate: Mapping[str, Any] | None,
) -> bool:
    if certificate is None or missing_required_certificate_fields(certificate):
        return False
    if not certificate_intervention_semantics_valid(certificate):
        return False
    current_prev_hash = certificate.get("prev_hash")
    if previous_certificate is None:
        if current_prev_hash not in (None, ""):
            return False
    elif current_prev_hash != previous_certificate.get("certificate_hash"):
        return False
    horizon_value = extract_certificate_validity_horizon(certificate)
    return horizon_value > 0 and _certificate_reliability_weight(certificate) >= 0.0


def _independent_certificate_validity(
    certificate: Mapping[str, Any] | None,
    previous_certificate: Mapping[str, Any] | None,
) -> bool:
    if certificate is None or missing_required_certificate_fields(certificate):
        return False
    if not certificate_intervention_semantics_valid(certificate):
        return False
    verdict = formal_validity_predicate(certificate, previous_certificate, w_min=0.0)
    return bool(verdict.valid)


def _scenario_fault_family(scenario_id: str) -> str:
    digest = hashlib.sha256(str(scenario_id).encode("utf-8")).digest()
    return FAULT_FAMILIES[digest[0] % len(FAULT_FAMILIES)]


def _validity_rank(status: str) -> int:
    order = {"nominal": 0, "watch": 1, "degraded": 2, "invalid": 3}
    return int(order.get(str(status), 0))


def _remap_path_prefix(value: Any, source_prefix: Path, target_prefix: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _remap_path_prefix(nested, source_prefix, target_prefix) for key, nested in value.items()
        }
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
        suffix = candidate_text[len(source_text) :].lstrip("/")
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


def _runtime_aligned_longitudinal_controller(
    state: dict[str, Any],
    *,
    constraints: Mapping[str, Any],
    policy_config: Mapping[str, Any],
) -> dict[str, float]:
    """Nominal proposal that agrees with configured ORIUS projection regions."""
    base_action = deterministic_longitudinal_controller(state)
    if not _bool_cfg(policy_config, "align_nominal_with_runtime_projection", False):
        return base_action

    base_accel = _f(base_action.get("acceleration_mps2"), 0.0)
    accel_min = _f(constraints.get("accel_min_mps2"), -6.0)
    accel_max = _f(constraints.get("accel_max_mps2"), 3.0)
    hard_headway_m = _f(constraints.get("hard_headway_m"), 5.0)
    hard_ttc_s = _f(constraints.get("hard_ttc_s"), 2.0)
    entry_headway_m = _f(constraints.get("entry_headway_m"), 10.0)
    entry_ttc_s = _f(constraints.get("entry_ttc_s"), 4.0)
    margin_m = max(0.0, _f(policy_config.get("nominal_projection_margin_m"), 0.0))
    margin_ttc_s = max(0.0, _f(policy_config.get("nominal_projection_ttc_margin_s"), 0.0))
    near_failsafe_accel = float(
        np.clip(_f(policy_config.get("near_failsafe_accel_mps2"), accel_min + 0.10), accel_min, accel_max)
    )
    entry_projection_accel = float(
        np.clip(
            _f(policy_config.get("entry_projection_accel_mps2"), near_failsafe_accel), accel_min, accel_max
        )
    )

    current_gap = _f(state.get("min_gap_m"), 30.0)
    predicted_gap_lower = _f(state.get("pred_relative_gap_lower_m"), current_gap)
    ego_speed = _f(state.get("ego_speed_mps"), 0.0)
    lead_speed = _f(state.get("lead_speed_mps"), ego_speed)
    closing_speed = max(0.0, ego_speed - lead_speed)
    predicted_ttc_s = (
        float("inf") if closing_speed <= 1e-9 else predicted_gap_lower / max(closing_speed, 1e-9)
    )
    current_ttc_s = _f(state.get("ttc_s"), predicted_ttc_s)
    ttc_probe_s = min(predicted_ttc_s, current_ttc_s) if math.isfinite(current_ttc_s) else predicted_ttc_s

    hard_projection_region = predicted_gap_lower <= hard_headway_m + margin_m or (
        math.isfinite(ttc_probe_s) and ttc_probe_s <= hard_ttc_s + margin_ttc_s
    )
    critical_region = current_gap <= 1.0 or (
        math.isfinite(current_ttc_s) and current_ttc_s <= 0.0 and current_gap <= 1.0
    )
    if hard_projection_region:
        return {
            "acceleration_mps2": float(np.clip(min(base_accel, near_failsafe_accel), accel_min, accel_max))
        }
    if critical_region:
        return {"acceleration_mps2": float(np.clip(min(base_accel, accel_min), accel_min, accel_max))}

    if _bool_cfg(policy_config, "align_nominal_entry_projection", False):
        entry_projection_region = predicted_gap_lower <= entry_headway_m + margin_m or (
            math.isfinite(ttc_probe_s) and ttc_probe_s <= entry_ttc_s + margin_ttc_s
        )
        if entry_projection_region:
            return {
                "acceleration_mps2": float(
                    np.clip(min(base_accel, entry_projection_accel), accel_min, accel_max)
                )
            }

    return {"acceleration_mps2": float(np.clip(base_accel, accel_min, accel_max))}


def _is_full_brake(action: Mapping[str, Any], brake_accel_mps2: float) -> bool:
    return _f(action.get("acceleration_mps2"), 0.0) <= float(brake_accel_mps2) + 1e-9


def _contract_margin(
    min_gap_m: float, ttc_s: float, headway_threshold_m: float, ttc_threshold_s: float
) -> float:
    ttc_margin = 1e6 if not math.isfinite(ttc_s) else float(ttc_s - ttc_threshold_s)
    return float(min(min_gap_m - headway_threshold_m, ttc_margin))


def _brake_hold_contract_status(
    *,
    true_metrics: Mapping[str, Any],
    observed_metrics: Mapping[str, Any] | None,
    uncertainty_meta: Mapping[str, Any],
    safe_action: Mapping[str, Any],
    repair_meta: Mapping[str, Any],
    certificate_valid: bool,
    headway_threshold_m: float,
    ttc_threshold_s: float,
    brake_accel_mps2: float,
) -> dict[str, Any]:
    validity_status = str(uncertainty_meta.get("validity_status", "nominal"))
    full_brake = _is_full_brake(safe_action, brake_accel_mps2)
    true_min_gap = _f(true_metrics.get("min_gap_m"), 0.0)
    true_ttc = _f(true_metrics.get("ttc_s"), float("inf"))
    hard_headway_threshold_m = min(float(headway_threshold_m), 5.0)
    hard_ttc_threshold_s = min(float(ttc_threshold_s), 2.0)
    full_brake_equivalent = bool(
        full_brake
        or (
            bool(repair_meta.get("projected_release", False))
            and bool(repair_meta.get("allow_near_failsafe_projected_validity", False))
            and _f(safe_action.get("acceleration_mps2"), 0.0)
            <= brake_accel_mps2 + _f(repair_meta.get("near_failsafe_projection_epsilon_mps2"), 0.25)
        )
    )
    true_requires_fallback = bool(
        bool(true_metrics.get("overlap"))
        or true_min_gap < hard_headway_threshold_m
        or (math.isfinite(true_ttc) and true_ttc < hard_ttc_threshold_s)
        or not certificate_valid
    )
    if observed_metrics is None:
        observed_requires_fallback = None
        observed_margin = None
    else:
        observed_min_gap = _f(observed_metrics.get("min_gap_m"), 0.0)
        observed_ttc = _f(observed_metrics.get("ttc_s"), float("inf"))
        observed_requires_fallback = bool(
            bool(observed_metrics.get("overlap"))
            or observed_min_gap < hard_headway_threshold_m
            or (math.isfinite(observed_ttc) and observed_ttc < hard_ttc_threshold_s)
            or bool(repair_meta.get("fallback_required", False))
        )
        observed_margin = (
            float(_f(safe_action.get("acceleration_mps2"), 0.0) - brake_accel_mps2)
            if observed_requires_fallback
            else _contract_margin(observed_min_gap, observed_ttc, headway_threshold_m, ttc_threshold_s)
        )
    true_margin = (
        float(_f(safe_action.get("acceleration_mps2"), 0.0) - brake_accel_mps2)
        if true_requires_fallback
        else _contract_margin(true_min_gap, true_ttc, headway_threshold_m, ttc_threshold_s)
    )
    return {
        "true_constraint_violated": bool(true_requires_fallback and not full_brake_equivalent),
        "observed_constraint_satisfied": (
            None if observed_requires_fallback is None else not observed_requires_fallback
        ),
        "true_margin": true_margin,
        "observed_margin": observed_margin,
        "release_requires_fallback": true_requires_fallback,
        "full_brake": full_brake,
        "validity_status": validity_status,
    }


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
        policy_cfg = self._policy_cfg()
        self.domain_id = "av_waymo"
        self._expected_cadence_s = _f(self._cfg.get("expected_cadence_s"), 0.1)
        self._accel_min = _f(self._cfg.get("accel_min_mps2"), -6.0)
        self._accel_max = _f(self._cfg.get("accel_max_mps2"), 3.0)
        self._min_headway_m = _f(self._cfg.get("min_headway_m"), 8.0)
        self._ttc_min_s = _f(self._cfg.get("ttc_min_s"), 3.0)
        self._entry_headway_m = _f(self._cfg.get("entry_headway_m"), 10.0)
        self._entry_ttc_s = _f(self._cfg.get("entry_ttc_s"), 4.0)
        self._prediction_backed_telemetry = _bool_cfg(policy_cfg, "prediction_backed_telemetry", False)
        self._hard_projection_requires_observed_confirmation = _bool_cfg(
            policy_cfg,
            "hard_projection_requires_observed_confirmation",
            False,
        )
        self._hard_projection_current_gap_buffer_m = _f(
            policy_cfg.get("hard_projection_current_gap_buffer_m"), 0.0
        )
        self._hard_projection_current_ttc_buffer_s = _f(
            policy_cfg.get("hard_projection_current_ttc_buffer_s"), 0.0
        )
        self._entry_projection_uses_current_state = _bool_cfg(
            policy_cfg, "entry_projection_uses_current_state", False
        )
        self._near_failsafe_projection_epsilon_mps2 = _f(
            policy_cfg.get("near_failsafe_projection_epsilon_mps2"), 0.25
        )
        self._near_failsafe_accel_mps2 = _f(
            policy_cfg.get("near_failsafe_accel_mps2"), self._accel_min + 0.10
        )
        self._entry_projection_accel_mps2 = _f(
            policy_cfg.get("entry_projection_accel_mps2"), self._near_failsafe_accel_mps2
        )
        self._degraded_projection_accel_mps2 = _f(
            policy_cfg.get("degraded_projection_accel_mps2"),
            self._near_failsafe_accel_mps2,
        )
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
                self._shift_engines[target_name] = ShiftAwareRuntimeEngine(
                    cfg=self._shift_cfg, state_path=state_path
                )

    def _policy_cfg(self) -> dict[str, Any]:
        policy = self._cfg.get("runtime_policy", {})
        return dict(policy) if isinstance(policy, Mapping) else {}

    def capability_profile(self) -> dict[str, Any]:
        return {
            "safety_surface_type": "waymo_longitudinal_headway_barrier",
            "repair_mode": "piecewise_hold_or_full_brake",
            "fallback_mode": "full_brake",
            "supports_multi_agent_eval": True,
            "supports_certos_eval": True,
        }

    def ingest_telemetry(self, raw_packet: dict[str, Any]) -> dict[str, Any]:
        dict(raw_packet)
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
        missing_fields: list[str] = []
        for field in fields:
            value = raw_packet.get(field)
            numeric = (
                _finite_float_or_none(value)
                if isinstance(value, float | int | np.floating | np.integer)
                else value
            )
            if value is None or (
                isinstance(value, float | int | np.floating | np.integer) and numeric is None
            ):
                if field in {"pred_ego_speed_center_mps", "target_ego_speed_mps_1s"}:
                    state[field] = _f(raw_packet.get("ego_speed_mps"), 0.0)
                    continue
                if field == "pred_ego_speed_lower_mps":
                    state[field] = max(0.0, _f(raw_packet.get("ego_speed_mps"), 0.0) - 1.0)
                    continue
                if field == "pred_ego_speed_upper_mps":
                    state[field] = _f(raw_packet.get("ego_speed_mps"), 0.0) + 1.0
                    continue
                if field in {"pred_relative_gap_center_m", "target_relative_gap_m_1s"}:
                    state[field] = _f(raw_packet.get("min_gap_m"), 30.0)
                    continue
                if field == "pred_relative_gap_lower_m":
                    state[field] = _f(raw_packet.get("min_gap_m"), 30.0) - 1.0
                    continue
                if field == "pred_relative_gap_upper_m":
                    state[field] = _f(raw_packet.get("min_gap_m"), 30.0) + 1.0
                    continue
                if field in {"ego_accel_mps2", "ego_jerk_mps3", "neighbor_instability", "shift_score"}:
                    state[field] = 0.0
                    continue
                if field == "min_gap_m" and int(_f(raw_packet.get("neighbor_count"), 0.0)) <= 0:
                    state[field] = 1e6
                    continue
                if field == "lead_speed_mps":
                    state[field] = _f(raw_packet.get("ego_speed_mps"), 0.0)
                    continue
                if field == "lead_rel_speed_mps":
                    state[field] = 0.0
                    continue
                missing_fields.append(field)
                state[field] = raw_packet.get(f"_hold_{field}", 0.0)
            else:
                state[field] = value
        state["ts_utc"] = raw_packet.get("ts_utc", "")
        state["scenario_id"] = raw_packet.get("scenario_id")
        state["shard_id"] = raw_packet.get("shard_id")
        state["ego_track_id"] = raw_packet.get("ego_track_id")
        state["neighbor_ids_csv"] = raw_packet.get("neighbor_ids_csv", "")
        state["telemetry_missing_count"] = int(len(missing_fields))
        state["telemetry_missing_fields"] = ",".join(missing_fields)
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
        planned_factor = float(
            np.clip(1.0 + 0.7 * (1.0 - float(reliability_w)) + 0.5 * shift_score, 1.0, max_inflation)
        )
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
        target_key = (
            "target_ego_speed_mps_1s" if target_name == "ego_speed_mps" else "target_relative_gap_m_1s"
        )
        target_true_val = _finite_float_or_none(state.get(target_key))
        abs_residual = 0.0 if target_true_val is None else abs(target_true_val - center)
        volatility_raw = (
            abs(_f(state.get("ego_accel_mps2"), 0.0))
            if target_name == "ego_speed_mps"
            else abs(_f(state.get("lead_rel_speed_mps"), 0.0))
        )
        volatility = float(
            np.clip(volatility_raw / (8.0 if target_name == "ego_speed_mps" else 10.0), 0.0, 1.0)
        )
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
            covered = (target_true_val >= center - final_half_width) and (
                target_true_val <= center + final_half_width
            )
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

    def compute_oqe(
        self, state: dict[str, Any], history: list[dict[str, Any]] | None = None
    ) -> tuple[float, dict[str, Any]]:
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
            closure_tier="runtime_contract_closed",
        )
        valid_ratio = float(np.clip(_f(state.get("reliability_proxy"), 1.0), 0.05, 1.0))
        instability = float(np.clip(_f(state.get("neighbor_instability"), 0.0), 0.0, 1.0))
        accel_penalty = 0.5 if abs(_f(state.get("ego_accel_mps2"), 0.0)) > 8.0 else 1.0
        jerk_penalty = 0.6 if abs(_f(state.get("ego_jerk_mps3"), 0.0)) > 15.0 else 1.0
        missing_penalty = 0.05 if int(_f(state.get("telemetry_missing_count"), 0.0)) > 0 else 1.0
        w_t = float(
            np.clip(
                base_w
                * valid_ratio
                * (1.0 - 0.5 * instability)
                * accel_penalty
                * jerk_penalty
                * missing_penalty,
                0.05,
                1.0,
            )
        )
        flags["actor_validity_ratio"] = valid_ratio
        flags["neighbor_instability"] = instability
        flags["impossible_acceleration"] = abs(_f(state.get("ego_accel_mps2"), 0.0)) > 8.0
        flags["impossible_jerk"] = abs(_f(state.get("ego_jerk_mps3"), 0.0)) > 15.0
        flags["telemetry_missing_count"] = int(_f(state.get("telemetry_missing_count"), 0.0))
        flags["telemetry_missing_fields"] = str(state.get("telemetry_missing_fields", ""))
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
        planned_contract_factor = float(
            np.clip(1.0 + 0.7 * (1.0 - float(reliability_w)) + 0.5 * shift_score, 1.0, 3.0)
        )
        update_runtime_state = bool(cfg.get("record_shift_runtime", False))
        telemetry_missing_count = int(_f(state.get("telemetry_missing_count"), 0.0))
        prediction_backed_telemetry = bool(self._prediction_backed_telemetry and telemetry_missing_count > 0)
        current_speed_estimate = (
            speed_center if prediction_backed_telemetry else _f(state.get("ego_speed_mps"), speed_center)
        )
        current_gap_estimate = (
            gap_center if prediction_backed_telemetry else _f(state.get("min_gap_m"), gap_center)
        )
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
        worst_meta = min(
            (speed_meta, gap_meta),
            key=lambda meta: (meta["validity_score"], -_validity_rank(meta["validity_status"])),
        )
        actual_final_factor = float(max(speed_meta["final_factor"], gap_meta["final_factor"]))
        validity_status = str(worst_meta["validity_status"])
        if telemetry_missing_count > 0 and not prediction_backed_telemetry:
            validity_status = "invalid"
        elif prediction_backed_telemetry and _validity_rank(validity_status) < _validity_rank("degraded"):
            validity_status = "degraded"
        shift_alert_flag = bool(speed_meta["shift_alert_flag"] or gap_meta["shift_alert_flag"])

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
            prev_meta_payload = (
                prev_meta.get("meta", {}) if isinstance(prev_meta.get("meta", {}), Mapping) else {}
            )
            prev_status = str(prev_meta_payload.get("validity_status", "nominal"))
            if _validity_rank(prev_status) > _validity_rank(validity_status):
                validity_status = prev_status
            shift_alert_flag = shift_alert_flag or bool(prev_meta_payload.get("shift_alert_flag", False))

        uncertainty = {
            "speed_lower_mps": max(0.0, speed_lower_final),
            "speed_upper_mps": speed_upper_final,
            "gap_lower_m": gap_lower_final,
            "gap_upper_m": gap_upper_final,
            "speed_center_mps": speed_center,
            "gap_center_m": gap_center,
            "lead_speed_mps": _f(state.get("lead_speed_mps"), speed_center),
            "current_speed_mps": current_speed_estimate,
            "current_gap_m": current_gap_estimate,
            "current_ttc_s": _f(state.get("ttc_s"), float("inf")),
            "neighbor_count": _f(state.get("neighbor_count"), 0.0),
            "fault_family": str(state.get("fault_family", "")),
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
                "validity_status": validity_status,
                "conditional_coverage_gap": float(
                    max(speed_meta["under_coverage_gap"], gap_meta["under_coverage_gap"])
                ),
                "coverage_group_key": str(worst_meta["coverage_group_key"]),
                "adaptive_quantile": float(
                    max(speed_meta["adaptive_quantile"], gap_meta["adaptive_quantile"])
                ),
                "runtime_interval_policy": str(worst_meta["runtime_interval_policy"]),
                "shift_alert_flag": bool(shift_alert_flag),
                "telemetry_missing_count": telemetry_missing_count,
                "prediction_backed_telemetry": prediction_backed_telemetry,
                "monotonicity_probe": prev_meta is not None,
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
                    "max_under_coverage_gap": float(
                        max((row.get("under_coverage_gap", 0.0) for row in rows), default=0.0)
                    ),
                    "final_validity_score": float(
                        engine.state.last_validity.validity_score if engine.state.last_validity else 1.0
                    ),
                    "final_validity_status": str(
                        engine.state.last_validity.validity_status
                        if engine.state.last_validity
                        else "nominal"
                    ),
                    "shift_alert_rate": float(
                        np.mean([row.get("shift_alert_flag", False) for row in engine.state.validity_trace])
                    )
                    if engine.state.validity_trace
                    else 0.0,
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

    def tighten_action_set(
        self, uncertainty: dict[str, Any], constraints: dict[str, Any], *, cfg: dict[str, Any]
    ) -> dict[str, Any]:
        del cfg
        current_speed = _f(uncertainty.get("current_speed_mps"), 0.0)
        speed_limit = _f(constraints.get("speed_limit_mps"), uncertainty.get("speed_limit_mps", 25.0))
        lead_speed = _f(uncertainty.get("lead_speed_mps"), current_speed)
        gap_lower = _f(uncertainty.get("gap_lower_m"), 30.0)
        current_gap = _f(uncertainty.get("current_gap_m"), gap_lower)
        neighbor_count = int(_f(uncertainty.get("neighbor_count"), 0.0))
        fault_family = str(uncertainty.get("fault_family", ""))
        dt_s = _f(constraints.get("dt_s"), 0.1)
        min_headway_m = _f(constraints.get("min_headway_m"), self._min_headway_m)
        ttc_min_s = _f(constraints.get("ttc_min_s"), self._ttc_min_s)
        hard_headway_m = _f(constraints.get("hard_headway_m"), 5.0)
        hard_ttc_s = _f(constraints.get("hard_ttc_s"), 2.0)
        entry_headway_m = _f(constraints.get("entry_headway_m"), self._entry_headway_m)
        entry_ttc_s = _f(constraints.get("entry_ttc_s"), self._entry_ttc_s)
        a_lower = _f(constraints.get("accel_min_mps2"), self._accel_min)
        a_upper = _f(constraints.get("accel_max_mps2"), self._accel_max)
        active: list[str] = []
        fallback_action = {"acceleration_mps2": float(_f(constraints.get("accel_min_mps2"), self._accel_min))}
        closing_speed = max(0.0, current_speed - lead_speed)
        predicted_ttc_lower_s = float("inf") if closing_speed <= 1e-9 else gap_lower / closing_speed
        current_ttc_s = _f(uncertainty.get("current_ttc_s"), predicted_ttc_lower_s)
        effective_gap_lower = min(gap_lower, current_gap)
        if math.isfinite(current_ttc_s):
            predicted_ttc_lower_s = min(predicted_ttc_lower_s, current_ttc_s)
        validity_status = str(uncertainty.get("meta", {}).get("validity_status", "nominal"))
        shift_alert_flag = bool(uncertainty.get("meta", {}).get("shift_alert_flag", False))
        shift_score = float(np.clip(_f(uncertainty.get("meta", {}).get("shift_score"), 0.0), 0.0, 1.0))
        telemetry_missing_count = int(_f(uncertainty.get("meta", {}).get("telemetry_missing_count"), 0.0))
        prediction_backed_telemetry = bool(
            uncertainty.get("meta", {}).get("prediction_backed_telemetry", False)
        )
        monotonicity_probe = bool(uncertainty.get("meta", {}).get("monotonicity_probe", False))
        fallback_reason = ""
        fallback_region = "hold_region"
        projected_release = False
        projection_reason = ""
        allow_near_failsafe_projected_validity = False
        near_failsafe_projection_epsilon = float(max(self._near_failsafe_projection_epsilon_mps2, 0.0))
        near_failsafe_accel = float(np.clip(self._near_failsafe_accel_mps2, a_lower, a_upper))
        entry_projection_accel = float(np.clip(self._entry_projection_accel_mps2, a_lower, a_upper))
        degraded_projection_accel = float(np.clip(self._degraded_projection_accel_mps2, a_lower, a_upper))
        if monotonicity_probe:
            entry_projection_accel = near_failsafe_accel
            degraded_projection_accel = near_failsafe_accel
        projected_release_margin = float(
            min(
                effective_gap_lower - hard_headway_m,
                (predicted_ttc_lower_s - hard_ttc_s) if math.isfinite(predicted_ttc_lower_s) else 1e6,
            )
        )

        hard_projection_observed_confirmed = (
            monotonicity_probe
            or not self._hard_projection_requires_observed_confirmation
            or current_gap <= hard_headway_m + max(self._hard_projection_current_gap_buffer_m, 0.0)
            or (
                math.isfinite(current_ttc_s)
                and current_ttc_s <= hard_ttc_s + max(self._hard_projection_current_ttc_buffer_s, 0.0)
            )
        )
        entry_gap_metric = (
            current_gap
            if self._entry_projection_uses_current_state and not monotonicity_probe
            else effective_gap_lower
        )
        entry_ttc_metric = (
            current_ttc_s
            if self._entry_projection_uses_current_state and not monotonicity_probe
            else predicted_ttc_lower_s
        )

        if current_speed + a_upper * dt_s > speed_limit:
            a_upper = min(a_upper, (speed_limit - current_speed) / max(dt_s, 1e-9))
            active.append("speed_limit")

        # --- CBF longitudinal barrier ---
        cbf_alpha = 0.5
        d_safe = current_speed**2 / 12.0 + 0.2 * current_speed + 0.5
        h_value = effective_gap_lower - d_safe
        denom = 0.166 * current_speed + 0.2 + 1e-9
        cbf_a_upper = (lead_speed - current_speed + cbf_alpha * h_value) / denom
        if cbf_a_upper < a_upper:
            a_upper = cbf_a_upper
            active.append("cbf_longitudinal_barrier")

        fallback_required = False
        if telemetry_missing_count > 0 and not prediction_backed_telemetry:
            fallback_required = True
            fallback_reason = "telemetry_missing_full_brake"
            fallback_region = "telemetry_missing"
            active.append("telemetry_missing_full_brake")
        elif current_gap <= 1.0:
            fallback_required = True
            fallback_reason = "critical_headway_full_brake"
            fallback_region = "critical_headway"
            active.append("critical_headway_full_brake")
        elif math.isfinite(current_ttc_s) and current_ttc_s <= 0.0 and current_gap <= 1.0:
            fallback_required = True
            fallback_reason = "critical_ttc_full_brake"
            fallback_region = "critical_ttc"
            active.append("critical_ttc_full_brake")
        elif (
            fault_family == "delay_jitter"
            and validity_status == "watch"
            and shift_score >= 0.25
            and neighbor_count >= 8
            and current_speed <= 0.01
        ):
            fallback_required = True
            fallback_reason = "temporal_fault_non_viable_full_brake"
            fallback_region = "temporal_fault_dense_neighbors"
            active.append("temporal_fault_non_viable_full_brake")
        elif validity_status == "invalid":
            fallback_required = True
            fallback_reason = "certificate_invalid_full_brake"
            fallback_region = "certificate_invalid"
            active.append("certificate_invalid_full_brake")
        elif effective_gap_lower <= hard_headway_m and hard_projection_observed_confirmed:
            projected_release = True
            projection_reason = "hard_headway_near_failsafe_projection"
            fallback_region = "hard_headway_projection"
            allow_near_failsafe_projected_validity = True
            a_lower = a_upper = near_failsafe_accel
            projected_release_margin = max(
                1e-6,
                fallback_action["acceleration_mps2"] + near_failsafe_projection_epsilon - near_failsafe_accel,
            )
            active.append("hard_headway_near_failsafe_projection")
        elif (
            math.isfinite(predicted_ttc_lower_s)
            and predicted_ttc_lower_s <= hard_ttc_s
            and hard_projection_observed_confirmed
        ):
            projected_release = True
            projection_reason = "hard_ttc_near_failsafe_projection"
            fallback_region = "hard_ttc_projection"
            allow_near_failsafe_projected_validity = True
            a_lower = a_upper = near_failsafe_accel
            projected_release_margin = max(
                1e-6,
                fallback_action["acceleration_mps2"] + near_failsafe_projection_epsilon - near_failsafe_accel,
            )
            active.append("hard_ttc_near_failsafe_projection")
        elif validity_status == "degraded" or shift_alert_flag:
            projected_release = True
            projection_reason = "certificate_degraded_projection"
            fallback_region = "certificate_degraded_projection"
            allow_near_failsafe_projected_validity = True
            a_lower = a_upper = degraded_projection_accel
            projected_release_margin = max(
                1e-6,
                fallback_action["acceleration_mps2"]
                + near_failsafe_projection_epsilon
                - degraded_projection_accel,
            )
            active.append("certificate_degraded_projection")
        elif entry_gap_metric <= entry_headway_m:
            projected_release = True
            projection_reason = "headway_predictive_entry_projection"
            fallback_region = "tight_headway_projection"
            allow_near_failsafe_projected_validity = True
            a_lower = a_upper = entry_projection_accel
            projected_release_margin = max(
                1e-6,
                fallback_action["acceleration_mps2"]
                + near_failsafe_projection_epsilon
                - entry_projection_accel,
            )
            active.append("headway_predictive_entry_barrier")
        elif math.isfinite(entry_ttc_metric) and entry_ttc_metric <= entry_ttc_s:
            projected_release = True
            projection_reason = "ttc_predictive_entry_projection"
            fallback_region = "low_ttc_projection"
            allow_near_failsafe_projected_validity = True
            a_lower = a_upper = entry_projection_accel
            projected_release_margin = max(
                1e-6,
                fallback_action["acceleration_mps2"]
                + near_failsafe_projection_epsilon
                - entry_projection_accel,
            )
            active.append("ttc_predictive_entry_barrier")

        if fallback_required:
            a_lower = a_upper = fallback_action["acceleration_mps2"]
        elif not projected_release:
            max_next_speed = lead_speed + effective_gap_lower / max(ttc_min_s + dt_s, 1e-9)
            ttc_upper = (max_next_speed - current_speed) / max(dt_s, 1e-9)
            if ttc_upper < a_upper:
                a_upper = ttc_upper
                active.append("ttc_clamp")

        if a_lower > a_upper:
            a_lower = a_upper = fallback_action["acceleration_mps2"]
            active.append("fallback_collapse")
            fallback_required = True
            fallback_reason = "fallback_collapse"
            fallback_region = "non_viable"

        return {
            "acceleration_mps2_lower": float(a_lower),
            "acceleration_mps2_upper": float(a_upper),
            "fallback_action": fallback_action,
            "active_constraints": active,
            "projection_surface": "waymo_longitudinal_barrier",
            "viable": not fallback_required,
            "fallback_required": fallback_required,
            "fallback_reason": fallback_reason,
            "fallback_region": fallback_region,
            "projected_release": bool(projected_release and not fallback_required),
            "projection_reason": projection_reason,
            "projected_release_margin": float(projected_release_margin),
            "allow_near_failsafe_projected_validity": bool(
                allow_near_failsafe_projected_validity and not fallback_required
            ),
            "near_failsafe_projection_epsilon_mps2": float(near_failsafe_projection_epsilon),
            "entry_barrier_triggered": "headway_predictive_entry_barrier" in active,
            "widening_factor": float(uncertainty.get("meta", {}).get("widening_factor", 1.0)),
            "predicted_ttc_lower_s": predicted_ttc_lower_s,
            "contract_headway_threshold_m": min_headway_m,
            "contract_ttc_threshold_s": ttc_min_s,
            "theorem_contract": "av_brake_hold_release",
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
        if bool(tightened_set.get("fallback_required", False)) or tightened_set.get("viable") is False:
            fallback_action = dict(
                tightened_set.get("fallback_action", {"acceleration_mps2": self._accel_min})
            )
            return {
                "acceleration_mps2": float(_f(fallback_action.get("acceleration_mps2"), self._accel_min))
            }, {
                "repaired": True,
                "mode": "fallback",
                "intervention_reason": str(tightened_set.get("fallback_reason", "full_brake")),
                "fallback_region": str(tightened_set.get("fallback_region", "non_viable")),
                "fallback_required": True,
                "widening_factor": float(tightened_set.get("widening_factor", 1.0)),
                "entry_barrier_triggered": bool(tightened_set.get("entry_barrier_triggered", False)),
                "theorem_contract": str(tightened_set.get("theorem_contract", "av_brake_hold_release")),
            }
        proposal = _f(candidate_action.get("acceleration_mps2"), 0.0)
        lower = _f(tightened_set.get("acceleration_mps2_lower"), self._accel_min)
        upper = _f(tightened_set.get("acceleration_mps2_upper"), self._accel_max)
        safe = float(np.clip(proposal, lower, upper))
        repaired = abs(safe - proposal) > 1e-9
        reason = None
        if repaired:
            if "hard_headway_near_failsafe_projection" in tightened_set.get("active_constraints", []):
                reason = "hard_headway_near_failsafe_projection"
            elif "hard_ttc_near_failsafe_projection" in tightened_set.get("active_constraints", []):
                reason = "hard_ttc_near_failsafe_projection"
            elif "headway_predictive_entry_barrier" in tightened_set.get("active_constraints", []):
                reason = str(tightened_set.get("projection_reason") or "headway_predictive_entry_barrier")
            elif "ttc_clamp" in tightened_set.get("active_constraints", []):
                reason = "ttc_clamp"
            elif "speed_limit" in tightened_set.get("active_constraints", []):
                reason = "speed_limit_clamp"
            elif bool(tightened_set.get("projected_release", False)):
                reason = str(tightened_set.get("projection_reason") or "projected_release")
            else:
                reason = "acceleration_clamp"
        mode = "projection" if bool(tightened_set.get("projected_release", False)) else "hold"
        return {"acceleration_mps2": safe}, {
            "repaired": repaired,
            "mode": mode,
            "intervention_reason": reason,
            "widening_factor": float(tightened_set.get("widening_factor", 1.0)),
            "entry_barrier_triggered": bool(tightened_set.get("entry_barrier_triggered", False)),
            "fallback_required": False,
            "fallback_region": str(tightened_set.get("fallback_region", "hold_region")),
            "theorem_contract": str(tightened_set.get("theorem_contract", "av_brake_hold_release")),
            "projected_release": bool(tightened_set.get("projected_release", False)),
            "allow_single_step_projected_validity": bool(tightened_set.get("projected_release", False)),
            "allow_near_failsafe_projected_validity": bool(
                tightened_set.get("allow_near_failsafe_projected_validity", False)
            ),
            "near_failsafe_projection_epsilon_mps2": float(
                tightened_set.get("near_failsafe_projection_epsilon_mps2", 0.25)
            ),
            "projected_release_margin": float(tightened_set.get("projected_release_margin", 0.0)),
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
        reliability_w = _f(reliability.get("w_t", reliability.get("w")), 1.0)
        validity_status = str(uncertainty.get("meta", {}).get("validity_status", "nominal"))
        step_index = int(cfg.get("step_index", 0) or 0)
        validity = domain_certificate_validity_semantics(
            domain="av",
            safe_action=safe_action,
            uncertainty=uncertainty,
            reliability_w=reliability_w,
            validity_status=validity_status,
            step_index=step_index,
            repair_meta=repair_meta,
            cfg={**dict(cfg), "fallback_accel_mps2": -6.0, "max_validity_horizon_steps": 10},
        )
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
            intervention_reason=repair_meta.get("intervention_reason")
            if isinstance(repair_meta, Mapping)
            else None,
            reliability_w=reliability_w,
            drift_flag=bool(drift.get("drift", False)),
            inflation=_f(uncertainty.get("meta", {}).get("widening_factor"), 1.0),
            validity_score=uncertainty.get("meta", {}).get("validity_score"),
            adaptive_quantile=uncertainty.get("meta", {}).get("adaptive_quantile"),
            conditional_coverage_gap=uncertainty.get("meta", {}).get("conditional_coverage_gap"),
            runtime_interval_policy=uncertainty.get("meta", {}).get("runtime_interval_policy"),
            coverage_group_key=uncertainty.get("meta", {}).get("coverage_group_key"),
            shift_alert_flag=uncertainty.get("meta", {}).get("shift_alert_flag"),
            assumptions_version="waymo-av-shift-aware-v1",
            guarantee_checks_passed=validity.guarantee_checks_passed,
            guarantee_fail_reasons=list(validity.guarantee_fail_reasons),
            validity_horizon_H_t=validity.validity_horizon_H_t,
            half_life_steps=validity.half_life_steps,
            expires_at_step=validity.expires_at_step,
            validity_status=validity_status,
            runtime_surface="waymo_motion_replay_dry_run",
            closure_tier="runtime_contract_closed",
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
        cert["validity_status"] = validity_status
        cert["coverage_group_key"] = uncertainty.get("meta", {}).get("coverage_group_key")
        cert["shift_alert_flag"] = uncertainty.get("meta", {}).get("shift_alert_flag")
        cert["true_margin"] = cfg.get("true_margin")
        cert["observed_margin"] = cfg.get("observed_margin")
        cert["intervention_trace_id"] = cfg.get("intervention_trace_id", command_id)
        cert["fallback_mode"] = (
            "full_brake"
            if isinstance(repair_meta, Mapping) and repair_meta.get("mode") == "fallback"
            else "projected_release"
            if isinstance(repair_meta, Mapping) and repair_meta.get("mode") == "projection"
            else "hold"
        )
        cert["fallback_region"] = (
            repair_meta.get("fallback_region") if isinstance(repair_meta, Mapping) else None
        )
        cert["fallback_required"] = (
            bool(repair_meta.get("fallback_required", False)) if isinstance(repair_meta, Mapping) else False
        )
        cert["theorem_contract"] = (
            repair_meta.get("theorem_contract", "av_brake_hold_release")
            if isinstance(repair_meta, Mapping)
            else "av_brake_hold_release"
        )
        cert["validity_scope"] = validity.validity_scope
        cert["validity_theorem_id"] = validity.validity_theorem_id
        cert["validity_theorem_contract"] = validity.validity_theorem_contract
        cert["projected_release"] = (
            bool(repair_meta.get("projected_release", False)) if isinstance(repair_meta, Mapping) else False
        )
        cert["projected_release_margin"] = (
            float(repair_meta.get("projected_release_margin", 0.0))
            if isinstance(repair_meta, Mapping)
            else 0.0
        )
        # CBF barrier metadata
        gap_val = _f(uncertainty.get("gap_lower_m"), 30.0)
        speed_val = _f(uncertainty.get("current_speed_mps"), 0.0)
        lead_speed_val = _f(uncertainty.get("lead_speed_mps"), speed_val)
        d_safe_val = speed_val**2 / 12.0 + 0.2 * speed_val + 0.5
        h_val = gap_val - d_safe_val
        cbf_alpha = 0.5
        h_dot_val = (lead_speed_val - speed_val) + cbf_alpha * h_val
        cert["cbf_barrier_h"] = float(h_val)
        cert["cbf_barrier_h_dot"] = float(h_dot_val)
        cert["cbf_enforcement_active"] = bool(h_val < d_safe_val * 0.5)
        cert["certificate_hash"] = recompute_certificate_hash(cert)
        return cert


def load_runtime_bundles(models_dir: str | Path, *, artifact_prefix: str = "waymo_av") -> dict[str, Any]:
    models_path = Path(models_dir)
    speed_bundle = load_model_bundle(models_path / f"{artifact_prefix}_ego_speed_mps_1s_bundle.pkl")
    gap_bundle = load_model_bundle(models_path / f"{artifact_prefix}_relative_gap_m_1s_bundle.pkl")
    return {"ego_speed_mps_1s": speed_bundle, "relative_gap_m_1s": gap_bundle}


def _build_runtime_lookup(step_features: pd.DataFrame) -> dict[tuple[str, int], pd.Series]:
    return {(str(row["scenario_id"]), int(row["step_index"])): row for _, row in step_features.iterrows()}


def _action_magnitude(action: Mapping[str, Any]) -> float:
    return sum(
        abs(float(value))
        for value in action.values()
        if isinstance(value, int | float) or (hasattr(value, "__float__") and not isinstance(value, bool))
    )


@dataclass
class _RuntimeMetricAccumulator:
    n_steps: int = 0
    violation_count: int = 0
    comparable_count: int = 0
    oasg_gap_count: int = 0
    cva_correct_count: int = 0
    intervention_count: int = 0
    fallback_count: int = 0
    useful_work_total: float = 0.0
    audit_fields_present: int = 0
    audit_fields_required: int = 0
    fallback_magnitude_count: int = 0
    fallback_monotonic_count: int = 0
    previous_fallback_magnitude: float | None = None
    in_expired_certificate: bool = False
    expire_start_step: int = 0
    recovery_latency_total: float = 0.0
    recovery_latency_count: int = 0

    def update(self, records: list[StepRecord]) -> None:
        for record in records:
            self.n_steps += 1
            true_violated = bool(record.resolved_true_constraint_violated())
            observed_satisfied = record.resolved_observed_constraint_satisfied()
            if true_violated:
                self.violation_count += 1
            if observed_satisfied is not None:
                self.comparable_count += 1
                if bool(observed_satisfied) and true_violated:
                    self.oasg_gap_count += 1
            if record.certificate_valid == record.certificate_predicted_valid:
                self.cva_correct_count += 1
            if record.resolved_intervened():
                self.intervention_count += 1
            if record.resolved_fallback_used():
                self.fallback_count += 1
                magnitude = _action_magnitude(record.action)
                if (
                    self.previous_fallback_magnitude is not None
                    and magnitude <= self.previous_fallback_magnitude + 1e-9
                ):
                    self.fallback_monotonic_count += 1
                self.previous_fallback_magnitude = magnitude
                self.fallback_magnitude_count += 1
            self.useful_work_total += float(record.useful_work)
            self.audit_fields_present += int(record.audit_fields_present)
            self.audit_fields_required += int(record.audit_fields_required)
            if not record.certificate_valid and not self.in_expired_certificate:
                self.in_expired_certificate = True
                self.expire_start_step = int(record.step)
            elif record.certificate_valid and self.in_expired_certificate:
                self.recovery_latency_total += float(int(record.step) - self.expire_start_step)
                self.recovery_latency_count += 1
                self.in_expired_certificate = False

    def summary_row(self, controller: str) -> dict[str, Any]:
        n_steps = max(int(self.n_steps), 1)
        tsvr = float(self.violation_count / n_steps)
        oasg = float(self.oasg_gap_count / self.comparable_count) if self.comparable_count else 0.0
        cva = float(self.cva_correct_count / n_steps)
        intervention_rate = float(self.intervention_count / n_steps)
        fallback_activation_rate = float(self.fallback_count / n_steps)
        audit_completeness = (
            float(np.clip(self.audit_fields_present / self.audit_fields_required, 0.0, 1.0))
            if self.audit_fields_required
            else 1.0
        )
        useful_work_mean = float(self.useful_work_total / n_steps)
        descent = (
            float(self.fallback_monotonic_count / (self.fallback_magnitude_count - 1))
            if self.fallback_magnitude_count >= 2
            else 1.0
        )
        useful_work_fraction = min(1.0, float(self.useful_work_total / n_steps))
        gdq = float(useful_work_fraction * (1.0 - tsvr) * descent)
        recovery_latency = (
            float(self.recovery_latency_total / self.recovery_latency_count)
            if self.recovery_latency_count
            else 0.0
        )
        return {
            "controller": controller,
            "tsvr": tsvr,
            "oasg": oasg,
            "cva": cva,
            "gdq": gdq,
            "intervention_rate": intervention_rate,
            "fallback_activation_rate": fallback_activation_rate,
            "useful_work_total": float(self.useful_work_total),
            "useful_work_mean": useful_work_mean,
            "audit_completeness": audit_completeness,
            "recovery_latency": recovery_latency,
            "n_steps": int(self.n_steps),
        }


def _update_runtime_coverage(
    coverage_accumulators: dict[tuple[str, str, str], dict[str, float]],
    trace_rows: list[dict[str, Any]],
) -> None:
    for row in trace_rows:
        controller = str(row.get("controller", ""))
        fault_family = str(row.get("fault_family", ""))
        for target_name, lower_col, upper_col, true_col in (
            ("ego_speed_mps", "pred_ego_speed_lower_mps", "pred_ego_speed_upper_mps", "target_ego_speed_1s"),
            (
                "relative_gap_m",
                "pred_relative_gap_lower_m",
                "pred_relative_gap_upper_m",
                "target_relative_gap_1s",
            ),
        ):
            y_true = _finite_float_or_none(row.get(true_col))
            lower = _finite_float_or_none(row.get(lower_col))
            upper = _finite_float_or_none(row.get(upper_col))
            if y_true is None or lower is None or upper is None:
                continue
            acc = coverage_accumulators.setdefault(
                (controller, fault_family, target_name),
                {"n": 0.0, "covered": 0.0, "width_sum": 0.0},
            )
            acc["n"] += 1.0
            acc["covered"] += float(lower <= y_true <= upper)
            acc["width_sum"] += float(upper - lower)


def _predict_runtime_telemetry(feature_row: pd.Series, bundles: dict[str, Any]) -> dict[str, Any]:
    feature_frame = pd.DataFrame([feature_row.to_dict()])
    speed_bundle = bundles["ego_speed_mps_1s"]
    gap_bundle = bundles["relative_gap_m_1s"]
    speed_center, speed_lower, speed_upper = predict_interval_from_bundle(speed_bundle, feature_frame)
    gap_center, gap_lower, gap_upper = predict_interval_from_bundle(gap_bundle, feature_frame)
    shift_score = float(
        max(
            estimate_shift_score(speed_bundle, feature_frame)[0],
            estimate_shift_score(gap_bundle, feature_frame)[0],
        )
    )
    return {
        "pred_ego_speed_center_mps": float(speed_center[0]),
        "pred_ego_speed_lower_mps": float(speed_lower[0]),
        "pred_ego_speed_upper_mps": float(speed_upper[0]),
        "pred_relative_gap_center_m": float(gap_center[0]),
        "pred_relative_gap_lower_m": float(gap_lower[0]),
        "pred_relative_gap_upper_m": float(gap_upper[0]),
        "shift_score": shift_score,
    }


def _attach_runtime_predictions(step_features: pd.DataFrame, bundles: dict[str, Any]) -> pd.DataFrame:
    predicted = step_features.copy(deep=False)
    for column in RUNTIME_PREDICTION_COLUMNS:
        predicted[column] = np.nan
    speed_bundle = bundles["ego_speed_mps_1s"]
    gap_bundle = bundles["relative_gap_m_1s"]
    total_rows = int(len(predicted))
    for start in range(0, total_rows, RUNTIME_PREDICTION_BATCH_ROWS):
        stop = min(start + RUNTIME_PREDICTION_BATCH_ROWS, total_rows)
        feature_frame = (
            predicted.iloc[start:stop].drop(columns=list(RUNTIME_PREDICTION_COLUMNS), errors="ignore").copy()
        )
        speed_center, speed_lower, speed_upper = predict_interval_from_bundle(speed_bundle, feature_frame)
        gap_center, gap_lower, gap_upper = predict_interval_from_bundle(gap_bundle, feature_frame)
        speed_shift = estimate_shift_score(speed_bundle, feature_frame)
        gap_shift = estimate_shift_score(gap_bundle, feature_frame)
        row_index = predicted.index[start:stop]
        predicted.loc[row_index, "pred_ego_speed_center_mps"] = speed_center
        predicted.loc[row_index, "pred_ego_speed_lower_mps"] = speed_lower
        predicted.loc[row_index, "pred_ego_speed_upper_mps"] = speed_upper
        predicted.loc[row_index, "pred_relative_gap_center_m"] = gap_center
        predicted.loc[row_index, "pred_relative_gap_lower_m"] = gap_lower
        predicted.loc[row_index, "pred_relative_gap_upper_m"] = gap_upper
        predicted.loc[row_index, "shift_score"] = np.maximum(speed_shift, gap_shift)
    return predicted


def _observed_margin_from_state(state: Mapping[str, Any]) -> float:
    gap = _f(state.get("min_gap_m"), 30.0)
    closing = _f(state.get("lead_rel_speed_mps"), 0.0)
    return float(gap - max(5.0, 2.0 * closing))


def _av_runtime_policy_action(
    controller_mode: str,
    observed: Mapping[str, Any],
    candidate_action: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    accel_min = _f(constraints.get("accel_min_mps2"), -6.0)
    accel_max = _f(constraints.get("accel_max_mps2"), 3.0)
    proposal = _f(candidate_action.get("acceleration_mps2"), 0.0)
    current_speed = _f(observed.get("ego_speed_mps"), 0.0)
    lead_speed = _f(observed.get("lead_speed_mps"), current_speed)
    gap = _f(observed.get("min_gap_m"), 1e6)
    closing = max(0.0, current_speed - lead_speed)
    ttc = float("inf") if closing <= 1e-9 else gap / closing
    if controller_mode == "baseline":
        safe = float(np.clip(proposal, accel_min, accel_max))
        family = "nominal_deterministic_controller"
    elif controller_mode == "predictor_only_no_runtime":
        safe = float(np.clip(proposal, accel_min, accel_max))
        family = "predictor_only_no_runtime"
    elif controller_mode == "robust_fixed_deceleration":
        safe = float(np.clip(min(proposal, -1.5 if current_speed > 0.1 else 0.0), accel_min, accel_max))
        family = "robust_fixed_deceleration"
    elif controller_mode == "rss_cbf_filter":
        if gap <= 5.0 or (math.isfinite(ttc) and ttc <= 2.0):
            safe = accel_min
        elif gap <= 10.0 or (math.isfinite(ttc) and ttc <= 4.0):
            safe = min(proposal, -2.0)
        else:
            d_safe = current_speed**2 / 12.0 + 0.2 * current_speed + 0.5
            cbf_upper = (lead_speed - current_speed + 0.5 * (gap - d_safe)) / (
                0.166 * current_speed + 0.2 + 1e-9
            )
            safe = min(proposal, cbf_upper)
        safe = float(np.clip(safe, accel_min, accel_max))
        family = "rss_cbf_filter"
    elif controller_mode == "nonreliability_conformal_runtime":
        safe = min(proposal, -1.0) if gap <= 10.0 or (math.isfinite(ttc) and ttc <= 4.0) else proposal
        safe = float(np.clip(safe, accel_min, accel_max))
        family = "nonreliability_conformal_runtime"
    elif controller_mode == "stale_certificate_no_temporal_guard":
        safe = min(proposal, -2.5) if gap <= 10.0 or (math.isfinite(ttc) and ttc <= 4.0) else proposal
        safe = float(np.clip(safe, accel_min, accel_max))
        family = "stale_certificate_no_temporal_guard"
    else:
        raise ValueError(f"Unknown AV runtime controller mode: {controller_mode}")
    return {"acceleration_mps2": safe}, {
        "mode": "policy",
        "repaired": abs(safe - proposal) > 1e-9,
        "intervention_reason": family if abs(safe - proposal) > 1e-9 else None,
        "fallback_required": False,
        "fallback_region": family,
        "theorem_contract": "av_brake_hold_release",
        "runtime_policy_family": family,
    }


def _run_episode(
    *,
    track: WaymoReplayTrackAdapter,
    scenario_id: str,
    step_features: pd.DataFrame,
    fault_family: str,
    controller_mode: str,
    adapter: WaymoAVDomainAdapter | None = None,
    previous_certificate: Mapping[str, Any] | None = None,
    runtime_policy_config: Mapping[str, Any] | None = None,
) -> tuple[list[StepRecord], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    adapter = adapter or WaymoAVDomainAdapter({"expected_cadence_s": 0.1})
    policy_config = dict(runtime_policy_config or {})
    feature_by_step = {int(row["step_index"]): row for _, row in step_features.iterrows()}
    scenario_steps = sorted(feature_by_step)
    if not scenario_steps:
        return [], [], [], None
    use_orius = controller_mode == "orius"
    always_brake = controller_mode == "always_brake"
    track.load_scenario(scenario_id, start_step_index=min(scenario_steps))
    last_certificate = dict(previous_certificate) if previous_certificate is not None else None
    history: list[dict[str, Any]] = []
    records: list[StepRecord] = []
    trace_rows: list[dict[str, Any]] = []
    cert_rows: list[dict[str, Any]] = []

    previous_speed = None
    previous_accel = None
    for step_index in scenario_steps:
        true_state = dict(track.true_state())
        observed = dict(track.observe(true_state, {"kind": fault_family}))
        feature_row = feature_by_step[int(step_index)]
        observed.update(
            {
                "pred_ego_speed_center_mps": float(feature_row["pred_ego_speed_center_mps"]),
                "pred_ego_speed_lower_mps": float(feature_row["pred_ego_speed_lower_mps"]),
                "pred_ego_speed_upper_mps": float(feature_row["pred_ego_speed_upper_mps"]),
                "pred_relative_gap_center_m": float(feature_row["pred_relative_gap_center_m"]),
                "pred_relative_gap_lower_m": float(feature_row["pred_relative_gap_lower_m"]),
                "pred_relative_gap_upper_m": float(feature_row["pred_relative_gap_upper_m"]),
                "shift_score": float(feature_row["shift_score"]),
            }
        )
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
        observed["neighbor_instability"] = float(
            max(0.0, abs(int(feature_row["neighbor_count"]) - int(true_state.get("neighbor_count", 0)))) / 8.0
        )
        observed["observed_margin"] = _observed_margin_from_state(observed)
        constraints = {
            "speed_limit_mps": _f(true_state.get("speed_limit_mps"), 25.0),
            "accel_min_mps2": -6.0,
            "accel_max_mps2": 3.0,
            "dt_s": 0.1,
            "min_headway_m": 8.0,
            "ttc_min_s": 3.0,
            "entry_headway_m": 10.0,
            "entry_ttc_s": 4.0,
        }
        constraint_overrides = policy_config.get("constraints", {})
        if isinstance(constraint_overrides, Mapping):
            constraints.update(dict(constraint_overrides))
        for key in (
            "min_headway_m",
            "ttc_min_s",
            "entry_headway_m",
            "entry_ttc_s",
            "hard_headway_m",
            "hard_ttc_s",
        ):
            if key in policy_config:
                constraints[key] = policy_config[key]
        if always_brake:
            candidate_action = {"acceleration_mps2": -6.0}
        elif use_orius and _bool_cfg(policy_config, "align_nominal_with_runtime_projection", False):
            candidate_action = _runtime_aligned_longitudinal_controller(
                observed,
                constraints=constraints,
                policy_config=policy_config,
            )
        else:
            candidate_action = deterministic_longitudinal_controller(observed)
        latency_start = time.perf_counter_ns()
        certificate_valid = False
        certificate_predicted_valid = False
        intervened = False
        fallback_used = always_brake
        certificate = None
        audit_fields_present = 0
        audit_fields_required = len(REQUIRED_CERTIFICATE_FIELDS)
        repair_meta: dict[str, Any] = {
            "mode": "fallback" if always_brake else "hold",
            "repaired": always_brake,
            "intervention_reason": "degenerate_always_brake" if always_brake else None,
            "fallback_required": always_brake,
            "fallback_region": "always_brake" if always_brake else "hold_region",
            "theorem_contract": "av_brake_hold_release",
        }
        theorem_contracts: Mapping[str, Any] | None = None

        if use_orius:
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
                    "step_index": int(step_index),
                    "intervention_trace_id": f"{scenario_id}-{step_index}",
                },
                prev_cert_hash=(last_certificate or {}).get("certificate_hash"),
                device_id=f"ego-{true_state.get('ego_track_id')}",
                zone_id=str(true_state.get("shard_id")),
                controller="waymo_orius_dry_run",
            )
            theorem_contracts = result["theorem_contracts"]
            safe_action = dict(result["safe_action"])
            certificate = dict(result["certificate"])
            # The universal layer augments the certificate after the adapter emits
            # its initial payload, so re-hash the final mapping before chaining
            # and persistence.
            certificate["certificate_hash"] = recompute_certificate_hash(certificate)
            certificate_predicted_valid = _predict_certificate_validity(certificate, last_certificate)
            certificate_valid = _independent_certificate_validity(certificate, last_certificate)
            last_certificate = dict(certificate)
            repair_meta = dict(result["repair_meta"])
            intervened = bool(repair_meta.get("repaired", False))
            fallback_used = str(repair_meta.get("mode", "")) == "fallback"
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
            if always_brake:
                safe_action = dict(candidate_action)
            else:
                safe_action, repair_meta = _av_runtime_policy_action(
                    controller_mode, observed, candidate_action, constraints
                )
                intervened = bool(repair_meta.get("repaired", False))
                fallback_used = False
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
        audit_fields_present, audit_fields_required = _certificate_required_field_counts(certificate)
        latency_us = (time.perf_counter_ns() - latency_start) / 1_000.0
        history.append(dict(observed))
        next_true = dict(track.step(safe_action))

        true_metrics = compute_state_safety_metrics(next_true)
        observed_metrics = (
            compute_state_safety_metrics(observed)
            if observed.get("min_gap_m") is not None
            and not (isinstance(observed.get("min_gap_m"), float) and math.isnan(observed.get("min_gap_m")))
            else None
        )
        contract_status = _brake_hold_contract_status(
            true_metrics=true_metrics,
            observed_metrics=observed_metrics,
            uncertainty_meta=uncertainty_meta,
            safe_action=safe_action,
            repair_meta=repair_meta,
            certificate_valid=certificate_valid if use_orius else True,
            headway_threshold_m=float(constraints["min_headway_m"]),
            ttc_threshold_s=float(constraints["ttc_min_s"]),
            brake_accel_mps2=float(constraints["accel_min_mps2"]),
        )
        trace_id = f"{scenario_id}-{step_index}-{controller_mode}"
        if use_orius:
            theorem_trace_fields = witness_trace_fields_from_result(
                domain="av",
                trace_id=trace_id,
                theorem_contracts=theorem_contracts,
                certificate_valid=certificate_valid,
                postcondition_passed=not bool(contract_status["true_constraint_violated"]),
                post_margin=float(contract_status["true_margin"]),
            )
        else:
            theorem_trace_fields = {
                "contract_id": AV_BRAKE_HOLD_CONTRACT_ID,
                "source_theorem": "T11",
                "t11_status": "missing",
                "t11_failed_obligations": "",
                "domain_postcondition_passed": not bool(contract_status["true_constraint_violated"]),
                "domain_postcondition_failure": "non_orius_controller",
            }
        records.append(
            StepRecord(
                step=int(step_index),
                true_state=_compact_step_state(true_state),
                observed_state=_compact_step_state(observed, include_runtime=True),
                action=safe_action,
                true_constraint_violated=bool(contract_status["true_constraint_violated"]),
                observed_constraint_satisfied=contract_status["observed_constraint_satisfied"],
                true_margin=float(contract_status["true_margin"]),
                observed_margin=(
                    None
                    if contract_status["observed_margin"] is None
                    else float(contract_status["observed_margin"])
                ),
                intervened=intervened,
                fallback_used=fallback_used,
                certificate_valid=certificate_valid,
                certificate_predicted_valid=certificate_predicted_valid,
                useful_work=float(max(0.0, next_true.get("ego_x_m", 0.0) - true_state.get("ego_x_m", 0.0))),
                audit_fields_present=audit_fields_present,
                audit_fields_required=audit_fields_required,
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
                "trace_id": trace_id,
                "scenario_id": scenario_id,
                "shard_id": str(true_state.get("shard_id", "")),
                "ego_track_id": int(true_state.get("ego_track_id", 0) or 0),
                "step_index": int(step_index),
                "fault_family": fault_family,
                "controller": controller_mode,
                "candidate_acceleration_mps2": float(candidate_action["acceleration_mps2"]),
                "safe_acceleration_mps2": float(safe_action["acceleration_mps2"]),
                "intervened": bool(intervened),
                "fallback_used": bool(fallback_used),
                "repair_mode": str(repair_meta.get("mode", "hold")),
                "intervention_reason": repair_meta.get("intervention_reason"),
                "fallback_region": repair_meta.get("fallback_region"),
                "projected_release": bool(repair_meta.get("projected_release", False)),
                "projected_release_margin": float(repair_meta.get("projected_release_margin", 0.0)),
                "runtime_policy_family": str(repair_meta.get("runtime_policy_family", "")),
                "theorem_contract": repair_meta.get("theorem_contract", "av_brake_hold_release"),
                "certificate_valid": bool(certificate_valid),
                "certificate_predicted_valid": bool(certificate_predicted_valid),
                "audit_fields_present": int(audit_fields_present),
                "audit_fields_required": int(audit_fields_required),
                "true_constraint_violated": bool(contract_status["true_constraint_violated"]),
                "observed_constraint_satisfied": contract_status["observed_constraint_satisfied"],
                "true_margin": float(contract_status["true_margin"]),
                "observed_margin": contract_status["observed_margin"],
                "release_requires_fallback": bool(contract_status["release_requires_fallback"]),
                "geometric_constraint_violated": bool(true_metrics["true_constraint_violated"]),
                "min_gap_m": float(true_metrics["min_gap_m"]),
                "ttc_s": float(true_metrics["ttc_s"]),
                "latency_us": float(latency_us),
                "reliability_w": float(result["reliability_w"])
                if use_orius
                else float(np.clip(observed["reliability_proxy"], 0.05, 1.0)),
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
                "certificate_schema_version": str(certificate.get("certificate_schema_version", ""))
                if certificate is not None
                else "",
                "certificate_hash": str(certificate.get("certificate_hash", ""))
                if certificate is not None
                else "",
                "prev_hash": str(certificate.get("prev_hash", "") or "") if certificate is not None else "",
                "issuer": str(certificate.get("issuer", "")) if certificate is not None else "",
                "domain": str(certificate.get("domain", "")) if certificate is not None else "",
                "action": json.dumps(certificate.get("action", {}), sort_keys=True)
                if certificate is not None
                else "",
                "theorem_contracts": json.dumps(certificate.get("theorem_contracts", {}), sort_keys=True)
                if certificate is not None
                else "",
                "validity_scope": str(certificate.get("validity_scope", "not_certified"))
                if certificate is not None
                else "not_certified",
                "validity_theorem_id": str(certificate.get("validity_theorem_id", ""))
                if certificate is not None
                else "",
                "validity_theorem_contract": str(certificate.get("validity_theorem_contract", ""))
                if certificate is not None
                else "",
                "coverage_group_key": str(uncertainty_meta.get("coverage_group_key", "global")),
                "shift_alert_flag": bool(uncertainty_meta.get("shift_alert_flag", False)),
                **theorem_trace_fields,
            }
        )
        if certificate is not None:
            cert_rows.append(certificate)
        previous_speed = current_speed
        previous_accel = accel
    return records, trace_rows, cert_rows, last_certificate


def run_runtime_dry_run(
    *,
    replay_windows_path: str | Path,
    step_features_path: str | Path,
    models_dir: str | Path,
    out_dir: str | Path,
    max_scenarios: int | None = None,
    artifact_prefix: str = "waymo_av",
    runtime_policy_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    replay_path = Path(replay_windows_path)
    step_features, test_scenarios = _load_runtime_test_step_features(
        step_features_path,
        max_scenarios=max_scenarios,
    )
    if step_features.empty:
        raise ValueError("No test step features were available for AV runtime dry run.")
    bundles = load_runtime_bundles(models_dir, artifact_prefix=artifact_prefix)
    step_features = _attach_runtime_predictions(step_features, bundles)
    # Avoid a full-frame sort on uncapped nuPlan holdouts. Episodes sort their
    # own step indices, so global ordering is not part of the runtime contract.
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
    policy_config = dict(runtime_policy_config or {})
    (out_path / "runtime_policy_config.json").write_text(
        json.dumps(policy_config, indent=2), encoding="utf-8"
    )
    orius_adapter_cfg = {
        "expected_cadence_s": 0.1,
        "shift_aware_uncertainty": shift_cfg,
        "shift_output_dir": str(shift_out_dir),
        "runtime_policy": policy_config,
    }
    for key in ("min_headway_m", "ttc_min_s", "entry_headway_m", "entry_ttc_s"):
        if key in policy_config:
            orius_adapter_cfg[key] = policy_config[key]
    orius_adapter = WaymoAVDomainAdapter(orius_adapter_cfg)

    controller_modes = [
        "baseline",
        "rss_cbf_filter",
        "robust_fixed_deceleration",
        "predictor_only_no_runtime",
        "nonreliability_conformal_runtime",
        "stale_certificate_no_temporal_guard",
        "always_brake",
        "orius",
    ]
    metric_accumulators = {name: _RuntimeMetricAccumulator() for name in controller_modes}
    coverage_accumulators: dict[tuple[str, str, str], dict[str, float]] = {}
    trace_path = stage_dir / "runtime_traces.csv"
    trace_buffer: list[dict[str, Any]] = []
    trace_columns: list[str] | None = None
    cert_buffer: list[dict[str, Any]] = []
    certificate_count = 0
    previous_orius_certificate: dict[str, Any] | None = None
    # Pre-load only the runtime test scenarios once; full replay parquet is too
    # large to materialize for uncapped nuPlan runs.
    baseline_track = WaymoReplayTrackAdapter(replay_path, scenario_ids=test_scenarios)
    orius_track = baseline_track
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    def flush_trace_buffer() -> None:
        nonlocal trace_buffer, trace_columns
        if not trace_buffer:
            return
        frame = pd.DataFrame(trace_buffer)
        if trace_columns is None:
            trace_columns = frame.columns.tolist()
            header = True
        else:
            header = False
            for column in trace_columns:
                if column not in frame.columns:
                    frame[column] = pd.NA
            frame = frame[trace_columns]
        frame.to_csv(trace_path, mode="a", index=False, header=header)
        trace_buffer = []

    def flush_certificate_buffer() -> None:
        nonlocal cert_buffer
        if not cert_buffer:
            return
        store_certificates_batch(cert_buffer, duckdb_path=str(audit_db_path), table_name=table_name)
        cert_buffer = []

    processed_scenarios = 0
    try:
        for scenario_id, scenario_features in step_features.groupby("scenario_id", sort=False):
            fault_family = _scenario_fault_family(str(scenario_id))
            policy_outputs: dict[
                str,
                tuple[list[StepRecord], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None],
            ] = {}
            for controller_mode in controller_modes:
                if controller_mode == "orius":
                    continue
                policy_outputs[controller_mode] = _run_episode(
                    track=baseline_track,
                    scenario_id=str(scenario_id),
                    step_features=scenario_features,
                    fault_family=fault_family,
                    controller_mode=controller_mode,
                )
            orius_records, orius_traces, orius_certs, previous_orius_certificate = _run_episode(
                track=orius_track,  # reuse pre-loaded adapter
                scenario_id=str(scenario_id),
                step_features=scenario_features,
                fault_family=fault_family,
                controller_mode="orius",
                adapter=orius_adapter,
                previous_certificate=previous_orius_certificate,
                runtime_policy_config=policy_config,
            )
            for controller_mode, (records, traces, _, _) in policy_outputs.items():
                metric_accumulators[controller_mode].update(records)
                _update_runtime_coverage(coverage_accumulators, traces)
                trace_buffer.extend(traces)
            metric_accumulators["orius"].update(orius_records)
            _update_runtime_coverage(coverage_accumulators, orius_traces)
            trace_buffer.extend(orius_traces)
            if len(trace_buffer) >= 50_000:
                flush_trace_buffer()
            if orius_certs:
                cert_buffer.extend(orius_certs)
                certificate_count += int(len(orius_certs))
                if len(cert_buffer) >= 50_000:
                    flush_certificate_buffer()
            processed_scenarios += 1
            if processed_scenarios == 1 or processed_scenarios % 1_000 == 0:
                pass
        flush_trace_buffer()
        flush_certificate_buffer()
    finally:
        if gc_was_enabled:
            gc.enable()
            gc.collect()

    if not trace_path.exists():
        pd.DataFrame().to_csv(trace_path, index=False)

    summary_rows = [
        metric_accumulators[controller_name].summary_row(controller_name)
        for controller_name in controller_modes
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = stage_dir / "runtime_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    coverage_rows = []
    for (controller_name, fault_family, target_name), acc in sorted(coverage_accumulators.items()):
        n = max(float(acc["n"]), 1.0)
        coverage_rows.append(
            {
                "controller": str(controller_name),
                "fault_family": str(fault_family),
                "target": str(target_name),
                "coverage": float(acc["covered"] / n),
                "mean_width": float(acc["width_sum"] / n),
            }
        )
    coverage_path = stage_dir / "fault_family_coverage.csv"
    pd.DataFrame(
        coverage_rows, columns=["controller", "fault_family", "target", "coverage", "mean_width"]
    ).to_csv(coverage_path, index=False)
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
        shutil.rmtree(final_shift_out_dir, ignore_errors=True)
    if shift_out_dir.exists():
        shift_out_dir.replace(final_shift_out_dir)
    shift_artifacts = _remap_path_prefix(shift_artifacts, shift_out_dir, final_shift_out_dir)
    equal_domain_artifacts = _write_equal_domain_artifacts("av", out_path)

    report = {
        "runtime_summary_csv": str(final_summary_path),
        "runtime_traces_csv": str(final_trace_path),
        "fault_family_coverage_csv": str(final_coverage_path),
        "audit_db_path": str(final_audit_db_path),
        "certificate_count": int(certificate_count),
        "scenario_count": int(len(test_scenarios)),
        "shift_aware_artifacts": shift_artifacts,
        **equal_domain_artifacts,
    }
    final_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    return report
