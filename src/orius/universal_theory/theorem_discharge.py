"""Executable T9/T10 assumption-discharge estimators.

These helpers compute empirical evidence for the draft universal extension
theorems.  They are evidence gates, not proof generators: a domain artifact can
pass only when the observed trace data discharges the numerical assumptions
declared by the theorem promotion package.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

NOMINAL_FAULT_LABELS = {
    "",
    "none",
    "no_fault",
    "nominal",
    "baseline",
    "nominal_deterministic_controller",
}


@dataclass(frozen=True)
class DischargeThresholds:
    """Promotion thresholds shared by T9/T10 discharge builders."""

    min_rows: int = 1000
    min_positive_rate: float = 1e-6
    boundary_margin: float = 0.5
    reliability_degradation_threshold: float = 0.95
    mixing_autocorrelation_max: float = 0.99
    mixing_max_lag: int = 128
    tv_bridge_epsilon: float = 0.05
    tv_histogram_bins: int = 10
    healthcare_spo2_boundary: float = 92.0
    healthcare_alert_boundary: float = 1.0


def iter_csv_rows(path: Path, *, max_rows: int | None = None) -> Iterator[dict[str, str]]:
    """Yield rows from a CSV file without loading the whole artifact."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            yield dict(row)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _reliability(row: Mapping[str, Any]) -> float | None:
    for key in ("reliability_w", "reliability", "validity_score"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _fault_degraded(row: Mapping[str, Any]) -> bool:
    label = str(row.get("fault_family", "")).strip().lower()
    return label not in NOMINAL_FAULT_LABELS


def _unsafe(row: Mapping[str, Any], thresholds: DischargeThresholds) -> bool:
    if _as_bool(row.get("true_constraint_violated")):
        return True
    if _as_bool(row.get("is_critical")) or _as_bool(row.get("critical_event")):
        return True
    alert_level = _safe_float(row.get("alert_level"))
    if alert_level is not None and alert_level >= thresholds.healthcare_alert_boundary:
        return True
    target = _safe_float(row.get("target"))
    if target is not None and target <= thresholds.healthcare_spo2_boundary:
        return True
    spo2 = _safe_float(row.get("spo2_pct"))
    return bool(spo2 is not None and spo2 <= thresholds.healthcare_spo2_boundary)


def _near_boundary(row: Mapping[str, Any], thresholds: DischargeThresholds) -> bool:
    if _unsafe(row, thresholds):
        return True
    for key in ("true_margin", "observed_margin", "projected_release_margin"):
        value = _safe_float(row.get(key))
        if value is not None and abs(value) <= thresholds.boundary_margin:
            return True
    target = _safe_float(row.get("target"))
    if target is not None and target <= thresholds.healthcare_spo2_boundary + thresholds.boundary_margin:
        return True
    spo2 = _safe_float(row.get("spo2_pct"))
    return bool(spo2 is not None and spo2 <= thresholds.healthcare_spo2_boundary + thresholds.boundary_margin)


def _observation_scalar(row: Mapping[str, Any]) -> float | None:
    for key in (
        "observed_margin",
        "observed_value",
        "forecast_spo2_pct",
        "forecast",
        "alert_level",
        "target_relative_gap_1s",
        "min_gap_m",
        "target",
        "spo2_pct",
        "true_margin",
    ):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    return None


def _autocorrelation_at_lag(values: list[float], lag: int) -> float | None:
    if lag < 1 or len(values) <= lag + 1:
        return None
    arr = np.asarray(values, dtype=float)
    if float(np.var(arr)) <= 1e-12:
        return None
    left = arr[:-lag]
    right = arr[lag:]
    if float(np.var(left)) <= 1e-12 or float(np.var(right)) <= 1e-12:
        return None
    corr = float(np.corrcoef(left, right)[0, 1])
    if not math.isfinite(corr):
        return None
    return corr


def _autocorrelation(values: list[float]) -> float | None:
    return _autocorrelation_at_lag(values, 1)


def _mixing_proxy(
    *,
    degradation_flags: list[float],
    reliability_deficits: list[float],
    thresholds: DischargeThresholds,
) -> dict[str, Any]:
    """Estimate a finite empirical mixing witness from reliability/degradation traces.

    The theorem needs a finite-domain witness, not necessarily a lag-1 witness.
    We therefore scan a bounded lag window and accept the first lag whose
    empirical autocorrelation drops under the configured threshold.  Reliability
    deficit is preferred because an all-degraded domain can still carry varying
    reliability information; the binary degraded flag is retained as a fallback.
    """

    max_lag = max(1, int(thresholds.mixing_max_lag))
    candidates = (
        ("reliability_deficit", reliability_deficits),
        ("degraded_indicator", degradation_flags),
    )
    degenerate_constant = False
    best_payload: dict[str, Any] | None = None

    for sequence_name, values in candidates:
        if len(values) < 3:
            continue
        arr = np.asarray(values, dtype=float)
        if float(np.var(arr)) <= 1e-12:
            degenerate_constant = True
            continue

        lag_payloads: list[dict[str, float | int]] = []
        selected: tuple[int, float] | None = None
        for lag in range(1, min(max_lag, len(values) - 2) + 1):
            corr = _autocorrelation_at_lag(values, lag)
            if corr is None:
                continue
            lag_payloads.append({"lag": int(lag), "autocorrelation": float(corr)})
            if selected is None and abs(corr) <= thresholds.mixing_autocorrelation_max:
                selected = (lag, corr)

        lag1 = next(
            (entry["autocorrelation"] for entry in lag_payloads if entry["lag"] == 1),
            None,
        )
        if selected is not None:
            selected_lag, selected_corr = selected
            return {
                "method": "bounded_multi_lag_autocorrelation",
                "sequence": sequence_name,
                "lag1_autocorrelation": lag1,
                "selected_lag": int(selected_lag),
                "selected_autocorrelation": float(selected_corr),
                "lag_autocorrelations": lag_payloads,
                "finite_mixing_proxy": True,
                "max_abs_autocorrelation": thresholds.mixing_autocorrelation_max,
                "max_lag": max_lag,
            }

        if best_payload is None or len(lag_payloads) > len(best_payload.get("lag_autocorrelations", [])):
            best_payload = {
                "method": "bounded_multi_lag_autocorrelation",
                "sequence": sequence_name,
                "lag1_autocorrelation": lag1,
                "selected_lag": None,
                "selected_autocorrelation": None,
                "lag_autocorrelations": lag_payloads,
                "finite_mixing_proxy": False,
                "max_abs_autocorrelation": thresholds.mixing_autocorrelation_max,
                "max_lag": max_lag,
            }

    if degenerate_constant:
        return {
            "method": "degenerate_constant_sequence",
            "sequence": "reliability_deficit_or_degraded_indicator",
            "lag1_autocorrelation": None,
            "selected_lag": 0,
            "selected_autocorrelation": 0.0,
            "lag_autocorrelations": [],
            "finite_mixing_proxy": True,
            "max_abs_autocorrelation": thresholds.mixing_autocorrelation_max,
            "max_lag": max_lag,
        }

    if best_payload is not None:
        return best_payload

    return {
        "method": "insufficient_sequence_length",
        "sequence": "none",
        "lag1_autocorrelation": None,
        "selected_lag": None,
        "selected_autocorrelation": None,
        "lag_autocorrelations": [],
        "finite_mixing_proxy": False,
        "max_abs_autocorrelation": thresholds.mixing_autocorrelation_max,
        "max_lag": max_lag,
    }


def _empirical_tv(left_values: list[float], right_values: list[float], bins: int) -> float | None:
    if not left_values or not right_values:
        return None
    values = np.asarray([*left_values, *right_values], dtype=float)
    if values.size == 0:
        return None
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if abs(max_value - min_value) <= 1e-12:
        return 0.0
    left_hist, edges = np.histogram(
        left_values, bins=max(2, int(bins)), range=(min_value, max_value), density=False
    )
    right_hist, _ = np.histogram(right_values, bins=edges, density=False)
    left_mass = left_hist.astype(float) / max(float(np.sum(left_hist)), 1.0)
    right_mass = right_hist.astype(float) / max(float(np.sum(right_hist)), 1.0)
    return float(0.5 * np.sum(np.abs(left_mass - right_mass)))


def _blocker(parts: list[str]) -> str:
    return "; ".join(parts)


def compute_t9_discharge_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    domain: str,
    artifact_source: str,
    thresholds: DischargeThresholds | None = None,
    artifact_exists: bool = True,
) -> dict[str, Any]:
    """Compute the empirical T9 assumption-discharge payload."""
    cfg = thresholds or DischargeThresholds()
    total_rows = 0
    usable_rows = 0
    reliability_invalid = 0
    degradation_count = 0
    boundary_count = 0
    witness_deficit_sum = 0.0
    witness_count = 0
    degradation_flags: list[float] = []
    reliability_deficits: list[float] = []

    for row in rows:
        total_rows += 1
        rel = _reliability(row)
        if rel is None or not (0.0 <= rel <= 1.0):
            reliability_invalid += 1
            continue
        usable_rows += 1
        degraded = rel < cfg.reliability_degradation_threshold or _fault_degraded(row) or _unsafe(row, cfg)
        boundary = _near_boundary(row, cfg)
        deficit = max(0.0, 1.0 - rel)
        degradation_flags.append(1.0 if degraded else 0.0)
        reliability_deficits.append(deficit)
        if degraded:
            degradation_count += 1
        if boundary:
            boundary_count += 1
        if degraded and boundary:
            witness_deficit_sum += deficit
            if deficit > 0.0:
                witness_count += 1

    degradation_rate = float(degradation_count / usable_rows) if usable_rows else 0.0
    boundary_rate = float(boundary_count / usable_rows) if usable_rows else 0.0
    witness_constant = float(witness_deficit_sum / usable_rows) if usable_rows else 0.0
    mixing_proxy = _mixing_proxy(
        degradation_flags=degradation_flags,
        reliability_deficits=reliability_deficits,
        thresholds=cfg,
    )
    finite_mixing = bool(mixing_proxy["finite_mixing_proxy"])

    blockers: list[str] = []
    if usable_rows < cfg.min_rows:
        blockers.append(f"min_rows: usable_rows={usable_rows} < {cfg.min_rows}")
    if reliability_invalid:
        blockers.append(f"reliability_sequence: invalid_or_missing_rows={reliability_invalid}")
    if degradation_rate <= cfg.min_positive_rate:
        blockers.append("degradation_rate: no persistent degradation signal")
    if boundary_rate <= cfg.min_positive_rate:
        blockers.append("boundary_reachability: no reachable boundary/unsafe-side mass")
    if witness_constant <= cfg.min_positive_rate:
        blockers.append("witness_constant: no positive reliability-degradation witness")
    if not finite_mixing:
        blockers.append("mixing_bridge: no finite bounded-lag geometric/phi-mixing proxy")

    ready = not blockers
    return {
        "theorem_id": "T9",
        "domain": str(domain),
        "artifact_source": str(artifact_source),
        "artifact_exists": bool(artifact_exists),
        "n_rows": int(total_rows),
        "n_usable_rows": int(usable_rows),
        "thresholds": asdict(cfg),
        "degradation_rate": degradation_rate,
        "boundary_reachability_rate": boundary_rate,
        "witness_constant": witness_constant,
        "witness_positive_count": int(witness_count),
        "mixing_proxy": {
            **mixing_proxy,
            "max_abs_lag1_autocorrelation": cfg.mixing_autocorrelation_max,
        },
        "applicability_status": "three_domain_empirical_discharge"
        if ready
        else "blocked_empirical_discharge",
        "witness_constant_status": "witness_constant_discharged"
        if witness_constant > cfg.min_positive_rate
        else "missing_positive_witness_constant",
        "degradation_rate_status": "degradation_persistence_discharged"
        if degradation_rate > cfg.min_positive_rate
        else "missing_degradation_persistence",
        "boundary_reachability_status": "boundary_reachability_discharged"
        if boundary_rate > cfg.min_positive_rate
        else "missing_boundary_reachability",
        "mixing_bridge_status": "geometric_phi_mixing_proxy_discharged"
        if finite_mixing
        else "missing_statistical_phi_mixing_estimate",
        "constants_status": "domain_witness_constants_discharged"
        if ready
        else "missing_domain_witness_constants",
        "assumptions_status": "A10b_A11_empirically_discharged_for_domain"
        if ready
        else "A10b_A11_not_discharged_for_domain",
        "promotion_ready": ready,
        "blocker": "" if ready else _blocker(blockers),
    }


def compute_t10_discharge_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    domain: str,
    artifact_source: str,
    thresholds: DischargeThresholds | None = None,
    artifact_exists: bool = True,
    auxiliary_unsafe_rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute the empirical T10 assumption-discharge payload."""
    cfg = thresholds or DischargeThresholds()
    total_rows = 0
    auxiliary_total_rows = 0
    usable_rows = 0
    reliability_invalid = 0
    auxiliary_invalid = 0
    boundary_count = 0
    unsafe_count = 0
    auxiliary_unsafe_count = 0
    safe_count = 0
    reliability_sum = 0.0
    unsafe_observations: list[float] = []
    safe_observations: list[float] = []

    def ingest_row(row: Mapping[str, Any], *, force_unsafe: bool = False) -> bool:
        nonlocal usable_rows
        nonlocal boundary_count
        nonlocal unsafe_count
        nonlocal auxiliary_unsafe_count
        nonlocal safe_count
        nonlocal reliability_sum
        rel = _reliability(row)
        obs = _observation_scalar(row)
        if rel is None or obs is None or not (0.0 <= rel <= 1.0):
            return False
        unsafe = bool(force_unsafe or _unsafe(row, cfg))
        if force_unsafe and not _unsafe(row, cfg):
            return False
        usable_rows += 1
        reliability_sum += rel
        boundary = _near_boundary(row, cfg) or unsafe
        if boundary:
            boundary_count += 1
        if unsafe:
            unsafe_count += 1
            if force_unsafe:
                auxiliary_unsafe_count += 1
            unsafe_observations.append(obs)
        else:
            safe_count += 1
            safe_observations.append(obs)
        return True

    for row in rows:
        total_rows += 1
        if not ingest_row(row):
            reliability_invalid += 1

    if auxiliary_unsafe_rows is not None:
        for row in auxiliary_unsafe_rows:
            auxiliary_total_rows += 1
            if not ingest_row(row, force_unsafe=True):
                auxiliary_invalid += 1

    mean_reliability = float(reliability_sum / usable_rows) if usable_rows else 0.0
    unsafe_mass = float(unsafe_count / usable_rows) if usable_rows else 0.0
    safe_mass = float(safe_count / usable_rows) if usable_rows else 0.0
    boundary_mass = float(boundary_count / usable_rows) if usable_rows else 0.0
    tv_estimate = _empirical_tv(unsafe_observations, safe_observations, cfg.tv_histogram_bins)
    tv_bound = min(1.0, mean_reliability + cfg.tv_bridge_epsilon)
    tv_passed = bool(tv_estimate is not None and tv_estimate <= tv_bound)
    le_cam_lower_bound = (
        float(0.5 * unsafe_mass * max(0.0, 1.0 - min(float(tv_estimate), 1.0)))
        if tv_estimate is not None
        else 0.0
    )

    blockers: list[str] = []
    if usable_rows < cfg.min_rows:
        blockers.append(f"min_rows: usable_rows={usable_rows} < {cfg.min_rows}")
    if reliability_invalid:
        blockers.append(f"reliability_sequence: invalid_or_missing_rows={reliability_invalid}")
    if auxiliary_invalid:
        blockers.append(f"auxiliary_unsafe_law: invalid_or_unmarked_rows={auxiliary_invalid}")
    if unsafe_mass <= cfg.min_positive_rate:
        blockers.append("unsafe_boundary_mass: no unsafe-side boundary mass p_t")
    if safe_mass <= cfg.min_positive_rate:
        blockers.append("boundary_testing_subproblem: missing safe-side comparison mass")
    if boundary_mass <= cfg.min_positive_rate:
        blockers.append("boundary_testing_subproblem: no boundary-testing mass")
    if not tv_passed:
        observed = "missing" if tv_estimate is None else f"{tv_estimate:.6f}"
        blockers.append(f"tv_bridge: empirical_tv={observed} exceeds reliability bound {tv_bound:.6f}")
    if tv_estimate is None:
        blockers.append("le_cam_inputs: missing paired safe/unsafe observation laws")

    ready = not blockers
    return {
        "theorem_id": "T10",
        "domain": str(domain),
        "artifact_source": str(artifact_source),
        "artifact_exists": bool(artifact_exists),
        "n_rows": int(total_rows),
        "n_primary_rows": int(total_rows),
        "n_auxiliary_rows": int(auxiliary_total_rows),
        "n_auxiliary_unsafe_rows": int(auxiliary_unsafe_count),
        "n_usable_rows": int(usable_rows),
        "thresholds": asdict(cfg),
        "mean_reliability": mean_reliability,
        "unsafe_boundary_mass": unsafe_mass,
        "safe_side_mass": safe_mass,
        "boundary_mass": boundary_mass,
        "tv_bridge": {
            "estimate": tv_estimate,
            "bound": tv_bound,
            "epsilon": cfg.tv_bridge_epsilon,
            "passed": tv_passed,
        },
        "le_cam_lower_bound": le_cam_lower_bound,
        "applicability_status": "boundary_testing_empirical_discharge"
        if ready
        else "blocked_boundary_testing_discharge",
        "tv_bridge_status": "tv_bridge_discharged" if tv_passed else "missing_domain_pair_observation_laws",
        "unsafe_boundary_mass_status": "unsafe_boundary_mass_discharged"
        if unsafe_mass > cfg.min_positive_rate
        else "missing_positive_p_t_artifact",
        "reliability_sequence_status": "reliability_sequence_valid"
        if reliability_invalid == 0 and usable_rows
        else "missing_or_invalid_reliability_sequence",
        "boundary_testing_subproblem_status": "boundary_testing_subproblem_constructed"
        if safe_mass > cfg.min_positive_rate and unsafe_mass > cfg.min_positive_rate
        else "not_constructed",
        "constants_status": "domain_boundary_constants_discharged"
        if ready
        else "boundary_mass_supplied_explicitly_not_universal",
        "assumptions_status": "A13_empirically_discharged_for_domain"
        if ready
        else "A13_not_three_domain_discharged",
        "promotion_ready": ready,
        "blocker": "" if ready else _blocker(blockers),
    }


def compute_t9_discharge_from_csv(
    path: Path,
    *,
    domain: str,
    artifact_source: str,
    thresholds: DischargeThresholds | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Compute a T9 artifact from a CSV trace file."""
    return compute_t9_discharge_from_rows(
        iter_csv_rows(path, max_rows=max_rows),
        domain=domain,
        artifact_source=artifact_source,
        thresholds=thresholds,
        artifact_exists=path.exists(),
    )


def compute_t10_discharge_from_csv(
    path: Path,
    *,
    domain: str,
    artifact_source: str,
    thresholds: DischargeThresholds | None = None,
    max_rows: int | None = None,
    auxiliary_unsafe_path: Path | None = None,
    auxiliary_unsafe_rows: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute a T10 artifact from a CSV trace file."""
    aux_rows = auxiliary_unsafe_rows
    if aux_rows is None and auxiliary_unsafe_path is not None:
        aux_rows = iter_csv_rows(auxiliary_unsafe_path)
    return compute_t10_discharge_from_rows(
        iter_csv_rows(path, max_rows=max_rows),
        domain=domain,
        artifact_source=artifact_source,
        thresholds=thresholds,
        artifact_exists=path.exists(),
        auxiliary_unsafe_rows=aux_rows,
    )


__all__ = [
    "DischargeThresholds",
    "compute_t9_discharge_from_csv",
    "compute_t9_discharge_from_rows",
    "compute_t10_discharge_from_csv",
    "compute_t10_discharge_from_rows",
    "iter_csv_rows",
]
