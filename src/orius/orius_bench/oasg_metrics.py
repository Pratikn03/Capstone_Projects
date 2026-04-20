"""Canonical OASG metric surfaces used by the submission-facing papers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .real_data_loader import load_vehicle_rows

__all__ = [
    "OASGMetricResult",
    "build_submission_domain_surfaces",
    "compute_oasg_signature",
    "signature_across_domains",
    "signature_latex_table",
    "signature_report",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
BATTERY_TRACE_PATH = REPO_ROOT / "reports" / "publication" / "48h_trace_final_de.csv"
AV_RUNTIME_TRACE_PATH = REPO_ROOT / "reports" / "orius_av" / "full_corpus" / "runtime_traces.csv"


@dataclass(frozen=True)
class OASGMetricResult:
    """Summary statistics for the Observation-Action Safety Gap."""

    signature: float
    exposure_rate: float
    severity: float
    blindness: float
    n_steps: int
    bootstrap_ci_95: tuple[float, float]
    domain_name: str


@dataclass(frozen=True)
class SubmissionDomainSurface:
    """Metric-ready domain inputs for the submission scope."""

    true_states: np.ndarray
    observations: np.ndarray
    reliability_scores: np.ndarray
    safe_set_check: Callable[[np.ndarray], bool]
    distance_to_boundary: Callable[[np.ndarray], float]


def _validate_inputs(
    true_states: np.ndarray,
    observations: np.ndarray,
    reliability_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_arr = np.asarray(true_states, dtype=float)
    obs_arr = np.asarray(observations, dtype=float)
    rel_arr = np.asarray(reliability_scores, dtype=float).reshape(-1)
    if true_arr.ndim != 2 or obs_arr.ndim != 2:
        raise ValueError("true_states and observations must be 2D arrays")
    if true_arr.shape != obs_arr.shape:
        raise ValueError("true_states and observations must have identical shape")
    if rel_arr.shape[0] != true_arr.shape[0]:
        raise ValueError("reliability_scores length must match the trajectory length")
    return true_arr, obs_arr, np.clip(rel_arr, 0.0, 1.0)


def compute_oasg_signature(
    true_states: np.ndarray,
    observations: np.ndarray,
    reliability_scores: np.ndarray,
    safe_set_check: Callable[[np.ndarray], bool],
    distance_to_boundary: Callable[[np.ndarray], float],
    domain_name: str = "unnamed",
    bootstrap_samples: int = 1000,
    random_seed: int = 42,
) -> OASGMetricResult:
    """Compute the canonical weighted-mean OASG signature."""

    true_arr, obs_arr, rel_arr = _validate_inputs(true_states, observations, reliability_scores)
    obs_safe = np.asarray([bool(safe_set_check(row)) for row in obs_arr], dtype=bool)
    true_unsafe = np.asarray([not bool(safe_set_check(row)) for row in true_arr], dtype=bool)
    degraded = rel_arr < 1.0
    gap_events = obs_safe & true_unsafe & degraded

    distances = np.asarray([float(distance_to_boundary(row)) for row in true_arr], dtype=float)
    distances_positive = np.maximum(distances, 0.0)
    gap_contributions = distances_positive * gap_events.astype(float)

    signature = float(np.mean(gap_contributions))
    exposure_rate = float(np.mean(gap_events.astype(float)))
    severity = float(np.mean(distances_positive[gap_events])) if gap_events.any() else 0.0
    blindness = float(gap_events.sum() / max(1, true_unsafe.sum()))

    rng = np.random.default_rng(random_seed)
    bootstrap_values = np.empty(max(int(bootstrap_samples), 1), dtype=float)
    for idx in range(bootstrap_values.size):
        sample_index = rng.integers(0, gap_contributions.size, size=gap_contributions.size)
        bootstrap_values[idx] = float(np.mean(gap_contributions[sample_index]))
    ci_low, ci_high = np.percentile(bootstrap_values, [2.5, 97.5])

    return OASGMetricResult(
        signature=signature,
        exposure_rate=exposure_rate,
        severity=severity,
        blindness=blindness,
        n_steps=int(gap_contributions.size),
        bootstrap_ci_95=(float(ci_low), float(ci_high)),
        domain_name=domain_name,
    )


def signature_across_domains(domain_data: dict[str, dict]) -> dict[str, OASGMetricResult]:
    """Evaluate the canonical OASG metric across multiple domains."""

    return {name: compute_oasg_signature(domain_name=name, **data) for name, data in domain_data.items()}


def signature_report(results: dict[str, OASGMetricResult]) -> str:
    """Generate a text report for the submission domains."""

    lines = ["OASG Signature Report", "=" * 76]
    lines.append(
        f"{'Domain':<24} {'Signature':>10} {'Exposure':>10} {'Severity':>10} {'Blindness':>11} {'95% CI':>15}"
    )
    lines.append("-" * 76)
    for name, result in results.items():
        ci_text = f"[{result.bootstrap_ci_95[0]:.4f}, {result.bootstrap_ci_95[1]:.4f}]"
        lines.append(
            f"{name:<24} {result.signature:>10.4f} {result.exposure_rate:>10.2%} "
            f"{result.severity:>10.4f} {result.blindness:>11.2%} {ci_text:>15}"
        )
    return "\n".join(lines)


def signature_latex_table(results: dict[str, OASGMetricResult]) -> str:
    """Render a submission-ready LaTeX table."""

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Domain & Signature & Exposure & Severity & Blindness \\",
        r"\midrule",
    ]
    for name, result in results.items():
        lines.append(
            f"{name} & {result.signature:.4f} & {result.exposure_rate:.3f} & "
            f"{result.severity:.4f} & {result.blindness:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _reliability_from_jumps(states: np.ndarray, scale: float) -> np.ndarray:
    diffs = np.linalg.norm(np.diff(states, axis=0, prepend=states[:1]), axis=1)
    normalized = diffs / max(float(scale), 1e-9)
    reliability = 1.0 - np.clip(normalized, 0.0, 0.8)
    reliability[np.arange(states.shape[0]) % 17 == 0] = np.minimum(
        reliability[np.arange(states.shape[0]) % 17 == 0],
        0.45,
    )
    return np.clip(reliability, 0.15, 1.0)


def _stale_clip_observations(
    true_states: np.ndarray,
    reliability: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    observations = np.asarray(true_states, dtype=float).copy()
    for idx in range(observations.shape[0]):
        if reliability[idx] >= 0.999:
            continue
        prev_idx = max(idx - 1, 0)
        stale = 0.75 * observations[prev_idx] + 0.25 * observations[idx]
        observations[idx] = np.clip(stale, lower, upper)
    return observations


def _build_box_surface(
    states: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    reliability: np.ndarray,
) -> SubmissionDomainSurface:
    lower_arr = np.asarray(lower, dtype=float)
    upper_arr = np.asarray(upper, dtype=float)
    reliability_arr = np.asarray(reliability, dtype=float).reshape(-1)

    def safe_set_check(state: np.ndarray) -> bool:
        return bool(np.all(state >= lower_arr) and np.all(state <= upper_arr))

    def distance_to_boundary(state: np.ndarray) -> float:
        below = np.maximum(lower_arr - state, 0.0)
        above = np.maximum(state - upper_arr, 0.0)
        return float(np.max(np.concatenate([below, above])))

    return SubmissionDomainSurface(
        true_states=np.asarray(states, dtype=float),
        observations=_stale_clip_observations(states, reliability_arr, lower_arr, upper_arr),
        reliability_scores=np.clip(reliability_arr, 0.0, 1.0),
        safe_set_check=safe_set_check,
        distance_to_boundary=distance_to_boundary,
    )


def _load_trace_rows(path: Path, *, controller_field: str, controller_value: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get(controller_field) == controller_value
        ]
    if not rows:
        raise ValueError(f"No rows found in {path} for {controller_field}={controller_value!r}")
    return rows


def _battery_surface() -> SubmissionDomainSurface:
    rows = _load_trace_rows(
        BATTERY_TRACE_PATH,
        controller_field="controller",
        controller_value="deterministic_lp",
    )
    true_states = np.asarray([[float(row["soc_true_mwh"])] for row in rows], dtype=float)
    observations = np.asarray([[float(row["soc_observed_mwh"])] for row in rows], dtype=float)
    reliability = np.asarray([float(row["reliability_w"]) for row in rows], dtype=float)
    lower = np.array([0.0], dtype=float)
    upper = np.array([10000.0], dtype=float)

    def safe_set_check(state: np.ndarray) -> bool:
        return bool(lower[0] <= state[0] <= upper[0])

    def distance_to_boundary(state: np.ndarray) -> float:
        return float(max(lower[0] - state[0], state[0] - upper[0], 0.0))

    return SubmissionDomainSurface(
        true_states=true_states,
        observations=observations,
        reliability_scores=np.clip(reliability, 0.0, 1.0),
        safe_set_check=safe_set_check,
        distance_to_boundary=distance_to_boundary,
    )


def _av_surface() -> SubmissionDomainSurface:
    trace_rows = _load_trace_rows(
        AV_RUNTIME_TRACE_PATH,
        controller_field="controller",
        controller_value="baseline",
    )
    true_states = np.asarray([[float(row["true_margin"])] for row in trace_rows], dtype=float)
    observations = np.asarray([[float(row["observed_margin"])] for row in trace_rows], dtype=float)
    reliability = np.asarray([float(row["reliability_w"]) for row in trace_rows], dtype=float)

    def safe_set_check(state: np.ndarray) -> bool:
        return bool(state[0] >= 0.0)

    def distance_to_boundary(state: np.ndarray) -> float:
        return float(max(0.0, -state[0]))

    return SubmissionDomainSurface(
        true_states=true_states,
        observations=observations,
        reliability_scores=np.clip(reliability, 0.0, 1.0),
        safe_set_check=safe_set_check,
        distance_to_boundary=distance_to_boundary,
    )


def build_submission_domain_surfaces() -> dict[str, dict]:
    """Return the Battery + AV domain surfaces allowed in the current submission scope."""

    domains = {
        "Battery": _battery_surface(),
        "Autonomous Vehicles": _av_surface(),
    }
    return {
        name: {
            "true_states": payload.true_states,
            "observations": payload.observations,
            "reliability_scores": payload.reliability_scores,
            "safe_set_check": payload.safe_set_check,
            "distance_to_boundary": payload.distance_to_boundary,
        }
        for name, payload in domains.items()
    }
