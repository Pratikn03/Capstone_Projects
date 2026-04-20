#!/usr/bin/env python3
"""Property Gap Closure: 3-domain experimental evaluation.

Runs all 6 Property Gap Closure modules against real data from battery
(OPSD energy), autonomous vehicle (Waymo-derived RSS), and healthcare
(BIDMC vitals) domains.  Produces LaTeX tables and CSV curve data.

Usage::

    .venv/bin/python3 scripts/run_property_gap_experiments.py

Outputs land in ``reports/property_gap_results/``.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path so imports resolve
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# Bypass dc3s/__init__.py which eagerly imports torch-dependent modules.
# We only need lightweight submodules (no torch required).
import importlib
import types

_dc3s_stub = types.ModuleType("orius.dc3s")
_dc3s_stub.__path__ = [str(_ROOT / "src" / "orius" / "dc3s")]
_dc3s_stub.__package__ = "orius.dc3s"
sys.modules.setdefault("orius.dc3s", _dc3s_stub)

from orius.orius_bench.oasg_metrics import compute_oasg_signature
from orius.dc3s.brownian_half_life import (
    certificate_half_life,
    empirical_half_life,
    validity_probability,
)
from orius.universal_theory.no_free_safety import construct_counterexample
from orius.dc3s.reliability_weighted_cp import calibrate_rwcp, predict_rwcp
from orius.dc3s.inflation_law_derived import (
    derived_k_q,
    derived_inflation,
    inflation_curve,
    verify_heuristic_vs_derived,
)
from orius.dc3s.reliability_constraints import (
    evaluate_constraint,
    constraint_tightening_curve,
    linear_margin,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_DIR = _ROOT / "reports" / "property_gap_results"
CURVES_DIR = OUT_DIR / "curves"
SEED = 42
RNG = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Config constants (from configs/dc3s*.yaml)
# ---------------------------------------------------------------------------
BATTERY_CFG = dict(
    k_q=0.5, k_d=0.3, alpha=0.10, cadence_s=3600, capacity_mwh=200.0,
    soc_min=0.1, soc_max=0.9,
)
AV_CFG = dict(
    k_q=0.5, k_d=0.3, alpha=0.10, cadence_s=0.25,
    speed_limit_mps=15.0, min_headway_m=12.0,
)
HC_CFG = dict(
    k_q=0.60, k_d=0.40, alpha=0.10, cadence_s=1.0,
    spo2_min=95.0, hr_min=50.0, hr_max=120.0, rr_min=5.0, rr_max=30.0,
)

FAULT_REGIMES = ["clean", "mild_noise", "moderate_noise", "staleness", "adversarial"]


# ===================================================================== #
#  DATA LOADING & FAULT INJECTION                                       #
# ===================================================================== #

@dataclass
class DomainData:
    """Container for domain-specific experiment inputs."""
    name: str
    true_states: np.ndarray       # (T, d)
    observations: dict[str, np.ndarray]   # fault_name -> (T, d)
    reliability: dict[str, np.ndarray]    # fault_name -> (T,)
    safe_set_check: Callable[[np.ndarray], bool]
    distance_to_boundary: Callable[[np.ndarray], float]
    step_sec: float
    k_q_heuristic: float
    alpha: float
    # For No Free Safety
    dynamics: Callable[[np.ndarray, np.ndarray], np.ndarray]
    initial_state: np.ndarray
    drift_per_step: float


def _load_battery() -> DomainData:
    """Load battery data and simulate SOC trajectory."""
    feat_path = _ROOT / "data" / "processed" / "features.parquet"
    df = pd.read_parquet(feat_path)
    load = df["load_mw"].dropna().values[:2000]  # first 2000 hours
    T = len(load)

    # Simulate SOC as a bounded random walk driven by load changes.
    # Scale so per-step changes are ~0.01 (physically: small grid imbalance
    # relative to a 200 MWh storage system).
    load_diff = np.diff(load, prepend=load[0])
    scale = float(np.std(load_diff)) * 20.0  # gives |delta| ~ 0.01
    if scale == 0:
        scale = 1.0
    soc = np.empty(T)
    soc[0] = 0.5
    for i in range(1, T):
        delta = -load_diff[i] / scale
        soc[i] = np.clip(soc[i - 1] + delta, 0.0, 1.0)

    true_states = soc.reshape(-1, 1)
    sigma_d = float(np.std(np.diff(soc)))

    # Fault regimes
    obs: dict[str, np.ndarray] = {}
    rel: dict[str, np.ndarray] = {}

    # Clean
    obs["clean"] = true_states.copy()
    rel["clean"] = np.ones(T)

    # Mild noise σ=0.02
    noise_mild = RNG.normal(0, 0.02, (T, 1))
    obs["mild_noise"] = np.clip(true_states + noise_mild, 0, 1)
    rel["mild_noise"] = np.full(T, 0.90)

    # Moderate noise σ=0.05
    noise_mod = RNG.normal(0, 0.05, (T, 1))
    obs["moderate_noise"] = np.clip(true_states + noise_mod, 0, 1)
    rel["moderate_noise"] = np.full(T, 0.70)

    # Staleness (freeze for 5 steps)
    stale_obs = true_states.copy()
    stale_rel = np.ones(T)
    for start in range(0, T - 5, 20):
        stale_obs[start + 1 : start + 6] = stale_obs[start]
        stale_rel[start + 1 : start + 6] = 0.30
    obs["staleness"] = stale_obs
    rel["staleness"] = stale_rel

    # Adversarial bias +0.03
    obs["adversarial"] = np.clip(true_states + 0.03, 0, 1)
    rel["adversarial"] = np.full(T, 0.60)

    soc_min, soc_max = BATTERY_CFG["soc_min"], BATTERY_CFG["soc_max"]

    def safe_set(x: np.ndarray) -> bool:
        v = float(x[0]) if x.ndim > 0 else float(x)
        return soc_min <= v <= soc_max

    def dist_boundary(x: np.ndarray) -> float:
        v = float(x[0]) if x.ndim > 0 else float(x)
        if soc_min <= v <= soc_max:
            return -min(v - soc_min, soc_max - v)  # negative inside
        return max(soc_min - v, v - soc_max)

    def dynamics_battery(x: np.ndarray, a: np.ndarray) -> np.ndarray:
        return np.clip(x + a, 0.0, 1.0)

    return DomainData(
        name="battery",
        true_states=true_states,
        observations=obs,
        reliability=rel,
        safe_set_check=safe_set,
        distance_to_boundary=dist_boundary,
        step_sec=BATTERY_CFG["cadence_s"],
        k_q_heuristic=BATTERY_CFG["k_q"],
        alpha=BATTERY_CFG["alpha"],
        dynamics=dynamics_battery,
        initial_state=np.array([0.5]),
        drift_per_step=sigma_d,
    )


def _load_av() -> DomainData:
    """Load Waymo-derived AV trajectories.

    Set AV_CRASH_DATA=1 to use the crash/near-crash dataset instead.
    """
    use_crash = os.environ.get("AV_CRASH_DATA", "0") == "1"
    if use_crash:
        av_path = _ROOT / "data" / "orius_av" / "crash" / "processed" / "crash_trajectories_orius.csv"
        print(f"  [AV] Using crash dataset: {av_path}")
    else:
        av_path = _ROOT / "data" / "orius_av" / "av" / "processed" / "av_trajectories_orius.csv"

    df = pd.read_csv(av_path)

    # Pick trajectories with ≥ 50 steps
    counts = df.groupby("vehicle_id").size()
    valid_ids = counts[counts >= 50].index.tolist()
    if not valid_ids:
        # Fallback: use synthetic data if Waymo-derived is too short
        av_path = _ROOT / "data" / "av" / "processed" / "av_trajectories_orius.csv"
        df = pd.read_csv(av_path)
        counts = df.groupby("vehicle_id").size()
        valid_ids = counts[counts >= 50].index.tolist()

    # When using crash dataset, prefer actual crash scenarios
    if use_crash and "is_crash_scenario" in df.columns:
        crash_ids = [v for v in valid_ids if df.loc[df["vehicle_id"] == v, "is_crash_scenario"].any()]
        normal_ids = [v for v in valid_ids if v not in crash_ids]
        # Mix: 7 crash + 3 normal (or whatever is available)
        pick = crash_ids[:7] + normal_ids[:3]
        if len(pick) < 10:
            pick += [v for v in valid_ids if v not in pick][:10 - len(pick)]
        valid_ids = pick
        print(f"  [AV] Crash mix: {len([v for v in valid_ids[:10] if v in crash_ids])} crash + "
              f"{len([v for v in valid_ids[:10] if v not in crash_ids])} normal")

    # Concatenate first 10 valid trajectories (up to ~2000 steps)
    frames = []
    for vid in valid_ids[:10]:
        frames.append(df[df["vehicle_id"] == vid].sort_values("step"))
    sub = pd.concat(frames, ignore_index=True)
    T = len(sub)

    speed = sub["speed_mps"].values
    pos = sub["position_m"].values
    lead_pos = sub["lead_position_m"].values
    gap = lead_pos - pos

    true_states = np.column_stack([speed, gap])  # (T, 2)
    sigma_speed = float(np.std(np.diff(speed)))
    sigma_gap = float(np.std(np.diff(gap)))
    sigma_d = float(np.sqrt(sigma_speed**2 + sigma_gap**2))

    obs: dict[str, np.ndarray] = {}
    rel: dict[str, np.ndarray] = {}

    obs["clean"] = true_states.copy()
    rel["clean"] = np.ones(T)

    # GPS noise on position → affects gap; use larger noise to stress boundary
    gps_noise = RNG.normal(0, 3.0, T)
    noisy_gap = gap + gps_noise
    obs["mild_noise"] = np.column_stack([speed, noisy_gap])
    rel["mild_noise"] = np.full(T, 0.85)

    # Speed bias: +5 m/s pushes some near the limit, plus random noise
    speed_noise = RNG.normal(5.0, 2.0, T)
    gap_noise_mod = RNG.normal(0, 5.0, T)
    obs["moderate_noise"] = np.column_stack([speed + speed_noise, gap + gap_noise_mod])
    rel["moderate_noise"] = np.full(T, 0.70)

    # Staleness (freeze 4 steps)
    stale_obs = true_states.copy()
    stale_rel = np.ones(T)
    for start in range(0, T - 4, 16):
        stale_obs[start + 1 : start + 5] = stale_obs[start]
        stale_rel[start + 1 : start + 5] = 0.25
    obs["staleness"] = stale_obs
    rel["staleness"] = stale_rel

    # Combined: GPS noise + speed bias + random
    adv_speed_noise = RNG.normal(5.0, 3.0, T)
    obs["adversarial"] = np.column_stack([speed + adv_speed_noise, noisy_gap])
    rel["adversarial"] = np.full(T, 0.50)

    # Use a tighter safe-set for the experiment to stress-test boundaries:
    # actual speeds are 0-18 m/s with p90=14.4, so 15 m/s limit puts ~9%
    # near/across boundary.  Headway at 12 m puts ~16% near boundary.
    slimit = AV_CFG["speed_limit_mps"]
    min_gap = AV_CFG["min_headway_m"]

    def safe_set(x: np.ndarray) -> bool:
        return float(x[0]) < slimit and float(x[1]) > min_gap

    def dist_boundary(x: np.ndarray) -> float:
        d_speed = float(x[0]) - slimit   # positive means overspeeding
        d_gap = min_gap - float(x[1])    # positive means gap too small
        worst = max(d_speed, d_gap)
        return worst  # positive outside, negative inside

    dt = AV_CFG["cadence_s"]

    def dynamics_av(x: np.ndarray, a: np.ndarray) -> np.ndarray:
        # x = [speed, gap], a = [accel]
        new_speed = max(0.0, float(x[0]) + float(a[0]) * dt)
        new_gap = float(x[1]) - float(a[0]) * dt * dt * 0.5  # approximate
        return np.array([new_speed, new_gap])

    return DomainData(
        name="av",
        true_states=true_states,
        observations=obs,
        reliability=rel,
        safe_set_check=safe_set,
        distance_to_boundary=dist_boundary,
        step_sec=AV_CFG["cadence_s"],
        k_q_heuristic=AV_CFG["k_q"],
        alpha=AV_CFG["alpha"],
        dynamics=dynamics_av,
        initial_state=np.array([float(speed[0]), float(gap[0])]),
        drift_per_step=sigma_d,
    )


def _load_healthcare() -> DomainData:
    """Load healthcare vitals.

    Set HC_MIMIC3=1 to use MIMIC-III PhysioNet data instead of BIDMC.
    """
    use_mimic = os.environ.get("HC_MIMIC3", "0") == "1"
    if use_mimic:
        hc_path = _ROOT / "data" / "healthcare" / "mimic3" / "processed" / "mimic3_healthcare_orius.csv"
        print(f"  [HC] Using MIMIC-III dataset: {hc_path}")
    else:
        hc_path = _ROOT / "data" / "healthcare" / "processed" / "healthcare_bidmc_orius.csv"

    df = pd.read_csv(hc_path)
    # Drop rows where any key column is NaN
    df = df.dropna(subset=["target", "forecast", "hr", "resp"]).copy()
    df["reliability"] = df["reliability"].fillna(1.0)

    # Use first patients with enough data, up to 2000 rows
    patient_ids = df["patient_id"].unique()
    frames = []
    total = 0
    for pid in patient_ids:
        chunk = df[df["patient_id"] == pid]
        frames.append(chunk)
        total += len(chunk)
        if total >= 2000:
            break
    sub = pd.concat(frames, ignore_index=True).head(2000)
    T = len(sub)

    spo2_true = sub["target"].values.astype(float)
    spo2_obs = sub["forecast"].values.astype(float)
    hr = sub["hr"].values.astype(float)
    resp = sub["resp"].values.astype(float)
    w_real = sub["reliability"].values.astype(float).clip(0.05, 1.0)

    true_states = np.column_stack([spo2_true, hr, resp])  # (T, 3)
    base_obs = np.column_stack([spo2_obs, hr, resp])

    sigma_d = float(np.std(np.diff(spo2_true)))

    obs: dict[str, np.ndarray] = {}
    rel: dict[str, np.ndarray] = {}

    # Clean: use actual forecast, real w_t
    obs["clean"] = base_obs.copy()
    rel["clean"] = w_real.copy()

    # Mild degradation: scale reliability × 0.8
    obs["mild_noise"] = base_obs.copy()
    rel["mild_noise"] = (w_real * 0.8).clip(0.05, 1.0)

    # Moderate degradation: add SpO2 noise + scale reliability × 0.6
    spo2_noisy = spo2_obs + RNG.normal(0, 1.5, T)
    obs["moderate_noise"] = np.column_stack([spo2_noisy, hr, resp])
    rel["moderate_noise"] = (w_real * 0.6).clip(0.05, 1.0)

    # Staleness: freeze SpO2 forecast in 10-step windows
    stale_spo2 = spo2_obs.copy()
    stale_rel = w_real.copy()
    for start in range(0, T - 10, 30):
        stale_spo2[start + 1 : start + 11] = stale_spo2[start]
        stale_rel[start + 1 : start + 11] = 0.15
    obs["staleness"] = np.column_stack([stale_spo2, hr, resp])
    rel["staleness"] = stale_rel

    # Adversarial: bias SpO2 up by 2 %, scale reliability × 0.4
    obs["adversarial"] = np.column_stack([spo2_obs + 2.0, hr, resp])
    rel["adversarial"] = (w_real * 0.4).clip(0.05, 1.0)

    spo2_min = HC_CFG["spo2_min"]
    hr_min, hr_max = HC_CFG["hr_min"], HC_CFG["hr_max"]
    rr_min, rr_max = HC_CFG["rr_min"], HC_CFG["rr_max"]

    def safe_set(x: np.ndarray) -> bool:
        return (
            float(x[0]) >= spo2_min
            and hr_min <= float(x[1]) <= hr_max
            and rr_min <= float(x[2]) <= rr_max
        )

    def dist_boundary(x: np.ndarray) -> float:
        margins = [
            float(x[0]) - spo2_min,
            float(x[1]) - hr_min,
            hr_max - float(x[1]),
            float(x[2]) - rr_min,
            rr_max - float(x[2]),
        ]
        min_margin = min(margins)
        return -min_margin  # positive outside, negative inside

    def dynamics_hc(x: np.ndarray, a: np.ndarray) -> np.ndarray:
        return x + a

    return DomainData(
        name="healthcare",
        true_states=true_states,
        observations=obs,
        reliability=rel,
        safe_set_check=safe_set,
        distance_to_boundary=dist_boundary,
        step_sec=HC_CFG["cadence_s"],
        k_q_heuristic=HC_CFG["k_q"],
        alpha=HC_CFG["alpha"],
        dynamics=dynamics_hc,
        initial_state=np.array([float(spo2_true[0]), float(hr[0]), float(resp[0])]),
        drift_per_step=sigma_d,
    )


# ===================================================================== #
#  EXPERIMENT 1: OASG SIGNATURE                                         #
# ===================================================================== #

def experiment_oasg(domains: list[DomainData]) -> list[dict[str, Any]]:
    """Compute OASG signatures across all domains and fault regimes."""
    rows: list[dict[str, Any]] = []
    for dd in domains:
        for fault in FAULT_REGIMES:
            res = compute_oasg_signature(
                true_states=dd.true_states,
                observations=dd.observations[fault],
                reliability_scores=dd.reliability[fault],
                safe_set_check=dd.safe_set_check,
                distance_to_boundary=dd.distance_to_boundary,
                domain_name=dd.name,
                bootstrap_samples=2000,
                random_seed=SEED,
            )
            rows.append(dict(
                domain=dd.name, fault=fault,
                sigma_oasg=res.signature,
                exposure_rate=res.exposure_rate,
                severity=res.severity,
                blindness=res.blindness,
                ci_low=res.bootstrap_ci_95[0],
                ci_high=res.bootstrap_ci_95[1],
                n_steps=res.n_steps,
            ))
            print(f"  OASG  {dd.name:12s} {fault:18s}  σ={res.signature:.6f}  "
                  f"exp={res.exposure_rate:.4f}  sev={res.severity:.4f}  "
                  f"blind={res.blindness:.4f}  CI=[{res.bootstrap_ci_95[0]:.6f}, {res.bootstrap_ci_95[1]:.6f}]")
    return rows


# ===================================================================== #
#  EXPERIMENT 2: CERTIFICATE HALF-LIFE                                  #
# ===================================================================== #

def experiment_half_life(domains: list[DomainData]) -> list[dict[str, Any]]:
    """Theoretical and empirical certificate half-life comparison."""
    rows: list[dict[str, Any]] = []
    for dd in domains:
        # Compute initial margin: median distance to boundary for safe states
        dists = np.array([dd.distance_to_boundary(x) for x in dd.true_states])
        # distance_to_boundary returns positive outside; for safe states it's negative
        margins = -dists[dists < 0]  # inside safe set → flip sign
        if len(margins) == 0:
            margins = np.array([0.01])
        d_0 = float(np.median(margins))
        sigma_d = dd.drift_per_step
        if sigma_d <= 0:
            sigma_d = 1e-6

        # Theoretical
        th = certificate_half_life(
            initial_margin=d_0,
            disturbance_std=sigma_d,
            step_duration_sec=dd.step_sec,
            domain_name=dd.name,
        )

        # Empirical: simulate certificate issuance per safe step, track violations
        violation_times: list[float] = []
        cert_start: int | None = None
        for t, x in enumerate(dd.true_states):
            if dd.safe_set_check(x):
                if cert_start is None:
                    cert_start = t
            else:
                if cert_start is not None:
                    violation_times.append(float(t - cert_start))
                    cert_start = None
        if cert_start is not None:
            # Certificate never violated → inf
            violation_times.append(float("inf"))

        emp = empirical_half_life(np.array(violation_times if violation_times else [float("inf")]))

        ratio = emp / th.half_life_steps if th.half_life_steps > 0 else float("inf")
        rows.append(dict(
            domain=dd.name,
            d_0=d_0,
            sigma_d=sigma_d,
            tau_half_theory=th.half_life_steps,
            tau_half_theory_sec=th.half_life_seconds,
            tau_half_empirical=emp,
            ratio=ratio,
        ))
        print(f"  HalfLife  {dd.name:12s}  d₀={d_0:.4f}  σ_d={sigma_d:.6f}  "
              f"τ½_th={th.half_life_steps:.1f} steps  τ½_emp={emp:.1f} steps  ratio={ratio:.2f}")

        # Validity decay curves
        t_range = np.arange(1, 1001)
        probs = np.array([validity_probability(float(t), d_0, sigma_d) for t in t_range])
        curve_df = pd.DataFrame({"t_steps": t_range, "validity_prob": probs})
        curve_df.to_csv(CURVES_DIR / f"validity_decay_{dd.name}.csv", index=False)

    return rows


# ===================================================================== #
#  EXPERIMENT 3: NO FREE SAFETY COUNTEREXAMPLES                         #
# ===================================================================== #

def experiment_no_free_safety(domains: list[DomainData]) -> list[dict[str, Any]]:
    """Construct counterexamples for quality-ignorant controllers."""
    rows: list[dict[str, Any]] = []
    for dd in domains:
        # Quality-ignorant controller: zero action
        dim = dd.initial_state.shape[0]

        def qi_controller(obs: np.ndarray, _dim=dim) -> np.ndarray:
            return np.zeros(_dim)

        result = construct_counterexample(
            quality_ignorant_controller=qi_controller,
            dynamics=dd.dynamics,
            safe_set_check=dd.safe_set_check,
            initial_state=dd.initial_state,
            horizon=50,
            random_seed=SEED,
        )

        divergence = float(np.linalg.norm(
            np.asarray(result.true_trajectory_faulty[-1])
            - np.asarray(result.true_trajectory_clean[-1])
        ))

        rows.append(dict(
            domain=dd.name,
            safe_clean=result.safety_outcome_clean,
            safe_stale=result.safety_outcome_faulty,
            divergence_norm=divergence,
            x_clean_final=np.asarray(result.true_trajectory_clean[-1]).tolist(),
            x_stale_final=np.asarray(result.true_trajectory_faulty[-1]).tolist(),
            conclusion=result.conclusion,
        ))
        print(f"  NoFree  {dd.name:12s}  clean={'SAFE' if result.safety_outcome_clean else 'UNSAFE'}  "
              f"stale={'SAFE' if result.safety_outcome_faulty else 'UNSAFE'}  divergence={divergence:.4f}")

    return rows


# ===================================================================== #
#  EXPERIMENT 4: RWCP vs STANDARD CP                                    #
# ===================================================================== #

def experiment_rwcp(domains: list[DomainData]) -> list[dict[str, Any]]:
    """Compare reliability-weighted vs standard conformal prediction."""
    alphas = [0.01, 0.05, 0.10, 0.20]
    rows: list[dict[str, Any]] = []

    for dd in domains:
        # Use clean true vs moderate_noise observations for nonconformity
        true_1d = dd.true_states[:, 0]
        obs_1d = dd.observations["moderate_noise"][:, 0]
        w_t = dd.reliability["moderate_noise"]

        T = len(true_1d)
        cal_end = int(0.6 * T)
        cal_true, test_true = true_1d[:cal_end], true_1d[cal_end:]
        cal_obs, test_obs = obs_1d[:cal_end], obs_1d[cal_end:]
        cal_w, test_w = w_t[:cal_end], w_t[cal_end:]

        cal_scores = np.abs(cal_true - cal_obs)
        test_scores = np.abs(test_true - test_obs)

        for alpha in alphas:
            # Standard CP: uniform quantile
            q_level = min(np.ceil((cal_end + 1) * (1 - alpha)) / cal_end, 1.0)
            std_threshold = float(np.quantile(cal_scores, q_level))
            std_lower, std_upper = test_obs - std_threshold, test_obs + std_threshold
            std_coverage = float(np.mean((test_true >= std_lower) & (test_true <= std_upper)))
            std_width = float(np.mean(std_upper - std_lower))

            # RWCP
            rwcp_result = calibrate_rwcp(
                nonconformity_scores=cal_scores,
                reliability_scores=cal_w,
                alpha=alpha,
                domain_name=dd.name,
            )
            rwcp_lower, rwcp_upper = predict_rwcp(test_obs, rwcp_result.quantile_threshold)
            rwcp_coverage = float(np.mean((test_true >= rwcp_lower) & (test_true <= rwcp_upper)))
            rwcp_width = float(np.mean(rwcp_upper - rwcp_lower))

            rows.append(dict(
                domain=dd.name, alpha=alpha, method="Standard CP",
                coverage=std_coverage, width=std_width, ess=float(cal_end),
            ))
            rows.append(dict(
                domain=dd.name, alpha=alpha, method="RWCP",
                coverage=rwcp_coverage, width=rwcp_width,
                ess=rwcp_result.effective_sample_size,
            ))
            print(f"  RWCP  {dd.name:12s}  α={alpha:.2f}  "
                  f"StdCP cov={std_coverage:.4f} w={std_width:.4f}  |  "
                  f"RWCP  cov={rwcp_coverage:.4f} w={rwcp_width:.4f} ESS={rwcp_result.effective_sample_size:.1f}")

    return rows


# ===================================================================== #
#  EXPERIMENT 5: DERIVED INFLATION LAW                                  #
# ===================================================================== #

def experiment_inflation(domains: list[DomainData]) -> list[dict[str, Any]]:
    """Compare derived k_q vs heuristic k_q."""
    rows: list[dict[str, Any]] = []

    for dd in domains:
        # Estimate observation noise σ from moderate_noise regime
        true_1d = dd.true_states[:, 0]
        obs_1d = dd.observations["moderate_noise"][:, 0]
        errors = np.abs(true_1d - obs_1d)
        sigma = float(np.std(errors))
        if sigma <= 0:
            sigma = 1e-6
        q_hat = float(np.quantile(errors, 1 - dd.alpha))
        if q_hat <= 0:
            q_hat = 1e-6

        comparison = verify_heuristic_vs_derived(
            k_q_heuristic=dd.k_q_heuristic,
            sigma=sigma,
            alpha=dd.alpha,
            q_hat=q_hat,
        )

        # Evaluate coverage under both inflation laws on test data
        T = len(true_1d)
        cal_end = int(0.6 * T)
        test_true = true_1d[cal_end:]
        test_obs = obs_1d[cal_end:]
        test_w = dd.reliability["moderate_noise"][cal_end:]

        # Heuristic inflation
        heur_gamma = 1.0 + dd.k_q_heuristic * (1.0 - test_w)
        heur_width = q_hat * heur_gamma
        heur_cov = float(np.mean(np.abs(test_true - test_obs) <= heur_width))

        # Derived inflation
        kq_d = comparison["k_q_derived"]
        deriv_gamma = 1.0 + kq_d * (1.0 - test_w)
        deriv_width = q_hat * deriv_gamma
        deriv_cov = float(np.mean(np.abs(test_true - test_obs) <= deriv_width))

        rows.append(dict(
            domain=dd.name,
            sigma=sigma,
            q_hat=q_hat,
            k_q_heuristic=dd.k_q_heuristic,
            k_q_derived=comparison["k_q_derived"],
            relative_dev=comparison["relative_deviation"],
            coverage_heuristic=heur_cov,
            coverage_derived=deriv_cov,
            mean_width_heuristic=float(np.mean(heur_width)),
            mean_width_derived=float(np.mean(deriv_width)),
        ))
        print(f"  Inflation  {dd.name:12s}  σ={sigma:.4f}  q̂={q_hat:.4f}  "
              f"k_q_h={dd.k_q_heuristic:.3f}  k_q_d={comparison['k_q_derived']:.3f}  "
              f"reldev={comparison['relative_deviation']:.3f}  "
              f"cov_h={heur_cov:.4f}  cov_d={deriv_cov:.4f}")

        # Inflation curve
        w_arr = np.linspace(0, 1, 101)
        gamma_arr = inflation_curve(w_arr, sigma, dd.alpha, q_hat)
        curve_df = pd.DataFrame({"w": w_arr, "gamma": gamma_arr})
        curve_df.to_csv(CURVES_DIR / f"inflation_{dd.name}.csv", index=False)

    return rows


# ===================================================================== #
#  EXPERIMENT 6: RELIABILITY CONSTRAINTS                                #
# ===================================================================== #

def experiment_constraints(domains: list[DomainData]) -> list[dict[str, Any]]:
    """Evaluate reliability-conditioned constraint tightening."""
    k_values = [0.5, 1.0, 2.0, 5.0]
    rows: list[dict[str, Any]] = []

    for dd in domains:
        # h(x) = -dist_boundary(x)  (positive inside safe set)
        dists = np.array([dd.distance_to_boundary(x) for x in dd.true_states])
        h_nominals = -dists  # positive inside, negative outside
        w_t = dd.reliability["moderate_noise"]
        T = len(h_nominals)

        for k in k_values:
            nominal_violations = 0
            effective_violations = 0
            early_warnings = 0

            for i in range(T):
                h_nom = float(h_nominals[i])
                cr = evaluate_constraint(h_nom, float(w_t[i]), k=k)
                nom_violated = h_nom < 0
                eff_violated = not cr.satisfied

                if nom_violated:
                    nominal_violations += 1
                if eff_violated:
                    effective_violations += 1
                if not nom_violated and eff_violated:
                    early_warnings += 1

            prevention_rate = early_warnings / max(1, nominal_violations) if nominal_violations > 0 else 0.0

            rows.append(dict(
                domain=dd.name, k=k,
                nominal_violations=nominal_violations,
                effective_violations=effective_violations,
                early_warnings=early_warnings,
                prevention_rate=prevention_rate,
                T=T,
            ))
            print(f"  Constraint  {dd.name:12s}  k={k:.1f}  "
                  f"nom_viol={nominal_violations}  eff_viol={effective_violations}  "
                  f"early_warn={early_warnings}  prevention={prevention_rate:.3f}")

        # Tightening curves (using median h_nominal)
        h_median = float(np.median(h_nominals[h_nominals > 0])) if np.any(h_nominals > 0) else 1.0
        w_arr = np.linspace(0, 1, 101)
        for k in k_values:
            h_eff_arr = constraint_tightening_curve(w_arr, h_median, k=k)
            curve_df = pd.DataFrame({"w": w_arr, f"h_eff_k{k}": h_eff_arr})
            curve_df.to_csv(CURVES_DIR / f"tightening_{dd.name}_k{k}.csv", index=False)

    return rows


# ===================================================================== #
#  LATEX TABLE GENERATION                                               #
# ===================================================================== #

def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def _build_table1(rows: list[dict]) -> str:
    """OASG Signature table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment 1: OASG Signature across domains and fault regimes.}",
        r"\label{tab:oasg}",
        r"\small",
        r"\begin{tabular}{ll rrrrr}",
        r"\toprule",
        r"Domain & Fault & $\sigma_{\mathrm{OASG}}$ & Exposure & Severity & Blindness & 95\% CI \\",
        r"\midrule",
    ]
    for r in rows:
        ci = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        lines.append(
            f"  {_latex_escape(r['domain'])} & {_latex_escape(r['fault'])} & "
            f"{r['sigma_oasg']:.6f} & {r['exposure_rate']:.4f} & "
            f"{r['severity']:.4f} & {r['blindness']:.4f} & {ci} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_table2(rows: list[dict]) -> str:
    """Certificate half-life table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment 2: Certificate half-life --- theoretical vs.\ empirical.}",
        r"\label{tab:halflife}",
        r"\begin{tabular}{l rr rr r}",
        r"\toprule",
        r"Domain & $d_0$ & $\sigma_d$ & $\tau_{1/2}^{\mathrm{th}}$ (steps) & $\tau_{1/2}^{\mathrm{emp}}$ (steps) & Ratio \\",
        r"\midrule",
    ]
    for r in rows:
        emp_str = f"{r['tau_half_empirical']:.1f}" if np.isfinite(r['tau_half_empirical']) else r"$\infty$"
        ratio_str = f"{r['ratio']:.2f}" if np.isfinite(r['ratio']) else r"$\infty$"
        lines.append(
            f"  {_latex_escape(r['domain'])} & {r['d_0']:.4f} & {r['sigma_d']:.6f} & "
            f"{r['tau_half_theory']:.1f} & {emp_str} & {ratio_str} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_table3(rows: list[dict]) -> str:
    """No Free Safety table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment 3: No Free Safety --- quality-ignorant counterexamples.}",
        r"\label{tab:nofreesafety}",
        r"\begin{tabular}{l cc r}",
        r"\toprule",
        r"Domain & Clean safe? & Stale safe? & $\|\Delta x\|$ \\",
        r"\midrule",
    ]
    for r in rows:
        clean = r"\checkmark" if r["safe_clean"] else r"$\times$"
        stale = r"\checkmark" if r["safe_stale"] else r"$\times$"
        lines.append(
            f"  {_latex_escape(r['domain'])} & {clean} & {stale} & "
            f"{r['divergence_norm']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_table4(rows: list[dict]) -> str:
    """RWCP vs Standard CP table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment 4: RWCP vs.\ Standard CP coverage and width.}",
        r"\label{tab:rwcp}",
        r"\small",
        r"\begin{tabular}{ll r rrr}",
        r"\toprule",
        r"Domain & $\alpha$ & Method & Coverage & Width & ESS \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"  {_latex_escape(r['domain'])} & {r['alpha']:.2f} & {r['method']} & "
            f"{r['coverage']:.4f} & {r['width']:.4f} & {r['ess']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_table5(rows: list[dict]) -> str:
    """Derived Inflation Law table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment 5: Derived inflation law vs.\ heuristic.}",
        r"\label{tab:inflation}",
        r"\begin{tabular}{l rr rr rr}",
        r"\toprule",
        r"Domain & $k_q^{\mathrm{heur}}$ & $k_q^{\mathrm{deriv}}$ & Rel.\ Dev. & Cov.\ (heur) & Cov.\ (deriv) & Width (h/d) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"  {_latex_escape(r['domain'])} & {r['k_q_heuristic']:.3f} & "
            f"{r['k_q_derived']:.3f} & {r['relative_dev']:.3f} & "
            f"{r['coverage_heuristic']:.4f} & {r['coverage_derived']:.4f} & "
            f"{r['mean_width_heuristic']:.4f}/{r['mean_width_derived']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def _build_table6(rows: list[dict]) -> str:
    """Reliability Constraints table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Experiment 6: Reliability-conditioned constraint tightening.}",
        r"\label{tab:constraints}",
        r"\begin{tabular}{lr rrrr}",
        r"\toprule",
        r"Domain & $k$ & Nom.\ Viol. & Eff.\ Viol. & Early Warn. & Prev.\ Rate \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"  {_latex_escape(r['domain'])} & {r['k']:.1f} & "
            f"{r['nominal_violations']} & {r['effective_violations']} & "
            f"{r['early_warnings']} & {r['prevention_rate']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ===================================================================== #
#  CROSS-DOMAIN SUMMARY                                                 #
# ===================================================================== #

def cross_domain_summary(
    oasg_rows: list[dict],
    hl_rows: list[dict],
    nfs_rows: list[dict],
    rwcp_rows: list[dict],
    infl_rows: list[dict],
    constr_rows: list[dict],
) -> str:
    """Print a human-readable cross-domain summary."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("CROSS-DOMAIN SUMMARY")
    lines.append("=" * 72)

    # OASG: mean signature per fault
    lines.append("\n--- OASG Signature (mean across domains) ---")
    for fault in FAULT_REGIMES:
        vals = [r["sigma_oasg"] for r in oasg_rows if r["fault"] == fault]
        lines.append(f"  {fault:18s}  mean σ_OASG = {np.mean(vals):.6f}")

    # Half-life ratios
    lines.append("\n--- Certificate Half-Life (empirical / theoretical) ---")
    for r in hl_rows:
        ratio_s = f"{r['ratio']:.2f}" if np.isfinite(r['ratio']) else "∞"
        lines.append(f"  {r['domain']:12s}  ratio = {ratio_s}")

    # No Free Safety
    lines.append("\n--- No Free Safety (counterexample divergence) ---")
    for r in nfs_rows:
        lines.append(f"  {r['domain']:12s}  ‖Δx‖ = {r['divergence_norm']:.4f}  "
                      f"clean={'safe' if r['safe_clean'] else 'UNSAFE'}  "
                      f"stale={'safe' if r['safe_stale'] else 'UNSAFE'}")

    # RWCP improvement at α=0.10
    lines.append("\n--- RWCP Coverage Improvement at α=0.10 ---")
    for domain in ["battery", "av", "healthcare"]:
        std_rows = [r for r in rwcp_rows
                    if r["domain"] == domain and r["alpha"] == 0.10 and r["method"] == "Standard CP"]
        rwcp_r = [r for r in rwcp_rows
                  if r["domain"] == domain and r["alpha"] == 0.10 and r["method"] == "RWCP"]
        if std_rows and rwcp_r:
            delta_cov = rwcp_r[0]["coverage"] - std_rows[0]["coverage"]
            delta_w = rwcp_r[0]["width"] - std_rows[0]["width"]
            lines.append(f"  {domain:12s}  Δcov = {delta_cov:+.4f}  Δwidth = {delta_w:+.4f}")

    # Inflation law deviation
    lines.append("\n--- Inflation Law: k_q heuristic vs derived ---")
    for r in infl_rows:
        lines.append(f"  {r['domain']:12s}  k_q_h={r['k_q_heuristic']:.3f}  "
                      f"k_q_d={r['k_q_derived']:.3f}  reldev={r['relative_dev']:.1%}")

    # Constraints at k=2.0
    lines.append("\n--- Constraint Tightening (k=2.0) ---")
    for r in constr_rows:
        if r["k"] == 2.0:
            lines.append(f"  {r['domain']:12s}  early_warnings={r['early_warnings']}  "
                          f"prevention_rate={r['prevention_rate']:.3f}")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


# ===================================================================== #
#  MAIN                                                                 #
# ===================================================================== #

def main() -> None:
    print("=" * 72)
    print("PROPERTY GAP CLOSURE: 3-DOMAIN EXPERIMENTAL EVALUATION")
    print("=" * 72)

    # Create output dirs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)

    # Phase A: Load data
    print("\n[Phase A] Loading domain data and injecting faults ...")
    battery = _load_battery()
    print(f"  Battery:    {battery.true_states.shape[0]} steps, SOC range "
          f"[{battery.true_states.min():.3f}, {battery.true_states.max():.3f}]")

    av = _load_av()
    print(f"  AV:         {av.true_states.shape[0]} steps, speed range "
          f"[{av.true_states[:, 0].min():.1f}, {av.true_states[:, 0].max():.1f}] m/s")

    hc = _load_healthcare()
    print(f"  Healthcare: {hc.true_states.shape[0]} steps, SpO2 range "
          f"[{hc.true_states[:, 0].min():.1f}, {hc.true_states[:, 0].max():.1f}]")

    domains = [battery, av, hc]

    # Phase B: OASG Signature
    print("\n[Phase B] Experiment 1: OASG Signature")
    oasg_rows = experiment_oasg(domains)

    # Phase C: Certificate Half-Life
    print("\n[Phase C] Experiment 2: Certificate Half-Life")
    hl_rows = experiment_half_life(domains)

    # Phase D: No Free Safety
    print("\n[Phase D] Experiment 3: No Free Safety Counterexamples")
    nfs_rows = experiment_no_free_safety(domains)

    # Phase E: RWCP vs Standard CP
    print("\n[Phase E] Experiment 4: RWCP vs Standard CP")
    rwcp_rows = experiment_rwcp(domains)

    # Phase F: Derived Inflation Law
    print("\n[Phase F] Experiment 5: Derived Inflation Law")
    infl_rows = experiment_inflation(domains)

    # Phase G: Reliability Constraints
    print("\n[Phase G] Experiment 6: Reliability-Conditioned Constraints")
    constr_rows = experiment_constraints(domains)

    # Phase H: Output
    print("\n[Phase H] Generating output ...")

    # LaTeX tables
    latex_content = "\n\n".join([
        r"% Auto-generated by run_property_gap_experiments.py",
        r"% Date: " + "2026-04-18",
        r"\usepackage{booktabs}",
        "",
        _build_table1(oasg_rows),
        _build_table2(hl_rows),
        _build_table3(nfs_rows),
        _build_table4(rwcp_rows),
        _build_table5(infl_rows),
        _build_table6(constr_rows),
    ])
    (OUT_DIR / "all_tables.tex").write_text(latex_content, encoding="utf-8")
    print(f"  LaTeX tables → {OUT_DIR / 'all_tables.tex'}")

    # JSON results
    all_results = dict(
        oasg=oasg_rows,
        half_life=hl_rows,
        no_free_safety=nfs_rows,
        rwcp=rwcp_rows,
        inflation=infl_rows,
        constraints=constr_rows,
    )
    (OUT_DIR / "results.json").write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )
    print(f"  JSON results → {OUT_DIR / 'results.json'}")

    # Summary
    summary = cross_domain_summary(oasg_rows, hl_rows, nfs_rows, rwcp_rows, infl_rows, constr_rows)
    print(summary)
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    print(f"\nAll outputs in {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
