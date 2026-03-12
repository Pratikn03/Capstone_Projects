"""CLI and orchestration for CPSBench-IoT runs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gridpulse.dc3s.certificate import compute_config_hash, compute_model_hash, make_certificate
from gridpulse.dc3s.calibration import build_uncertainty_set, build_uncertainty_set_kappa
from gridpulse.dc3s.drift import PageHinkleyDetector
from gridpulse.dc3s.ftit import update as update_ftit_state
from gridpulse.dc3s.guarantee_checks import evaluate_guarantee_checks
from gridpulse.dc3s.quality import compute_reliability
from gridpulse.dc3s.rac_cert import RACCertModel, compute_dispatch_sensitivity, normalize_sensitivity
from gridpulse.dc3s.shield import repair_action
from gridpulse.forecasting.uncertainty.cqr import RegimeCQR
from gridpulse.forecasting.uncertainty.conformal import load_conformal
from gridpulse.optimizer import optimize_dispatch
from gridpulse.optimizer.robust_dispatch import (
    CVaRDispatchConfig,
    RobustDispatchConfig,
    optimize_cvar_dispatch,
    optimize_robust_dispatch,
)

from .baselines import aci_conformal_dispatch, scenario_robust_dispatch
from .scenario_mpc import scenario_mpc_dispatch
from .metrics import compute_all_metrics
from .plant import BatteryPlant
from .scenarios import FAULT_COLUMNS, generate_episode
from .telemetry_soc import SOCTelemetryChannel, SOCTelemetryFaultConfig


REQUIRED_OUTPUTS = (
    "dc3s_main_table.csv",
    "dc3s_fault_breakdown.csv",
    "cpsbench_merged_sweep.csv",
    "calibration_plot.png",
    "violation_vs_cost_curve.png",
    "fig_violation_rate.png",
    "fig_violation_severity_p95.png",
    "fig_true_soc_violation_vs_dropout.png",
    "fig_true_soc_severity_p95_vs_dropout.png",
    "dc3s_run_summary.json",
)

FAULT_SWEEP_LEVELS: dict[str, list[float]] = {
    "dropout": [0.0, 0.05, 0.10, 0.20, 0.30],
    "delay_seconds": [0.0, 1.0, 5.0, 15.0],
    "out_of_order": [0.0, 0.05, 0.15],
    "spike_sigma": [0.0, 1.0, 2.0, 3.0],
}


@dataclass
class _DC3SLoopState:
    detector: PageHinkleyDetector
    prev_event: Mapping[str, Any] | None = None
    prev_hash: str | None = None
    prev_inflation: float | None = None
    last_net_mw: float = 0.0
    sigma_sq: float | None = None
    adaptive_state: dict[str, Any] = field(default_factory=dict)


def _deep_update(base: dict[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _load_optimization_cfg() -> dict[str, Any]:
    for p in ("configs/optimization.yaml", "configs/optimize.yaml"):
        payload = _load_yaml(p)
        if payload:
            return payload
    return {}


def _resolve_dc3s_cfg_path() -> Path:
    env_path = os.getenv("GRIDPULSE_DC3S_CONFIG", "").strip()
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
    return Path("configs/dc3s.yaml")


def _load_dc3s_cfg() -> dict[str, Any]:
    payload = _load_yaml(_resolve_dc3s_cfg_path())
    dc3s = payload.get("dc3s", {}) if isinstance(payload, dict) else {}
    if not isinstance(dc3s, dict):
        dc3s = {}
    dc3s.setdefault("ambiguity", {})

    env_lambda = os.getenv("GRIDPULSE_DC3S_LAMBDA_MW")
    if env_lambda:
        try:
            dc3s["ambiguity"]["lambda_mw"] = float(env_lambda)
        except ValueError:
            pass

    env_quantile = os.getenv("GRIDPULSE_DC3S_LAMBDA_QUANTILE")
    if env_quantile:
        try:
            dc3s["ambiguity"]["lambda_quantile"] = float(env_quantile)
        except ValueError:
            pass

    env_learn = os.getenv("GRIDPULSE_DC3S_LEARN_LAMBDA")
    if env_learn:
        dc3s["ambiguity"]["learn_lambda_from_quantile"] = env_learn.strip().lower() in {"1", "true", "yes", "on"}
    return dc3s


def _load_uncertainty_cfg() -> dict[str, Any]:
    payload = _load_yaml("configs/uncertainty.yaml")
    return payload if isinstance(payload, dict) else {}


def _battery_constraints(cfg: dict[str, Any]) -> dict[str, float]:
    battery = dict(cfg.get("battery", {}))
    capacity = float(battery.get("capacity_mwh", 100.0))
    max_power = float(battery.get("max_power_mw", 50.0))
    min_soc = float(battery.get("min_soc_mwh", 0.0))
    max_soc = float(battery.get("max_soc_mwh", capacity))
    init_soc = float(battery.get("initial_soc_mwh", capacity * 0.5))

    eff = float(battery.get("efficiency", 0.95))
    charge_eff = float(battery.get("charge_efficiency", eff))
    discharge_eff = float(battery.get("discharge_efficiency", eff))
    dt = float(cfg.get("time_step_hours", 1.0))

    return {
        "capacity_mwh": capacity,
        "max_power_mw": max_power,
        "max_charge_mw": float(battery.get("max_charge_mw", max_power)),
        "max_discharge_mw": float(battery.get("max_discharge_mw", max_power)),
        "min_soc_mwh": min_soc,
        "max_soc_mwh": max_soc,
        "initial_soc_mwh": init_soc,
        "charge_efficiency": charge_eff,
        "discharge_efficiency": discharge_eff,
        "degradation_cost_per_mwh": float(battery.get("degradation_cost_per_mwh", 10.0)),
        "max_grid_import_mw": float(cfg.get("grid", {}).get("max_import_mw", 500.0)),
        "time_step_hours": dt,
    }


def _ensure_out_dir(out_dir: str | Path) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_telemetry_events(x_obs: pd.DataFrame, event_log: pd.DataFrame) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for idx in range(len(x_obs)):
        payload: dict[str, Any] = {
            "ts_utc": pd.to_datetime(event_log.loc[idx, "arrived_timestamp"], utc=True).isoformat(),
            "device_id": "bench-device",
            "zone_id": "DE",
            "load_mw": float(x_obs.loc[idx, "load_mw"]),
            "renewables_mw": float(x_obs.loc[idx, "renewables_mw"]),
        }
        for fault_col in FAULT_COLUMNS:
            payload[fault_col] = bool(event_log.loc[idx, fault_col])
        events.append(payload)
    return events


def _soc_fault_config_for_episode(
    *,
    scenario: str,
    event_log: pd.DataFrame,
    seed: int,
    fault_overrides: Mapping[str, Any] | None,
) -> SOCTelemetryFaultConfig:
    overrides = dict(fault_overrides or {})

    dropout_prob = float(event_log["dropout"].mean()) if "dropout" in event_log.columns else 0.0
    stale_prob = float(event_log["stale_sensor"].mean()) if "stale_sensor" in event_log.columns else 0.0

    if scenario in {"dropout", "drift_combo"} and dropout_prob == 0.0:
        dropout_prob = 0.20
    if scenario in {"stale_sensor", "drift_combo"} and stale_prob == 0.0:
        stale_prob = 0.20

    if "soc_dropout_prob" in overrides:
        dropout_prob = float(overrides["soc_dropout_prob"])
    if "soc_stale_prob" in overrides:
        stale_prob = float(overrides["soc_stale_prob"])
    if "dropout_prob" in overrides:
        dropout_prob = float(overrides["dropout_prob"])
    if "stale_prob" in overrides:
        stale_prob = float(overrides["stale_prob"])

    noise_std = float(overrides.get("soc_noise_std_mwh", 0.25))

    return SOCTelemetryFaultConfig(
        dropout_prob=min(max(dropout_prob, 0.0), 0.95),
        stale_prob=min(max(stale_prob, 0.0), 0.95),
        noise_std_mwh=max(0.0, noise_std),
        seed=int(seed),
    )


@lru_cache(maxsize=2)
def _load_load_conformal(target: str = "load_mw") -> Any | None:
    unc_cfg = _load_uncertainty_cfg()
    artifacts_dir = Path(unc_cfg.get("artifacts_dir", "artifacts/uncertainty"))
    path = artifacts_dir / f"{target}_conformal.json"
    if not path.exists():
        return None
    try:
        return load_conformal(path)
    except Exception:
        return None


def _regime_runtime_cfg() -> dict[str, Any]:
    unc_cfg = _load_uncertainty_cfg()
    regime_cfg = unc_cfg.get("regime_cqr", {}) if isinstance(unc_cfg, dict) else {}
    if not isinstance(regime_cfg, dict):
        regime_cfg = {}
    artifacts_dir = Path(unc_cfg.get("artifacts_dir", "artifacts/uncertainty"))
    default_artifact = str(artifacts_dir / "{target}_regime_cqr.json")
    return {
        "enabled": bool(regime_cfg.get("enabled", False)),
        "artifact_path": str(regime_cfg.get("artifact_path", default_artifact)),
        "policy": str(regime_cfg.get("quantile_backend_policy", "fallback")).strip().lower(),
    }


def _rac_runtime_cfg() -> dict[str, Any]:
    unc_cfg = _load_uncertainty_cfg()
    rac_cfg = unc_cfg.get("rac_cert", {}) if isinstance(unc_cfg, dict) else {}
    if not isinstance(rac_cfg, dict):
        rac_cfg = {}
    artifacts_dir = Path(unc_cfg.get("artifacts_dir", "artifacts/uncertainty"))
    default_artifact = str(artifacts_dir / "{target}_rac_cert.json")
    return {
        "enabled": bool(rac_cfg.get("enabled", True)),
        "artifact_path": str(rac_cfg.get("artifact_path", default_artifact)),
        "sens_eps_mw": float(rac_cfg.get("sens_eps_mw", 25.0)),
        "sens_norm_ref": float(rac_cfg.get("sens_norm_ref", 0.5)),
        "policy": str(rac_cfg.get("quantile_backend_policy", "strict")).strip().lower(),
    }


@lru_cache(maxsize=2)
def _load_regime_cqr(target: str = "load_mw") -> RegimeCQR | None:
    cfg = _regime_runtime_cfg()
    if not cfg["enabled"]:
        return None

    path = Path(str(cfg["artifact_path"]).format(target=target))
    if not path.exists():
        if cfg["policy"] == "strict":
            strict_runtime = os.getenv("GRIDPULSE_REQUIRE_REGIME_CQR", "").strip().lower() in {"1", "true", "yes", "on"}
            if strict_runtime:
                raise RuntimeError(
                    f"Regime CQR enabled with strict policy but missing artifact: {path}. "
                    "Run scripts/train_regime_cqr.py first."
                )
        return None


@lru_cache(maxsize=2)
def _load_rac_cert(target: str = "load_mw") -> RACCertModel | None:
    cfg = _rac_runtime_cfg()
    if not cfg["enabled"]:
        return None
    path = Path(str(cfg["artifact_path"]).format(target=target))
    if not path.exists():
        strict_runtime = os.getenv("GRIDPULSE_REQUIRE_RAC_CERT", "").strip().lower() in {"1", "true", "yes", "on"}
        if strict_runtime:
            raise RuntimeError(
                f"RAC-Cert enabled with strict policy but missing artifact: {path}. "
                "Run scripts/train_regime_cqr.py first."
            )
        return None
    try:
        return RACCertModel.from_json(path.read_text(encoding="utf-8"))
    except Exception:
        if cfg["policy"] == "strict":
            raise
        return None

    try:
        return RegimeCQR.from_json(path.read_text(encoding="utf-8"))
    except Exception:
        if cfg["policy"] == "strict":
            raise
        return None


def _base_interval_bounds(load_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ci = _load_load_conformal("load_mw")
    if ci is not None:
        try:
            lo, hi = ci.predict_interval(np.asarray(load_window, dtype=float))
            lo = np.maximum(0.0, np.asarray(lo, dtype=float))
            hi = np.asarray(hi, dtype=float)
            if lo.shape == hi.shape == load_window.shape:
                return lo, hi
        except Exception:
            pass

    q = np.maximum(75.0, 0.08 * np.abs(load_window))
    return np.maximum(0.0, load_window - q), load_window + q


def _cqr_bounds(load_window: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_lower, base_upper = _base_interval_bounds(np.asarray(load_window, dtype=float))
    regime = _load_regime_cqr("load_mw")
    if regime is None:
        return base_lower, base_upper

    try:
        lo, hi, _bins = regime.predict_interval(
            y_context=np.asarray(load_window, dtype=float),
            q_lo=np.asarray(base_lower, dtype=float),
            q_hi=np.asarray(base_upper, dtype=float),
        )
        lo = np.maximum(0.0, np.asarray(lo, dtype=float))
        hi = np.asarray(hi, dtype=float)
        if lo.shape == hi.shape == np.asarray(load_window).shape:
            return lo, hi
    except Exception:
        pass
    return base_lower, base_upper


def _rac_bounds(
    *,
    load_window: np.ndarray,
    dc3s_cfg: Mapping[str, Any] | None = None,
    w_t: float = 1.0,
    drift_flag: bool = False,
    sensitivity_t: float = 0.0,
    sensitivity_norm: float | None = None,
    prev_inflation: float | None = None,
    sigma_sq: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    load_arr = np.asarray(load_window, dtype=float).reshape(-1)
    base_lower, base_upper = _cqr_bounds(load_arr)
    base_half = np.maximum(0.0, 0.5 * (base_upper - base_lower))

    cfg = dict(dc3s_cfg or _load_dc3s_cfg())
    cfg.setdefault("ambiguity", {})
    cfg.setdefault("reliability", {})
    cfg.setdefault("rac_cert", {})

    rac_runtime = _rac_runtime_cfg()
    rac_model = _load_rac_cert("load_mw")
    q_for_build = np.asarray(base_half, dtype=float)
    if rac_model is not None:
        try:
            q_for_build = np.asarray(rac_model.qhat_for_context(load_arr, horizon=len(load_arr)), dtype=float)
            cfg["rac_cert"] = _deep_update(
                dict(cfg.get("rac_cert", {})),
                {
                    "enabled": True,
                    "alpha": float(rac_model.cfg.alpha),
                    "n_vol_bins": int(rac_model.cfg.n_vol_bins),
                    "vol_window": int(rac_model.cfg.vol_window),
                    "beta_reliability": float(rac_model.cfg.beta_reliability),
                    "beta_sensitivity": float(rac_model.cfg.beta_sensitivity),
                    "k_sensitivity": float(rac_model.cfg.k_sensitivity),
                    "infl_max": float(rac_model.cfg.infl_max),
                    "sens_eps_mw": float(rac_model.cfg.sens_eps_mw),
                    "sens_norm_ref": float(rac_model.cfg.sens_norm_ref),
                    "qhat_shrink_tau": float(rac_model.cfg.qhat_shrink_tau),
                    "max_q_multiplier": float(rac_model.cfg.max_q_multiplier),
                    "min_w": float(rac_model.cfg.min_w),
                },
            )
        except Exception:
            q_for_build = np.asarray(base_half, dtype=float)

    cfg["rac_cert"] = _deep_update(
        dict(cfg.get("rac_cert", {})),
        {
            "enabled": True,
            "sens_eps_mw": float(rac_runtime["sens_eps_mw"]),
            "sens_norm_ref": float(rac_runtime["sens_norm_ref"]),
        },
    )
    cfg["sensitivity_t"] = float(sensitivity_t)
    if sensitivity_norm is not None:
        cfg["sensitivity_norm"] = float(np.clip(float(sensitivity_norm), 0.0, 1.0))
    else:
        cfg.pop("sensitivity_norm", None)

    inflation_law = str(cfg.get("inflation_law", "linear")).strip().lower()
    if inflation_law not in {"linear", "kappa"}:
        raise ValueError(f"Unsupported inflation_law: {inflation_law}")

    if inflation_law == "kappa":
        sigma_sq_used = float(sigma_sq) if sigma_sq is not None else float(np.var(np.abs(load_arr - 0.5 * (base_lower + base_upper))))
        lower, upper, meta = build_uncertainty_set_kappa(
            yhat=load_arr,
            q=np.asarray(q_for_build, dtype=float),
            w_t=float(w_t),
            drift_flag=bool(drift_flag),
            cfg=cfg,
            sigma_sq=sigma_sq_used,
            prev_inflation=prev_inflation,
        )
    else:
        lower, upper, meta = build_uncertainty_set(
            yhat=load_arr,
            q=np.asarray(q_for_build, dtype=float),
            w_t=float(w_t),
            drift_flag=bool(drift_flag),
            cfg=cfg,
            prev_inflation=prev_inflation,
            base_lower=np.asarray(base_lower, dtype=float),
            base_upper=np.asarray(base_upper, dtype=float),
        )
    meta["coverage_lb_t"] = float(max(0.0, 1.0 - float(cfg.get("alpha0", 0.10))))
    return lower, upper, meta


def _controller_step_deterministic(
    *,
    load_window: np.ndarray,
    renew_window: np.ndarray,
    price_window: np.ndarray,
    carbon_window: np.ndarray,
    optimization_cfg: Mapping[str, Any],
    dc3s_cfg: Mapping[str, Any],
    observed_soc_mwh: float,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(dict(optimization_cfg)))
    cfg.setdefault("battery", {})
    cfg["battery"]["initial_soc_mwh"] = float(observed_soc_mwh)

    try:
        dispatch = optimize_dispatch(
            forecast_load=load_window.tolist(),
            forecast_renewables=renew_window.tolist(),
            forecast_price=price_window.tolist(),
            forecast_carbon_kg=carbon_window.tolist(),
            config=cfg,
        )
    except Exception:
        dispatch = {"battery_charge_mw": [0.0], "battery_discharge_mw": [0.0]}

    lower_90, upper_90, rac_meta = _rac_bounds(
        load_window=np.asarray(load_window, dtype=float),
        dc3s_cfg={**dict(dc3s_cfg), "inflation_law": "linear"},
        w_t=1.0,
        drift_flag=False,
        sensitivity_t=0.0,
        sensitivity_norm=0.0,
    )
    return {
        "proposed_charge_mw": float(np.asarray(dispatch.get("battery_charge_mw", [0.0]), dtype=float)[0]),
        "proposed_discharge_mw": float(np.asarray(dispatch.get("battery_discharge_mw", [0.0]), dtype=float)[0]),
        "safe_charge_mw": float(np.asarray(dispatch.get("battery_charge_mw", [0.0]), dtype=float)[0]),
        "safe_discharge_mw": float(np.asarray(dispatch.get("battery_discharge_mw", [0.0]), dtype=float)[0]),
        "interval_lower": np.asarray(lower_90, dtype=float),
        "interval_upper": np.asarray(upper_90, dtype=float),
        "solver_status": "deterministic",
        "w_t": float(rac_meta.get("w_t", 1.0)),
        "delta_mw": float(rac_meta.get("delta_mw", 0.0)),
        "interval_width": float(rac_meta.get("interval_width", max(0.0, upper_90[0] - lower_90[0]) if len(lower_90) else 0.0)),
        "sensitivity_t": float(rac_meta.get("sensitivity_t", 0.0)),
        "sensitivity_norm": float(rac_meta.get("sensitivity_norm", 0.0)),
        "q_eff": float(rac_meta.get("q_eff", 0.0)),
        "q_multiplier": float(rac_meta.get("q_multiplier", 1.0)),
        "rac_inflation": float(rac_meta.get("inflation", 1.0)),
        "coverage_lb_t": float(rac_meta.get("coverage_lb_t", 0.0)),
        "guarantee_checks_passed": True,
        "certificate": None,
        "cvar_eta": np.nan,
        "cvar_cost": np.nan,
    }


def _controller_step_robust_fixed(
    *,
    load_window: np.ndarray,
    renew_window: np.ndarray,
    price_window: np.ndarray,
    optimization_cfg: Mapping[str, Any],
    dc3s_cfg: Mapping[str, Any],
    observed_soc_mwh: float,
) -> dict[str, Any]:
    constraints = _battery_constraints(dict(optimization_cfg))
    lower, upper, rac_meta = _rac_bounds(
        load_window=np.asarray(load_window, dtype=float),
        dc3s_cfg={**dict(dc3s_cfg), "inflation_law": "linear"},
        w_t=1.0,
        drift_flag=False,
        sensitivity_t=0.0,
        sensitivity_norm=0.0,
    )

    rcfg = RobustDispatchConfig(
        battery_capacity_mwh=constraints["capacity_mwh"],
        battery_max_charge_mw=constraints["max_charge_mw"],
        battery_max_discharge_mw=constraints["max_discharge_mw"],
        battery_charge_efficiency=constraints["charge_efficiency"],
        battery_discharge_efficiency=constraints["discharge_efficiency"],
        battery_initial_soc_mwh=float(observed_soc_mwh),
        battery_min_soc_mwh=constraints["min_soc_mwh"],
        battery_max_soc_mwh=constraints["max_soc_mwh"],
        max_grid_import_mw=constraints["max_grid_import_mw"],
        default_price_per_mwh=float(np.mean(price_window)),
        degradation_cost_per_mwh=constraints["degradation_cost_per_mwh"],
        risk_weight_worst_case=float(dict(optimization_cfg).get("robust", {}).get("risk_weight_worst_case", 1.0)),
        time_step_hours=constraints["time_step_hours"],
        solver_name=str(dict(optimization_cfg).get("solver_name", "appsi_highs")),
    )

    solver_status = "error"
    try:
        robust = optimize_robust_dispatch(
            load_lower_bound=lower.tolist(),
            load_upper_bound=upper.tolist(),
            renewables_forecast=renew_window.tolist(),
            price=price_window.tolist(),
            config=rcfg,
            verbose=False,
        )
        solver_status = str(robust.get("solver_status", "ok"))
    except Exception:
        robust = {"battery_charge_mw": [0.0], "battery_discharge_mw": [0.0], "feasible": False}

    ch = float(np.asarray(robust.get("battery_charge_mw", [0.0]), dtype=float)[0])
    dis = float(np.asarray(robust.get("battery_discharge_mw", [0.0]), dtype=float)[0])
    return {
        "proposed_charge_mw": ch,
        "proposed_discharge_mw": dis,
        "safe_charge_mw": ch,
        "safe_discharge_mw": dis,
        "interval_lower": np.asarray(lower, dtype=float),
        "interval_upper": np.asarray(upper, dtype=float),
        "solver_status": solver_status,
        "w_t": float(rac_meta.get("w_t", 1.0)),
        "delta_mw": float(rac_meta.get("delta_mw", 0.0)),
        "interval_width": float(rac_meta.get("interval_width", max(0.0, upper[0] - lower[0]) if len(lower) else 0.0)),
        "sensitivity_t": float(rac_meta.get("sensitivity_t", 0.0)),
        "sensitivity_norm": float(rac_meta.get("sensitivity_norm", 0.0)),
        "q_eff": float(rac_meta.get("q_eff", 0.0)),
        "q_multiplier": float(rac_meta.get("q_multiplier", 1.0)),
        "rac_inflation": float(rac_meta.get("inflation", 1.0)),
        "coverage_lb_t": float(rac_meta.get("coverage_lb_t", 0.0)),
        "guarantee_checks_passed": True,
        "certificate": None,
        "cvar_eta": np.nan,
        "cvar_cost": np.nan,
    }


def _sample_load_scenarios(
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    n_scenarios: int,
    seed: int,
) -> np.ndarray:
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.size != hi.size:
        raise ValueError("lower and upper must have the same length")
    if np.any(lo > hi):
        raise ValueError("lower cannot exceed upper")
    rng = np.random.default_rng(int(seed))
    return rng.uniform(lo, hi, size=(int(n_scenarios), lo.size))


def _controller_step_cvar_interval(
    *,
    load_window: np.ndarray,
    renew_window: np.ndarray,
    price_window: np.ndarray,
    optimization_cfg: Mapping[str, Any],
    dc3s_cfg: Mapping[str, Any],
    observed_soc_mwh: float,
    random_seed: int,
) -> dict[str, Any]:
    constraints = _battery_constraints(dict(optimization_cfg))
    lower, upper, rac_meta = _rac_bounds(
        load_window=np.asarray(load_window, dtype=float),
        dc3s_cfg={**dict(dc3s_cfg), "inflation_law": "linear"},
        w_t=1.0,
        drift_flag=False,
        sensitivity_t=0.0,
        sensitivity_norm=0.0,
    )

    cvar_cfg = dict(dict(optimization_cfg).get("cvar", {}))
    n_scenarios = int(cvar_cfg.get("n_scenarios", 20))
    beta = float(cvar_cfg.get("beta", 0.90))
    risk_weight_cvar = float(cvar_cfg.get("risk_weight_cvar", 1.0))

    scenarios = _sample_load_scenarios(
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        n_scenarios=n_scenarios,
        seed=int(random_seed),
    )

    cfg = CVaRDispatchConfig(
        battery_capacity_mwh=constraints["capacity_mwh"],
        battery_max_charge_mw=constraints["max_charge_mw"],
        battery_max_discharge_mw=constraints["max_discharge_mw"],
        battery_charge_efficiency=constraints["charge_efficiency"],
        battery_discharge_efficiency=constraints["discharge_efficiency"],
        battery_initial_soc_mwh=float(observed_soc_mwh),
        battery_min_soc_mwh=constraints["min_soc_mwh"],
        battery_max_soc_mwh=constraints["max_soc_mwh"],
        max_grid_import_mw=constraints["max_grid_import_mw"],
        default_price_per_mwh=float(np.mean(price_window)),
        degradation_cost_per_mwh=constraints["degradation_cost_per_mwh"],
        risk_weight_worst_case=float(dict(optimization_cfg).get("robust", {}).get("risk_weight_worst_case", 1.0)),
        time_step_hours=constraints["time_step_hours"],
        solver_name=str(dict(optimization_cfg).get("solver_name", "appsi_highs")),
        beta=beta,
        n_scenarios=int(max(2, n_scenarios)),
        risk_weight_cvar=max(0.0, min(1.0, risk_weight_cvar)),
        scenario_seed=int(random_seed),
    )

    solver_status = "error"
    cvar_eta = np.nan
    cvar_cost = np.nan
    try:
        sol = optimize_cvar_dispatch(
            load_scenarios=scenarios,
            renewables_forecast=renew_window.tolist(),
            price=price_window.tolist(),
            config=cfg,
            verbose=False,
        )
        solver_status = str(sol.get("solver_status", "ok"))
        ch = float(np.asarray(sol.get("battery_charge_mw", [0.0]), dtype=float)[0])
        dis = float(np.asarray(sol.get("battery_discharge_mw", [0.0]), dtype=float)[0])
        if sol.get("eta") is not None:
            cvar_eta = float(sol.get("eta"))
        if sol.get("cvar_cost") is not None:
            cvar_cost = float(sol.get("cvar_cost"))
    except Exception:
        ch = 0.0
        dis = 0.0

    return {
        "proposed_charge_mw": ch,
        "proposed_discharge_mw": dis,
        "safe_charge_mw": ch,
        "safe_discharge_mw": dis,
        "interval_lower": np.asarray(lower, dtype=float),
        "interval_upper": np.asarray(upper, dtype=float),
        "solver_status": solver_status,
        "w_t": float(rac_meta.get("w_t", 1.0)),
        "delta_mw": float(rac_meta.get("delta_mw", 0.0)),
        "interval_width": float(rac_meta.get("interval_width", max(0.0, upper[0] - lower[0]) if len(lower) else 0.0)),
        "sensitivity_t": float(rac_meta.get("sensitivity_t", 0.0)),
        "sensitivity_norm": float(rac_meta.get("sensitivity_norm", 0.0)),
        "q_eff": float(rac_meta.get("q_eff", 0.0)),
        "q_multiplier": float(rac_meta.get("q_multiplier", 1.0)),
        "rac_inflation": float(rac_meta.get("inflation", 1.0)),
        "coverage_lb_t": float(rac_meta.get("coverage_lb_t", 0.0)),
        "guarantee_checks_passed": True,
        "certificate": None,
        "cvar_eta": float(cvar_eta) if np.isfinite(cvar_eta) else np.nan,
        "cvar_cost": float(cvar_cost) if np.isfinite(cvar_cost) else np.nan,
    }


def _controller_step_dc3s(
    *,
    load_window: np.ndarray,
    renew_window: np.ndarray,
    price_window: np.ndarray,
    carbon_window: np.ndarray,
    load_true_t: float,
    observed_soc_mwh: float,
    current_true_soc_mwh: float,
    telemetry_event: Mapping[str, Any],
    optimization_cfg: Mapping[str, Any],
    dc3s_cfg: Mapping[str, Any],
    state: _DC3SLoopState,
    command_id: str,
    controller_name: str = "dc3s_wrapped",
    law_override: str | None = None,
) -> dict[str, Any]:
    _ = carbon_window
    constraints = _battery_constraints(dict(optimization_cfg))
    cfg_runtime = dict(dc3s_cfg)
    if law_override is not None:
        cfg_runtime["law"] = str(law_override)
    if controller_name != "dc3s_wrapped":
        cfg_runtime["inflation_law"] = "linear"

    event = dict(telemetry_event)
    if "ts_utc" not in event:
        event["ts_utc"] = str(command_id)

    expected_cadence_s = float(dc3s_cfg.get("expected_cadence_s", 3600.0))
    ftit_cfg = {**dict(cfg_runtime.get("ftit", {})), "law": str(cfg_runtime.get("law", "linear")).strip().lower()}
    w_t, flags = compute_reliability(
        event,
        state.prev_event,
        expected_cadence_s=expected_cadence_s,
        reliability_cfg=dc3s_cfg.get("reliability", {}),
        adaptive_state=state.adaptive_state,
        ftit_cfg=ftit_cfg,
    )
    residual = abs(float(load_true_t) - float(load_window[0]))
    drift = state.detector.update(residual)
    ftit_state = update_ftit_state(
        adaptive_state=state.adaptive_state,
        fault_flags=flags.get("fault_flags", {}),
        constraints={
            **constraints,
            "current_soc_mwh": float(observed_soc_mwh),
        },
        cfg=ftit_cfg,
        stale_tracker=flags.get("stale_tracker"),
        sigma2_observation=float(residual) ** 2,
    )
    state.adaptive_state = ftit_state["adaptive_state"]
    if str(cfg_runtime.get("law", "linear")).strip().lower() == "ftit_ro":
        cfg_runtime["ftit_runtime"] = {"sigma2": float(ftit_state["sigma2"])}

    rcfg = RobustDispatchConfig(
        battery_capacity_mwh=constraints["capacity_mwh"],
        battery_max_charge_mw=constraints["max_charge_mw"],
        battery_max_discharge_mw=constraints["max_discharge_mw"],
        battery_charge_efficiency=constraints["charge_efficiency"],
        battery_discharge_efficiency=constraints["discharge_efficiency"],
        battery_initial_soc_mwh=float(observed_soc_mwh),
        battery_min_soc_mwh=constraints["min_soc_mwh"],
        battery_max_soc_mwh=constraints["max_soc_mwh"],
        max_grid_import_mw=constraints["max_grid_import_mw"],
        default_price_per_mwh=float(np.mean(price_window)),
        degradation_cost_per_mwh=constraints["degradation_cost_per_mwh"],
        risk_weight_worst_case=float(dict(optimization_cfg).get("robust", {}).get("risk_weight_worst_case", 1.0)),
        time_step_hours=constraints["time_step_hours"],
        solver_name=str(dict(optimization_cfg).get("solver_name", "appsi_highs")),
    )

    rac_cfg = dc3s_cfg.get("rac_cert", {}) if isinstance(dc3s_cfg.get("rac_cert"), Mapping) else {}
    sens_eps = float(rac_cfg.get("sens_eps_mw", _rac_runtime_cfg()["sens_eps_mw"]))
    sens_norm_ref = float(rac_cfg.get("sens_norm_ref", _rac_runtime_cfg()["sens_norm_ref"]))
    probe_mode = str(rac_cfg.get("sensitivity_probe", "heuristic")).strip().lower()
    drift_flag = bool(drift.get("drift", False))

    if probe_mode in {"finite_diff", "fd", "robust_fd"}:
        def _robust_probe(load_probe: np.ndarray) -> tuple[float, float]:
            probe_lower, probe_upper, _ = _rac_bounds(
                load_window=np.asarray(load_probe, dtype=float),
                dc3s_cfg=cfg_runtime,
                w_t=float(w_t),
                drift_flag=drift_flag,
                sensitivity_t=0.0,
                sensitivity_norm=0.0,
                sigma_sq=state.sigma_sq,
            )
            probe_sol = optimize_robust_dispatch(
                load_lower_bound=probe_lower.tolist(),
                load_upper_bound=probe_upper.tolist(),
                renewables_forecast=renew_window.tolist(),
                price=price_window.tolist(),
                config=rcfg,
                verbose=False,
            )
            ch_probe = float(np.asarray(probe_sol.get("battery_charge_mw", [0.0]), dtype=float)[0])
            dis_probe = float(np.asarray(probe_sol.get("battery_discharge_mw", [0.0]), dtype=float)[0])
            return ch_probe, dis_probe

        try:
            sensitivity_t = compute_dispatch_sensitivity(
                load_window=np.asarray(load_window, dtype=float),
                dispatch_probe=_robust_probe,
                sens_eps_mw=float(sens_eps),
            )
        except Exception:
            sensitivity_t = 0.0
    else:
        if len(load_window) > 1:
            slope = abs(float(load_window[1]) - float(load_window[0])) / max(abs(float(load_window[0])), 1.0)
        else:
            slope = 0.0
        sensitivity_t = float(np.clip(slope, 0.0, 1.0))
    sensitivity_norm = normalize_sensitivity(sensitivity_t, norm_ref=sens_norm_ref)

    lower, upper, widen_meta = _rac_bounds(
        load_window=np.asarray(load_window, dtype=float),
        dc3s_cfg=cfg_runtime,
        w_t=float(w_t),
        drift_flag=drift_flag,
        sensitivity_t=float(sensitivity_t),
        sensitivity_norm=float(sensitivity_norm),
        prev_inflation=state.prev_inflation,
        sigma_sq=state.sigma_sq,
    )
    state.prev_inflation = float(widen_meta.get("inflation", 1.0))
    widen_meta["gamma_mw"] = float(ftit_state["gamma_mw"])
    widen_meta["e_t_mwh"] = float(ftit_state["e_t_mwh"])
    widen_meta["soc_tube_lower_mwh"] = float(ftit_state["soc_tube_lower_mwh"])
    widen_meta["soc_tube_upper_mwh"] = float(ftit_state["soc_tube_upper_mwh"])

    solver_status = "error"
    proposed_charge = 0.0
    proposed_discharge = 0.0
    try:
        robust = optimize_robust_dispatch(
            load_lower_bound=lower.tolist(),
            load_upper_bound=upper.tolist(),
            renewables_forecast=renew_window.tolist(),
            price=price_window.tolist(),
            config=rcfg,
            verbose=False,
        )
        solver_status = str(robust.get("solver_status", "ok"))
        proposed_charge = float(np.asarray(robust.get("battery_charge_mw", [0.0]), dtype=float)[0])
        proposed_discharge = float(np.asarray(robust.get("battery_discharge_mw", [0.0]), dtype=float)[0])
    except Exception:
        robust = {"battery_charge_mw": [0.0], "battery_discharge_mw": [0.0], "feasible": False}

    # Projection-repair the robust first-step action.
    repair_cfg = json.loads(json.dumps(dict(dc3s_cfg)))
    repair_cfg.setdefault("shield", {})
    repair_cfg["shield"]["mode"] = str(dc3s_cfg.get("shield", {}).get("mode", "projection"))
    cvar_cfg = dict(dc3s_cfg.get("shield", {}).get("cvar", {}))
    repair_constraints = {
        **constraints,
        "current_soc_mwh": float(observed_soc_mwh),
        "last_net_mw": float(state.last_net_mw),
        "ramp_mw": float(dc3s_cfg.get("shield", {}).get("max_ramp_mw", 0.0) or 0.0),
        "solver_name": str(dict(optimization_cfg).get("solver_name", "appsi_highs")),
        "cvar_beta": float(cvar_cfg.get("beta", 0.90)),
        "cvar_n_scenarios": int(cvar_cfg.get("n_scenarios", 20)),
        "cvar_risk_weight": float(cvar_cfg.get("risk_weight_cvar", 1.0)),
        "scenario_seed": int(cvar_cfg.get("scenario_seed", 0)),
    }
    if str(cfg_runtime.get("law", "linear")).strip().lower() == "ftit_ro":
        repair_constraints["ftit_soc_min_mwh"] = float(ftit_state["soc_tube_lower_mwh"])
        repair_constraints["ftit_soc_max_mwh"] = float(ftit_state["soc_tube_upper_mwh"])
    safe_action, repair_meta = repair_action(
        a_star={"charge_mw": proposed_charge, "discharge_mw": proposed_discharge},
        state={"current_soc_mwh": float(observed_soc_mwh)},
        uncertainty_set={
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "meta": {**widen_meta, "flags": flags},
            "renewables_forecast": renew_window.tolist(),
            "price": price_window.tolist(),
        },
        constraints=repair_constraints,
        cfg={**repair_cfg, "law": str(cfg_runtime.get("law", "linear"))},
    )

    safe_charge = float(safe_action.get("charge_mw", 0.0))
    safe_discharge = float(safe_action.get("discharge_mw", 0.0))
    delta_mw = float(abs(safe_charge - proposed_charge) + abs(safe_discharge - proposed_discharge))

    guarantee_ok, guarantee_reasons, _ = evaluate_guarantee_checks(
        current_soc=float(current_true_soc_mwh),
        action={"charge_mw": safe_charge, "discharge_mw": safe_discharge},
        constraints=repair_constraints,
    )

    config_hash = compute_config_hash(json.dumps(dc3s_cfg, sort_keys=True).encode("utf-8"))
    cert = make_certificate(
        command_id=command_id,
        device_id=str(event.get("device_id", "bench-device")),
        zone_id=str(event.get("zone_id", "DE")),
        controller=controller_name,
        proposed_action={"charge_mw": proposed_charge, "discharge_mw": proposed_discharge},
        safe_action={"charge_mw": safe_charge, "discharge_mw": safe_discharge},
        uncertainty={
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "meta": {
                **widen_meta,
                "delta_mw": float(delta_mw),
                "solver_status": solver_status,
                "repair_meta": repair_meta,
            },
        },
        reliability={"w_t": float(w_t), "flags": flags},
        drift=drift,
        model_hash=compute_model_hash([]),
        config_hash=config_hash,
        prev_hash=state.prev_hash,
        dispatch_plan=None,
        intervened=bool(delta_mw > 1e-9),
        intervention_reason="projection_repair" if delta_mw > 1e-9 else "none",
        reliability_w=float(w_t),
        drift_flag=bool(drift.get("drift", False)),
        inflation=float(widen_meta.get("inflation", 1.0)),
        guarantee_checks_passed=bool(guarantee_ok),
        guarantee_fail_reasons=guarantee_reasons,
        assumptions_version=str(dc3s_cfg.get("assumptions_version", "dc3s-assumptions-v1")),
        gamma_mw=float(ftit_state["gamma_mw"]),
        e_t_mwh=float(ftit_state["e_t_mwh"]),
        soc_tube_lower_mwh=float(ftit_state["soc_tube_lower_mwh"]),
        soc_tube_upper_mwh=float(ftit_state["soc_tube_upper_mwh"]),
    )
    cert["w_t"] = float(w_t)
    cert["delta_mw"] = float(delta_mw)
    cert["interval_width"] = float(widen_meta.get("interval_width", 0.0))
    cert["solver_status"] = solver_status
    cert["guarantee_checks_passed"] = bool(guarantee_ok)
    cert["lambda_mw_used"] = float(widen_meta.get("lambda_mw_used", 0.0))
    cert["coverage_lb_t"] = float(widen_meta.get("coverage_lb_t", 0.0))
    cert["sensitivity_t"] = float(widen_meta.get("sensitivity_t", sensitivity_t))
    cert["sensitivity_norm"] = float(widen_meta.get("sensitivity_norm", sensitivity_norm))
    cert["q_eff"] = float(widen_meta.get("q_eff", 0.0))
    cert["q_multiplier"] = float(widen_meta.get("q_multiplier", 1.0))

    state.prev_event = event
    state.prev_hash = str(cert.get("certificate_hash"))
    state.last_net_mw = float(safe_discharge - safe_charge)

    return {
        "proposed_charge_mw": proposed_charge,
        "proposed_discharge_mw": proposed_discharge,
        "safe_charge_mw": safe_charge,
        "safe_discharge_mw": safe_discharge,
        "interval_lower": lower,
        "interval_upper": upper,
        "solver_status": solver_status,
        "w_t": float(w_t),
        "delta_mw": float(delta_mw),
        "interval_width": float(widen_meta.get("interval_width", 0.0)),
        "lambda_mw_used": float(widen_meta.get("lambda_mw_used", 0.0)),
        "sensitivity_t": float(widen_meta.get("sensitivity_t", sensitivity_t)),
        "sensitivity_norm": float(widen_meta.get("sensitivity_norm", sensitivity_norm)),
        "q_eff": float(widen_meta.get("q_eff", 0.0)),
        "q_multiplier": float(widen_meta.get("q_multiplier", 1.0)),
        "rac_inflation": float(widen_meta.get("inflation", 1.0)),
        "coverage_lb_t": float(widen_meta.get("coverage_lb_t", 0.0)),
        "guarantee_checks_passed": bool(guarantee_ok),
        "certificate": cert,
        "cvar_eta": np.nan,
        "cvar_cost": np.nan,
    }


def _init_controller_buffers(n: int) -> dict[str, Any]:
    return {
        "proposed_charge_mw": np.zeros(n, dtype=float),
        "proposed_discharge_mw": np.zeros(n, dtype=float),
        "safe_charge_mw": np.zeros(n, dtype=float),
        "safe_discharge_mw": np.zeros(n, dtype=float),
        "soc_true_mwh": np.zeros(n, dtype=float),
        "soc_observed_mwh": np.zeros(n, dtype=float),
        "interval_lower": np.zeros(n, dtype=float),
        "interval_upper": np.zeros(n, dtype=float),
        "w_t": np.ones(n, dtype=float),
        "delta_mw": np.zeros(n, dtype=float),
        "interval_width": np.zeros(n, dtype=float),
        "lambda_mw_used": np.zeros(n, dtype=float),
        "rac_sensitivity": np.zeros(n, dtype=float),
        "rac_sensitivity_norm": np.zeros(n, dtype=float),
        "rac_q_multiplier": np.ones(n, dtype=float),
        "rac_inflation": np.ones(n, dtype=float),
        "cvar_eta": np.full(n, np.nan, dtype=float),
        "cvar_cost": np.full(n, np.nan, dtype=float),
        "guarantee_checks_passed": np.ones(n, dtype=float),
        "bms_trip_mask": np.zeros(n, dtype=float),
        "certificates": [None] * n,
        "solver_status": ["ok"] * n,
        "expected_cost_usd": 0.0,
        "carbon_kg": 0.0,
    }


def _compute_cqr_group_metrics(y_true: np.ndarray, y_context: np.ndarray) -> dict[str, float]:
    lower, upper, _ = _rac_bounds(
        load_window=np.asarray(y_context, dtype=float),
        dc3s_cfg={**_load_dc3s_cfg(), "inflation_law": "linear"},
        w_t=1.0,
        drift_flag=False,
        sensitivity_t=0.0,
        sensitivity_norm=0.0,
    )
    y = np.asarray(y_true, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if not (len(y) == len(lo) == len(hi)):
        return {
            "cqr_picp_group_low": np.nan,
            "cqr_picp_group_mid": np.nan,
            "cqr_picp_group_high": np.nan,
            "cqr_width_group_low": np.nan,
            "cqr_width_group_mid": np.nan,
            "cqr_width_group_high": np.nan,
        }

    width = hi - lo
    covered = (y >= lo) & (y <= hi)
    vol = pd.Series(y).rolling(window=24, min_periods=6).std().fillna(0.0).to_numpy()
    q1, q2 = np.quantile(vol, [1.0 / 3.0, 2.0 / 3.0]) if len(vol) > 2 else (0.0, 0.0)
    labels = np.where(vol <= q1, "low", np.where(vol <= q2, "mid", "high"))

    out: dict[str, float] = {}
    for label in ("low", "mid", "high"):
        mask = labels == label
        out[f"cqr_picp_group_{label}"] = float(np.mean(covered[mask])) if np.any(mask) else np.nan
        out[f"cqr_width_group_{label}"] = float(np.mean(width[mask])) if np.any(mask) else np.nan
    return out


def run_single(
    *,
    scenario: str,
    seed: int,
    horizon: int = 168,
    fault_overrides: dict[str, Any] | None = None,
    dc3s_overrides: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run one scenario+seed pair and return metrics row payloads."""
    x_obs, x_true, event_log = generate_episode(
        scenario=scenario,
        seed=seed,
        horizon=horizon,
        fault_overrides=fault_overrides,
    )

    optimization_cfg = _load_optimization_cfg()
    dc3s_cfg = _load_dc3s_cfg()
    if dc3s_overrides:
        dc3s_cfg = _deep_update(dc3s_cfg, dc3s_overrides)

    constraints = _battery_constraints(optimization_cfg)

    load_obs = x_obs["load_mw"].to_numpy(dtype=float)
    renew_obs = x_obs["renewables_mw"].to_numpy(dtype=float)
    load_true = x_true["load_mw"].to_numpy(dtype=float)
    renew_true = x_true["renewables_mw"].to_numpy(dtype=float)
    price = x_obs["price_per_mwh"].to_numpy(dtype=float)
    carbon = x_obs["carbon_kg_per_mwh"].to_numpy(dtype=float)
    telemetry_events = _to_telemetry_events(x_obs=x_obs, event_log=event_log)

    n = len(load_obs)
    stepwise_controllers = ("deterministic_lp", "robust_fixed_interval", "cvar_interval", "dc3s_wrapped", "dc3s_ftit")
    controllers = stepwise_controllers + ("aci_conformal", "scenario_robust", "scenario_mpc")
    cqr_group_metrics = _compute_cqr_group_metrics(load_true, load_obs)
    sigma_sq_episode = float(np.var(np.abs(load_true - load_obs)))

    results: dict[str, dict[str, Any]] = {}
    for controller in stepwise_controllers:
        plant = BatteryPlant(
            soc_mwh=float(constraints["initial_soc_mwh"]),
            min_soc_mwh=float(constraints["min_soc_mwh"]),
            max_soc_mwh=float(constraints["max_soc_mwh"]),
            charge_eff=float(constraints["charge_efficiency"]),
            discharge_eff=float(constraints["discharge_efficiency"]),
            dt_hours=float(constraints["time_step_hours"]),
        )
        soc_channel = SOCTelemetryChannel(
            _soc_fault_config_for_episode(
                scenario=scenario,
                event_log=event_log,
                seed=int(seed),
                fault_overrides=fault_overrides,
            )
        )
        dc3s_state = _DC3SLoopState(
            detector=PageHinkleyDetector.from_state(None, cfg=dc3s_cfg.get("drift", {})),
            sigma_sq=sigma_sq_episode,
        )

        buf = _init_controller_buffers(n)
        for t in range(n):
            t_end = n
            load_window = load_obs[t:t_end]
            renew_window = renew_obs[t:t_end]
            price_window = price[t:t_end]
            carbon_window = carbon[t:t_end]
            current_true_soc = float(plant.soc_mwh)
            observed_soc, _ = soc_channel.observe(current_true_soc)

            if controller == "deterministic_lp":
                step = _controller_step_deterministic(
                    load_window=load_window,
                    renew_window=renew_window,
                    price_window=price_window,
                    carbon_window=carbon_window,
                    optimization_cfg=optimization_cfg,
                    dc3s_cfg=dc3s_cfg,
                    observed_soc_mwh=float(observed_soc),
                )
            elif controller == "robust_fixed_interval":
                step = _controller_step_robust_fixed(
                    load_window=load_window,
                    renew_window=renew_window,
                    price_window=price_window,
                    optimization_cfg=optimization_cfg,
                    dc3s_cfg=dc3s_cfg,
                    observed_soc_mwh=float(observed_soc),
                )
            elif controller == "cvar_interval":
                step = _controller_step_cvar_interval(
                    load_window=load_window,
                    renew_window=renew_window,
                    price_window=price_window,
                    optimization_cfg=optimization_cfg,
                    dc3s_cfg=dc3s_cfg,
                    observed_soc_mwh=float(observed_soc),
                    random_seed=int(seed * 100000 + t),
                )
            elif controller == "dc3s_wrapped":
                step = _controller_step_dc3s(
                    load_window=load_window,
                    renew_window=renew_window,
                    price_window=price_window,
                    carbon_window=carbon_window,
                    load_true_t=float(load_true[t]),
                    observed_soc_mwh=float(observed_soc),
                    current_true_soc_mwh=current_true_soc,
                    telemetry_event=telemetry_events[t],
                    optimization_cfg=optimization_cfg,
                    dc3s_cfg=dc3s_cfg,
                    state=dc3s_state,
                    command_id=f"{scenario}-{seed}-{controller}-{t:04d}",
                    controller_name="dc3s_wrapped",
                    law_override="linear",
                )
            else:
                step = _controller_step_dc3s(
                    load_window=load_window,
                    renew_window=renew_window,
                    price_window=price_window,
                    carbon_window=carbon_window,
                    load_true_t=float(load_true[t]),
                    observed_soc_mwh=float(observed_soc),
                    current_true_soc_mwh=current_true_soc,
                    telemetry_event=telemetry_events[t],
                    optimization_cfg=optimization_cfg,
                    dc3s_cfg=dc3s_cfg,
                    state=dc3s_state,
                    command_id=f"{scenario}-{seed}-{controller}-{t:04d}",
                    controller_name="dc3s_ftit",
                    law_override="ftit_ro",
                )

            safe_charge = float(step["safe_charge_mw"])
            safe_discharge = float(step["safe_discharge_mw"])
            guarantee_ok = bool(step.get("guarantee_checks_passed", True))

            if not guarantee_ok:
                buf["bms_trip_mask"][t] = 1.0
                applied_charge = 0.0
                applied_discharge = 0.0
            else:
                applied_charge = safe_charge
                applied_discharge = safe_discharge

            next_soc = float(plant.step(charge_mw=applied_charge, discharge_mw=applied_discharge))
            violation_after_apply = bool(next_soc < constraints["min_soc_mwh"] or next_soc > constraints["max_soc_mwh"])
            cert = step.get("certificate")
            if isinstance(cert, dict):
                cert["true_soc_violation_after_apply"] = violation_after_apply
                buf["certificates"][t] = cert

            buf["proposed_charge_mw"][t] = float(step["proposed_charge_mw"])
            buf["proposed_discharge_mw"][t] = float(step["proposed_discharge_mw"])
            buf["safe_charge_mw"][t] = safe_charge
            buf["safe_discharge_mw"][t] = safe_discharge
            buf["soc_true_mwh"][t] = next_soc
            buf["soc_observed_mwh"][t] = float(observed_soc)
            buf["interval_lower"][t] = float(np.asarray(step["interval_lower"], dtype=float)[0])
            buf["interval_upper"][t] = float(np.asarray(step["interval_upper"], dtype=float)[0])
            buf["w_t"][t] = float(step.get("w_t", 1.0))
            buf["delta_mw"][t] = float(step.get("delta_mw", 0.0))
            buf["interval_width"][t] = float(step.get("interval_width", max(0.0, buf["interval_upper"][t] - buf["interval_lower"][t])))
            buf["lambda_mw_used"][t] = float(step.get("lambda_mw_used", 0.0))
            buf["rac_sensitivity"][t] = float(step.get("sensitivity_t", 0.0))
            buf["rac_sensitivity_norm"][t] = float(step.get("sensitivity_norm", 0.0))
            buf["rac_q_multiplier"][t] = float(step.get("q_multiplier", 1.0))
            buf["rac_inflation"][t] = float(step.get("rac_inflation", step.get("inflation", 1.0)))
            buf["cvar_eta"][t] = float(step.get("cvar_eta", np.nan))
            buf["cvar_cost"][t] = float(step.get("cvar_cost", np.nan))
            buf["guarantee_checks_passed"][t] = 1.0 if guarantee_ok else 0.0
            buf["solver_status"][t] = str(step.get("solver_status", "ok"))

            grid_import_true = max(
                0.0,
                float(load_true[t] - renew_true[t] - applied_discharge + applied_charge),
            )
            dt = float(constraints.get("time_step_hours", 1.0))
            deg = float(constraints.get("degradation_cost_per_mwh", 10.0))
            buf["expected_cost_usd"] += float(price[t]) * grid_import_true * dt
            buf["expected_cost_usd"] += deg * (abs(applied_charge) + abs(applied_discharge)) * dt
            buf["carbon_kg"] += float(carbon[t]) * grid_import_true * dt

        results[controller] = buf

    aci_result = aci_conformal_dispatch(
        load_forecast=load_obs,
        renewables_forecast=renew_obs,
        load_true=load_true,
        telemetry_events=telemetry_events,
        price=price,
        optimization_cfg=optimization_cfg,
        command_prefix=f"{scenario}-{seed}-aci_conformal",
    )
    aci_buf = _init_controller_buffers(n)
    aci_buf["proposed_charge_mw"] = np.asarray(aci_result["proposed_charge_mw"], dtype=float)
    aci_buf["proposed_discharge_mw"] = np.asarray(aci_result["proposed_discharge_mw"], dtype=float)
    aci_buf["safe_charge_mw"] = np.asarray(aci_result["safe_charge_mw"], dtype=float)
    aci_buf["safe_discharge_mw"] = np.asarray(aci_result["safe_discharge_mw"], dtype=float)
    aci_buf["soc_true_mwh"] = np.asarray(aci_result["soc_mwh"], dtype=float)
    aci_buf["soc_observed_mwh"] = np.asarray(aci_result["soc_mwh"], dtype=float)
    aci_buf["interval_lower"] = np.asarray(aci_result["interval_lower"], dtype=float)
    aci_buf["interval_upper"] = np.asarray(aci_result["interval_upper"], dtype=float)
    aci_buf["interval_width"] = aci_buf["interval_upper"] - aci_buf["interval_lower"]
    aci_buf["certificates"] = list(aci_result["certificates"])
    aci_buf["solver_status"] = ["ok"] * n
    dt = float(constraints.get("time_step_hours", 1.0))
    deg = float(constraints.get("degradation_cost_per_mwh", 10.0))
    aci_grid_import_true = np.maximum(
        0.0,
        load_true - renew_true - aci_buf["safe_discharge_mw"] + aci_buf["safe_charge_mw"],
    )
    aci_expected_cost = float(
        np.sum(price * aci_grid_import_true * dt + deg * (np.abs(aci_buf["safe_charge_mw"]) + np.abs(aci_buf["safe_discharge_mw"])) * dt)
    )
    aci_carbon = float(np.sum(carbon * aci_grid_import_true * dt))
    aci_result["expected_cost_usd"] = aci_expected_cost
    aci_result["carbon_kg"] = aci_carbon
    if isinstance(aci_result.get("dispatch_plan"), dict):
        aci_result["dispatch_plan"]["expected_cost_usd"] = aci_expected_cost
        aci_result["dispatch_plan"]["carbon_kg"] = aci_carbon
    aci_buf["expected_cost_usd"] = aci_expected_cost
    aci_buf["carbon_kg"] = aci_carbon
    results["aci_conformal"] = aci_buf

    scenario_robust_result = scenario_robust_dispatch(
        load_forecast=load_obs,
        renewables_forecast=renew_obs,
        load_true=load_true,
        price=price,
        optimization_cfg=optimization_cfg,
        seed=seed,
    )
    scenario_buf = _init_controller_buffers(n)
    scenario_buf["proposed_charge_mw"] = np.asarray(scenario_robust_result["proposed_charge_mw"], dtype=float)
    scenario_buf["proposed_discharge_mw"] = np.asarray(scenario_robust_result["proposed_discharge_mw"], dtype=float)
    scenario_buf["safe_charge_mw"] = np.asarray(scenario_robust_result["safe_charge_mw"], dtype=float)
    scenario_buf["safe_discharge_mw"] = np.asarray(scenario_robust_result["safe_discharge_mw"], dtype=float)
    scenario_buf["soc_true_mwh"] = np.asarray(scenario_robust_result["soc_mwh"], dtype=float)
    scenario_buf["soc_observed_mwh"] = np.asarray(scenario_robust_result["soc_mwh"], dtype=float)
    scenario_buf["interval_lower"] = np.asarray(scenario_robust_result["interval_lower"], dtype=float)
    scenario_buf["interval_upper"] = np.asarray(scenario_robust_result["interval_upper"], dtype=float)
    scenario_buf["interval_width"] = scenario_buf["interval_upper"] - scenario_buf["interval_lower"]
    scenario_buf["certificates"] = list(scenario_robust_result["certificates"])
    scenario_status = str(scenario_robust_result.get("dispatch_plan", {}).get("solver_status", "ok"))
    scenario_buf["solver_status"] = [scenario_status] * n
    scenario_grid_import_true = np.maximum(
        0.0,
        load_true - renew_true - scenario_buf["safe_discharge_mw"] + scenario_buf["safe_charge_mw"],
    )
    scenario_expected_cost = float(
        np.sum(price * scenario_grid_import_true * dt + deg * (np.abs(scenario_buf["safe_charge_mw"]) + np.abs(scenario_buf["safe_discharge_mw"])) * dt)
    )
    scenario_carbon = float(np.sum(carbon * scenario_grid_import_true * dt))
    scenario_robust_result["expected_cost_usd"] = scenario_expected_cost
    scenario_robust_result["carbon_kg"] = scenario_carbon
    if isinstance(scenario_robust_result.get("dispatch_plan"), dict):
        scenario_robust_result["dispatch_plan"]["expected_cost_usd"] = scenario_expected_cost
        scenario_robust_result["dispatch_plan"]["carbon_kg"] = scenario_carbon
    scenario_buf["expected_cost_usd"] = scenario_expected_cost
    scenario_buf["carbon_kg"] = scenario_carbon
    results["scenario_robust"] = scenario_buf

    # --- scenario_mpc: receding-horizon MPC baseline ---
    mpc_result = scenario_mpc_dispatch(
        load_forecast=load_obs,
        renewables_forecast=renew_obs,
        load_true=load_true,
        price=price,
        optimization_cfg=optimization_cfg,
        seed=seed,
    )
    mpc_buf = _init_controller_buffers(n)
    mpc_buf["proposed_charge_mw"] = np.asarray(mpc_result["proposed_charge_mw"], dtype=float)
    mpc_buf["proposed_discharge_mw"] = np.asarray(mpc_result["proposed_discharge_mw"], dtype=float)
    mpc_buf["safe_charge_mw"] = np.asarray(mpc_result["safe_charge_mw"], dtype=float)
    mpc_buf["safe_discharge_mw"] = np.asarray(mpc_result["safe_discharge_mw"], dtype=float)
    mpc_buf["soc_true_mwh"] = np.asarray(mpc_result["soc_mwh"], dtype=float)
    mpc_buf["soc_observed_mwh"] = np.asarray(mpc_result["soc_mwh"], dtype=float)
    mpc_buf["interval_lower"] = np.asarray(mpc_result["interval_lower"], dtype=float)
    mpc_buf["interval_upper"] = np.asarray(mpc_result["interval_upper"], dtype=float)
    mpc_buf["interval_width"] = mpc_buf["interval_upper"] - mpc_buf["interval_lower"]
    mpc_buf["certificates"] = list(mpc_result["certificates"])
    mpc_status = str(mpc_result.get("dispatch_plan", {}).get("solver_status", "ok"))
    mpc_buf["solver_status"] = [mpc_status] * n
    mpc_grid_import_true = np.maximum(
        0.0,
        load_true - renew_true - mpc_buf["safe_discharge_mw"] + mpc_buf["safe_charge_mw"],
    )
    mpc_expected_cost = float(
        np.sum(price * mpc_grid_import_true * dt + deg * (np.abs(mpc_buf["safe_charge_mw"]) + np.abs(mpc_buf["safe_discharge_mw"])) * dt)
    )
    mpc_carbon = float(np.sum(carbon * mpc_grid_import_true * dt))
    mpc_result["expected_cost_usd"] = mpc_expected_cost
    mpc_result["carbon_kg"] = mpc_carbon
    if isinstance(mpc_result.get("dispatch_plan"), dict):
        mpc_result["dispatch_plan"]["expected_cost_usd"] = mpc_expected_cost
        mpc_result["dispatch_plan"]["carbon_kg"] = mpc_carbon
    mpc_buf["expected_cost_usd"] = mpc_expected_cost
    mpc_buf["carbon_kg"] = mpc_carbon
    results["scenario_mpc"] = mpc_buf

    baseline_cost = float(results["deterministic_lp"]["expected_cost_usd"])

    main_rows: list[dict[str, Any]] = []
    fault_rows: list[dict[str, Any]] = []
    for controller in controllers:
        res = results[controller]
        metrics = compute_all_metrics(
            y_true=load_true,
            y_pred=load_obs,
            lower_90=res["interval_lower"],
            upper_90=res["interval_upper"],
            proposed_charge_mw=res["proposed_charge_mw"],
            proposed_discharge_mw=res["proposed_discharge_mw"],
            safe_charge_mw=res["safe_charge_mw"],
            safe_discharge_mw=res["safe_discharge_mw"],
            soc_mwh=res["soc_true_mwh"],
            true_soc_mwh=res["soc_true_mwh"],
            constraints=constraints,
            certificates=res["certificates"],
            event_log=event_log,
            bms_trip_mask=res["bms_trip_mask"],
            load_true=load_true,
            renewables_true=renew_true,
        )

        cost_delta_pct = None
        if baseline_cost > 0:
            cost_delta_pct = 100.0 * (float(res["expected_cost_usd"]) - baseline_cost) / baseline_cost

        cvar_eta_vals = np.asarray(res["cvar_eta"], dtype=float)
        cvar_cost_vals = np.asarray(res["cvar_cost"], dtype=float)
        cvar_eta_mean = float(np.nanmean(cvar_eta_vals)) if np.isfinite(cvar_eta_vals).any() else np.nan
        cvar_cost_mean = float(np.nanmean(cvar_cost_vals)) if np.isfinite(cvar_cost_vals).any() else np.nan
        adaptive_width_mean = float(np.mean(res["interval_width"])) if len(res["interval_width"]) else np.nan
        adaptive_width_p95 = float(np.quantile(res["interval_width"], 0.95)) if len(res["interval_width"]) else np.nan

        row = {
            "scenario": scenario,
            "seed": int(seed),
            "controller": controller,
            "policy": controller,
            "expected_cost_usd": float(res["expected_cost_usd"]),
            "carbon_kg": float(res["carbon_kg"]),
            "cost_delta_pct": float(cost_delta_pct) if cost_delta_pct is not None else None,
            "mean_reliability_w": float(np.mean(res["w_t"])),
            "mean_delta_mw": float(np.mean(res["delta_mw"])),
            "mean_interval_width_cert": float(np.mean(res["interval_width"])),
            "solver_status_ok_rate": float(
                np.mean(
                    np.asarray(
                        [
                            (s in {"ok", "deterministic", "optimal"})
                            or ("optimal" in str(s).lower())
                            for s in res["solver_status"]
                        ],
                        dtype=float,
                    )
                )
            ),
            "guarantee_checks_passed_rate": float(np.mean(res["guarantee_checks_passed"])),
            "adaptive_width_mean": adaptive_width_mean,
            "adaptive_width_p95": adaptive_width_p95,
            "lambda_mw_used_mean": float(np.mean(res["lambda_mw_used"])) if len(res["lambda_mw_used"]) else np.nan,
            "rac_sensitivity_mean": float(np.mean(res["rac_sensitivity"])) if len(res["rac_sensitivity"]) else np.nan,
            "rac_sensitivity_p95": float(np.quantile(res["rac_sensitivity"], 0.95)) if len(res["rac_sensitivity"]) else np.nan,
            "rac_q_multiplier_mean": float(np.mean(res["rac_q_multiplier"])) if len(res["rac_q_multiplier"]) else np.nan,
            "rac_q_multiplier_p95": float(np.quantile(res["rac_q_multiplier"], 0.95)) if len(res["rac_q_multiplier"]) else np.nan,
            "rac_inflation_mean": float(np.mean(res["rac_inflation"])) if len(res["rac_inflation"]) else np.nan,
            "cvar_eta": cvar_eta_mean,
            "cvar_cost": cvar_cost_mean,
            **cqr_group_metrics,
            **metrics,
        }
        main_rows.append(row)

        violated = np.asarray(metrics["true_soc_violation_mask"], dtype=bool)
        interventions = (
            np.abs(np.asarray(res["proposed_charge_mw"], dtype=float) - np.asarray(res["safe_charge_mw"], dtype=float)) > 1e-6
        ) | (
            np.abs(np.asarray(res["proposed_discharge_mw"], dtype=float) - np.asarray(res["safe_discharge_mw"], dtype=float)) > 1e-6
        )

        for fault_col in FAULT_COLUMNS:
            mask = event_log[fault_col].to_numpy(dtype=int) > 0
            fault_rows.append(
                {
                    "scenario": scenario,
                    "seed": int(seed),
                    "controller": controller,
                    "fault_type": fault_col,
                    "fault_count": int(mask.sum()),
                    "true_soc_violation_rate_at_fault": float(np.mean(violated[mask])) if mask.any() else 0.0,
                    "violation_rate_at_fault": float(np.mean(violated[mask])) if mask.any() else 0.0,
                    "intervention_rate_at_fault": float(np.mean(interventions[mask])) if mask.any() else 0.0,
                }
            )

    return {"main_rows": main_rows, "fault_rows": fault_rows}


def _plot_calibration(main_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in main_df.groupby("controller", sort=True):
        ax.scatter(sub["picp_90"], sub["mean_interval_width"], label=controller, alpha=0.85)
    ax.axvline(0.90, color="black", linestyle="--", linewidth=1.2, label="Nominal 90%")
    ax.set_xlabel("PICP@90")
    ax.set_ylabel("Mean Interval Width")
    ax.set_title("CPSBench Calibration vs Sharpness")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_violation_vs_cost(main_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, sub in main_df.groupby("controller", sort=True):
        ax.scatter(sub["expected_cost_usd"], sub["true_soc_violation_rate"], label=controller, alpha=0.85)
    ax.set_xlabel("Expected Cost (USD)")
    ax.set_ylabel("True SOC Violation Rate")
    ax.set_title("Violation vs Cost Tradeoff")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_true_soc_curves(main_df: pd.DataFrame, out_violation: Path, out_severity: Path) -> None:
    subset = main_df[main_df["scenario"].isin(["dropout", "drift_combo"])].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    if subset.empty:
        ax.text(0.5, 0.5, "No dropout/drift_combo rows in run", ha="center", va="center")
    else:
        for controller, sub in subset.groupby("controller", sort=True):
            ax.plot(sub["seed"], sub["true_soc_violation_rate"], marker="o", linestyle="-", label=controller)
        ax.legend()
    ax.set_xlabel("Seed")
    ax.set_ylabel("True SOC Violation Rate")
    ax.set_title("True SOC Violation Rate (faulted telemetry)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_violation, dpi=220)
    plt.close(fig)

    sev_col = "true_soc_violation_severity_p95_mwh" if "true_soc_violation_severity_p95_mwh" in subset.columns else "true_soc_violation_severity_p95"
    fig, ax = plt.subplots(figsize=(8, 5))
    if subset.empty:
        ax.text(0.5, 0.5, "No dropout/drift_combo rows in run", ha="center", va="center")
    else:
        for controller, sub in subset.groupby("controller", sort=True):
            ax.plot(sub["seed"], sub[sev_col], marker="o", linestyle="-", label=controller)
        ax.legend()
    ax.set_xlabel("Seed")
    ax.set_ylabel("True SOC Violation Severity P95 (MWh)")
    ax.set_title("True SOC Violation Severity (P95)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_severity, dpi=220)
    plt.close(fig)


def _plot_sweep_metric(
    *,
    sweep_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    grouped = (
        sweep_df.groupby(["fault_dimension", "severity", "controller"], as_index=False)[metric_col]
        .mean(numeric_only=True)
        .sort_values(["fault_dimension", "severity", "controller"])
    )
    for (fault_dimension, controller), sub in grouped.groupby(["fault_dimension", "controller"], sort=True):
        ax.plot(
            sub["severity"].to_numpy(dtype=float),
            sub[metric_col].to_numpy(dtype=float),
            marker="o",
            label=f"{fault_dimension}:{controller}",
            alpha=0.9,
        )
    ax.set_xlabel("Fault Severity")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_fault_sweep(
    *,
    seeds: Iterable[int],
    horizon: int,
    out_dir: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dim, levels in FAULT_SWEEP_LEVELS.items():
        scenario = "dropout"
        if dim == "delay_seconds":
            scenario = "delay_jitter"
        elif dim == "out_of_order":
            scenario = "out_of_order"
        elif dim == "spike_sigma":
            scenario = "spikes"

        for level in levels:
            overrides: dict[str, Any] = {}
            if dim == "dropout":
                overrides["dropout_rate"] = float(level)
                overrides["soc_dropout_prob"] = float(level)
            elif dim == "delay_seconds":
                overrides["delay_seconds"] = float(level)
                overrides["delay_rate"] = 0.50 if float(level) > 0 else 0.0
                overrides["soc_stale_prob"] = 0.35 if float(level) >= 5.0 else 0.0
            elif dim == "out_of_order":
                overrides["out_of_order_rate"] = float(level)
            elif dim == "spike_sigma":
                overrides["spike_sigma"] = float(level)

            for seed in seeds:
                payload = run_single(
                    scenario=scenario,
                    seed=int(seed),
                    horizon=horizon,
                    fault_overrides=overrides,
                )
                for row in payload["main_rows"]:
                    rows.append(
                        {
                            "fault_dimension": dim,
                            "severity": float(level),
                            "scenario": scenario,
                            **row,
                        }
                    )

    sweep_df = (
        pd.DataFrame(rows)
        .sort_values(["fault_dimension", "severity", "scenario", "seed", "controller"])
        .reset_index(drop=True)
    )
    sweep_df.to_csv(out_dir / "cpsbench_merged_sweep.csv", index=False, float_format="%.6f")
    return sweep_df


def run_suite(
    *,
    scenarios: Iterable[str],
    seeds: Iterable[int],
    out_dir: str | Path = "reports/publication",
    horizon: int = 168,
    fault_overrides: dict[str, Any] | None = None,
    dc3s_param_overrides: dict[str, Any] | None = None,
    include_fault_sweep: bool = True,
) -> dict[str, Any]:
    """Run CPSBench suite and persist canonical publication artifacts.

    Args:
        scenarios: Scenario names to run (e.g. ["nominal", "dropout"]).
        seeds: Random seeds for episode generation.
        out_dir: Directory for output artifacts.
        horizon: Episode length in hours.
        fault_overrides: Optional fault parameter overrides passed to generate_episode.
            Supports keys from scenarios.py (e.g. load_scale, dropout_rate, etc.)
            Enables cross-region transfer evaluation (US-scale vs DE-scale).
        dc3s_param_overrides: Optional DC³S config overrides (e.g. k_quality, k_drift,
            infl_max). Enables ablation sweeps without modifying config files.
        include_fault_sweep: When False, skip the internal fault sweep and its extra
            artifacts. Default remains True for the canonical publication workflow.
    """
    out = _ensure_out_dir(out_dir)
    scenarios_list = list(scenarios)
    seeds_list = [int(s) for s in seeds]

    all_main_rows: list[dict[str, Any]] = []
    all_fault_rows: list[dict[str, Any]] = []
    for scenario in scenarios_list:
        for seed in seeds_list:
            payload = run_single(
                scenario=scenario,
                seed=int(seed),
                horizon=horizon,
                fault_overrides=fault_overrides,
                dc3s_overrides=dc3s_param_overrides,
            )
            all_main_rows.extend(payload["main_rows"])
            all_fault_rows.extend(payload["fault_rows"])


    main_df = pd.DataFrame(all_main_rows).sort_values(["scenario", "seed", "controller"]).reset_index(drop=True)
    if "true_soc_violation_mask" in main_df.columns:
        main_df = main_df.drop(columns=["true_soc_violation_mask"])
    fault_df = pd.DataFrame(all_fault_rows).sort_values(["scenario", "seed", "fault_type", "controller"]).reset_index(drop=True)

    main_csv = out / "dc3s_main_table.csv"
    fault_csv = out / "dc3s_fault_breakdown.csv"
    calibration_png = out / "calibration_plot.png"
    violation_png = out / "violation_vs_cost_curve.png"
    summary_json = out / "dc3s_run_summary.json"
    fig_violation_rate = out / "fig_violation_rate.png"
    fig_violation_severity = out / "fig_violation_severity_p95.png"
    fig_true_soc_rate = out / "fig_true_soc_violation_vs_dropout.png"
    fig_true_soc_severity = out / "fig_true_soc_severity_p95_vs_dropout.png"

    main_df.to_csv(main_csv, index=False, float_format="%.6f")
    fault_df.to_csv(fault_csv, index=False, float_format="%.6f")
    _plot_calibration(main_df, calibration_png)
    _plot_violation_vs_cost(main_df, violation_png)

    artifacts = {
        "dc3s_main_table.csv": str(main_csv),
        "dc3s_fault_breakdown.csv": str(fault_csv),
        "calibration_plot.png": str(calibration_png),
        "violation_vs_cost_curve.png": str(violation_png),
        "dc3s_run_summary.json": str(summary_json),
    }
    rows_sweep = 0
    if include_fault_sweep:
        sweep_df = run_fault_sweep(seeds=seeds_list, horizon=horizon, out_dir=out)
        rows_sweep = int(len(sweep_df))
        sev_col = "true_soc_violation_severity_p95_mwh" if "true_soc_violation_severity_p95_mwh" in sweep_df.columns else "true_soc_violation_severity_p95"
        _plot_sweep_metric(
            sweep_df=sweep_df,
            metric_col="true_soc_violation_rate",
            ylabel="True SOC Violation Rate",
            title="Violation Rate vs Fault Severity",
            out_path=fig_violation_rate,
        )
        _plot_sweep_metric(
            sweep_df=sweep_df,
            metric_col=sev_col,
            ylabel="True SOC Violation Severity P95 (MWh)",
            title="Violation Severity (P95) vs Fault Severity",
            out_path=fig_violation_severity,
        )
        _plot_true_soc_curves(main_df=main_df, out_violation=fig_true_soc_rate, out_severity=fig_true_soc_severity)
        artifacts.update(
            {
                "cpsbench_merged_sweep.csv": str(out / "cpsbench_merged_sweep.csv"),
                "fig_violation_rate.png": str(fig_violation_rate),
                "fig_violation_severity_p95.png": str(fig_violation_severity),
                "fig_true_soc_violation_vs_dropout.png": str(fig_true_soc_rate),
                "fig_true_soc_severity_p95_vs_dropout.png": str(fig_true_soc_severity),
            }
        )

    summary = {
        "scenarios": scenarios_list,
        "seeds": seeds_list,
        "horizon": int(horizon),
        "rows_main": int(len(main_df)),
        "rows_fault_breakdown": int(len(fault_df)),
        "rows_sweep": rows_sweep,
        "controller_summary": {
            controller: {
                "mean_true_soc_violation_rate": float(sub["true_soc_violation_rate"].mean()),
                "mean_true_soc_violation_severity_p95_mwh": float(
                    sub[
                        "true_soc_violation_severity_p95_mwh"
                        if "true_soc_violation_severity_p95_mwh" in sub.columns
                        else "true_soc_violation_severity_p95"
                    ].mean()
                ),
                "mean_intervention_rate": float(sub["intervention_rate"].mean()),
                "mean_cost_usd": float(sub["expected_cost_usd"].dropna().mean()) if not sub["expected_cost_usd"].dropna().empty else None,
            }
            for controller, sub in main_df.groupby("controller", sort=True)
        },
        "artifacts": artifacts,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPSBench-IoT benchmark")
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out-dir", type=str, default="reports/publication")
    parser.add_argument("--horizon", type=int, default=168)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_suite(
        scenarios=[args.scenario],
        seeds=[int(args.seed)],
        out_dir=args.out_dir,
        horizon=int(args.horizon),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
