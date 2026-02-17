"""Pipeline orchestration: run."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Any

import yaml

from gridpulse.utils.logging import setup_logging
from gridpulse.utils.registry import register_models

def _repo_root() -> Path:
    # Key: orchestrate end-to-end pipeline steps
    # run.py -> src/gridpulse/pipeline/run.py, so repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hash_paths(paths: Iterable[Path], base: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        if p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file():
                    rel = fp.relative_to(base)
                    h.update(str(rel).encode("utf-8"))
                    h.update(_hash_file(fp).encode("utf-8"))
        elif p.is_file():
            rel = p.relative_to(base)
            h.update(str(rel).encode("utf-8"))
            h.update(_hash_file(p).encode("utf-8"))
    return h.hexdigest()


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _pip_freeze(repo_root: Path, run_dir: Path, log: logging.Logger) -> None:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], cwd=repo_root)
        (run_dir / "pip_freeze.txt").write_bytes(out)
        log.info("Saved pip freeze to %s", run_dir / "pip_freeze.txt")
    except Exception as exc:
        log.warning("pip freeze failed: %s", exc)


def _snapshot_configs(repo_root: Path, run_dir: Path) -> None:
    cfg_dir = repo_root / "configs"
    if not cfg_dir.exists():
        return
    out_dir = run_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fp in cfg_dir.glob("*.yaml"):
        shutil.copy2(fp, out_dir / fp.name)


@dataclass
class SignalConfig:
    enabled: bool
    path: Path | None


@dataclass
class WeatherConfig:
    enabled: bool
    path: Path | None


@dataclass
class HolidaysConfig:
    enabled: bool
    path: Path | None
    country: str


@dataclass(frozen=True)
class ResearchDatasetSpec:
    key: str
    split_path: Path
    models_dir: Path
    forecast_cfg_path: Path
    output_csv: Path
    uncertainty_artifacts_dir: Path


def _load_signal_config(cfg_path: Path, repo_root: Path, log: logging.Logger) -> SignalConfig:
    if not cfg_path.exists():
        return SignalConfig(False, None)
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.warning("Failed to parse %s: %s", cfg_path, exc)
        return SignalConfig(False, None)
    signals_cfg = payload.get("signals", {}) if isinstance(payload, dict) else {}
    enabled = bool(signals_cfg.get("enabled", False))
    if not enabled:
        return SignalConfig(False, None)
    signal_path = signals_cfg.get("file")
    if not signal_path:
        log.warning("signals.enabled is true but signals.file is missing in %s", cfg_path)
        return SignalConfig(True, None)
    path = (repo_root / signal_path).resolve() if not Path(signal_path).is_absolute() else Path(signal_path)
    return SignalConfig(True, path)


def _load_weather_config(cfg_path: Path, repo_root: Path, log: logging.Logger) -> WeatherConfig:
    if not cfg_path.exists():
        return WeatherConfig(False, None)
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.warning("Failed to parse %s: %s", cfg_path, exc)
        return WeatherConfig(False, None)
    weather_cfg = payload.get("weather", {}) if isinstance(payload, dict) else {}
    enabled = bool(weather_cfg.get("enabled", False))
    if not enabled:
        return WeatherConfig(False, None)
    weather_path = weather_cfg.get("file")
    if not weather_path:
        log.warning("weather.enabled is true but weather.file is missing in %s", cfg_path)
        return WeatherConfig(True, None)
    path = (repo_root / weather_path).resolve() if not Path(weather_path).is_absolute() else Path(weather_path)
    return WeatherConfig(True, path)


def _load_holidays_config(cfg_path: Path, repo_root: Path, log: logging.Logger) -> HolidaysConfig:
    if not cfg_path.exists():
        return HolidaysConfig(False, None, "DE")
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.warning("Failed to parse %s: %s", cfg_path, exc)
        return HolidaysConfig(False, None, "DE")
    holidays_cfg = payload.get("holidays", {}) if isinstance(payload, dict) else {}
    enabled = bool(holidays_cfg.get("enabled", False))
    if not enabled:
        return HolidaysConfig(False, None, str(holidays_cfg.get("country", "DE")))
    holiday_path = holidays_cfg.get("file")
    country = str(holidays_cfg.get("country", "DE"))
    if not holiday_path:
        log.warning("holidays.enabled is true but holidays.file is missing in %s", cfg_path)
        return HolidaysConfig(True, None, country)
    path = (repo_root / holiday_path).resolve() if not Path(holiday_path).is_absolute() else Path(holiday_path)
    return HolidaysConfig(True, path, country)


def _run(cmd: list[str], log: logging.Logger) -> None:
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _snapshot_artifacts(repo_root: Path, run_dir: Path, log: logging.Logger) -> None:
    # copy models
    models_src = repo_root / "artifacts" / "models"
    models_dst = run_dir / "models"
    if models_src.exists():
        models_dst.mkdir(parents=True, exist_ok=True)
        for fp in models_src.glob("*"):
            if fp.is_file():
                shutil.copy2(fp, models_dst / fp.name)

    # copy reports
    reports_dst = run_dir / "reports"
    reports_dst.mkdir(parents=True, exist_ok=True)

    report_files = [
        repo_root / "reports" / "formal_evaluation_report.md",
        repo_root / "reports" / "ml_vs_dl_comparison.md",
        repo_root / "reports" / "multi_horizon_backtest.json",
        repo_root / "reports" / "walk_forward_report.json",
        repo_root / "reports" / "impact_comparison.md",
        repo_root / "reports" / "impact_comparison.json",
        repo_root / "reports" / "impact_summary.csv",
        repo_root / "reports" / "research_metrics.csv",
        repo_root / "reports" / "research_metrics_de.csv",
        repo_root / "reports" / "research_metrics_us.csv",
    ]
    for fp in report_files:
        if fp.exists():
            shutil.copy2(fp, reports_dst / fp.name)

    figures_src = repo_root / "reports" / "figures"
    if figures_src.exists():
        figures_dst = reports_dst / "figures"
        figures_dst.mkdir(parents=True, exist_ok=True)
        for fp in figures_src.glob("*"):
            if fp.is_file():
                shutil.copy2(fp, figures_dst / fp.name)

    cards_src = repo_root / "reports" / "model_cards"
    if cards_src.exists():
        cards_dst = reports_dst / "model_cards"
        cards_dst.mkdir(parents=True, exist_ok=True)
        for fp in cards_src.glob("*.md"):
            shutil.copy2(fp, cards_dst / fp.name)

    log.info("Snapshot saved to %s", run_dir)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def _resolve_gbm_model_path(target: str, forecast_cfg: dict[str, Any], models_dir: Path) -> Path:
    models = forecast_cfg.get("models", {}) if isinstance(forecast_cfg, dict) else {}
    explicit = models.get(target) if isinstance(models, dict) else None
    if explicit:
        candidate = Path(str(explicit))
        if not candidate.is_absolute():
            # models_dir is expected at <repo_root>/artifacts/models.
            repo_root = models_dir.parent.parent
            candidate = repo_root / candidate
        is_gbm_named = candidate.suffix.lower() == ".pkl" and "gbm" in candidate.stem.lower()
        if not is_gbm_named:
            raise RuntimeError(
                f"Configured model for {target} must be a GBM pickle (*.pkl with gbm in name): {candidate}"
            )
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Missing configured GBM model for {target}: {candidate}")

    matches = sorted(models_dir.glob(f"gbm_*_{target}.pkl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing GBM model for {target} under {models_dir}")


def _forecast_with_gbm(
    context_df,
    target: str,
    horizon: int,
    forecast_cfg: dict[str, Any],
    models_dir: Path,
) -> dict[str, Any]:
    import numpy as np

    from gridpulse.forecasting.predict import load_model_bundle, predict_next_24h

    model_path = _resolve_gbm_model_path(target, forecast_cfg, models_dir)
    bundle = load_model_bundle(model_path)
    if bundle.get("model_type") != "gbm":
        raise RuntimeError(f"Expected GBM model bundle for {target}, got {bundle.get('model_type')}")

    pred = predict_next_24h(context_df, bundle, horizon=horizon)
    yhat = np.asarray(pred.get("forecast", []), dtype=float)
    if yhat.size != horizon:
        raise RuntimeError(f"GBM forecast for {target} has length {yhat.size}, expected {horizon}")
    return pred


def _build_base_load_intervals(
    load_forecast,
    load_pred_payload: dict[str, Any],
    uncertainty_cfg: dict[str, Any],
    artifacts_dir: Path,
) -> tuple[Any, Any]:
    import numpy as np

    lower = np.asarray(load_forecast, dtype=float).copy()
    upper = np.asarray(load_forecast, dtype=float).copy()
    horizon = lower.size

    conformal_used = False
    if bool(uncertainty_cfg.get("enabled", False)):
        conformal_path = artifacts_dir / "load_mw_conformal.json"
        if conformal_path.exists():
            try:
                from gridpulse.forecasting.uncertainty import load_conformal

                ci = load_conformal(conformal_path)
                lo, hi = ci.predict_interval(np.asarray(load_forecast, dtype=float))
                lo_arr = np.asarray(lo, dtype=float)
                hi_arr = np.asarray(hi, dtype=float)
                if lo_arr.size == horizon and hi_arr.size == horizon:
                    lower, upper = lo_arr, hi_arr
                    conformal_used = True
            except Exception:
                conformal_used = False

    if not conformal_used:
        quantiles = load_pred_payload.get("quantiles", {}) if isinstance(load_pred_payload, dict) else {}
        q10 = quantiles.get("0.1") if isinstance(quantiles, dict) else None
        q90 = quantiles.get("0.9") if isinstance(quantiles, dict) else None
        if q10 is not None and q90 is not None:
            lo_arr = np.asarray(q10, dtype=float)
            hi_arr = np.asarray(q90, dtype=float)
            if lo_arr.size == horizon and hi_arr.size == horizon:
                lower, upper = lo_arr, hi_arr

    lower = np.maximum(lower, 0.0)
    upper = np.maximum(upper, lower)
    return lower, upper


def _resolve_calibration_npz_path(
    repo_root: Path,
    uncertainty_cfg: dict[str, Any],
    uncertainty_artifacts_dir: Path,
    target: str = "load_mw",
) -> Path | None:
    """Resolve calibration NPZ path used to learn FACI scale bounds."""
    template = str(
        uncertainty_cfg.get("calibration_npz", "artifacts/backtests/{target}_calibration.npz")
    )
    candidates: list[Path] = []

    try:
        templated = Path(template.format(target=target))
    except Exception:
        templated = Path(template)
    if not templated.is_absolute():
        templated = repo_root / templated
    candidates.append(templated)

    # Dataset-specific fallback if uncertainty artifacts point to eia930.
    if "eia930" in str(uncertainty_artifacts_dir).lower():
        candidates.append(repo_root / "artifacts" / "backtests_eia930" / f"{target}_calibration.npz")

    # Final fallback to default location.
    candidates.append(repo_root / "artifacts" / "backtests" / f"{target}_calibration.npz")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _estimate_faci_scale_bounds(
    repo_root: Path,
    uncertainty_cfg: dict[str, Any],
    uncertainty_artifacts_dir: Path,
    target: str,
    eps: float,
    log: logging.Logger,
) -> tuple[float, float] | None:
    """
    Learn interval-width scale bounds from calibration residuals.

    This is deliberately data-driven:
    - Uses historical calibration (y_true, y_pred) + conformal interval widths.
    - Computes required scale = |error| / half_width for each calibration point.
    - Uses quantiles of that empirical scale distribution as soft bounds.
    """
    import numpy as np

    faci_cfg = uncertainty_cfg.get("faci", {}) if isinstance(uncertainty_cfg, dict) else {}
    if not bool(faci_cfg.get("learn_scale_bounds", True)):
        return None

    q_low = float(faci_cfg.get("scale_lower_quantile", 0.10))
    q_high = float(faci_cfg.get("scale_upper_quantile", 0.90))
    min_samples = int(faci_cfg.get("min_scale_samples", 50))
    if not (0.0 <= q_low < q_high <= 1.0):
        log.warning("Invalid FACI scale quantiles (%s, %s); skipping learned bounds.", q_low, q_high)
        return None

    calibration_npz = _resolve_calibration_npz_path(
        repo_root=repo_root,
        uncertainty_cfg=uncertainty_cfg,
        uncertainty_artifacts_dir=uncertainty_artifacts_dir,
        target=target,
    )
    if calibration_npz is None:
        return None

    conformal_path = uncertainty_artifacts_dir / f"{target}_conformal.json"
    if not conformal_path.exists():
        return None

    try:
        from gridpulse.forecasting.uncertainty import load_conformal

        with np.load(calibration_npz) as data:
            y_true = np.asarray(data["y_true"], dtype=float).reshape(-1)
            y_pred = np.asarray(data["y_pred"], dtype=float).reshape(-1)

        if y_true.size == 0 or y_pred.size == 0:
            return None
        n = min(y_true.size, y_pred.size)
        y_true = y_true[:n]
        y_pred = y_pred[:n]

        ci = load_conformal(conformal_path)
        lo, hi = ci.predict_interval(y_pred)
        lo_arr = np.asarray(lo, dtype=float).reshape(-1)[:n]
        hi_arr = np.asarray(hi, dtype=float).reshape(-1)[:n]

        half_width = np.maximum(0.5 * (hi_arr - lo_arr), max(float(eps), 1e-9))
        required_half_width = np.abs(y_true - y_pred)
        scales = required_half_width / half_width
        scales = scales[np.isfinite(scales)]

        if scales.size < min_samples:
            return None

        lower = float(np.quantile(scales, q_low))
        upper = float(np.quantile(scales, q_high))
        lower = max(lower, max(float(eps), 1e-9))
        upper = max(upper, lower + max(float(eps), 1e-9))

        log.info(
            "Learned FACI scale bounds from %s (%d samples): [%.4f, %.4f]",
            calibration_npz,
            int(scales.size),
            lower,
            upper,
        )
        return (lower, upper)
    except Exception as exc:
        log.warning("Failed to learn FACI scale bounds; continuing without bounds: %s", exc)
        return None


def _apply_faci_online(
    load_true,
    base_lower,
    base_upper,
    alpha: float,
    gamma: float,
    eps: float,
    scale_bounds: tuple[float, float] | None = None,
    history_quantiles: tuple[float, float] = (0.05, 0.95),
    history_warmup: int = 8,
) -> tuple[Any, Any]:
    import numpy as np

    from gridpulse.forecasting.uncertainty import AdaptiveConformal

    load_true_arr = np.asarray(load_true, dtype=float)
    base_lower_arr = np.asarray(base_lower, dtype=float)
    base_upper_arr = np.asarray(base_upper, dtype=float)

    if load_true_arr.ndim != 1 or base_lower_arr.ndim != 1 or base_upper_arr.ndim != 1:
        raise ValueError("FACI inputs must be 1D arrays")
    horizon = load_true_arr.size
    if base_lower_arr.size != horizon or base_upper_arr.size != horizon:
        raise ValueError("FACI interval arrays must match load_true horizon")
    if np.any(base_lower_arr > base_upper_arr):
        raise ValueError("base_lower must be <= base_upper")
    q_low, q_high = history_quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError("history_quantiles must satisfy 0 <= low < high <= 1")
    if history_warmup < 0:
        raise ValueError("history_warmup must be >= 0")
    if scale_bounds is not None:
        lb, ub = scale_bounds
        if lb <= 0 or ub <= 0 or lb >= ub:
            raise ValueError("scale_bounds must satisfy 0 < lower < upper")

    dyn_lower = base_lower_arr.copy()
    dyn_upper = base_upper_arr.copy()
    dyn_lower[0] = max(dyn_lower[0], 0.0)
    dyn_upper[0] = max(dyn_upper[0], dyn_lower[0])

    faci = AdaptiveConformal(
        alpha=float(alpha),
        gamma=float(gamma),
        mode="global",
        eps=float(eps),
    )
    eps_safe = max(float(eps), 1e-9)
    observed_scales: list[float] = []

    for t in range(1, horizon):
        prev_lo = float(dyn_lower[t - 1])
        prev_hi = float(dyn_upper[t - 1])
        prev_width = max(prev_hi - prev_lo, eps_safe)

        upd_lo, upd_hi = faci.update(
            y_true=np.asarray([load_true_arr[t - 1]], dtype=float),
            y_pred_interval=(
                np.asarray([prev_lo], dtype=float),
                np.asarray([prev_hi], dtype=float),
            ),
        )
        upd_lo_val = float(np.asarray(upd_lo, dtype=float).reshape(-1)[0])
        upd_hi_val = float(np.asarray(upd_hi, dtype=float).reshape(-1)[0])
        upd_width = max(upd_hi_val - upd_lo_val, eps_safe)

        raw_scale = upd_width / prev_width

        # Bound scale using learned historical statistics. This avoids "hard caps"
        # and keeps adaptation tied to empirical residual behavior.
        lower_bound = None
        upper_bound = None
        if scale_bounds is not None:
            lower_bound, upper_bound = float(scale_bounds[0]), float(scale_bounds[1])
        if history_warmup > 0 and len(observed_scales) >= history_warmup:
            hist = np.asarray(observed_scales, dtype=float)
            hist_low = float(np.quantile(hist, q_low))
            hist_high = float(np.quantile(hist, q_high))
            lower_bound = hist_low if lower_bound is None else max(lower_bound, hist_low)
            upper_bound = hist_high if upper_bound is None else min(upper_bound, hist_high)

        # Always respect any available upper/lower limits.
        # If intersections collapse (upper < lower), collapse to a single point
        # instead of silently dropping one of the bounds.
        lower = -np.inf if lower_bound is None else float(lower_bound)
        upper = np.inf if upper_bound is None else float(upper_bound)
        if upper < lower:
            upper = lower
        scale = float(np.clip(raw_scale, lower, upper))
        observed_scales.append(scale)

        mid = 0.5 * (base_lower_arr[t] + base_upper_arr[t])
        half_width = max(0.5 * (base_upper_arr[t] - base_lower_arr[t]), eps_safe)
        dyn_lower[t] = mid - half_width * scale
        dyn_upper[t] = mid + half_width * scale

    dyn_lower = np.maximum(dyn_lower, 0.0)
    dyn_upper = np.maximum(dyn_upper, dyn_lower)
    return dyn_lower, dyn_upper


def _build_robust_dispatch_config(opt_cfg: dict[str, Any]):
    from gridpulse.optimizer.robust_dispatch import RobustDispatchConfig

    battery = opt_cfg.get("battery", {}) if isinstance(opt_cfg, dict) else {}
    grid = opt_cfg.get("grid", {}) if isinstance(opt_cfg, dict) else {}

    capacity = float(battery.get("capacity_mwh", 100.0))
    max_power = float(battery.get("max_power_mw", 50.0))
    max_charge = float(battery.get("max_charge_mw", max_power))
    max_discharge = float(battery.get("max_discharge_mw", max_power))
    # Prefer regime-A efficiency when available so robust and deterministic
    # optimizers share the Phase-3 battery physics baseline.
    efficiency = float(
        battery.get(
            "efficiency_regime_a",
            battery.get("efficiency", 0.95),
        )
    )
    min_soc = float(battery.get("min_soc_mwh", 0.0))
    initial_soc = float(battery.get("initial_soc_mwh", capacity / 2.0))
    max_soc = float(battery.get("max_soc_mwh", capacity))
    max_import = float(grid.get("max_import_mw", grid.get("max_draw_mw", 500.0)))
    default_price = float(grid.get("price_per_mwh", grid.get("price_usd_per_mwh", 60.0)))
    degradation = float(battery.get("degradation_cost_per_mwh", 10.0))

    return RobustDispatchConfig(
        battery_capacity_mwh=capacity,
        battery_max_charge_mw=max_charge,
        battery_max_discharge_mw=max_discharge,
        battery_charge_efficiency=efficiency,
        battery_discharge_efficiency=efficiency,
        battery_initial_soc_mwh=initial_soc,
        battery_min_soc_mwh=min_soc,
        battery_max_soc_mwh=max_soc,
        max_grid_import_mw=max_import,
        default_price_per_mwh=default_price,
        degradation_cost_per_mwh=degradation,
        risk_weight_worst_case=float(opt_cfg.get("robust", {}).get("risk_weight_worst_case", 1.0)),
        time_step_hours=1.0,
        solver_name="appsi_highs",
    )


def _merge_nested_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
        else:
            merged[key] = value
    return merged


def _resolve_research_operational_config(
    cfg: dict[str, Any],
    dataset_key: str,
) -> dict[str, Any]:
    research_op = cfg.get("research_operational", {}) if isinstance(cfg, dict) else {}
    if not isinstance(research_op, dict):
        return {}

    merged = {k: v for k, v in research_op.items() if k != "datasets"}
    datasets = research_op.get("datasets", {})
    if isinstance(datasets, dict):
        dataset_override = datasets.get(dataset_key, {})
        if isinstance(dataset_override, dict):
            merged = _merge_nested_dict(merged, dataset_override)
    return merged


def _apply_operational_load_stress(
    load_lower_bound,
    load_upper_bound,
    context_df,
    research_operational_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
    """
    Expand interval tails from empirical uncertainty and recent net-load volatility.

    This is intentionally data-driven:
    - Uses interval-width quantiles from FACI output.
    - Uses context net-load ramp quantiles to capture anomaly behavior.
    """
    import numpy as np

    lower = np.asarray(load_lower_bound, dtype=float).reshape(-1)
    upper = np.asarray(load_upper_bound, dtype=float).reshape(-1)
    if lower.size != upper.size:
        raise ValueError("load_lower_bound and load_upper_bound must have identical length")

    stress_cfg = (
        research_operational_cfg.get("stress", {})
        if isinstance(research_operational_cfg, dict)
        else {}
    )
    if not bool(stress_cfg.get("enabled", False)):
        return lower, upper, {
            "applied": False,
            "load_stress_additive_mw": 0.0,
            "stress_interval_multiplier": 1.0,
        }

    eps_safe = 1e-9
    width = np.maximum(upper - lower, eps_safe)
    width_q = min(max(float(stress_cfg.get("width_quantile", 0.75)), 0.0), 1.0)
    width_ref = float(np.quantile(width, width_q)) if width.size else 0.0

    vol_ref = 0.0
    if {"load_mw", "wind_mw", "solar_mw"}.issubset(set(context_df.columns)):
        net_load = (
            context_df["load_mw"].to_numpy(dtype=float)
            - context_df["wind_mw"].to_numpy(dtype=float)
            - context_df["solar_mw"].to_numpy(dtype=float)
        )
        if net_load.size >= 3:
            ramp = np.abs(np.diff(net_load))
            vol_q = min(max(float(stress_cfg.get("volatility_quantile", 0.90)), 0.0), 1.0)
            vol_ref = float(np.quantile(ramp, vol_q))

    # Blend interval and realized-volatility scales to avoid overreacting to
    # either source in isolation.
    base_scale = width_ref if vol_ref <= 0 else 0.5 * (width_ref + vol_ref)
    stress_scale = max(0.0, float(stress_cfg.get("stress_scale", 0.75)))
    additive = max(float(stress_cfg.get("min_additive_mw", 0.0)), stress_scale * base_scale)
    max_additive = float(stress_cfg.get("max_additive_mw", 0.0))
    if max_additive > 0:
        additive = min(additive, max_additive)

    lower_relax = min(max(float(stress_cfg.get("lower_relax_fraction", 0.15)), 0.0), 1.0)
    step_min = max(0.0, float(stress_cfg.get("step_scale_min", 0.85)))
    step_max = max(step_min, float(stress_cfg.get("step_scale_max", 1.25)))
    step_scale = np.clip(width / max(width_ref, eps_safe), step_min, step_max)

    stressed_upper = upper + additive * step_scale
    stressed_lower = np.maximum(0.0, lower - lower_relax * additive * step_scale)
    stressed_upper = np.maximum(stressed_upper, stressed_lower)

    # Cap interval widening to keep stress scenarios realistic and avoid
    # pathological over-conservatism.
    max_rel_expansion = max(0.0, float(stress_cfg.get("max_relative_expansion", 0.60)))
    base_width = np.maximum(upper - lower, eps_safe)
    capped_width = np.minimum(stressed_upper - stressed_lower, base_width * (1.0 + max_rel_expansion))
    mid = 0.5 * (stressed_upper + stressed_lower)
    stressed_lower = np.maximum(0.0, mid - 0.5 * capped_width)
    stressed_upper = np.maximum(stressed_lower, mid + 0.5 * capped_width)

    stress_interval_multiplier = float(np.mean((stressed_upper - stressed_lower) / width))
    return stressed_lower, stressed_upper, {
        "applied": True,
        "load_stress_additive_mw": float(additive),
        "stress_interval_multiplier": stress_interval_multiplier,
    }


def _build_window_operational_config(
    base_config: dict[str, Any],
    context_df,
    load_lower_bound,
    load_upper_bound,
    renewables_forecast,
    dataset_key: str,
    log: logging.Logger,
) -> dict[str, Any]:
    """
    Build a per-window optimization config where uncertainty has operational impact.

    This keeps the logic data-driven:
    - Grid cap is tightened from historical net-load quantiles (plus small headroom),
      bounded by the global cap and a feasibility floor from forecast net-load.
    - Battery reserve floor is applied as a minimum SoC fraction of capacity.
    """
    import copy
    import numpy as np

    cfg = copy.deepcopy(base_config if isinstance(base_config, dict) else {})
    battery = cfg.setdefault("battery", {})
    grid = cfg.setdefault("grid", {})
    research_op = _resolve_research_operational_config(cfg, dataset_key)
    load_lower_arr = np.asarray(load_lower_bound, dtype=float).reshape(-1)
    load_upper_arr = np.asarray(load_upper_bound, dtype=float).reshape(-1)
    if load_lower_arr.size != load_upper_arr.size:
        raise ValueError("load_lower_bound and load_upper_bound must align for operational config")

    reserve_cfg = research_op.get("reserve_soc", {}) if isinstance(research_op, dict) else {}
    if bool(reserve_cfg.get("enabled", True)):
        capacity = float(battery.get("capacity_mwh", 100.0))
        reserve_frac = float(reserve_cfg.get("min_soc_fraction", 0.05))
        reserve_abs = float(reserve_cfg.get("min_soc_mwh", 0.0))
        reserve_mwh = max(reserve_abs, max(0.0, reserve_frac) * max(0.0, capacity))
        battery["min_soc_mwh"] = max(float(battery.get("min_soc_mwh", 0.0)), reserve_mwh)
        if "initial_soc_mwh" in battery:
            battery["initial_soc_mwh"] = max(float(battery["initial_soc_mwh"]), float(battery["min_soc_mwh"]))

    terminal_cfg = research_op.get("terminal_soc", {}) if isinstance(research_op, dict) else {}
    if bool(terminal_cfg.get("enabled", True)):
        capacity = float(battery.get("capacity_mwh", 100.0))
        max_soc = float(battery.get("max_soc_mwh", capacity))
        target_frac = float(terminal_cfg.get("target_soc_fraction", 0.15))
        target_abs = float(terminal_cfg.get("target_soc_mwh", 0.0))
        target_base = max(target_abs, max(0.0, target_frac) * max(0.0, capacity))

        width = np.maximum(load_upper_arr - load_lower_arr, 0.0)
        width_q = min(max(float(terminal_cfg.get("uncertainty_quantile", 0.90)), 0.0), 1.0)
        width_ref = float(np.quantile(width, width_q)) if width.size else 0.0
        uncertainty_to_soc = max(0.0, float(terminal_cfg.get("uncertainty_to_soc_mwh_per_mw", 0.0)))
        uncertainty_buffer = uncertainty_to_soc * width_ref
        resolved_terminal_target = min(max_soc, target_base + uncertainty_buffer)

        # Optional hard enforcement through min SoC floor; by default terminal risk is soft (metric-time).
        if bool(terminal_cfg.get("enforce_as_min_soc", False)):
            battery["min_soc_mwh"] = max(float(battery.get("min_soc_mwh", 0.0)), resolved_terminal_target)
            if "initial_soc_mwh" in battery:
                battery["initial_soc_mwh"] = max(
                    float(battery["initial_soc_mwh"]),
                    float(battery["min_soc_mwh"]),
                )

        cfg.setdefault("research_operational", {})
        if not isinstance(cfg["research_operational"], dict):
            cfg["research_operational"] = {}
        cfg["research_operational"].setdefault("terminal_soc", {})
        if isinstance(cfg["research_operational"]["terminal_soc"], dict):
            cfg["research_operational"]["terminal_soc"]["resolved_target_mwh"] = float(resolved_terminal_target)

    cap_cfg = research_op.get("grid_cap", {}) if isinstance(research_op, dict) else {}
    if bool(cap_cfg.get("enabled", True)):
        q_hist = float(cap_cfg.get("net_load_quantile", 0.95))
        q_hist = min(max(q_hist, 0.0), 1.0)
        q_forecast = float(cap_cfg.get("forecast_feasibility_quantile", 0.90))
        q_forecast = min(max(q_forecast, 0.0), 1.0)
        headroom = max(0.0, float(cap_cfg.get("headroom_mw", 500.0)))
        feasibility_margin = max(0.0, float(cap_cfg.get("feasibility_margin_mw", 500.0)))
        min_cap = max(0.0, float(cap_cfg.get("min_cap_mw", 1000.0)))
        tightness = max(0.0, float(cap_cfg.get("tightness_factor", 1.0)))
        min_feasibility_ratio = min(max(float(cap_cfg.get("min_feasibility_ratio", 0.98)), 0.0), 1.0)

        static_cap = float(grid.get("max_import_mw", grid.get("max_draw_mw", 500.0)))
        if static_cap <= 0:
            static_cap = float("inf")

        if {"load_mw", "wind_mw", "solar_mw"}.issubset(set(context_df.columns)):
            hist_net = (
                context_df["load_mw"].to_numpy(dtype=float)
                - context_df["wind_mw"].to_numpy(dtype=float)
                - context_df["solar_mw"].to_numpy(dtype=float)
            )
            hist_net = np.maximum(hist_net, 0.0)
            cap_hist = float(np.quantile(hist_net, q_hist)) + headroom if hist_net.size else static_cap
        else:
            cap_hist = static_cap

        width = np.maximum(load_upper_arr - load_lower_arr, 0.0)
        width_q = min(max(float(cap_cfg.get("uncertainty_width_quantile", 0.90)), 0.0), 1.0)
        width_ref = float(np.quantile(width, width_q)) if width.size else 0.0
        uncertainty_headroom_factor = max(0.0, float(cap_cfg.get("uncertainty_headroom_factor", 0.0)))
        uncertainty_headroom = uncertainty_headroom_factor * width_ref

        forecast_net = np.maximum(load_upper_arr - np.asarray(renewables_forecast, dtype=float), 0.0)
        cap_feasible = (
            float(np.quantile(forecast_net, q_forecast)) + feasibility_margin + uncertainty_headroom
            if forecast_net.size
            else 0.0
        )

        adaptive_cap = max(min_cap, cap_hist, cap_feasible) * tightness
        adaptive_cap = max(adaptive_cap, min_cap, cap_feasible * min_feasibility_ratio)
        adaptive_cap = min(adaptive_cap, static_cap)
        grid["max_import_mw"] = float(adaptive_cap)
        log.debug(
            "Applied adaptive grid cap %.2f MW (hist=%.2f, feasible=%.2f, static=%.2f, tightness=%.3f)",
            adaptive_cap,
            cap_hist,
            cap_feasible,
            static_cap,
            tightness,
        )

    return cfg


def _optimize_robust_dispatch(*args, **kwargs):
    from gridpulse.optimizer.robust_dispatch import optimize_robust_dispatch

    return optimize_robust_dispatch(*args, **kwargs)


def _calculate_vss(*args, **kwargs):
    from gridpulse.evaluation.regret import calculate_vss

    return calculate_vss(*args, **kwargs)


def _calculate_evpi(*args, **kwargs):
    from gridpulse.evaluation.regret import calculate_evpi

    return calculate_evpi(*args, **kwargs)


def _build_research_dataset_specs(
    repo_root: Path,
    uncertainty_cfg: dict[str, Any],
) -> dict[str, ResearchDatasetSpec]:
    artifacts_dir_cfg = uncertainty_cfg.get("artifacts_dir", "artifacts/uncertainty")
    base_uncertainty_dir = Path(str(artifacts_dir_cfg))
    if not base_uncertainty_dir.is_absolute():
        base_uncertainty_dir = repo_root / base_uncertainty_dir

    return {
        "de": ResearchDatasetSpec(
            key="de",
            split_path=repo_root / "data" / "processed" / "splits" / "test.parquet",
            models_dir=repo_root / "artifacts" / "models",
            forecast_cfg_path=repo_root / "configs" / "forecast.yaml",
            output_csv=repo_root / "reports" / "research_metrics_de.csv",
            uncertainty_artifacts_dir=base_uncertainty_dir,
        ),
        "us": ResearchDatasetSpec(
            key="us",
            split_path=repo_root / "data" / "processed" / "us_eia930" / "splits" / "test.parquet",
            models_dir=repo_root / "artifacts" / "models_eia930",
            forecast_cfg_path=repo_root / "configs" / "forecast_eia930.yaml",
            output_csv=repo_root / "reports" / "research_metrics_us.csv",
            uncertainty_artifacts_dir=base_uncertainty_dir / "eia930",
        ),
    }


def _run_research_step(
    repo_root: Path,
    run_id: str,
    log: logging.Logger,
    research_horizon: int,
    research_window_step: int,
    research_gamma: float,
    split_path: Path,
    models_dir: Path,
    output_csv: Path,
    uncertainty_artifacts_dir: Path,
    deterministic_config: dict[str, Any],
    forecast_cfg: dict[str, Any],
    uncertainty_cfg: dict[str, Any],
    dataset_key: str = "de",
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    if research_horizon <= 0:
        raise ValueError("research_horizon must be > 0")
    if research_window_step <= 0:
        raise ValueError("research_window_step must be > 0")
    if research_gamma < 0:
        raise ValueError("research_gamma must be >= 0")

    if not split_path.exists():
        raise FileNotFoundError(f"Missing research split: {split_path}")

    df = pd.read_parquet(split_path).sort_values("timestamp").reset_index(drop=True)
    required_cols = ["timestamp", "load_mw", "wind_mw", "solar_mw"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Research split missing required columns: {missing_cols}")

    window_starts = list(range(research_horizon, len(df) - research_horizon + 1, research_window_step))
    if not window_starts:
        raise RuntimeError(
            f"Not enough rows in test split ({len(df)}) for horizon {research_horizon} "
            f"with step {research_window_step}"
        )

    conformal_cfg = uncertainty_cfg.get("conformal", {}) if isinstance(uncertainty_cfg, dict) else {}
    faci_cfg = uncertainty_cfg.get("faci", {}) if isinstance(uncertainty_cfg, dict) else {}
    alpha = float(conformal_cfg.get("alpha", 0.10))
    eps = float(conformal_cfg.get("eps", 1e-6))
    history_q_low = float(faci_cfg.get("history_lower_quantile", 0.05))
    history_q_high = float(faci_cfg.get("history_upper_quantile", 0.95))
    history_warmup = int(faci_cfg.get("history_warmup", 8))
    artifacts_dir = uncertainty_artifacts_dir
    learned_scale_bounds = _estimate_faci_scale_bounds(
        repo_root=repo_root,
        uncertainty_cfg=uncertainty_cfg,
        uncertainty_artifacts_dir=artifacts_dir,
        target="load_mw",
        eps=eps,
        log=log,
    )

    unmet_penalty = 10000.0
    now_utc = datetime.utcnow().isoformat()
    rows: list[dict[str, Any]] = []

    for idx, start in enumerate(window_starts):
        end = start + research_horizon
        context_df = df.iloc[:start]
        window_df = df.iloc[start:end]

        load_pred = _forecast_with_gbm(context_df, "load_mw", research_horizon, forecast_cfg, models_dir)
        wind_pred = _forecast_with_gbm(context_df, "wind_mw", research_horizon, forecast_cfg, models_dir)
        solar_pred = _forecast_with_gbm(context_df, "solar_mw", research_horizon, forecast_cfg, models_dir)

        load_forecast = np.asarray(load_pred["forecast"], dtype=float)
        wind_forecast = np.asarray(wind_pred["forecast"], dtype=float)
        solar_forecast = np.asarray(solar_pred["forecast"], dtype=float)
        renewables_forecast = wind_forecast + solar_forecast

        load_true = window_df["load_mw"].to_numpy(dtype=float)
        renewables_true = (
            window_df["wind_mw"].to_numpy(dtype=float)
            + window_df["solar_mw"].to_numpy(dtype=float)
        )

        base_lower, base_upper = _build_base_load_intervals(
            load_forecast=load_forecast,
            load_pred_payload=load_pred,
            uncertainty_cfg=uncertainty_cfg,
            artifacts_dir=artifacts_dir,
        )
        dyn_lower, dyn_upper = _apply_faci_online(
            load_true=load_true,
            base_lower=base_lower,
            base_upper=base_upper,
            alpha=alpha,
            gamma=research_gamma,
            eps=eps,
            scale_bounds=learned_scale_bounds,
            history_quantiles=(history_q_low, history_q_high),
            history_warmup=history_warmup,
        )
        research_op_cfg = _resolve_research_operational_config(deterministic_config, dataset_key)
        stressed_lower, stressed_upper, stress_meta = _apply_operational_load_stress(
            load_lower_bound=dyn_lower,
            load_upper_bound=dyn_upper,
            context_df=context_df,
            research_operational_cfg=research_op_cfg,
        )

        window_config = _build_window_operational_config(
            base_config=deterministic_config,
            context_df=context_df,
            load_lower_bound=stressed_lower,
            load_upper_bound=stressed_upper,
            renewables_forecast=renewables_forecast,
            dataset_key=dataset_key,
            log=log,
        )
        robust_cfg = _build_robust_dispatch_config(window_config)
        window_grid_cfg = window_config.get("grid", {}) if isinstance(window_config, dict) else {}
        window_battery_cfg = window_config.get("battery", {}) if isinstance(window_config, dict) else {}
        window_research_cfg = (
            window_config.get("research_operational", {})
            if isinstance(window_config, dict)
            else {}
        )
        window_terminal_cfg = (
            window_research_cfg.get("terminal_soc", {})
            if isinstance(window_research_cfg, dict)
            else {}
        )
        operational_grid_cap = float(
            window_grid_cfg.get("max_import_mw", window_grid_cfg.get("max_draw_mw", np.nan))
        )
        reserve_soc_mwh = float(window_battery_cfg.get("min_soc_mwh", 0.0))
        terminal_soc_target_mwh = float(window_terminal_cfg.get("resolved_target_mwh", np.nan))

        if "price_eur_mwh" in window_df.columns:
            price = window_df["price_eur_mwh"].to_numpy(dtype=float)
        elif "price_usd_mwh" in window_df.columns:
            price = window_df["price_usd_mwh"].to_numpy(dtype=float)
        else:
            price = None
        if price is not None:
            if np.any(price < 0):
                log.warning(
                    "Negative price values detected in %s; clipping to 0.0 for robust optimization.",
                    split_path,
                )
            price = np.maximum(price, 0.0)

        robust_solution = _optimize_robust_dispatch(
            load_lower_bound=stressed_lower,
            load_upper_bound=stressed_upper,
            renewables_forecast=renewables_forecast,
            price=price,
            config=robust_cfg,
            verbose=False,
        )
        if not bool(robust_solution.get("feasible", False)):
            raise RuntimeError(
                f"Robust dispatch infeasible on window {idx}: {robust_solution.get('solver_status')}"
            )

        vss = _calculate_vss(
            load_true=load_true,
            renewables_true=renewables_true,
            load_forecast=load_forecast,
            renewables_forecast=renewables_forecast,
            load_lower_bound=stressed_lower,
            load_upper_bound=stressed_upper,
            price=price,
            deterministic_config=window_config,
            robust_config=robust_cfg,
            unmet_load_penalty_per_mwh=unmet_penalty,
        )
        evpi_robust = _calculate_evpi(
            load_true=load_true,
            renewables_true=renewables_true,
            load_forecast=load_forecast,
            renewables_forecast=renewables_forecast,
            load_lower_bound=stressed_lower,
            load_upper_bound=stressed_upper,
            price=price,
            deterministic_config=window_config,
            robust_config=robust_cfg,
            unmet_load_penalty_per_mwh=unmet_penalty,
            actual_model="robust",
        )
        evpi_deterministic = _calculate_evpi(
            load_true=load_true,
            renewables_true=renewables_true,
            load_forecast=load_forecast,
            renewables_forecast=renewables_forecast,
            load_lower_bound=stressed_lower,
            load_upper_bound=stressed_upper,
            price=price,
            deterministic_config=window_config,
            robust_config=robust_cfg,
            unmet_load_penalty_per_mwh=unmet_penalty,
            actual_model="deterministic",
        )

        rows.append(
            {
                "run_id": run_id,
                "timestamp_utc": now_utc,
                "row_type": "window",
                "window_idx": int(idx),
                "window_start": int(start),
                "window_end": int(end),
                "horizon": int(research_horizon),
                "evpi": float(evpi_robust["evpi"]),
                "evpi_robust": float(evpi_robust["evpi"]),
                "evpi_deterministic": float(evpi_deterministic["evpi"]),
                "vss": float(vss["vss"]),
                "robust_actual_realized_cost": float(evpi_robust["actual_realized_cost"]),
                "robust_perfect_info_cost": float(evpi_robust["perfect_info_cost"]),
                "deterministic_actual_realized_cost": float(evpi_deterministic["actual_realized_cost"]),
                "deterministic_perfect_info_cost": float(evpi_deterministic["perfect_info_cost"]),
                "robust_total_cost": float(robust_solution["total_cost"]),
                "robust_feasible": bool(robust_solution.get("feasible", False)),
                "solver_status": str(robust_solution.get("solver_status", "")),
                "mean_dynamic_interval_width": float(np.mean(dyn_upper - dyn_lower)),
                "mean_base_interval_width": float(np.mean(base_upper - base_lower)),
                "mean_stressed_interval_width": float(np.mean(stressed_upper - stressed_lower)),
                "operational_grid_cap_mw": operational_grid_cap,
                "reserve_soc_mwh": reserve_soc_mwh,
                "terminal_soc_target_mwh": terminal_soc_target_mwh,
                "load_stress_additive_mw": float(stress_meta.get("load_stress_additive_mw", 0.0)),
                "stress_interval_multiplier": float(stress_meta.get("stress_interval_multiplier", 1.0)),
                "faci_scale_bound_lower": (
                    float(learned_scale_bounds[0]) if learned_scale_bounds is not None else np.nan
                ),
                "faci_scale_bound_upper": (
                    float(learned_scale_bounds[1]) if learned_scale_bounds is not None else np.nan
                ),
                "unmet_load_penalty_per_mwh": float(unmet_penalty),
            }
        )

    new_df = pd.DataFrame(rows)
    summary_numeric_cols = [
        "horizon",
        "evpi",
        "evpi_robust",
        "evpi_deterministic",
        "vss",
        "robust_actual_realized_cost",
        "robust_perfect_info_cost",
        "deterministic_actual_realized_cost",
        "deterministic_perfect_info_cost",
        "robust_total_cost",
        "mean_dynamic_interval_width",
        "mean_base_interval_width",
        "mean_stressed_interval_width",
        "operational_grid_cap_mw",
        "reserve_soc_mwh",
        "terminal_soc_target_mwh",
        "load_stress_additive_mw",
        "stress_interval_multiplier",
        "faci_scale_bound_lower",
        "faci_scale_bound_upper",
    ]
    summary_row: dict[str, Any] = {
        "run_id": run_id,
        "timestamp_utc": now_utc,
        "row_type": "run_summary",
        "window_idx": None,
        "window_start": None,
        "window_end": None,
        "robust_feasible": bool(new_df["robust_feasible"].all()),
        "solver_status": "summary",
        "operational_grid_cap_mw": float(pd.to_numeric(new_df["operational_grid_cap_mw"], errors="coerce").mean()),
        "reserve_soc_mwh": float(pd.to_numeric(new_df["reserve_soc_mwh"], errors="coerce").mean()),
        "terminal_soc_target_mwh": float(pd.to_numeric(new_df["terminal_soc_target_mwh"], errors="coerce").mean()),
        "load_stress_additive_mw": float(pd.to_numeric(new_df["load_stress_additive_mw"], errors="coerce").mean()),
        "stress_interval_multiplier": float(
            pd.to_numeric(new_df["stress_interval_multiplier"], errors="coerce").mean()
        ),
        "faci_scale_bound_lower": (
            float(learned_scale_bounds[0]) if learned_scale_bounds is not None else np.nan
        ),
        "faci_scale_bound_upper": (
            float(learned_scale_bounds[1]) if learned_scale_bounds is not None else np.nan
        ),
        "unmet_load_penalty_per_mwh": float(unmet_penalty),
    }
    for col in summary_numeric_cols:
        summary_row[col] = float(pd.to_numeric(new_df[col], errors="coerce").mean())

    new_df = pd.concat([new_df, pd.DataFrame([summary_row])], ignore_index=True)
    column_order = [
        "run_id",
        "timestamp_utc",
        "row_type",
        "window_idx",
        "window_start",
        "window_end",
        "horizon",
        "evpi",
        "evpi_robust",
        "evpi_deterministic",
        "vss",
        "robust_actual_realized_cost",
        "robust_perfect_info_cost",
        "deterministic_actual_realized_cost",
        "deterministic_perfect_info_cost",
        "robust_total_cost",
        "robust_feasible",
        "solver_status",
        "mean_dynamic_interval_width",
        "mean_base_interval_width",
        "mean_stressed_interval_width",
        "operational_grid_cap_mw",
        "reserve_soc_mwh",
        "terminal_soc_target_mwh",
        "load_stress_additive_mw",
        "stress_interval_multiplier",
        "faci_scale_bound_lower",
        "faci_scale_bound_upper",
        "unmet_load_penalty_per_mwh",
    ]
    new_df = new_df[column_order]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        for col in column_order:
            if col not in existing.columns:
                existing[col] = np.nan
        merged = pd.concat([existing[column_order], new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(output_csv, index=False)

    log.info(
        "Research metrics appended to %s (%d window rows + 1 summary row)",
        output_csv,
        len(rows),
    )
    return {
        "rows_written": int(len(new_df)),
        "window_rows": int(len(rows)),
        "output_csv": str(output_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GridPulse pipeline orchestrator")
    parser.add_argument(
        "--steps",
        default="data,train,reports,research",
        help="Comma-separated steps: data,train,reports,research",
    )
    parser.add_argument("--all", action="store_true", help="Run all steps (data,train,reports,research)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if cache is unchanged")
    parser.add_argument("--run-id", default=None, help="Override run id (default: timestamp)")
    parser.add_argument(
        "--research-horizon",
        type=int,
        default=None,
        help="Research evaluation horizon; defaults to task.horizon_hours from train config (fallback 24)",
    )
    parser.add_argument(
        "--research-window-step",
        type=int,
        default=None,
        help="Rolling research window stride; defaults to research horizon",
    )
    parser.add_argument(
        "--research-gamma",
        type=float,
        default=0.05,
        help="AdaptiveConformal FACI step size",
    )
    parser.add_argument(
        "--research-datasets",
        default="de,us",
        help="Comma-separated research datasets to process: de,us",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    steps = (
        ["data", "train", "reports", "research"]
        if args.all
        else [s.strip() for s in args.steps.split(",") if s.strip()]
    )

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(file_path=str(run_dir / "pipeline.log"))
    log = logging.getLogger("gridpulse.pipeline")

    cache_path = repo_root / ".cache" / "pipeline.json"
    cache = _load_cache(cache_path)

    raw_csv = repo_root / "data" / "raw" / "time_series_60min_singleindex.csv"
    data_cfg = repo_root / "configs" / "data.yaml"
    train_cfg = repo_root / "configs" / "train_forecast.yaml"
    uncertainty_cfg_path = repo_root / "configs" / "uncertainty.yaml"
    optimization_cfg_path = repo_root / "configs" / "optimization.yaml"
    signal_cfg = _load_signal_config(data_cfg, repo_root, log)
    weather_cfg = _load_weather_config(data_cfg, repo_root, log)
    holidays_cfg = _load_holidays_config(data_cfg, repo_root, log)

    raw_hash = _hash_paths([raw_csv], repo_root) if raw_csv.exists() else None
    data_cfg_hash = _hash_paths([data_cfg], repo_root) if data_cfg.exists() else None
    train_cfg_hash = _hash_paths([train_cfg], repo_root) if train_cfg.exists() else None
    signals_hash = (
        _hash_paths([signal_cfg.path], repo_root) if signal_cfg.enabled and signal_cfg.path and signal_cfg.path.exists() else None
    )

    features_path = repo_root / "data" / "processed" / "features.parquet"
    research_results: dict[str, dict[str, Any]] = {}

    if "data" in steps:
        if not raw_csv.exists():
            raise FileNotFoundError(f"Missing raw CSV: {raw_csv}")
        if (
            not args.force
            and cache.get("raw_hash") == raw_hash
            and cache.get("data_cfg_hash") == data_cfg_hash
            and cache.get("signals_hash") == signals_hash
            and features_path.exists()
        ):
            log.info("Data step skipped (cache hit)")
        else:
            _run([sys.executable, "-m", "gridpulse.data_pipeline.validate_schema", "--in", "data/raw", "--report", "reports/data_quality_report.md"], log)
            build_cmd = [
                sys.executable,
                "-m",
                "gridpulse.data_pipeline.build_features",
                "--in",
                "data/raw",
                "--out",
                "data/processed",
            ]
            if signal_cfg.enabled:
                if signal_cfg.path and signal_cfg.path.exists():
                    build_cmd += ["--signals", str(signal_cfg.path)]
                else:
                    log.warning("Signals enabled but file missing; proceeding without signals.")
            if weather_cfg.enabled:
                if weather_cfg.path and weather_cfg.path.exists():
                    build_cmd += ["--weather", str(weather_cfg.path)]
                else:
                    log.warning("Weather enabled but file missing; proceeding without weather.")
            if holidays_cfg.enabled:
                if holidays_cfg.path and holidays_cfg.path.exists():
                    build_cmd += ["--holidays", str(holidays_cfg.path)]
                else:
                    log.warning("Holidays enabled but file missing; proceeding without holidays.")
                build_cmd += ["--holiday-country", holidays_cfg.country]
            _run(build_cmd, log)
            _run([sys.executable, "-m", "gridpulse.data_pipeline.split_time_series", "--in", "data/processed/features.parquet", "--out", "data/processed/splits"], log)

        cache["raw_hash"] = raw_hash
        cache["data_cfg_hash"] = data_cfg_hash
        cache["signals_hash"] = signals_hash
        if features_path.exists():
            cache["features_hash"] = _hash_paths([features_path], repo_root)

    if "train" in steps:
        if not features_path.exists():
            raise FileNotFoundError("Missing data/processed/features.parquet. Run data step first.")
        train_hash = _hash_paths([features_path, train_cfg], repo_root)
        if not args.force and cache.get("train_hash") == train_hash and (repo_root / "artifacts" / "models").exists():
            log.info("Train step skipped (cache hit)")
        else:
            _run([sys.executable, "-m", "gridpulse.forecasting.train", "--config", "configs/train_forecast.yaml"], log)
            register_models(repo_root / "artifacts" / "models", repo_root / "artifacts" / "registry" / "models.json", run_id=run_id)
        cache["train_hash"] = train_hash

    if "reports" in steps:
        if not args.force and cache.get("reports_hash") == cache.get("train_hash"):
            log.info("Reports step skipped (cache hit)")
        else:
            _run([sys.executable, "scripts/build_reports.py"], log)
        cache["reports_hash"] = cache.get("train_hash")

    if "research" in steps:
        uncertainty_cfg = _load_yaml(uncertainty_cfg_path)
        deterministic_cfg = _load_yaml(optimization_cfg_path)
        dataset_specs = _build_research_dataset_specs(repo_root, uncertainty_cfg)

        research_datasets = [d.strip().lower() for d in str(args.research_datasets).split(",") if d.strip()]
        if not research_datasets:
            raise ValueError("research_datasets must include at least one of: de,us")
        invalid = [d for d in research_datasets if d not in dataset_specs]
        if invalid:
            raise ValueError(f"Unsupported research_datasets: {invalid}. Allowed values: {sorted(dataset_specs)}")

        task_cfg = _load_yaml(train_cfg).get("task", {})
        default_horizon = int(task_cfg.get("horizon_hours", 24)) if isinstance(task_cfg, dict) else 24
        research_horizon = int(args.research_horizon or default_horizon)
        research_window_step = int(args.research_window_step or research_horizon)
        if research_horizon <= 0:
            raise ValueError("research_horizon must be > 0")
        if research_window_step <= 0:
            raise ValueError("research_window_step must be > 0")

        for dataset in research_datasets:
            spec = dataset_specs[dataset]
            if not spec.split_path.exists():
                raise FileNotFoundError(f"Missing research split for {dataset}: {spec.split_path}")
            if not spec.forecast_cfg_path.exists():
                raise FileNotFoundError(f"Missing forecast config for {dataset}: {spec.forecast_cfg_path}")

            forecast_cfg = _load_yaml(spec.forecast_cfg_path)
            load_model = _resolve_gbm_model_path("load_mw", forecast_cfg, spec.models_dir)
            wind_model = _resolve_gbm_model_path("wind_mw", forecast_cfg, spec.models_dir)
            solar_model = _resolve_gbm_model_path("solar_mw", forecast_cfg, spec.models_dir)

            research_hash = _hash_paths(
                [
                    spec.split_path,
                    spec.forecast_cfg_path,
                    uncertainty_cfg_path,
                    optimization_cfg_path,
                    load_model,
                    wind_model,
                    solar_model,
                ],
                repo_root,
            )

            cache_key = f"research_hash_{dataset}"
            if (
                not args.force
                and cache.get(cache_key) == research_hash
                and spec.output_csv.exists()
            ):
                log.info("Research step for %s skipped (cache hit)", dataset)
                research_results[dataset] = {
                    "rows_written": 0,
                    "window_rows": 0,
                    "output_csv": str(spec.output_csv),
                }
            else:
                research_results[dataset] = _run_research_step(
                    repo_root=repo_root,
                    run_id=run_id,
                    log=log,
                    research_horizon=research_horizon,
                    research_window_step=research_window_step,
                    research_gamma=float(args.research_gamma),
                    split_path=spec.split_path,
                    models_dir=spec.models_dir,
                    output_csv=spec.output_csv,
                    uncertainty_artifacts_dir=spec.uncertainty_artifacts_dir,
                    deterministic_config=deterministic_cfg,
                    forecast_cfg=forecast_cfg,
                    uncertainty_cfg=uncertainty_cfg,
                    dataset_key=dataset,
                )
            cache[cache_key] = research_hash

        cache["research_hash"] = "|".join(
            f"{dataset}:{cache.get(f'research_hash_{dataset}', '')}" for dataset in research_datasets
        )
        cache["research_last_run_id"] = run_id

    # snapshot outputs
    _snapshot_artifacts(repo_root, run_dir, log)
    _snapshot_configs(repo_root, run_dir)
    _pip_freeze(repo_root, run_dir, log)

    research_rows_written_total = int(
        sum(int(result.get("rows_written", 0)) for result in research_results.values())
    )
    research_outputs = {dataset: result.get("output_csv") for dataset, result in research_results.items()}

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "git_commit": _git_commit(repo_root),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "raw_hash": cache.get("raw_hash"),
        "data_cfg_hash": cache.get("data_cfg_hash"),
        "features_hash": cache.get("features_hash"),
        "train_hash": cache.get("train_hash"),
        "reports_hash": cache.get("reports_hash"),
        "research_hash": cache.get("research_hash"),
        "research_hash_de": cache.get("research_hash_de"),
        "research_hash_us": cache.get("research_hash_us"),
        "signals_hash": cache.get("signals_hash"),
        "research_rows_written": research_rows_written_total,
        "research_rows_written_de": int(research_results.get("de", {}).get("rows_written", 0)),
        "research_rows_written_us": int(research_results.get("us", {}).get("rows_written", 0)),
        "research_output_csv": next(iter(research_outputs.values()), None),
        "research_outputs": research_outputs,
        "steps": steps,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _save_cache(cache_path, cache)
    log.info("Pipeline complete: %s", run_id)


if __name__ == "__main__":
    main()
