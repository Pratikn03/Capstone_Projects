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


def _apply_faci_online(
    load_true,
    base_lower,
    base_upper,
    alpha: float,
    gamma: float,
    eps: float,
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

        scale = upd_width / prev_width
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
    efficiency = float(battery.get("efficiency", battery.get("efficiency_regime_a", 0.95)))
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
        time_step_hours=1.0,
        solver_name="appsi_highs",
    )


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

    robust_cfg = _build_robust_dispatch_config(deterministic_config)

    conformal_cfg = uncertainty_cfg.get("conformal", {}) if isinstance(uncertainty_cfg, dict) else {}
    alpha = float(conformal_cfg.get("alpha", 0.10))
    eps = float(conformal_cfg.get("eps", 1e-6))
    artifacts_dir = uncertainty_artifacts_dir

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
        )

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
            load_lower_bound=dyn_lower,
            load_upper_bound=dyn_upper,
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
            load_lower_bound=dyn_lower,
            load_upper_bound=dyn_upper,
            price=price,
            deterministic_config=deterministic_config,
            robust_config=robust_cfg,
            unmet_load_penalty_per_mwh=unmet_penalty,
        )
        evpi_robust = _calculate_evpi(
            load_true=load_true,
            renewables_true=renewables_true,
            load_forecast=load_forecast,
            renewables_forecast=renewables_forecast,
            load_lower_bound=dyn_lower,
            load_upper_bound=dyn_upper,
            price=price,
            deterministic_config=deterministic_config,
            robust_config=robust_cfg,
            unmet_load_penalty_per_mwh=unmet_penalty,
            actual_model="robust",
        )
        evpi_deterministic = _calculate_evpi(
            load_true=load_true,
            renewables_true=renewables_true,
            load_forecast=load_forecast,
            renewables_forecast=renewables_forecast,
            load_lower_bound=dyn_lower,
            load_upper_bound=dyn_upper,
            price=price,
            deterministic_config=deterministic_config,
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
