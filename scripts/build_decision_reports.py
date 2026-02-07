"""Generate decision robustness and metrics reports using real dispatch evaluation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.forecasting.predict import load_model_bundle, predict_next_24h
from gridpulse.forecasting.uncertainty.conformal import load_conformal
from gridpulse.optimizer.baselines import grid_only_dispatch
from gridpulse.optimizer.impact import impact_summary
from gridpulse.optimizer.lp_dispatch import optimize_dispatch


def _load_cfg(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _resolve_model_path(target: str, cfg: dict, models_dir: Path) -> Path | None:
    explicit = cfg.get("models", {}).get(target)
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
    order = cfg.get("fallback_order", ["lstm", "tcn", "gbm"])
    patterns = []
    for kind in order:
        if kind == "gbm":
            patterns.append(f"gbm_*_{target}.pkl")
        else:
            patterns.append(f"{kind}_{target}.pt")
    for pat in patterns:
        for p in models_dir.glob(pat):
            if p.exists():
                return p
    return None


def _forecast(df: pd.DataFrame, target: str, horizon: int, cfg: dict, models_dir: Path) -> list[float] | None:
    model_path = _resolve_model_path(target, cfg, models_dir)
    if not model_path:
        return None
    bundle = load_model_bundle(model_path)
    try:
        pred = predict_next_24h(df, bundle, horizon=horizon)
    except Exception:
        return None
    return pred.get("forecast")


def _conformal_bounds(target: str, yhat: np.ndarray, cfg: dict) -> dict | None:
    if not cfg.get("enabled", False):
        return None
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts/uncertainty"))
    path = artifacts_dir / f"{target}_conformal.json"
    if not path.exists():
        return None
    ci = load_conformal(path)
    if ci.q_h is not None and len(ci.q_h) != len(yhat):
        return None
    lower, upper = ci.predict_interval(np.asarray(yhat, dtype=float))
    return {"lower": lower, "upper": upper}


def _check_bounds(name: str, arr: np.ndarray, lower: float | None, upper: float | None, tol: float = 1e-6) -> list[str]:
    issues = []
    if lower is not None and np.any(arr < lower - tol):
        issues.append(f"{name} below lower bound")
    if upper is not None and np.any(arr > upper + tol):
        issues.append(f"{name} above upper bound")
    return issues


def _validate_plan(plan: dict, load: np.ndarray, renew: np.ndarray, cfg: dict) -> tuple[bool, list[str]]:
    battery = cfg.get("battery", {})
    grid = cfg.get("grid", {})
    capacity = float(battery.get("capacity_mwh", 10.0))
    min_soc = float(battery.get("min_soc_mwh", 0.0))
    max_charge = float(battery.get("max_charge_mw", battery.get("max_power_mw", 2.0)))
    max_discharge = float(battery.get("max_discharge_mw", battery.get("max_power_mw", 2.0)))
    max_import = float(grid.get("max_import_mw", grid.get("max_draw_mw", 50.0)))

    grid_mw = np.asarray(plan.get("grid_mw", []), dtype=float)
    charge = np.asarray(plan.get("battery_charge_mw", []), dtype=float)
    discharge = np.asarray(plan.get("battery_discharge_mw", []), dtype=float)
    soc = np.asarray(plan.get("soc_mwh", []), dtype=float)
    curtail = np.asarray(plan.get("curtailment_mw", np.zeros_like(grid_mw)), dtype=float)
    unmet = np.asarray(plan.get("unmet_load_mw", np.zeros_like(grid_mw)), dtype=float)

    issues = []
    issues += _check_bounds("grid_mw", grid_mw, 0.0, max_import)
    issues += _check_bounds("battery_charge_mw", charge, 0.0, max_charge)
    issues += _check_bounds("battery_discharge_mw", discharge, 0.0, max_discharge)
    issues += _check_bounds("soc_mwh", soc, min_soc, capacity)

    residual = grid_mw + discharge - charge - curtail + unmet - (load - renew)
    if np.any(np.abs(residual) > 1e-4):
        issues.append("energy_balance_violation")

    return len(issues) == 0, issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default="reports/decision_robustness.md")
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args()

    opt_cfg = _load_cfg("configs/optimization.yaml")
    forecast_cfg = _load_cfg("configs/forecast.yaml")
    unc_cfg = _load_cfg("configs/uncertainty.yaml")

    test_path = Path("data/processed/splits/test.parquet")
    if not test_path.exists():
        raise SystemExit("Missing data/processed/splits/test.parquet. Run data pipeline.")

    df = pd.read_parquet(test_path).sort_values("timestamp").reset_index(drop=True)
    horizon = args.horizon
    if len(df) < horizon:
        raise SystemExit("Not enough rows in test split for requested horizon.")

    window = df.tail(horizon)
    context_df = df.iloc[:-horizon] if len(df) > horizon else df

    load = window["load_mw"].to_numpy()
    wind = window["wind_mw"].to_numpy() if "wind_mw" in window.columns else np.zeros_like(load)
    solar = window["solar_mw"].to_numpy() if "solar_mw" in window.columns else np.zeros_like(load)
    if "price_eur_mwh" in window.columns:
        price = window["price_eur_mwh"].to_numpy()
    elif "price_usd_mwh" in window.columns:
        price = window["price_usd_mwh"].to_numpy()
    else:
        price = None
    carbon = window["carbon_kg_per_mwh"].to_numpy() if "carbon_kg_per_mwh" in window.columns else None
    renew = wind + solar

    models_dir = Path("artifacts/models")
    f_load = _forecast(context_df, "load_mw", horizon, forecast_cfg, models_dir) or load.tolist()
    f_wind = _forecast(context_df, "wind_mw", horizon, forecast_cfg, models_dir) or wind.tolist()
    f_solar = _forecast(context_df, "solar_mw", horizon, forecast_cfg, models_dir) or solar.tolist()

    f_load = np.asarray(f_load, dtype=float)
    f_wind = np.asarray(f_wind, dtype=float)
    f_solar = np.asarray(f_solar, dtype=float)
    f_renew = f_wind + f_solar

    load_bounds = _conformal_bounds("load_mw", f_load, unc_cfg)
    wind_bounds = _conformal_bounds("wind_mw", f_wind, unc_cfg)
    solar_bounds = _conformal_bounds("solar_mw", f_solar, unc_cfg)

    renew_bounds = None
    if wind_bounds and solar_bounds:
        renew_bounds = {
            "lower": np.asarray(wind_bounds["lower"]) + np.asarray(solar_bounds["lower"]),
            "upper": np.asarray(wind_bounds["upper"]) + np.asarray(solar_bounds["upper"]),
        }

    risk_cfg = opt_cfg.get("risk", {})
    use_risk = bool(risk_cfg.get("enabled", False)) and load_bounds is not None and renew_bounds is not None

    baseline = grid_only_dispatch(load, renew, opt_cfg, price_series=price, carbon_series=carbon)
    optimized = optimize_dispatch(f_load, f_renew, opt_cfg, forecast_price=price, forecast_carbon_kg=carbon)
    oracle = optimize_dispatch(load, renew, opt_cfg, forecast_price=price, forecast_carbon_kg=carbon)
    risk_plan = None
    if use_risk:
        risk_plan = optimize_dispatch(
            f_load,
            f_renew,
            opt_cfg,
            forecast_price=price,
            forecast_carbon_kg=carbon,
            load_interval=load_bounds,
            renewables_interval=renew_bounds,
        )

    plan = risk_plan or optimized
    ok, issues = _validate_plan(plan, load, renew, opt_cfg)

    impact = impact_summary(baseline, plan)
    regret = plan.get("expected_cost_usd", 0.0) - oracle.get("expected_cost_usd", 0.0)

    note = "intervals missing; risk mode disabled" if bool(risk_cfg.get("enabled", False)) and not use_risk else "ok"
    lines = [
        "# Decision Robustness Report\n",
        f"- Horizon: {horizon} hours\n",
        f"- Risk-aware plan used: **{bool(risk_plan)}**\n",
        f"- Risk interval status: **{note}**\n",
        f"- Feasible: **{ok}**\n",
        f"- Violations: **{', '.join(issues) if issues else 'none'}**\n",
        "\n## Cost & Carbon\n",
        f"- Baseline cost: **{impact.get('baseline_cost_usd'):.2f}**\n",
        f"- Plan cost: **{impact.get('optimized_cost_usd'):.2f}**\n",
        f"- Cost savings: **{impact.get('cost_savings_usd'):.2f}**\n",
        f"- Baseline carbon: **{impact.get('baseline_carbon_kg'):.2f}**\n",
        f"- Plan carbon: **{impact.get('optimized_carbon_kg'):.2f}**\n",
        f"- Carbon reduction: **{impact.get('carbon_reduction_kg'):.2f}**\n",
        f"- Regret vs oracle (cost): **{regret:.2f}**\n",
    ]

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")

    metrics_lines = [
        "# Decision Metrics Report\n",
        f"- Feasibility violations: **{0 if ok else 1}**\n",
        f"- Regret vs oracle (cost): **{regret:.2f}**\n",
        f"- Cost delta vs baseline: **{impact.get('cost_savings_usd'):.2f}**\n",
        f"- Carbon delta vs baseline: **{impact.get('carbon_reduction_kg'):.2f}**\n",
    ]
    metrics_path = out_path.with_name("decision_metrics.md")
    metrics_path.write_text("".join(metrics_lines), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
