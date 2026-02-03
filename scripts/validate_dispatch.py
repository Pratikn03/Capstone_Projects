"""Validate optimizer outputs against basic feasibility constraints."""
from __future__ import annotations

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.optimizer.lp_dispatch import optimize_dispatch


def _load_features() -> pd.DataFrame | None:
    path = Path("data/processed/features.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path).sort_values("timestamp")


def _load_config() -> dict:
    cfg_path = Path("configs/optimization.yaml")
    if not cfg_path.exists():
        return {}
    import yaml

    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _check_bounds(name: str, arr: np.ndarray, lower: float | None, upper: float | None, tol: float = 1e-6) -> dict:
    if lower is not None:
        low_ok = bool(np.all(arr >= lower - tol))
    else:
        low_ok = True
    if upper is not None:
        high_ok = bool(np.all(arr <= upper + tol))
    else:
        high_ok = True
    return {
        "name": name,
        "ok": low_ok and high_ok,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "lower": lower,
        "upper": upper,
    }


def main() -> None:
    cfg = _load_config()
    battery = cfg.get("battery", {})
    grid = cfg.get("grid", {})

    capacity = float(battery.get("capacity_mwh", 10.0))
    min_soc = float(battery.get("min_soc_mwh", 0.0))
    max_charge = float(battery.get("max_charge_mw", battery.get("max_power_mw", 2.0)))
    max_discharge = float(battery.get("max_discharge_mw", battery.get("max_power_mw", 2.0)))
    max_import = float(grid.get("max_import_mw", grid.get("max_draw_mw", 50.0)))

    df = _load_features()
    if df is None or df.empty:
        # Fallback synthetic window if processed data is missing.
        horizon = 24
        load = np.full(horizon, 8000.0)
        wind = np.full(horizon, 2000.0)
        solar = np.full(horizon, 1000.0)
        price = None
        carbon = None
    else:
        horizon = 24
        window = df.tail(horizon)
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
    plan = optimize_dispatch(load, renew, cfg, forecast_price=price, forecast_carbon_kg=carbon)

    grid_mw = np.asarray(plan.get("grid_mw", []), dtype=float)
    charge_mw = np.asarray(plan.get("battery_charge_mw", []), dtype=float)
    discharge_mw = np.asarray(plan.get("battery_discharge_mw", []), dtype=float)
    soc = np.asarray(plan.get("soc_mwh", []), dtype=float)
    curtail = np.asarray(plan.get("curtailment_mw", np.zeros_like(grid_mw)), dtype=float)
    unmet = np.asarray(plan.get("unmet_load_mw", np.zeros_like(grid_mw)), dtype=float)

    checks = []
    if len(grid_mw):
        checks.append(_check_bounds("grid_mw", grid_mw, 0.0, max_import))
    if len(charge_mw):
        checks.append(_check_bounds("battery_charge_mw", charge_mw, 0.0, max_charge))
    if len(discharge_mw):
        checks.append(_check_bounds("battery_discharge_mw", discharge_mw, 0.0, max_discharge))
    if len(soc):
        checks.append(_check_bounds("soc_mwh", soc, min_soc, capacity))

    # Energy balance residual (grid + discharge - charge - curtail + unmet ≈ load - renew).
    residual = grid_mw + discharge_mw - charge_mw - curtail + unmet - (load - renew)
    residual_ok = bool(np.all(np.abs(residual) <= 1e-4))

    report = {
        "horizon": int(horizon),
        "checks": checks,
        "energy_balance_ok": residual_ok,
        "max_residual": float(np.max(np.abs(residual))) if len(residual) else None,
        "note": plan.get("note"),
    }

    out_path = Path("reports/dispatch_validation.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Dispatch Validation Report\n\n"]
    lines.append(f"- Horizon: {horizon}\n")
    lines.append(f"- Energy balance OK: {residual_ok}\n")
    lines.append(f"- Max residual: {report['max_residual']}\n\n")
    lines.append("## Constraint Checks\n")
    lines.append("| Name | OK | Min | Max | Lower | Upper |\n")
    lines.append("|---|---|---:|---:|---:|---:|\n")
    for item in checks:
        lines.append(
            f"| {item['name']} | {item['ok']} | {item['min']:.4f} | {item['max']:.4f} | {item['lower']} | {item['upper']} |\n"
        )
    if report.get("note"):
        lines.append(f"\n- Note: {report['note']}\n")
    out_path.write_text("".join(lines), encoding="utf-8")

    json_path = Path("reports/dispatch_validation.json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
