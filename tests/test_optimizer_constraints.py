"""Constraint checks for the dispatch optimizer."""
import numpy as np

from gridpulse.optimizer import optimize_dispatch


def test_optimizer_respects_bounds():
    load = [10.0, 12.0, 9.0]
    ren = [3.0, 4.0, 2.0]
    cfg = {
        "battery": {
            "capacity_mwh": 5.0,
            "max_power_mw": 2.0,
            "efficiency": 0.95,
            "min_soc_mwh": 0.5,
            "initial_soc_mwh": 2.5,
        },
        "grid": {"max_import_mw": 50.0, "price_per_mwh": 50.0, "carbon_kg_per_mwh": 0.0},
        "penalties": {"curtailment_per_mw": 500.0, "unmet_load_per_mw": 10000.0},
        "objective": {"cost_weight": 1.0, "carbon_weight": 0.0},
    }

    out = optimize_dispatch(load, ren, cfg)
    grid = np.asarray(out["grid_mw"], dtype=float)
    charge = np.asarray(out["battery_charge_mw"], dtype=float)
    discharge = np.asarray(out["battery_discharge_mw"], dtype=float)
    soc = np.asarray(out["soc_mwh"], dtype=float)
    curtail = np.asarray(out["curtailment_mw"], dtype=float)
    unmet = np.asarray(out["unmet_load_mw"], dtype=float)

    assert np.all(grid >= -1e-6)
    assert np.all(charge >= -1e-6)
    assert np.all(discharge >= -1e-6)
    assert np.all(soc >= cfg["battery"]["min_soc_mwh"] - 1e-6)
    assert np.all(soc <= cfg["battery"]["capacity_mwh"] + 1e-6)
    assert np.all(charge <= cfg["battery"]["max_power_mw"] + 1e-6)
    assert np.all(discharge <= cfg["battery"]["max_power_mw"] + 1e-6)

    residual = grid + discharge - charge - curtail + unmet - (np.asarray(load) - np.asarray(ren))
    assert np.all(np.abs(residual) <= 1e-4)


def test_optimizer_risk_bounds_apply():
    cfg = {
        "battery": {
            "capacity_mwh": 1.0,
            "max_power_mw": 0.0,
            "efficiency": 1.0,
            "min_soc_mwh": 0.0,
            "initial_soc_mwh": 0.0,
        },
        "grid": {"max_import_mw": 100.0, "price_per_mwh": 1.0, "carbon_kg_per_mwh": 0.0},
        "penalties": {"curtailment_per_mw": 0.0, "unmet_load_per_mw": 10000.0},
        "objective": {"cost_weight": 1.0, "carbon_weight": 0.0},
        "risk": {
            "enabled": True,
            "mode": "worst_case_interval",
            "load_bound": "upper",
            "renew_bound": "lower",
        },
    }

    out = optimize_dispatch(
        [10.0],
        [0.0],
        cfg,
        load_interval={"lower": [8.0], "upper": [12.0]},
        renewables_interval={"lower": [0.0], "upper": [1.0]},
    )
    assert abs(out["grid_mw"][0] - 12.0) <= 1e-6
