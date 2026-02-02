"""Tests for test optimizer."""
import numpy as np

from gridpulse.optimizer import optimize_dispatch


def test_optimize_dispatch_shapes():
    # Key: test setup and assertions
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
    assert len(out["grid_mw"]) == 3
    assert len(out["battery_charge_mw"]) == 3
    assert len(out["battery_discharge_mw"]) == 3
    assert len(out["soc_mwh"]) == 3

    # Non-negative checks
    assert np.all(np.asarray(out["grid_mw"]) >= 0.0)
    assert np.all(np.asarray(out["battery_charge_mw"]) >= 0.0)
    assert np.all(np.asarray(out["battery_discharge_mw"]) >= 0.0)
