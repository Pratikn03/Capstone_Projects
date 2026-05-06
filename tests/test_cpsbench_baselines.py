import numpy as np
import pandas as pd

from orius.cpsbench_iot.baselines import naive_safe_clip_dispatch


def test_naive_safe_clip_dispatch():
    horizon = 24
    load = np.full(horizon, 1000.0)
    ren = np.full(horizon, 0.0)
    price = np.full(horizon, 50.0)
    carbon = np.full(horizon, 400.0)
    ts = pd.Series(pd.date_range("2026-01-01T00:00:00Z", periods=horizon, freq="h", tz="UTC"))

    cfg = {
        "battery": {
            "capacity_mwh": 100.0,
            "max_power_mw": 50.0,
            "min_soc_mwh": 0.0,
            "initial_soc_mwh": 50.0,
            "efficiency": 1.0,
        },
        "time_step_hours": 1.0,
    }

    res = naive_safe_clip_dispatch(
        load_forecast=load,
        renewables_forecast=ren,
        price=price,
        carbon=carbon,
        timestamps=ts,
        optimization_cfg=cfg,
    )

    assert res["policy"] == "naive_safe_clip"
    assert len(res["safe_charge_mw"]) == horizon
    assert len(res["soc_mwh"]) == horizon

    # The heuristic charges between 0-5 hours UTC (which are indices 0-5 in this array).
    assert res["proposed_charge_mw"][0] == 0.60 * 50.0
    assert res["proposed_discharge_mw"][0] == 0.0

    # Hours 17-21 discharge
    assert res["proposed_charge_mw"][18] == 0.0
    assert res["proposed_discharge_mw"][18] == 0.60 * 50.0

    # Ensure safe limits were respected
    assert np.all(res["safe_charge_mw"] <= 50.0)
    assert np.all(res["safe_discharge_mw"] <= 50.0)
    assert np.all(res["soc_mwh"] >= 0.0)
    assert np.all(res["soc_mwh"] <= 100.0)
    assert "expected_cost_usd" in res
    assert "carbon_kg" in res
    assert "interval_lower" in res
    assert "interval_upper" in res
