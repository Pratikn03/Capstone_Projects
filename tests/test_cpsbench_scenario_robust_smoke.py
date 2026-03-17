from __future__ import annotations

import numpy as np

from orius.cpsbench_iot.baselines import scenario_robust_dispatch
from orius.cpsbench_iot.scenarios import generate_episode


def test_scenario_robust_dispatch_smoke() -> None:
    horizon = 24
    x_obs, x_true, _event_log = generate_episode("drift_combo", seed=11, horizon=horizon)

    res = scenario_robust_dispatch(
        load_forecast=x_obs["load_mw"].to_numpy(dtype=float),
        renewables_forecast=x_obs["renewables_mw"].to_numpy(dtype=float),
        load_true=x_true["load_mw"].to_numpy(dtype=float),
        price=x_obs["price_per_mwh"].to_numpy(dtype=float),
        seed=11,
        n_scenarios=10,
    )

    assert res["policy"] == "scenario_robust"
    for key in (
        "proposed_charge_mw",
        "proposed_discharge_mw",
        "safe_charge_mw",
        "safe_discharge_mw",
        "soc_mwh",
        "interval_lower",
        "interval_upper",
        "certificates",
    ):
        assert len(res[key]) == horizon

    for key in (
        "proposed_charge_mw",
        "proposed_discharge_mw",
        "safe_charge_mw",
        "safe_discharge_mw",
        "soc_mwh",
        "interval_lower",
        "interval_upper",
    ):
        assert np.all(np.isfinite(np.asarray(res[key], dtype=float)))

    lower = np.asarray(res["interval_lower"], dtype=float)
    upper = np.asarray(res["interval_upper"], dtype=float)
    assert np.all(lower <= upper)
    assert np.isfinite(float(res["expected_cost_usd"]))
    assert isinstance(res["constraints"], dict)
    assert isinstance(res["dispatch_plan"], dict)
    assert bool(res["dispatch_plan"].get("feasible", False)) in {True, False}
