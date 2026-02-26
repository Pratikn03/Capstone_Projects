from __future__ import annotations

import numpy as np
import pytest

from gridpulse.optimizer.robust_dispatch import CVaRDispatchConfig, optimize_cvar_dispatch


def test_optimize_cvar_dispatch_schema_and_feasible_flag() -> None:
    load_scenarios = np.array(
        [
            [120.0, 125.0, 130.0, 128.0],
            [118.0, 124.0, 129.0, 127.0],
            [122.0, 126.0, 131.0, 130.0],
        ],
        dtype=float,
    )
    renew = np.array([20.0, 21.0, 22.0, 22.0], dtype=float)
    price = np.array([60.0, 62.0, 61.0, 63.0], dtype=float)

    cfg = CVaRDispatchConfig(
        battery_capacity_mwh=200.0,
        battery_max_charge_mw=40.0,
        battery_max_discharge_mw=40.0,
        battery_charge_efficiency=0.95,
        battery_discharge_efficiency=0.95,
        battery_initial_soc_mwh=100.0,
        battery_min_soc_mwh=20.0,
        battery_max_soc_mwh=180.0,
        max_grid_import_mw=500.0,
        default_price_per_mwh=60.0,
        degradation_cost_per_mwh=2.0,
        time_step_hours=1.0,
        solver_name="appsi_highs",
        beta=0.9,
        n_scenarios=3,
        risk_weight_cvar=1.0,
    )

    try:
        result = optimize_cvar_dispatch(
            load_scenarios=load_scenarios,
            renewables_forecast=renew,
            price=price,
            config=cfg,
            verbose=False,
        )
    except RuntimeError as exc:
        if "HiGHS solver is required" in str(exc):
            pytest.skip("HiGHS solver unavailable in test environment")
        raise

    assert "battery_charge_mw" in result
    assert "battery_discharge_mw" in result
    assert "solver_status" in result
    assert "feasible" in result
    assert len(result["battery_charge_mw"]) == 4
    assert len(result["battery_discharge_mw"]) == 4
    assert isinstance(result["feasible"], bool)
