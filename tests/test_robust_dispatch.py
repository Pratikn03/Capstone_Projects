"""Tests for Pyomo-based robust dispatch optimization."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyomo.environ")

import pyomo.environ as pyo

import gridpulse.optimizer.robust_dispatch as rd


def _highs_available() -> bool:
    solver = pyo.SolverFactory("appsi_highs")
    if solver is None:
        return False
    try:
        return bool(solver.available(exception_flag=False))
    except Exception:
        return False


def _require_highs() -> None:
    if not _highs_available():
        pytest.skip("appsi_highs is not available in this environment")


def _base_config() -> rd.RobustDispatchConfig:
    return rd.RobustDispatchConfig(
        battery_capacity_mwh=8.0,
        battery_max_charge_mw=3.0,
        battery_max_discharge_mw=3.0,
        battery_charge_efficiency=0.95,
        battery_discharge_efficiency=0.95,
        battery_initial_soc_mwh=4.0,
        battery_min_soc_mwh=1.0,
        battery_max_soc_mwh=7.0,
        max_grid_import_mw=20.0,
        default_price_per_mwh=50.0,
        degradation_cost_per_mwh=2.0,
    )


def test_dro_returns_expected_keys_and_shapes() -> None:
    _require_highs()
    cfg = _base_config()

    out = rd.optimize_robust_dispatch(
        load_lower_bound=[8.0, 9.0, 7.0],
        load_upper_bound=[10.0, 12.0, 9.0],
        renewables_forecast=[2.0, 3.0, 2.0],
        price=[40.0, 60.0, 50.0],
        config=cfg,
    )

    expected_keys = {
        "battery_charge_mw",
        "battery_discharge_mw",
        "soc_mwh_lower",
        "soc_mwh_upper",
        "grid_import_mw_lower",
        "grid_import_mw_upper",
        "worst_case_cost",
        "degradation_cost",
        "total_cost",
        "feasible",
        "solver_status",
        "binding_scenario",
    }
    assert expected_keys.issubset(out.keys())
    assert out["feasible"] is True

    for key in [
        "battery_charge_mw",
        "battery_discharge_mw",
        "soc_mwh_lower",
        "soc_mwh_upper",
        "grid_import_mw_lower",
        "grid_import_mw_upper",
    ]:
        assert len(out[key]) == 3


def test_soc_bounds_hold_in_lower_and_upper_scenarios() -> None:
    _require_highs()
    cfg = _base_config()

    out = rd.optimize_robust_dispatch(
        load_lower_bound=[8.0, 10.0, 9.0, 8.0],
        load_upper_bound=[12.0, 14.0, 13.0, 12.0],
        renewables_forecast=[4.0, 3.0, 4.0, 5.0],
        price=55.0,
        config=cfg,
    )

    soc_lower = np.asarray(out["soc_mwh_lower"], dtype=float)
    soc_upper = np.asarray(out["soc_mwh_upper"], dtype=float)

    assert np.all(soc_lower >= cfg.battery_min_soc_mwh - 1e-6)
    assert np.all(soc_lower <= cfg.battery_max_soc_mwh + 1e-6)
    assert np.all(soc_upper >= cfg.battery_min_soc_mwh - 1e-6)
    assert np.all(soc_upper <= cfg.battery_max_soc_mwh + 1e-6)


def test_worst_case_objective_matches_binding_scenario_cost() -> None:
    _require_highs()
    cfg = _base_config()
    price = np.asarray([30.0, 50.0, 70.0, 40.0], dtype=float)

    out = rd.optimize_robust_dispatch(
        load_lower_bound=[6.0, 7.0, 8.0, 6.0],
        load_upper_bound=[11.0, 12.0, 13.0, 12.0],
        renewables_forecast=[3.0, 2.0, 2.0, 3.0],
        price=price,
        config=cfg,
    )

    grid_lower = np.asarray(out["grid_import_mw_lower"], dtype=float)
    grid_upper = np.asarray(out["grid_import_mw_upper"], dtype=float)
    charge = np.asarray(out["battery_charge_mw"], dtype=float)
    discharge = np.asarray(out["battery_discharge_mw"], dtype=float)

    cost_lower = float(np.sum(price * grid_lower))
    cost_upper = float(np.sum(price * grid_upper))
    worst = max(cost_lower, cost_upper)
    expected_binding = "lower" if cost_lower >= cost_upper - 1e-9 else "upper"

    degradation = cfg.degradation_cost_per_mwh * float(np.sum(charge + discharge))

    assert np.isclose(out["scenario_cost_lower"], cost_lower, atol=1e-5)
    assert np.isclose(out["scenario_cost_upper"], cost_upper, atol=1e-5)
    assert np.isclose(out["worst_case_cost"], worst, atol=1e-5)
    assert out["binding_scenario"] == expected_binding
    assert np.isclose(out["degradation_cost"], degradation, atol=1e-5)
    assert np.isclose(out["total_cost"], worst + degradation, atol=1e-5)


def test_infeasible_when_grid_import_cap_too_tight() -> None:
    _require_highs()
    cfg = rd.RobustDispatchConfig(
        battery_capacity_mwh=1.0,
        battery_max_charge_mw=0.0,
        battery_max_discharge_mw=0.0,
        battery_charge_efficiency=1.0,
        battery_discharge_efficiency=1.0,
        battery_initial_soc_mwh=0.0,
        battery_min_soc_mwh=0.0,
        battery_max_soc_mwh=0.0,
        max_grid_import_mw=2.0,
    )

    out = rd.optimize_robust_dispatch(
        load_lower_bound=[10.0, 10.0],
        load_upper_bound=[10.0, 10.0],
        renewables_forecast=[0.0, 0.0],
        config=cfg,
    )

    assert out["feasible"] is False
    assert out["total_cost"] is None
    assert "infeasible" in out["solver_status"].lower()


def test_input_validation_for_length_and_bound_order() -> None:
    cfg = _base_config()

    with pytest.raises(ValueError, match="load_lower_bound must be <= load_upper_bound"):
        rd.optimize_robust_dispatch(
            load_lower_bound=[5.0, 7.0],
            load_upper_bound=[4.0, 6.0],
            renewables_forecast=[1.0, 1.0],
            config=cfg,
        )

    with pytest.raises(ValueError, match="renewables_forecast length"):
        rd.optimize_robust_dispatch(
            load_lower_bound=[5.0, 6.0, 7.0],
            load_upper_bound=[6.0, 7.0, 8.0],
            renewables_forecast=[1.0, 1.0],
            config=cfg,
        )

    with pytest.raises(ValueError, match="price must contain non-negative"):
        rd.optimize_robust_dispatch(
            load_lower_bound=[5.0, 6.0],
            load_upper_bound=[6.0, 7.0],
            renewables_forecast=[1.0, 1.0],
            price=[10.0, -5.0],
            config=cfg,
        )


def test_no_uncertainty_mode_via_equal_bounds_is_supported() -> None:
    _require_highs()
    cfg = _base_config()

    out = rd.optimize_robust_dispatch(
        load_lower_bound=[9.0, 11.0, 10.0],
        load_upper_bound=[9.0, 11.0, 10.0],
        renewables_forecast=[2.0, 2.0, 2.0],
        price=[45.0, 45.0, 45.0],
        config=cfg,
    )

    assert out["feasible"] is True
    assert np.isclose(out["scenario_cost_lower"], out["scenario_cost_upper"], atol=1e-6)


def test_solver_missing_raises_clear_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _base_config()

    def _raise(_: str) -> None:
        raise RuntimeError("HiGHS solver is required for robust dispatch")

    monkeypatch.setattr(rd, "_ensure_highs_solver_available", _raise)

    with pytest.raises(RuntimeError, match="HiGHS solver is required"):
        rd.optimize_robust_dispatch(
            load_lower_bound=[5.0, 6.0],
            load_upper_bound=[6.0, 7.0],
            renewables_forecast=[1.0, 1.0],
            config=cfg,
        )
