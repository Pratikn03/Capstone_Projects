"""Tests for stochastic-programming metrics in evaluation.regret."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import gridpulse.evaluation.regret as rg


@dataclass
class DummyRobustConfig:
    max_grid_import_mw: float = 100.0
    default_price_per_mwh: float = 50.0
    degradation_cost_per_mwh: float = 2.0
    time_step_hours: float = 1.0


def test_calculate_evpi_robust_uses_truth_for_perfect_info(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_solve_robust_dispatch(
        load_lower_bound: np.ndarray,
        load_upper_bound: np.ndarray,
        renewables_forecast: np.ndarray,
        price: np.ndarray,
        robust_config: DummyRobustConfig,
    ) -> dict:
        calls.append(
            {
                "load_lower_bound": np.asarray(load_lower_bound, dtype=float),
                "load_upper_bound": np.asarray(load_upper_bound, dtype=float),
                "renewables_forecast": np.asarray(renewables_forecast, dtype=float),
                "price": np.asarray(price, dtype=float),
            }
        )
        if len(calls) == 1:
            return {"battery_charge_mw": [0.0, 0.0], "battery_discharge_mw": [0.0, 0.0]}
        return {"battery_charge_mw": [0.0, 0.0], "battery_discharge_mw": [5.0, 5.0]}

    monkeypatch.setattr(rg, "_solve_robust_dispatch", fake_solve_robust_dispatch)

    load_true = np.array([10.0, 10.0])
    renewables_true = np.array([0.0, 0.0])
    out = rg.calculate_evpi(
        actual_model="robust",
        load_true=load_true,
        renewables_true=renewables_true,
        load_forecast=[8.0, 12.0],
        renewables_forecast=[0.0, 0.0],
        load_lower_bound=[7.0, 11.0],
        load_upper_bound=[9.0, 13.0],
        price=[1.0, 1.0],
        robust_config=DummyRobustConfig(degradation_cost_per_mwh=0.0),
    )

    assert len(calls) == 2
    np.testing.assert_allclose(calls[1]["load_lower_bound"], load_true)
    np.testing.assert_allclose(calls[1]["load_upper_bound"], load_true)
    np.testing.assert_allclose(calls[1]["renewables_forecast"], renewables_true)
    assert out["evpi"] == pytest.approx(10.0)


def test_calculate_evpi_deterministic_formula(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_solve_deterministic_dispatch(
        load_forecast: np.ndarray,
        renewables_forecast: np.ndarray,
        deterministic_config: dict,
        price: np.ndarray,
    ) -> dict:
        calls.append({"price": np.asarray(price, dtype=float)})
        if len(calls) == 1:
            return {"battery_charge_mw": [1.0], "battery_discharge_mw": [0.0]}
        return {"battery_charge_mw": [0.0], "battery_discharge_mw": [1.0]}

    monkeypatch.setattr(rg, "_solve_deterministic_dispatch", fake_solve_deterministic_dispatch)

    out = rg.calculate_evpi(
        actual_model="deterministic",
        load_true=[10.0],
        renewables_true=[0.0],
        load_forecast=[10.0],
        renewables_forecast=[0.0],
        price=[2.0],
        deterministic_config={"grid": {"max_import_mw": 100.0}, "battery": {"degradation_cost_per_mwh": 1.0}},
    )

    assert out["actual_model"] == "deterministic"
    assert out["actual_realized_cost"] == pytest.approx(23.0)
    assert out["perfect_info_cost"] == pytest.approx(19.0)
    assert out["evpi"] == pytest.approx(4.0)


def test_calculate_vss_returns_det_minus_robust(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_det(
        load_forecast: np.ndarray,
        renewables_forecast: np.ndarray,
        deterministic_config: dict,
        price: np.ndarray,
    ) -> dict:
        return {"battery_charge_mw": [1.0], "battery_discharge_mw": [0.0]}

    def fake_rob(
        load_lower_bound: np.ndarray,
        load_upper_bound: np.ndarray,
        renewables_forecast: np.ndarray,
        price: np.ndarray,
        robust_config: DummyRobustConfig,
    ) -> dict:
        return {"battery_charge_mw": [0.0], "battery_discharge_mw": [1.0]}

    monkeypatch.setattr(rg, "_solve_deterministic_dispatch", fake_det)
    monkeypatch.setattr(rg, "_solve_robust_dispatch", fake_rob)

    out = rg.calculate_vss(
        load_true=[10.0],
        renewables_true=[0.0],
        load_forecast=[10.0],
        renewables_forecast=[0.0],
        load_lower_bound=[9.0],
        load_upper_bound=[11.0],
        price=[2.0],
        deterministic_config={"grid": {"max_import_mw": 100.0}, "battery": {"degradation_cost_per_mwh": 1.0}},
        robust_config=DummyRobustConfig(),
    )

    assert out["deterministic_realized_cost"] == pytest.approx(23.0)
    assert out["robust_realized_cost"] == pytest.approx(20.0)
    assert out["vss"] == pytest.approx(3.0)


def test_realized_cost_includes_unmet_penalty_when_grid_cap_exceeded() -> None:
    out = rg._evaluate_realized_cost(
        load_true=np.array([20.0]),
        renewables_true=np.array([0.0]),
        charge=np.array([0.0]),
        discharge=np.array([0.0]),
        price=np.array([1.0]),
        max_grid_import=10.0,
        degradation_cost_per_mwh=0.0,
        unmet_load_penalty_per_mwh=100.0,
        time_step_hours=1.0,
    )

    assert out["grid_cost"] == pytest.approx(10.0)
    assert out["unmet_penalty_cost"] == pytest.approx(1000.0)
    assert out["total_cost"] == pytest.approx(1010.0)


def test_price_scalar_broadcast_and_shape_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    det_prices: list[np.ndarray] = []
    robust_prices: list[np.ndarray] = []

    def fake_det(
        load_forecast: np.ndarray,
        renewables_forecast: np.ndarray,
        deterministic_config: dict,
        price: np.ndarray,
    ) -> dict:
        det_prices.append(np.asarray(price, dtype=float))
        return {"battery_charge_mw": np.zeros_like(load_forecast), "battery_discharge_mw": np.zeros_like(load_forecast)}

    def fake_rob(
        load_lower_bound: np.ndarray,
        load_upper_bound: np.ndarray,
        renewables_forecast: np.ndarray,
        price: np.ndarray,
        robust_config: DummyRobustConfig,
    ) -> dict:
        robust_prices.append(np.asarray(price, dtype=float))
        return {"battery_charge_mw": np.zeros_like(load_lower_bound), "battery_discharge_mw": np.zeros_like(load_lower_bound)}

    monkeypatch.setattr(rg, "_solve_deterministic_dispatch", fake_det)
    monkeypatch.setattr(rg, "_solve_robust_dispatch", fake_rob)

    rg.calculate_vss(
        load_true=[10.0, 11.0],
        renewables_true=[0.0, 0.0],
        load_forecast=[10.0, 11.0],
        renewables_forecast=[0.0, 0.0],
        load_lower_bound=[9.0, 10.0],
        load_upper_bound=[11.0, 12.0],
        price=5.0,
        robust_config=DummyRobustConfig(),
    )

    np.testing.assert_allclose(det_prices[0], np.array([5.0, 5.0]))
    np.testing.assert_allclose(robust_prices[0], np.array([5.0, 5.0]))

    with pytest.raises(ValueError, match="renewables_true length"):
        rg.calculate_vss(
            load_true=[10.0, 11.0],
            renewables_true=[0.0],
            load_forecast=[10.0, 11.0],
            renewables_forecast=[0.0, 0.0],
            load_lower_bound=[9.0, 10.0],
            load_upper_bound=[11.0, 12.0],
            price=5.0,
            robust_config=DummyRobustConfig(),
        )


def test_invalid_bounds_raise() -> None:
    with pytest.raises(ValueError, match="load_lower_bound must be <= load_upper_bound"):
        rg.calculate_vss(
            load_true=[10.0],
            renewables_true=[0.0],
            load_forecast=[10.0],
            renewables_forecast=[0.0],
            load_lower_bound=[12.0],
            load_upper_bound=[11.0],
            robust_config=DummyRobustConfig(),
        )


def test_terminal_soc_penalty_affects_vss_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_det(
        load_forecast: np.ndarray,
        renewables_forecast: np.ndarray,
        deterministic_config: dict,
        price: np.ndarray,
    ) -> dict:
        return {"battery_charge_mw": [0.0], "battery_discharge_mw": [10.0]}

    def fake_rob(
        load_lower_bound: np.ndarray,
        load_upper_bound: np.ndarray,
        renewables_forecast: np.ndarray,
        price: np.ndarray,
        robust_config: DummyRobustConfig,
    ) -> dict:
        return {"battery_charge_mw": [10.0], "battery_discharge_mw": [0.0]}

    monkeypatch.setattr(rg, "_solve_deterministic_dispatch", fake_det)
    monkeypatch.setattr(rg, "_solve_robust_dispatch", fake_rob)

    deterministic_config = {
        "grid": {"max_import_mw": 100.0},
        "battery": {
            "capacity_mwh": 100.0,
            "initial_soc_mwh": 50.0,
            "efficiency": 1.0,
            "degradation_cost_per_mwh": 0.0,
        },
        "research_operational": {
            "terminal_soc": {
                "enabled": True,
                "target_soc_mwh": 60.0,
                "penalty_per_mwh_shortfall": 50.0,
            }
        },
    }

    robust_config = DummyRobustConfig(degradation_cost_per_mwh=0.0)
    robust_config.battery_initial_soc_mwh = 50.0
    robust_config.battery_charge_efficiency = 1.0
    robust_config.battery_discharge_efficiency = 1.0

    out = rg.calculate_vss(
        load_true=[0.0],
        renewables_true=[0.0],
        load_forecast=[0.0],
        renewables_forecast=[0.0],
        load_lower_bound=[0.0],
        load_upper_bound=[0.0],
        price=[0.0],
        deterministic_config=deterministic_config,
        robust_config=robust_config,
        unmet_load_penalty_per_mwh=0.0,
    )

    assert out["deterministic_realized_cost"] == pytest.approx(1000.0)
    assert out["robust_realized_cost"] == pytest.approx(0.0)
    assert out["vss"] == pytest.approx(1000.0)


def test_generate_stochastic_metrics_report_writes_csv_with_summary_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def fake_calculate_evpi(*args, **kwargs) -> dict:
        if kwargs.get("actual_model") == "robust":
            return {
                "evpi": 10.0,
                "actual_realized_cost": 110.0,
                "perfect_info_cost": 100.0,
                "actual_model": "robust",
                "horizon": 2,
            }
        return {
            "evpi": 5.0,
            "actual_realized_cost": 105.0,
            "perfect_info_cost": 100.0,
            "actual_model": "deterministic",
            "horizon": 2,
        }

    def fake_calculate_vss(*args, **kwargs) -> dict:
        return {
            "vss": 3.0,
            "deterministic_realized_cost": 105.0,
            "robust_realized_cost": 102.0,
            "horizon": 2,
        }

    monkeypatch.setattr(rg, "calculate_evpi", fake_calculate_evpi)
    monkeypatch.setattr(rg, "calculate_vss", fake_calculate_vss)

    scenarios = [
        {
            "scenario": "S1",
            "load_true": [10.0, 10.0],
            "renewables_true": [0.0, 0.0],
            "load_forecast": [9.0, 11.0],
            "renewables_forecast": [0.0, 0.0],
            "load_lower_bound": [8.0, 10.0],
            "load_upper_bound": [10.0, 12.0],
        },
        {
            "scenario": "S2",
            "load_true": [11.0, 9.0],
            "renewables_true": [0.0, 0.0],
            "load_forecast": [10.0, 10.0],
            "renewables_forecast": [0.0, 0.0],
            "load_lower_bound": [9.0, 9.0],
            "load_upper_bound": [11.0, 11.0],
            "price": [50.0, 50.0],
        },
    ]

    out_csv = tmp_path / "stochastic_metrics.csv"
    df = rg.generate_stochastic_metrics_report(
        scenarios=scenarios,
        output_csv=out_csv,
        unmet_load_penalty_per_mwh=10000.0,
    )

    assert out_csv.exists()
    assert set(df["row_type"]) == {"scenario", "summary_mean"}
    assert (df["row_type"] == "scenario").sum() == 2
    assert (df["row_type"] == "summary_mean").sum() == 1

    required_columns = {
        "row_type",
        "scenario",
        "horizon",
        "evpi",
        "evpi_robust",
        "evpi_deterministic",
        "vss",
        "robust_actual_realized_cost",
        "robust_perfect_info_cost",
        "deterministic_actual_realized_cost",
        "deterministic_perfect_info_cost",
        "unmet_load_penalty_per_mwh",
    }
    assert required_columns.issubset(df.columns)
