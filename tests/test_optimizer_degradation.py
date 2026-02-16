"""Physics-informed battery degradation tests for MILP dispatch."""

import numpy as np

from gridpulse.optimizer import optimize_dispatch


def _base_cfg(initial_soc: float, degradation_cost: float = 0.0) -> dict:
    return {
        "battery": {
            "capacity_mwh": 10.0,
            "max_power_mw": 2.0,
            "max_charge_mw": 2.0,
            "max_discharge_mw": 2.0,
            "efficiency_regime_a": 0.98,
            "efficiency_regime_b": 0.90,
            "efficiency_soc_split": 0.80,
            "degradation_cost_per_mwh": degradation_cost,
            "min_soc_mwh": 0.0,
            "initial_soc_mwh": initial_soc,
        },
        "grid": {
            "max_import_mw": 100.0,
            "price_per_mwh": 1.0,
            "carbon_kg_per_mwh": 0.0,
        },
        "penalties": {
            "curtailment_per_mw": 1000.0,
            "unmet_load_per_mw": 10000.0,
            "peak_per_mw": 0.0,
        },
        "objective": {
            "cost_weight": 1.0,
            "carbon_weight": 0.0,
        },
    }


def test_piecewise_efficiency_regime_a_applies_below_split():
    cfg = _base_cfg(initial_soc=0.0, degradation_cost=0.0)
    out = optimize_dispatch([0.0], [10.0], cfg)

    charge = float(out["battery_charge_mw"][0])
    discharge = float(out["battery_discharge_mw"][0])
    soc = float(out["soc_mwh"][0])

    assert charge > 1e-6
    assert abs(discharge) <= 1e-6
    assert abs(soc - (0.0 + 0.98 * charge)) <= 1e-6


def test_piecewise_efficiency_regime_b_applies_above_split():
    cfg = _base_cfg(initial_soc=9.0, degradation_cost=0.0)
    out = optimize_dispatch([0.0], [10.0], cfg)

    charge = float(out["battery_charge_mw"][0])
    discharge = float(out["battery_discharge_mw"][0])
    soc = float(out["soc_mwh"][0])

    assert charge > 1e-6
    assert abs(soc - (9.0 + 0.90 * charge - discharge / 0.90)) <= 1e-6


def test_regime_boundary_at_80_percent_uses_regime_a():
    cfg = _base_cfg(initial_soc=8.0, degradation_cost=0.0)
    out = optimize_dispatch([0.0], [10.0], cfg)

    charge = float(out["battery_charge_mw"][0])
    discharge = float(out["battery_discharge_mw"][0])
    soc = float(out["soc_mwh"][0])

    assert charge > 1e-6
    assert abs(discharge) <= 1e-6
    assert abs(soc - (8.0 + 0.98 * charge)) <= 1e-6


def test_throughput_cost_penalizes_microcycling():
    base_cfg = {
        "battery": {
            "capacity_mwh": 10.0,
            "max_power_mw": 10.0,
            "max_charge_mw": 10.0,
            "max_discharge_mw": 10.0,
            "efficiency_regime_a": 0.98,
            "efficiency_regime_b": 0.90,
            "efficiency_soc_split": 0.80,
            "min_soc_mwh": 0.0,
            "initial_soc_mwh": 0.0,
        },
        "grid": {
            "max_import_mw": 100.0,
            "price_per_mwh": 1.0,
            "carbon_kg_per_mwh": 0.0,
        },
        "penalties": {
            "curtailment_per_mw": 0.0,
            "unmet_load_per_mw": 10000.0,
            "peak_per_mw": 5.0,
        },
        "objective": {
            "cost_weight": 1.0,
            "carbon_weight": 0.0,
        },
    }

    cfg_no_deg = {**base_cfg, "battery": {**base_cfg["battery"], "degradation_cost_per_mwh": 0.0}}
    cfg_deg = {**base_cfg, "battery": {**base_cfg["battery"], "degradation_cost_per_mwh": 10.0}}

    out_no_deg = optimize_dispatch([0.0, 10.0], [0.0, 0.0], cfg_no_deg)
    out_deg = optimize_dispatch([0.0, 10.0], [0.0, 0.0], cfg_deg)

    throughput_no_deg = float(np.sum(out_no_deg["battery_charge_mw"]) + np.sum(out_no_deg["battery_discharge_mw"]))
    throughput_deg = float(np.sum(out_deg["battery_charge_mw"]) + np.sum(out_deg["battery_discharge_mw"]))

    assert throughput_deg <= throughput_no_deg + 1e-8


def test_legacy_efficiency_backward_compatibility():
    cfg = {
        "battery": {
            "capacity_mwh": 10.0,
            "max_power_mw": 2.0,
            "efficiency": 0.93,
            "min_soc_mwh": 0.0,
            "initial_soc_mwh": 0.0,
            "degradation_cost_per_mwh": 0.0,
        },
        "grid": {
            "max_import_mw": 100.0,
            "price_per_mwh": 1.0,
            "carbon_kg_per_mwh": 0.0,
        },
        "penalties": {
            "curtailment_per_mw": 1000.0,
            "unmet_load_per_mw": 10000.0,
            "peak_per_mw": 0.0,
        },
        "objective": {
            "cost_weight": 1.0,
            "carbon_weight": 0.0,
        },
    }

    out = optimize_dispatch([0.0], [10.0], cfg)
    charge = float(out["battery_charge_mw"][0])
    soc = float(out["soc_mwh"][0])

    assert charge > 1e-6
    assert abs(soc - 0.93 * charge) <= 1e-6
    assert "battery_degradation_cost_usd" in out
