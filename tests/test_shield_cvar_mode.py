from __future__ import annotations

from gridpulse.dc3s.shield import repair_action


def test_repair_action_cvar_mode_returns_safe_action_and_meta() -> None:
    a_star = {"charge_mw": 15.0, "discharge_mw": 0.0}
    state = {"current_soc_mwh": 80.0}
    uncertainty_set = {
        "lower": [100.0, 101.0, 102.0],
        "upper": [120.0, 121.0, 122.0],
        "renewables_forecast": [20.0, 20.0, 20.0],
        "price": [60.0, 60.0, 60.0],
        "meta": {"drift_flag": False},
    }
    constraints = {
        "capacity_mwh": 200.0,
        "max_power_mw": 50.0,
        "max_charge_mw": 50.0,
        "max_discharge_mw": 50.0,
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 190.0,
        "current_soc_mwh": 80.0,
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.95,
        "max_grid_import_mw": 500.0,
        "default_price_per_mwh": 60.0,
        "degradation_cost_per_mwh": 2.0,
        "time_step_hours": 1.0,
        "solver_name": "appsi_highs",
        "cvar_beta": 0.90,
        "cvar_n_scenarios": 6,
        "cvar_risk_weight": 1.0,
        "scenario_seed": 7,
    }
    cfg = {"shield": {"mode": "robust_resolve_cvar", "reserve_soc_pct_drift": 0.0}}

    safe, meta = repair_action(
        a_star=a_star,
        state=state,
        uncertainty_set=uncertainty_set,
        constraints=constraints,
        cfg=cfg,
    )

    assert safe["charge_mw"] >= 0.0
    assert safe["discharge_mw"] >= 0.0
    assert not (safe["charge_mw"] > 0.0 and safe["discharge_mw"] > 0.0)
    assert meta["mode"] == "robust_resolve_cvar"
    assert "robust_meta" in meta
    assert "solver_status" in meta["robust_meta"] or "reason" in meta["robust_meta"]
