import pytest

from orius.cpsbench_iot.metrics import (
    compute_control_metrics,
    compute_forecast_metrics,
    compute_trace_metrics,
)


def test_compute_forecast_metrics():
    y_true = [10.0, 20.0, 30.0]
    y_pred = [11.0, 19.0, 32.0]
    lower_90 = [8.0, 15.0, 25.0]
    upper_90 = [12.0, 22.0, 35.0]
    
    res = compute_forecast_metrics(y_true=y_true, y_pred=y_pred, lower_90=lower_90, upper_90=upper_90)
    
    # mae = (1.0 + 1.0 + 2.0) / 3 = 1.333...
    assert res["mae"] == pytest.approx(4.0 / 3.0)
    # Coverage: 10 in [8, 12], 20 in [15, 22], 30 in [25, 35] -> 3/3
    assert res["picp_90"] == 1.0


def test_compute_control_metrics():
    constraints = {"max_power_mw": 100.0, "min_soc_mwh": 0.0, "max_soc_mwh": 1000.0}
    
    # The proposed controller asks for 150MW charge, which violates max_power.
    p_ch = [150.0]
    p_dis = [0.0]
    
    # The safety shield caps it at 100MW.
    s_ch = [100.0]
    s_dis = [0.0]
    
    soc = [500.0]
    
    res = compute_control_metrics(
        proposed_charge_mw=p_ch,
        proposed_discharge_mw=p_dis,
        safe_charge_mw=s_ch,
        safe_discharge_mw=s_dis,
        soc_mwh=soc,
        constraints=constraints
    )
    
    # Proposed limit violated -> 1.0
    assert res["unsafe_command_rate"] == 1.0
    
    # Safe action is within bounds -> 0.0
    assert res["violation_rate"] == 0.0
    
    # Action modified -> 1.0
    assert res["intervention_rate"] == 1.0
    
    # No SOC violations
    assert res["true_soc_violation_rate"] == 0.0


def test_compute_trace_metrics():
    certs = [
        {"command_id": "c1", "certificate_hash": "h1", "proposed_action": {}, "safe_action": {}},
        None,
        {"command_id": "c2"} # Missing fields (safe_action, etc.)
    ]
    
    res = compute_trace_metrics(certs, required_fields=["command_id", "certificate_hash"])
    
    # 2 out of 3 traces have certificate objects
    assert res["certificate_presence_rate"] == pytest.approx(2.0 / 3.0)
    
    # 2nd object is missing 'certificate_hash'
    assert res["certificate_missing_fields"] == 1.0 
