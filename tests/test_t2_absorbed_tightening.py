from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from orius.certos.domain_policies import BatteryGovernancePolicy
from orius.certos.runtime import CertOSConfig, CertOSRuntime
from orius.dc3s.guarantee_checks import next_soc
from orius.dc3s.safety_filter_theory import check_tightened_soc_invariance, tightened_soc_bounds


def _constraints() -> dict[str, float]:
    return {
        "min_soc_mwh": 10.0,
        "max_soc_mwh": 90.0,
        "capacity_mwh": 100.0,
        "time_step_hours": 1.0,
        "charge_efficiency": 1.0,
        "discharge_efficiency": 1.0,
    }


@given(
    base_margin=st.floats(min_value=0.0, max_value=20.0, allow_nan=False, allow_infinity=False),
    model_error=st.floats(min_value=0.0, max_value=8.0, allow_nan=False, allow_infinity=False),
    current_unit=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    action_unit=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    epsilon_unit=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    charge_mode=st.booleans(),
)
def test_t2_absorbed_tightening_randomized(
    base_margin: float,
    model_error: float,
    current_unit: float,
    action_unit: float,
    epsilon_unit: float,
    charge_mode: bool,
) -> None:
    constraints = _constraints()
    lower, upper = tightened_soc_bounds(
        min_soc_mwh=constraints["min_soc_mwh"],
        max_soc_mwh=constraints["max_soc_mwh"],
        error_bound_mwh=0.0,
        q_rac_mwh=base_margin,
        model_error_mwh=model_error,
    )
    current_soc = lower + current_unit * (upper - lower)
    if charge_mode:
        max_charge = max(0.0, upper - current_soc)
        action = {"charge_mw": action_unit * max_charge, "discharge_mw": 0.0}
    else:
        max_discharge = max(0.0, current_soc - lower)
        action = {"charge_mw": 0.0, "discharge_mw": action_unit * max_discharge}

    result = check_tightened_soc_invariance(
        current_soc_obs=current_soc,
        action=action,
        constraints=constraints,
        q_rac_mwh=base_margin,
        model_error_mwh=model_error,
        error_bound_mwh=0.0,
    )

    projected = next_soc(
        current_soc=current_soc,
        action=action,
        dt_hours=constraints["time_step_hours"],
        charge_efficiency=constraints["charge_efficiency"],
        discharge_efficiency=constraints["discharge_efficiency"],
    )
    epsilon = epsilon_unit * model_error
    true_next = projected + epsilon

    assert result["observed_safe"] is True
    assert result["true_safe_if_bound_holds"] is True
    assert constraints["min_soc_mwh"] <= true_next <= constraints["max_soc_mwh"]


def test_t2_runtime_assertion_rejects_battery_postcondition_violation() -> None:
    runtime = CertOSRuntime(
        config=CertOSConfig(
            governance_policy=BatteryGovernancePolicy(),
        )
    )
    with pytest.raises(AssertionError, match="T2 postcondition failed"):
        runtime.validate_and_step(
            observed_soc_mwh=89.0,
            proposed_action={"charge_mw": 5.0, "discharge_mw": 0.0},
            safe_action={"charge_mw": 5.0, "discharge_mw": 0.0},
            validity_horizon=10,
            observed_state={"current_soc_mwh": 89.0},
            constraints={
                **_constraints(),
                "epsilon_model_mwh": 2.0,
            },
        )
