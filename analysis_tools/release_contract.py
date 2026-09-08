"""Proposed conditional one-step energy contract. Copied into src only in disposable audit run.
Supplied error bounds are premises, not independently verified physical measurements.
"""
from __future__ import annotations
import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


def _finite(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real number, not a string or boolean')
    value=float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'{name} must be finite and within its domain')
    return value


def assess_battery_release(*, current_soc_mwh: float, action: Mapping[str, Any], constraints: Mapping[str, Any], state_error_bound_mwh: float | None, transition_error_bound_mwh: float | None) -> dict[str, Any]:
    """Total transition error is an energy bound for one timestep, not a standard deviation."""
    center=_finite(current_soc_mwh,'current_soc_mwh')
    charge=_finite(action.get('charge_mw',0.),'charge_mw',nonnegative=True)
    discharge=_finite(action.get('discharge_mw',0.),'discharge_mw',nonnegative=True)
    if charge > 0 and discharge > 0:
        raise ValueError('simultaneous charge and discharge is not supported')
    low=_finite(constraints.get('min_soc_mwh'),'min_soc_mwh')
    high=_finite(constraints.get('max_soc_mwh'),'max_soc_mwh')
    dt=_finite(constraints.get('time_step_hours'),'time_step_hours')
    eta_c=_finite(constraints.get('charge_efficiency'),'charge_efficiency')
    eta_d=_finite(constraints.get('discharge_efficiency'),'discharge_efficiency')
    if low>high or dt<=0 or not 0<eta_c<=1 or not 0<eta_d<=1:
        raise ValueError('invalid energy limits, timestep or efficiency')
    max_power=_finite(constraints.get('max_power_mw'),'max_power_mw',nonnegative=True)
    max_charge=_finite(constraints.get('max_charge_mw',max_power),'max_charge_mw',nonnegative=True)
    max_discharge=_finite(constraints.get('max_discharge_mw',max_power),'max_discharge_mw',nonnegative=True)
    e=None if state_error_bound_mwh is None else _finite(state_error_bound_mwh,'state_error_bound_mwh',nonnegative=True)
    m=None if transition_error_bound_mwh is None else _finite(transition_error_bound_mwh,'transition_error_bound_mwh',nonnegative=True)
    reasons=[]
    if e is None:reasons.append('state_error_bound_missing')
    if m is None:reasons.append('transition_error_bound_missing')
    if charge>max_charge+1e-12 or discharge>max_discharge+1e-12:reasons.append('power_bounds')
    r={'supported':False,'reasons':reasons,'validity_horizon_steps':0,'expiry_lower_bound':None,'basis':'conditional_one_step_bounded_energy_error','state_error_bound_mwh':e,'transition_error_bound_mwh':m,'independently_validated_premises':False}
    if e is None or m is None:return r
    state_lo=center-e;state_hi=center+e;delta=dt*(eta_c*charge-discharge/eta_d)
    next_lo=state_lo+delta-m;next_hi=state_hi+delta+m
    r.update({'current_lower_mwh':state_lo,'current_upper_mwh':state_hi,'next_lower_mwh':next_lo,'next_upper_mwh':next_hi,'action_delta_mwh':delta})
    if state_lo<low-1e-12 or state_hi>high+1e-12:reasons.append('current_state_interval_outside_limits')
    if next_lo<low-1e-12 or next_hi>high+1e-12:reasons.append('next_state_interval_outside_limits')
    r['supported']=not reasons;r['validity_horizon_steps']=1 if r['supported'] else 0
    return r
