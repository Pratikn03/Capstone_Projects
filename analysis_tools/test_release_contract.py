import ast
import importlib.util
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

def helper():
    p=ROOT/'src/orius/dc3s/release_contract.py'
    assert p.exists(),'Explicit energy-state release contract is absent'
    spec=importlib.util.spec_from_file_location('audit_contract',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m.assess_battery_release

def call(**kw):
    v=dict(current_soc_mwh=.005,action={'charge_mw':0.,'discharge_mw':.001},constraints={'min_soc_mwh':.001,'max_soc_mwh':.009,'max_power_mw':.006,'time_step_hours':.25,'charge_efficiency':.95,'discharge_efficiency':.95},state_error_bound_mwh=.0001,transition_error_bound_mwh=.00002)
    v.update(kw);return helper()(**v)

def test_missing_state_bound_is_not_certified():
    r=call(state_error_bound_mwh=None);assert not r['supported'] and r['validity_horizon_steps']==0
    assert 'state_error_bound_missing' in r['reasons']

def test_missing_transition_bound_is_not_certified():
    assert not call(transition_error_bound_mwh=None)['supported']

def test_energy_units_and_analytic_next_interval():
    r=call();delta=-.001*.25/.95
    assert r['supported'] and r['validity_horizon_steps']==1
    assert r['next_lower_mwh']==pytest.approx(.005+delta-.00012)
    assert r['next_upper_mwh']==pytest.approx(.005+delta+.00012)

def test_no_multistep_claim_from_single_step_contract():
    r=call(action={'charge_mw':0.,'discharge_mw':0.});assert r['validity_horizon_steps']==1 and r['expiry_lower_bound'] is None

def test_uncertain_current_state_outside_limits_is_unsupported():
    r=call(current_soc_mwh=.009,state_error_bound_mwh=.0001)
    assert not r['supported'] and 'current_state_interval_outside_limits' in r['reasons']

def test_large_transition_error_blocks_nominally_safe_action():
    assert not call(transition_error_bound_mwh=.008)['supported']

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.,True])
@pytest.mark.parametrize('name',['state_error_bound_mwh','transition_error_bound_mwh'])
def test_invalid_declared_bounds_rejected(name,bad):
    with pytest.raises(ValueError):call(**{name:bad})

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True])
def test_invalid_current_energy_rejected(bad):
    with pytest.raises(ValueError):call(current_soc_mwh=bad)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.,True])
def test_invalid_command_rejected(bad):
    with pytest.raises(ValueError):call(action={'charge_mw':0.,'discharge_mw':bad})

def test_simultaneous_command_rejected():
    with pytest.raises(ValueError):call(action={'charge_mw':.001,'discharge_mw':.001})

def test_boundary_equality_supported_under_zero_bounds():
    assert call(current_soc_mwh=.001,action={'charge_mw':0.,'discharge_mw':0.},state_error_bound_mwh=0.,transition_error_bound_mwh=0.)['supported']

def test_each_supported_action_contains_independent_corner_evolutions():
    from itertools import product
    for state,e,m,c,d,dt in product([.002,.005,.008],[0,.0001,.001],[0,.00002],[0,.001],[0,.001],[.05,.25,1.]):
        if c and d:continue
        cfg={'min_soc_mwh':.001,'max_soc_mwh':.009,'max_power_mw':.006,'time_step_hours':dt,'charge_efficiency':.93,'discharge_efficiency':.89}
        r=call(current_soc_mwh=state,state_error_bound_mwh=e,transition_error_bound_mwh=m,action={'charge_mw':c,'discharge_mw':d},constraints=cfg)
        if r['supported']:
            for actual,error in product([state-e,state+e],[-m,m]):
                nxt=actual+.93*c*dt-d*dt/.89+error
                assert .001-1e-12<=nxt<=.009+1e-12

def test_api_does_not_send_load_power_bounds_to_energy_horizon():
    tree=ast.parse((ROOT/'services/api/routers/dc3s.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ['certificate_validity_horizon','certificate_expiration_bound']:
            for kw in node.keywords:
                if kw.arg in ['interval_lower_mwh','interval_upper_mwh']:
                    assert not any(isinstance(x,ast.Name) and x.id in ['lower','upper'] for x in ast.walk(kw.value)), 'MW load interval routed into MWh state horizon'

def test_api_schema_accepts_explicit_energy_bounds_and_rejects_invalid_values():
    from services.api.routers.dc3s import DC3SStepRequest
    from pydantic import ValidationError
    assert 'state_error_bound_mwh' in DC3SStepRequest.model_fields
    assert 'transition_error_bound_mwh' in DC3SStepRequest.model_fields
    for bad in [float('nan'),float('inf'),-1.,True]:
        with pytest.raises(ValidationError):
            DC3SStepRequest(device_id='case',current_soc_mwh=.005,state_error_bound_mwh=bad,transition_error_bound_mwh=0.)
