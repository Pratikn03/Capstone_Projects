"""Additional native execution; preserves failures and never labels nameplate limits certified."""
from __future__ import annotations
import copy, hashlib, json, os, sys, traceback
from pathlib import Path
from collections import Counter
import original_pipeline_audit as A
import numpy as np
import pandas as pd
import yaml


def data_contract_profiles():
    # Evaluate the SAME held-out records under both contracts. No successful-row filtering.
    frame=pd.read_parquet(A.OUT/'public_hourly.parquet');split=json.loads((A.OUT/'splits.json').read_text())
    d=A.runtime_workspace();os.chdir(d)
    from fastapi.testclient import TestClient
    from services.api.main import app
    from services.api.config import get_api_keys,load_uncertainty_config,load_serving_config
    from orius.dc3s.certificate import verify_certificate,verify_certificate_signature
    from orius.certos.runtime import CertOSRuntime
    get_api_keys.cache_clear();load_uncertainty_config.cache_clear();load_serving_config.cache_clear()
    headers={'X-ORIUS-Key':'audit-key-local'};allresults={}
    try:
      with TestClient(app,raise_server_exceptions=True) as client:
        health=client.get('/health').json();badkey=client.post('/dc3s/step',json={'device_id':'unauth','current_soc_mwh':.005,'horizon':1}).status_code
        for profile,lo,hi in [('initial_10_90_percent',.1,.9),('nameplate_0_100_percent_DIAGNOSTIC_ONLY',0.,1.)]:
            os.environ['ORIUS_BMS_MIN_SOC_PCT']=str(lo);os.environ['ORIUS_BMS_MAX_SOC_PCT']=str(hi)
            get_api_keys.cache_clear();load_serving_config.cache_clear();load_uncertainty_config.cache_clear()
            opt=yaml.safe_load((d/'configs/optimization.yaml').read_text());opt['battery']['min_soc_mwh']=lo*.01024;opt['battery']['max_soc_mwh']=hi*.01024;opt['robust']['min_soc_mwh']=lo*.01024
            (d/'configs/optimization.yaml').write_text(yaml.safe_dump(opt));(d/'configs/optimize.yaml').write_text(yaml.safe_dump(opt))
            dc=yaml.safe_load((d/'configs/dc3s.yaml').read_text());dc['dc3s']['audit']['duckdb_path']=str(d/(profile+'.duckdb'));(d/'configs/dc3s.yaml').write_text(yaml.safe_dump(dc))
            rows=[];gov=CertOSRuntime();previous=None
            for j,ix in enumerate(split['test'][:48]):
                row=frame.iloc[ix];frame.iloc[:ix+1].to_parquet(d/'features.parquet',index=False);energy=float(row.soc_reported*.01024)
                opt['battery']['initial_soc_mwh']=energy;opt['robust']['initial_soc_mwh']=energy
                (d/'configs/optimization.yaml').write_text(yaml.safe_dump(opt));(d/'configs/optimize.yaml').write_text(yaml.safe_dump(opt))
                controller='deterministic' if j%2==0 else 'robust'
                req={'device_id':profile+'-'+controller,'zone_id':'DE','current_soc_mwh':energy,'telemetry_event':{'ts_utc':row.timestamp.isoformat(),'load_mw':float(row.load_mw),'renewables_mw':float(row.solar_mw),'mode':'shadow'},'horizon':1,'controller':controller,'include_certificate':True,'enqueue_iot':False}
                record={'index':int(ix),'timestamp':str(row.timestamp),'reported_energy_mwh':energy,'controller':controller,'inside_declared_point_envelope':lo*.01024<=energy<=hi*.01024}
                try:
                    r=client.post('/dc3s/step',json=req,headers=headers);record['status_code']=r.status_code;record['body']=r.json()
                    if r.status_code==200:
                        body=record['body'];ar=client.get('/dc3s/audit/'+body['command_id'],headers=headers);cert=ar.json()
                        record['audit_status']=ar.status_code
                        record['certificate_verification']=verify_certificate(cert,require_signature=True,signature_secret=os.environ['ORIUS_CERTIFICATE_SIGNING_KEY'])
                        record['signature_valid']=verify_certificate_signature(cert,secret=os.environ['ORIUS_CERTIFICATE_SIGNING_KEY'])
                        tampered=copy.deepcopy(cert);tampered['safe_action']['discharge_mw']=float(tampered['safe_action'].get('discharge_mw',0))+1
                        record['tampered_signature_rejected']=not verify_certificate_signature(tampered,secret=os.environ['ORIUS_CERTIFICATE_SIGNING_KEY'])
                        act=body['safe_action'];record['point_next_energy_mwh']=energy+.95*float(act['charge_mw'])-float(act['discharge_mw'])/.95
                        h=1 if body['guarantee_checks_passed'] else 0
                        try:
                            state=gov.validate_and_step(energy,body['proposed_action'],act,h,observed_state={'current_soc_mwh':energy},constraints={'min_soc_mwh':lo*.01024,'max_soc_mwh':hi*.01024,'capacity_mwh':.01024,'time_step_hours':1.,'charge_efficiency':.95,'discharge_efficiency':.95})
                            record['certos']={'status':state.status,'fallback_active':state.fallback_active,'invariant_failures':gov.check_invariants(state),'connection':'explicit audit composition, not existing API wiring'}
                        except Exception as e:record['certos']={'error':repr(e)}
                        req['telemetry_event']['mode']='active';req['device_id']='active-'+req['device_id']
                        active=client.post('/dc3s/step',json=req,headers=headers);record['active_status']=active.status_code;record['active_body']=active.json()
                except Exception as e:
                    record['status_code']=500;record['error']=repr(e);record['traceback']=traceback.format_exc()
                rows.append(record)
            A.save('profiles/'+profile+'.json',rows)
            allresults[profile]={'requests':len(rows),'status_counts':dict(Counter(r['status_code'] for r in rows)),'audit_retrieved':sum(r.get('audit_status')==200 for r in rows),'signature_valid':sum(r.get('signature_valid') is True for r in rows),'tamper_rejected':sum(r.get('tampered_signature_rejected') is True for r in rows),'guarantee_passed':sum(r.get('body',{}).get('guarantee_checks_passed',False) for r in rows),'point_state_in_envelope':sum(r['inside_declared_point_envelope'] for r in rows),'active_status_counts':dict(Counter(r.get('active_status') for r in rows if 'active_status' in r)),'first_errors':[r for r in rows if r['status_code']!=200][:2],'scope':'nameplate profile is an explicitly changed diagnostic contract, not certified safety or recovery of independent true state'}
        A.save('native_service_summary.json',{'profiles':allresults,'health':health,'unauthorized_status':badkey,'lifespan_executed':True,'original_implementations_unchanged':True})
        return allresults
    finally:os.chdir(A.ROOT)


def native_cpsbench():
    from orius.cpsbench_iot.runner import run_single
    from orius.cpsbench_iot.scenarios import DEFAULT_SCENARIOS
    os.chdir(A.ROOT);rows=[]
    for scenario in DEFAULT_SCENARIOS:
      for seed in [11,22]:
        try:
            out=run_single(scenario=scenario,seed=seed,horizon=24,return_controller_buffers=True,controllers_filter=['deterministic_lp','robust_fixed_interval','dc3s_wrapped','dc3s_ftit'])
            payload={k:v for k,v in out.items() if k!='event_log'};payload['event_log']=out['event_log'].to_dict('records')
            A.save(f'cpsbench/{scenario}_{seed}.json',payload)
            rows+=out['main_rows']
        except Exception as e:rows.append({'scenario':scenario,'seed':seed,'error':repr(e),'traceback':traceback.format_exc()})
    A.save('cpsbench_summary.json',rows)
    return {'scenario_seed_pairs':12,'rows':len(rows),'errors':[r for r in rows if 'error' in r],'source':'original CPSBench run_single with original configuration and its built-in synthetic episode generator','historical_CQR_artifacts_present':False,'fallback_interval_behavior':'unchanged original loader behavior; this is not historical calibrated checkpoint reproduction'}


def source_unit_trace():
    # Static data-flow evidence plus independent arithmetic; not a production patch.
    from orius.universal_theory.battery_instantiation import certificate_validity_horizon
    constraints={'min_soc_mwh':.001024,'max_soc_mwh':.009216,'time_step_hours':1.,'charge_efficiency':.95,'discharge_efficiency':.95}
    same={'safe_action':{'charge_mw':0.,'discharge_mw':0.},'constraints':constraints,'sigma_d':.000001,'max_steps':24}
    a=certificate_validity_horizon(interval_lower_mwh:.0001,interval_upper_mwh:.00012,**same)


if __name__=='__main__':
    os.chdir(A.ROOT)
    A.phase('source_snapshot',A.source_snapshot)
    A.phase('public_data',A.public_data)
    A.phase('public_training',A.public_training)
    A.phase('native_service_profiles',data_contract_profiles)
    A.phase('native_cpsbench',native_cpsbench)
    A.save('execution_completion.json',{'attempted_original_api':True,'maintained_function_bodies_patched':False,'history_checkpoints_reconstructed':False,'independent_reference_state':False,'physical_or_customer_validation':False,'independent_review':False})
