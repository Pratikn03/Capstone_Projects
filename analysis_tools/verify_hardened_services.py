"""Exercise proposed source correction using genuine trained models and isolated local queues."""
from pathlib import Path
import copy,json,os,traceback
from collections import Counter
import pandas as pd
import yaml
import original_pipeline_audit as A
import execute_native_services as S


def check_profiles():
    result=S.data_contract_profiles()
    from orius.dc3s.certificate import verify_certificate,verify_certificate_signature
    for p in (A.OUT/'profiles').glob('*.json'):
        rows=json.loads(p.read_text())
        for r in rows:
            if r['status_code']!=200:continue
            cert=r['body']['certificate'];secret=os.environ['ORIUS_CERTIFICATE_SIGNING_KEY']
            sig=verify_certificate_signature(cert,secret=secret)
            r['signature_verification_raw']=sig;r['signature_valid']=bool(sig['valid'])
            bad=copy.deepcopy(cert);bad['safe_action']['discharge_mw']+=1.
            tamper=verify_certificate_signature(bad,secret=secret)
            r['tamper_verification_raw']=tamper;r['tampered_signature_rejected']=not tamper['valid']
        A.save('profiles/'+p.name,rows)
        result[p.stem].update({'signature_valid':sum(r.get('signature_valid') is True for r in rows),'tamper_rejected':sum(r.get('tampered_signature_rejected') is True for r in rows)})
    A.save('native_service_summary.json',{'profiles':result,'patch_applied_to_ephemeral_checkout':True,'main_unchanged':True,'implementation':'original maintained modules plus explicitly supplied release-contract patch','signature_metric_correction':'Use verifier result.valid, not Python truthiness of a dictionary'})
    return result


def explicit_contract_fixtures():
    from fastapi.testclient import TestClient
    from services.api.main import app
    from services.api.config import get_api_keys,load_serving_config,load_uncertainty_config
    from orius.dc3s.certificate import verify_certificate_signature
    frame=pd.read_parquet(A.OUT/'public_hourly.parquet');split=json.loads((A.OUT/'splits.json').read_text());ix=split['test'][0]
    d=A.OUT/'runtime_workspace';os.chdir(d)
    try:
        frame.iloc[:ix+1].to_parquet(d/'features.parquet',index=False)
        os.environ['ORIUS_IOT_DUCKDB_PATH']=str(d/'fixture_queue.duckdb')
        os.environ['ORIUS_BMS_MIN_SOC_PCT']='0';os.environ['ORIUS_BMS_MAX_SOC_PCT']='1'
        opt=yaml.safe_load((d/'configs/optimization.yaml').read_text());opt['battery']['initial_soc_mwh']=.00512;opt['robust']['initial_soc_mwh']=.00512
        (d/'configs/optimization.yaml').write_text(yaml.safe_dump(opt));(d/'configs/optimize.yaml').write_text(yaml.safe_dump(opt))
        get_api_keys.cache_clear();load_serving_config.cache_clear();load_uncertainty_config.cache_clear()
        headers={'X-ORIUS-Key':'audit-key-local'};row=frame.iloc[ix]
        base={'device_id':'analytic-midpoint-NOT-measured-state','zone_id':'DE','current_soc_mwh':.00512,'telemetry_event':{'ts_utc':row.timestamp.isoformat(),'load_mw':float(row.load_mw),'renewables_mw':float(row.solar_mw),'mode':'active'},'horizon':1,'controller':'deterministic','include_certificate':True,'enqueue_iot':False}
        variants=[('missing_bounds',{},400),('declared_analytic_bounds',{'state_error_bound_mwh':.00005,'transition_error_bound_mwh':.00001},200),('unusable_bounds',{'state_error_bound_mwh':.1,'transition_error_bound_mwh':.00001},400),('queue_missing_bounds',{'enqueue_iot':True,'telemetry_event':dict(base['telemetry_event'],mode='shadow')},400),('queue_analytic_bounds',{'enqueue_iot':True,'state_error_bound_mwh':.00005,'transition_error_bound_mwh':.00001},200)]
        output=[]
        with TestClient(app) as client:
            for name,extra,wanted in variants:
                req=copy.deepcopy(base);req.update(extra);req['device_id']+=name
                r=client.post('/dc3s/step',json=req,headers=headers)
                rec={'name':name,'expected_status':wanted,'observed_status':r.status_code,'body':r.json(),'passed':r.status_code==wanted,'reference':'deliberately chosen numerical midpoint and error bounds; NOT independently validated physical state'}
                if r.status_code==200:
                    body=r.json();rec['signature_valid']=verify_certificate_signature(body['certificate'],secret=os.environ['ORIUS_CERTIFICATE_SIGNING_KEY'])['valid']
                    sa=body['safe_action'];nxt=.00512+.95*sa['charge_mw']-sa['discharge_mw']/.95
                    rec['independent_analytic_next_interval_mwh']=[nxt-.00006,nxt+.00006]
                    rec['analytic_corner_check']=nxt-.00006>=0 and nxt+.00006<=.01024
                    if extra.get('enqueue_iot'):
                        qr=client.get('/iot/command/next',params={'device_id':req['device_id'],'peek':'true'},headers=headers)
                        rec['local_queue_peek_status']=qr.status_code;rec['local_queue_peek']=qr.json()
                output.append(rec)
        A.save('explicit_contract_fixtures.json',output)
        assert all(r['passed'] for r in output),[(r['name'],r['observed_status']) for r in output]
        return {'cases':len(output),'matched_expected':sum(r['passed'] for r in output),'physical_device_connected':False,'no_client_or_lab_independent_review_claim':True}
    finally:os.chdir(A.ROOT)


if __name__=='__main__':
    os.chdir(A.ROOT)
    A.phase('source_snapshot',A.source_snapshot)
    A.phase('public_data',A.public_data)
    A.phase('public_training',A.public_training)
    A.phase('patched_native_service_profiles',check_profiles)
    A.phase('explicit_contract_fixtures',explicit_contract_fixtures)
    A.save('patched_execution_scope.json',{'public_forecasts':'newly fitted using original training/loading/prediction code','native_service':'original modules with reviewable hardening patch','physical_state_error':'unknown for the public measurements; live release blocked','analytic_fixtures':'new conditional arithmetic examples, not laboratory outcomes','original_historical_checkpoints':'absent in repository checkout','independent_laboratory_review':False})
