"""Original ORIUS execution audit. Offline actions only; no production credentials.
New orchestration/data adaptation, unchanged upstream scientific implementations.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, os, pickle, secrets, shutil, subprocess, sys, tarfile, time, traceback
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit_output' / 'integration'
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
BASE = '31de1c97ac96fa4e12a8f6c437318033bde7cf72'
DATA_COMMIT = '5884d253a5267fb240b7a8df6fa9e4d49a905167'
STATUS = {}

def js(v):
    if isinstance(v, dict): return {str(k):js(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [js(x) for x in v]
    if hasattr(v, 'tolist'): return js(v.tolist())
    if hasattr(v, 'item'): return js(v.item())
    if isinstance(v, float) and not math.isfinite(v): return None
    if isinstance(v, Path): return str(v)
    return v

def save(name, data):
    p=OUT/name; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(js(data), indent=2, sort_keys=True, default=str))
    return p

def phase(name, fn):
    start=time.perf_counter()
    try:
        result=fn(); STATUS[name]={'status':'completed','seconds':time.perf_counter()-start,'result':result}
    except Exception as e:
        STATUS[name]={'status':'failed','seconds':time.perf_counter()-start,'error':repr(e),'traceback':traceback.format_exc()}
    save('phase_status.json',STATUS)
    print('AUDIT_PHASE',name,json.dumps(js(STATUS[name]),default=str),flush=True)
    return STATUS[name].get('result')

def command(name, cmd, timeout=150):
    p=OUT/(name+'.log'); start=time.perf_counter()
    with p.open('w') as f:
        try: r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout);rc=r.returncode
        except subprocess.TimeoutExpired:rc=124
    return {'command':cmd,'returncode':rc,'seconds':time.perf_counter()-start,'log':str(p.relative_to(ROOT)),'tail':p.read_text(errors='replace')[-3500:]}

def source_snapshot():
    hashes={}
    for folder in ['src','services','scripts','configs','tests']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix in ['.py','.yaml','.yml','.toml','.json']:
                if '__pycache__' in p.parts:continue
                hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    save('source_sha256.json',hashes)
    with tarfile.open(OUT/'original_source.tar.gz','w:gz') as z:
        for rel in hashes:z.add(ROOT/rel,arcname=rel)
        for rel in ['LICENSE','README.md','pyproject.toml','requirements.lock.txt','pytest.ini','ORIUS_REPRODUCIBILITY.md']:
            if (ROOT/rel).exists():z.add(ROOT/rel,arcname=rel)
    models=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.suffix in ['.pkl','.pt','.joblib','.parquet'] and 'audit_output' not in p.parts]
    return {'base':BASE,'source_files':len(hashes),'preexisting_model_or_feature_artifacts':models,'whole_application_initialization':'not replaced or bypassed'}

def public_data():
    import requests,numpy as np,pandas as pd
    allrows=[]; inventory=[]
    partitions=['2025-08-a.csv','2025-08-b.csv','2025-09-a.csv']
    for part in partitions:
        url=f'https://raw.githubusercontent.com/OpenCEM-platform/opencem-dataset/{DATA_COMMIT}/data/measurements/{part}'
        r=requests.get(url,timeout=90);r.raise_for_status();raw=r.content
        (OUT/'data').mkdir(exist_ok=True);(OUT/'data'/part).write_bytes(raw)
        rd=csv.reader(io.StringIO(raw.decode('utf-8-sig')));header=next(rd); widths=Counter();valid=0
        needed=['read_ts','inverter','battvolt','battcurr','battsoc','battchgpower','outsumw','pv1power']
        assert all(k in header for k in needed),header
        for lineno,row in enumerate(rd,2):
            widths[len(row)]+=1
            if len(row)!=len(header):continue
            x={k:row[header.index(k)] for k in needed}; x['source_file']=part;x['source_line']=lineno
            allrows.append(x);valid+=1
        inventory.append({'file':part,'url':url,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'header_width':len(header),'row_widths':dict(widths),'accepted_exact_width':valid})
    save('data_inventory.json',inventory)
    df=pd.DataFrame(allrows)
    for k in ['read_ts','inverter','battvolt','battcurr','battsoc','battchgpower','outsumw','pv1power']:df[k]=pd.to_numeric(df[k],errors='coerce')
    counts=df.groupby('inverter').size(); inv=int(counts.idxmax())
    df=df[df.inverter==inv].copy();df['timestamp']=pd.to_datetime(df.read_ts,unit='s',utc=True)
    duplicates=int(df.duplicated(['inverter','read_ts']).sum());df=df.drop_duplicates(['inverter','read_ts']).sort_values('timestamp')
    save('data_quality.json',{'inverter_chosen_by_row_count_not_outcome':inv,'rows':len(df),'duplicates_removed':duplicates,'missing':df.isna().sum().to_dict(),'ambiguous_current_over_200A':int((df.battcurr.abs()>200).sum()),'no_current_sign_correction_applied':True})
    good=df[(df.outsumw>=0)&(df.outsumw<=16000)&(df.pv1power>=0)&(df.pv1power<=20000)&df.battsoc.between(0,100)].copy()
    hour=good.set_index('timestamp').resample('1h',label='right',closed='left').agg({'outsumw':'mean','pv1power':'mean','battsoc':'last','battvolt':'mean','battcurr':'mean','read_ts':'count'})
    hour=hour[hour.read_ts>=3].dropna(subset=['outsumw','pv1power','battsoc']).reset_index()
    hour['load_mw']=hour.outsumw/1e6;hour['solar_mw']=hour.pv1power/1e6
    hour['load_now']=hour.load_mw;hour['solar_now']=hour.solar_mw;hour['soc_reported']=hour.battsoc/100
    hour['load_prev']=hour.load_mw.shift(1)
    hour['hour_sin']=np.sin(2*np.pi*hour.timestamp.dt.hour/24);hour['hour_cos']=np.cos(2*np.pi*hour.timestamp.dt.hour/24)
    hour['next_load']=hour.load_mw.shift(-1);hour['next_solar']=hour.solar_mw.shift(-1)
    hour['target_timestamp']=hour.timestamp.shift(-1)
    consecutive=(hour.timestamp.diff().dt.total_seconds()==3600)&((hour.target_timestamp-hour.timestamp).dt.total_seconds()==3600)
    frame=hour[consecutive].dropna(subset=['load_prev','next_load','next_solar']).copy().reset_index(drop=True)
    if len(frame)<100:raise RuntimeError(f'Only {len(frame)} usable consecutive hourly examples. No synthetic substitution allowed.')
    n=len(frame);a=int(n*.6);b=int(n*.8)
    split={'train':list(range(a)),'calibration':list(range(a+1,b)),'test':list(range(b+1,n))}
    assert set(split['train']).isdisjoint(split['calibration']) and set(split['test']).isdisjoint(split['calibration'])
    frame.to_parquet(OUT/'public_hourly.parquet',index=False)
    save('hourly_records.json',frame.to_dict('records'))
    save('splits.json',split)
    return {'selected_inverter':inv,'raw_records_all_widths':sum(sum(x['row_widths'].values()) for x in inventory),'exact_width_records':len(allrows),'eligible_hourly_examples':len(frame),'split_counts':{k:len(v) for k,v in split.items()},'interval': [str(frame.timestamp.iloc[0]),str(frame.target_timestamp.iloc[-1])],'reference_state':'reported inverter SOC; not independent physical truth','row_selection':'exact header width; finite load/PV and reported SOC; at least 3 observations/hour; adjacent hours only'}

def public_training():
    import numpy as np,pandas as pd
    from orius.forecasting.ml_gbm import train_gbm
    from orius.forecasting.uncertainty.conformal import ConformalConfig,ConformalInterval,save_conformal
    from orius.forecasting.predict import load_model_bundle,predict_next_24h
    frame=pd.read_parquet(OUT/'public_hourly.parquet');split=json.loads((OUT/'splits.json').read_text())
    feats=['load_now','solar_now','load_prev','hour_sin','hour_cos']
    modeldir=OUT/'models';modeldir.mkdir(exist_ok=True);uncdir=OUT/'uncertainty';uncdir.mkdir(exist_ok=True)
    records={}
    for target,nextcol in [('load_mw','next_load'),('solar_mw','next_solar')]:
        train=frame.iloc[split['train']];cal=frame.iloc[split['calibration']];test=frame.iloc[split['test']]
        kind,model=train_gbm(train[feats].to_numpy(),train[nextcol].to_numpy(),params={'backend':'lightgbm','n_estimators':100,'learning_rate':.05,'num_leaves':15,'min_child_samples':10,'random_state':42,'n_jobs':1,'verbosity':-1})
        p=modeldir/f'{target}.pkl';p.write_bytes(pickle.dumps({'model_type':'gbm','model':model,'feature_cols':feats,'target':target,'horizon':1,'training_source':'OpenCEM public measured power; new fit, not historical paper artifact'}))
        digest=hashlib.sha256(p.read_bytes()).hexdigest();Path(str(p)+'.sha256').write_text(digest);p.with_suffix('.sha256').write_text(digest)
        calpred=np.asarray(model.predict(cal[feats].to_numpy()));ci=ConformalInterval(ConformalConfig(alpha=.1,horizon_wise=False,rolling=False));ci.fit_calibration(cal[nextcol].to_numpy(),calpred)
        save_conformal(uncdir/f'{target}_conformal.json',ci,meta={'scope':'held-out temporal power residuals; not physical SOC error; exchangeability unverified'})
        loaded=load_model_bundle(p);pred=[]
        for ix in split['test']:
            r=predict_next_24h(frame.iloc[:ix+1],loaded,horizon=1);pred.append(r['forecast'][0])
            assert pd.Timestamp(r['timestamp'][0])==frame.target_timestamp.iloc[ix]
        y=test[nextcol].to_numpy();pr=np.asarray(pred);q=float(ci.q_global)
        persistence=test['load_now' if target=='load_mw' else 'solar_now'].to_numpy()
        records[target]={'backend':kind,'feature_cols':feats,'model_sha256':digest,'train_count':len(train),'calibration_count':len(cal),'test_count':len(test),'q_global_mw':q,'test_mae_mw':float(np.mean(abs(pr-y))),'persistence_mae_mw':float(np.mean(abs(persistence-y))),'test_empirical_power_interval_coverage':float(np.mean(abs(pr-y)<=q)),'test_prediction':pr.tolist(),'test_actual':y.tolist(),'test_timestamps':test.target_timestamp.astype(str).tolist()}
    save('training_results.json',records)
    return {k:{a:b for a,b in v.items() if a not in ['test_prediction','test_actual','test_timestamps']} for k,v in records.items()}

def runtime_workspace():
    import yaml
    d=OUT/'runtime_workspace';d.mkdir(exist_ok=True);shutil.copytree(ROOT/'configs',d/'configs',dirs_exist_ok=True)
    cfg={'data':{'features_path':str(d/'features.parquet')},'models':{'load_mw':str(OUT/'models/load_mw.pkl'),'solar_mw':str(OUT/'models/solar_mw.pkl'),'wind_mw':str(d/'no_wind_model.pkl')},'fallback_order':['gbm']}
    (d/'configs/forecast.yaml').write_text(yaml.safe_dump(cfg))
    (d/'configs/uncertainty.yaml').write_text(yaml.safe_dump({'artifacts_dir':str(OUT/'uncertainty')}))
    opt={'time_step_hours':1.,'objective':{'cost_weight':1.,'carbon_weight':0.},'battery':{'capacity_mwh':.01024,'max_power_mw':.006,'max_charge_mw':.006,'max_discharge_mw':.006,'min_soc_mwh':.001024,'max_soc_mwh':.009216,'initial_soc_mwh':.00512,'charge_efficiency':.95,'discharge_efficiency':.95,'efficiency':.95,'degradation_cost_per_mwh':10.},'grid':{'max_import_mw':.05,'price_per_mwh':70.,'carbon_kg_per_mwh':400.},'penalties':{'unmet_load_per_mw':10000.,'curtailment_per_mw':0.,'peak_per_mw':0.},'robust':{'min_soc_mwh':.001024,'initial_soc_mwh':.00512,'risk_weight_worst_case':1.}}
    (d/'configs/optimization.yaml').write_text(yaml.safe_dump(opt));(d/'configs/optimize.yaml').write_text(yaml.safe_dump(opt))
    dc=yaml.safe_load((d/'configs/dc3s.yaml').read_text());dc['dc3s']['audit']={'duckdb_path':str(d/'audit.duckdb'),'table_name':'dispatch_certificates','state_table_name':'dc3s_online_state'}
    (d/'configs/dc3s.yaml').write_text(yaml.safe_dump(dc))
    os.environ.update({'ORIUS_ENV':'test','ORIUS_AUTH_DISABLED_FOR_TESTS':'0','ORIUS_BMS_CAPACITY_MWH':'.01024','ORIUS_BMS_MAX_POWER_MW':'.006','ORIUS_BMS_MIN_SOC_PCT':'.1','ORIUS_BMS_MAX_SOC_PCT':'.9','ORIUS_UNCERTAINTY_CONFIG':str(d/'configs/uncertainty.yaml'),'ORIUS_API_KEYS':json.dumps({'audit-key-local':['read','write']}),'ORIUS_CERTIFICATE_SIGNING_KEY':secrets.token_hex(32),'ORIUS_REQUIRE_CERT_SIGNATURE':'1'})
    save('lab_contract.json',{'scope':'offline public-data serving and engineering-constraint contract; no physical actions','nameplate_energy_mwh':.01024,'analyst_selected_energy_limits_mwh':[.001024,.009216],'analyst_selected_power_limit_mw':.006,'assumed_efficiencies':.95,'timestep_hours':1.,'data_reference':'reported SOC; not independent truth','uncertainty':'power residual calibration from disjoint chronological split; physical SOC error not supplied','independent_engineer_review':False,'historical_paper_artifact_reproduction':False,'permitted_actions':'local computation and local queue only; no driver addresses/credentials'})
    return d

def api_execution():
    import numpy as np,pandas as pd,yaml
    frame=pd.read_parquet(OUT/'public_hourly.parquet');split=json.loads((OUT/'splits.json').read_text())
    d=runtime_workspace();os.chdir(d)
    try:
        from fastapi.testclient import TestClient
        from services.api.main import app
        from services.api.config import get_api_keys,load_uncertainty_config,load_serving_config
        get_api_keys.cache_clear();load_uncertainty_config.cache_clear();load_serving_config.cache_clear()
        from orius.dc3s.certificate import verify_certificate
        from orius.certos.runtime import CertOSRuntime
        client=TestClient(app,raise_server_exceptions=False);headers={'X-ORIUS-Key':'audit-key-local'}
        unauthorized=client.post('/dc3s/step',json={'device_id':'audit','current_soc_mwh':.005,'horizon':1}).status_code
        rows=[];certos=[];gov=CertOSRuntime()
        ids=[i for i in split['test'] if .1<=frame.soc_reported.iloc[i]<=.9][:48]
        if not ids:raise RuntimeError('No held-out SOC rows inside the declared analysis envelope.')
        for j,ix in enumerate(ids):
            row=frame.iloc[ix];frame.iloc[:ix+1].to_parquet(d/'features.parquet',index=False)
            energy=float(row.soc_reported*.01024)
            opt=yaml.safe_load((d/'configs/optimization.yaml').read_text());opt['battery']['initial_soc_mwh']=energy;opt['robust']['initial_soc_mwh']=energy
            (d/'configs/optimization.yaml').write_text(yaml.safe_dump(opt));(d/'configs/optimize.yaml').write_text(yaml.safe_dump(opt))
            req={'device_id':'opencem-shadow','zone_id':'DE','current_soc_mwh':energy,'telemetry_event':{'ts_utc':row.timestamp.isoformat(),'load_mw':float(row.load_mw),'renewables_mw':float(row.solar_mw),'mode':'shadow'},'horizon':1,'controller':'deterministic','include_certificate':True,'enqueue_iot':False}
            r=client.post('/dc3s/step',json=req,headers=headers)
            try:b=r.json()
            except Exception:b={'body':r.text[:3000]}
            rec={'source_timestamp':str(row.timestamp),'reported_energy_mwh':energy,'status_code':r.status_code,'body':b}
            if r.status_code==200:
                a=client.get('/dc3s/audit/'+b['command_id'],headers=headers);audit=a.json();v=verify_certificate(audit,require_signature=True,signature_secret=os.environ['ORIUS_CERTIFICATE_SIGNING_KEY'])
                rec['audit_status']=a.status_code;rec['signature_verification']=v
                act=b['safe_action'];rec['independent_point_next_energy_mwh']=energy+.95*float(act['charge_mw'])-float(act['discharge_mw'])/.95
                h=1 if b['guarantee_checks_passed'] else 0
                try:
                    state=gov.validate_and_step(energy,b['proposed_action'],act,h,observed_state={'current_soc_mwh':energy},constraints={'min_soc_mwh':.001024,'max_soc_mwh':.009216,'capacity_mwh':.01024,'time_step_hours':1.,'charge_efficiency':.95,'discharge_efficiency':.95})
                    certos.append({'timestamp':str(row.timestamp),'status':state.status,'fallback':state.fallback_active,'invariant_failures':gov.check_invariants(state),'note':'audit harness calls original CertOS separately; API does not natively call this class'})
                except Exception as e:certos.append({'error':repr(e)})
            rows.append(rec)
            if j<4: print('API_CASE',j,json.dumps(js(rec),default=str)[:3500],flush=True)
        save('api_cases.json',rows);save('certos_cases.json',certos)
        # Exercise the actual fail-closed active gate with exactly the same last public row.
        req['telemetry_event']['mode']='active';req['device_id']='opencem-active-gate-check'
        ar=client.post('/dc3s/step',json=req,headers=headers)
        try:ab=ar.json()
        except Exception:ab={'text':ar.text}
        save('active_gate_check.json',{'status_code':ar.status_code,'body':ab,'physical_device_connected':False})
        return {'requests':len(rows),'http_status_counts':dict(Counter(x['status_code'] for x in rows)),'unauthenticated_status':unauthorized,'audit_retrievals':sum(x.get('audit_status')==200 for x in rows),'guarantee_pass_count':sum(x['body'].get('guarantee_checks_passed',False) for x in rows),'signature_valid_count':sum(bool(x.get('signature_verification',{}).get('valid',False)) for x in rows),'certos_cases':len(certos),'active_gate_status':ar.status_code,'no_monkeypatch_or_stub':True,'models':'native GBM training function; new real-data fit; not the missing historical trained artifacts'}
    finally:os.chdir(ROOT)

def standalone_kernel():
    import pandas as pd,yaml
    from orius.dc3s.pipeline import run_dc3s_step
    from orius.dc3s.battery_adapter import BatteryDomainAdapter
    from orius.dc3s.guarantee_checks import evaluate_guarantee_checks
    from orius.forecasting.predict import load_model_bundle,predict_next_24h
    from orius.forecasting.uncertainty.conformal import load_conformal
    frame=pd.read_parquet(OUT/'public_hourly.parquet');split=json.loads((OUT/'splits.json').read_text())
    model=load_model_bundle(OUT/'models/load_mw.pkl');ci=load_conformal(OUT/'uncertainty/load_mw_conformal.json');cfg=yaml.safe_load((ROOT/'configs/dc3s.yaml').read_text())['dc3s']
    cfg['shift_aware_uncertainty']={'enabled':False}
    rows=[];last=None;prev=None
    for ix in split['test'][:24]:
        r=frame.iloc[ix];event={'ts':r.timestamp.isoformat(),'load_mw':float(r.load_mw)};e=float(r.soc_reported*.01024)
        state=SimpleNamespace(current_soc_mwh=e,min_soc_mwh=.001024,max_soc_mwh=.009216,capacity_mwh=.01024,max_power_mw=.006)
        yhat=float(predict_next_24h(frame.iloc[:ix+1],model,horizon=1)['forecast'][0])
        out=run_dc3s_step(event=event,last_event=last,yhat=yhat,q=float(ci.q_global),candidate_action={'charge_mw':0.,'discharge_mw':min(.006,max(0.,yhat))},domain_adapter=BatteryDomainAdapter(),state=state,cfg=cfg,prev_cert_hash=prev,model_hash=hashlib.sha256((OUT/'models/load_mw.pkl').read_bytes()).hexdigest())
        passed,reasons,next_e=evaluate_guarantee_checks(current_soc=e,action=out['safe_action'],constraints={'min_soc_mwh':.001024,'max_soc_mwh':.009216,'max_power_mw':.006,'time_step_hours':1.,'charge_efficiency':.95,'discharge_efficiency':.95})
        rows.append({'timestamp':str(r.timestamp),'safe_action':out['safe_action'],'reliability':out['reliability_w'],'inflation':out['inflation'],'external_original_checker_passed':passed,'external_original_checker_reasons':reasons,'point_next_energy_mwh':next_e,'certificate':out['certificate']})
        last=event;prev=out['certificate'].get('certificate_hash')
    save('five_stage_kernel.json',rows)
    return {'five_stage_calls':len(rows),'original_checker_passes':sum(r['external_original_checker_passed'] for r in rows),'configuration':'original kernel with new public-data trained model inputs; optional shift-aware mode explicitly disabled','no_original_source_patch':True}

if __name__=='__main__':
    os.chdir(ROOT)
    phase('source_snapshot',source_snapshot)
    phase('maintained_benchmark',lambda: command('bench',['python','scripts/run_orius_bench_release.py','--seeds','2','--horizon','24','--out',str(OUT/'bench')],120))
    phase('public_data',public_data)
    phase('public_training',public_training)
    phase('original_api',api_execution)
    phase('original_five_stage_kernel',standalone_kernel)
    phase('published_claim_check',lambda: command('paper_claims',['python','scripts/validate_paper_claims.py','--tex','orius_book.tex'],90))
    save('final_scope.json',{'complete_source_checked_out':True,'installed_pinned_dependencies':'see workflow installation receipt','application_initialization_replaced':False,'model_or_forecast_mocked':False,'source_patched_by_this_driver':False,'new_training_on_public_data':True,'original_historical_trained_artifacts_available':False,'public_SOC_is_independent_truth':False,'physical_device_commanded':False,'independent_human_lab_review':False,'scientific_historical_result_reproduction':False})
    print('FINAL_PHASES',json.dumps(js(STATUS),default=str),flush=True)
