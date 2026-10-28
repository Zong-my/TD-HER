"""One frozen analytical test pass. Metadata/P0/prefixes only, no test truth.

Output guards retain runs. Per-event statuses preserve the complete M0 index.
Only Codex-owned output files are written. No PSS/E imports/calls occur.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import csv,hashlib,json,sys,time,platform
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
A=ROOT.parent;P=A.parent;SHARED=work_path('analytical/shared')
sys.path[:0]=[str(ROOT),str(A/'reproduction_v1')]
from analytical_predictor_v1 import predict
from causal_inputs_v2 import read_frozen_window,read_post_event_times_metadata
RUN_ID='physical_causal_sfr_fnp_v2'
OUT=work_path('analytical/shared/analytical')/RUN_ID
DATA=work_path('data/ieee39_v8_80_10_10')
G={}


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while part:=f.read(1024*1024):h.update(part)
    return h.hexdigest()


def init_worker():
    G['cfg']=json.loads(((REPO_ROOT / 'configs/analytical/analytical_operational_config_v1.json')).read_text())
    G['case']=json.loads((work_path('analytical/adaptation_v1/results_sav_v2/sav_bound_case_map.json')).read_text())
    G['bank']=json.loads((work_path('analytical/adaptation_v1/results_fnp_conditioned_v1/bank.json')).read_text())
    G['conv']=dict(np.load(work_path('analytical/adaptation_v1/results_fnp_conditioned_v1/offline_convolution_table.npz')))


def base_rows(e):
    return [{
        'interface_version':'1.0.0','manifest_id':e['manifest_id'],'event_id':e['event_id'],
        'scenario':e['scenario'],'active_split':e['active_split'],'row_index':e['row_index'],
        'method':method,'run_id':RUN_ID,'parameter_set_id':'historical_sav_dyr_known_h_linear_v1',
        'target_kind':target,'f_base_hz':60.,'delta_f_hat_hz':None,'t_extremum_hat_rel_s':None,
        'prediction_status':'INPUT_UNAVAILABLE','reason':'not evaluated',
        't_obs_end_rel_s':e['t_obs_end_rel_s'],'measured_compute_s':None,
        'timing_status':'UNAVAILABLE','timing_scope_id':'causal_array_state_and_method_v1',
        'reproduction_record_id':'original_domain_20260906_v1',
        'input_access_record_id':'physical_causal_sfr_fnp_input_ledger_v1',
        'qualification':'known P0 and unconstrained linear governor approximation'
    } for method,target in [('CLASSICAL_SFR','signed_max_abs_excursion'),('FNP','first_downward_nadir')]]


def worker(e):
    rows=base_rows(e);diag={'event_id':e['event_id'],'scenario':e['scenario'],'row_index':e['row_index'],
                         'disturbance_category_for_cohort_only':e['disturbance_category']}
    if e['disturbance_category'] not in G['cfg']['primary_event_categories']:
        for r in rows:r.update(prediction_status='NOT_APPLICABLE',reason='short-circuit event outside frozen power-imbalance cohort')
        diag['cohort']='OUTSIDE_PRIMARY_COHORT';return rows,diag
    try:
        path=P/e['raw_path']
        if sha(path)!=e['raw_sha256']:raise ValueError('Raw hash differs from verified M0 event')
        start=time.perf_counter();w=read_frozen_window(path,trigger_s=float(e['t0_abs_s']))
        ts=read_post_event_times_metadata(path,trigger_s=float(e['t0_abs_s']))
        diag['file_read_seconds_excluded_from_online']=time.perf_counter()-start
        if format(w['observation_end_s'],'.7f')!=format(float(e['t_obs_end_rel_s']),'.7f'):
            raise ValueError('Cutoff differs at M0 serialized seven-decimal precision')
        res,state=predict(w,float(e['h_inertia']),G['case'],G['bank'],G['conv'],ts,G['cfg'])
        for r in rows:
            x=res[r['method']]
            r.update(prediction_status=x['status'],reason=x['reason'],t_obs_end_rel_s=w['observation_end_s'],
                delta_f_hat_hz=x['delta_f_hat_hz'],t_extremum_hat_rel_s=x['t_extremum_hat_rel_s'],
                measured_compute_s=x['compute_s'],timing_status='MEASURED')
            if r['prediction_status']!='OK':
                r['delta_f_hat_hz']=None;r['t_extremum_hat_rel_s']=None
        diag.update(state=state,model_details={k:v['native_details'] for k,v in res.items()},
                    raw_hash_verified=True,measurement_read_end_row=w['measurement_read_end_row'],
                    last_recorded_rel_s=float(ts[-1]),time_metadata_row_count=len(ts))
    except Exception as exc:
        for r in rows:r.update(prediction_status='INPUT_UNAVAILABLE',reason=type(exc).__name__+': '+str(exc))
        diag['failure']=type(exc).__name__+': '+str(exc)
    return rows,diag


def main():
    if OUT.exists():raise RuntimeError('Frozen output directory already exists; never overwrite a test run')
    out_parent=OUT.parent;out_parent.mkdir(parents=True,exist_ok=True);OUT.mkdir()
    index_path=SHARED/'event_index.csv'
    code=[ROOT/p for p in ['run_frozen_analytical_v2.py','analytical_predictor_v1.py','operational_inputs.py','causal_inputs.py','causal_inputs_v2.py','governor_models.py']]+[A/'reproduction_v1'/p for p in ['classical_sfr.py','fnp_liu2020.py']]
    inputs=[index_path,(REPO_ROOT / 'configs/analytical/analytical_operational_config_v1.json'),work_path('analytical/adaptation_v1/results_sav_v2/sav_bound_case_map.json'),
            work_path('analytical/adaptation_v1/results_fnp_conditioned_v1/bank.json'),work_path('analytical/adaptation_v1/results_fnp_conditioned_v1/offline_convolution_table.npz')]
    freeze={'interface_version':'1.0.0','run_id':RUN_ID,'freeze_time_utc':datetime.now(timezone.utc).isoformat(),
            'code_sha256':{str(p):sha(p) for p in code},'input_hashes':{str(p):sha(p) for p in inputs},
            'parameter_selection_split_ids':['source_paper_cases','verified_equipment_equations','active_val_75_prefixes'],
            'test_truth_used_for_model_selection':False,'test_prediction_errors_seen':False,
            'new_psse_runs':0,'G4':'GO_UNDER_DECLARED_P0_AND_LINEAR_APPROXIMATION',
            'correction_record':'execution/codex_analytical/adaptation_v1/TIME_SERIALIZATION_CORRECTION.md',
            'supersedes_run':'physical_causal_sfr_fnp_v1',
            'prior_evaluation_exposure':'v1 scorer loaded frozen labels and saved predictions, but all analytical rows were unavailable and no analytical errors or rankings existed; model/estimator/parameter choices remain unchanged',
            'workers':12,'python':sys.version,'numpy':np.__version__,'platform':platform.platform(),
            'timing_scope':'shared causal-array estimator/state checks plus method parameter mapping/range lookup/inference; file IO excluded; worker timings under 12-process workload'}
    # Written before consuming any test P0 feature values or measurement arrays.
    (OUT/'pre_execution_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    with index_path.open() as f:events=list(csv.DictReader(f))
    assert len(events)==15524 and len({e['event_id'] for e in events})==15524
    split_sources={'test':'test_ms10.csv','cross_cond_test':'cross_cond_test_ms10.csv','cross_cond_topo_test':'cross_cond_topo_test_ms10.csv'}
    source_hashes={};metadata={}
    for split,name in split_sources.items():
        p=DATA/'csv'/name
        # Only identities and the existing pre-event feature are selected.
        metadata[split]=pd.read_csv(p,usecols=['file_name','distu_kind','h_inertia'])
        source_hashes[str(p)]=sha(p)
    for e in events:
        e['row_index']=int(e['row_index']);r=metadata[e['active_split']].iloc[e['row_index']]
        if r['file_name']!=e['source_filename'] or r['distu_kind']!=e['disturbance_category']:
            raise ValueError('CSV identity differs from M0 index')
        e['h_inertia']=float(r['h_inertia'])
    (OUT/'metadata_binding.json').write_text(json.dumps({'selected_columns':['file_name','distu_kind','h_inertia'],
        'csv_hashes':source_hashes,'event_count':len(events),'identities_all_match':True},indent=2)+'\n')
    fields=list(base_rows(events[0])[0]);counts=Counter();started=time.perf_counter()
    with (OUT/'analytical_predictions.csv').open('w',newline='') as f,(OUT/'input_diagnostics.jsonl').open('w') as d:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        with ProcessPoolExecutor(max_workers=12,initializer=init_worker) as pool:
            for n,(rows,diag) in enumerate(pool.map(worker,events,chunksize=4),1):
                writer.writerows(rows);d.write(json.dumps(diag,separators=(',',':'))+'\n')
                for r in rows:counts[(r['scenario'],r['method'],r['prediction_status'],r['reason'])]+=1
                if n%250==0 or n==len(events):
                    f.flush();d.flush();print(json.dumps({'events_complete':n,'total':len(events),'elapsed_s':round(time.perf_counter()-started,2)}),flush=True)
    with (OUT/'analytical_coverage.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['scenario','method','prediction_status','reason','count'])
        for k,v in sorted(counts.items()):w.writerow([*k,v])
    prediction_hash=sha(OUT/'analytical_predictions.csv')
    manifest={**freeze,'manifest_id':events[0]['manifest_id'],'method':['CLASSICAL_SFR','FNP'],
        'config_sha256':sha((REPO_ROOT / 'configs/analytical/analytical_operational_config_v1.json')),'prediction_sha256':prediction_hash,
        'reproduction_record_id':'execution/codex_analytical/reproduction_v1/REPRODUCTION_REPORT.md',
        'reproduction_status':'SFR official reduced response PASS with supplementary mismatch retained; FNP original numerical cases PASS',
        'runtime_environment':{'python':sys.version,'workers':12,'blas_threads':1},
        'event_count':len(events),'prediction_rows':2*len(events),'complete_time_utc':datetime.now(timezone.utc).isoformat(),
        'wall_seconds':time.perf_counter()-started,'output_hashes':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
    (OUT/'prediction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'COMPLETE':True,'event_count':len(events),'prediction_rows':2*len(events),
        'prediction_sha256':prediction_hash,'wall_seconds':manifest['wall_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
