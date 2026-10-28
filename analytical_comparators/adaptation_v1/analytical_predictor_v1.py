"""Model-only predictor with explicit state, native targets and online timing.

No filenames, event category, target values or future trajectory values enter
this module. The caller separately applies the predeclared benchmark cohort.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
from time import perf_counter
import numpy as np
from operational_inputs import assemble
from governor_models import physical_sfr_reduction
from classical_sfr import Parameters,step_response
from fnp_liu2020 import predict_from_coefficients


def predict(window,h_inertia,case,bank,convolution,post_times,config):
    start=perf_counter();state=assemble(window,h_inertia,case,config)
    weights=np.asarray(config['frozen_coi_weights'],dtype=float)
    pre=float(np.asarray(window['frequency_deviation_hz'])[0]@weights/weights.sum())
    bad=[]
    for d in state['branch_diagnostics']:
        u=next(u for u in state['units'] if u['bus']==d['bus'])
        tol=config['initial_limit_proxy_tolerance_mw']/u['mbase_mva']
        if not d['lower_bound_machine_pu']-tol<=d['p0_proxy_machine_pu']<=d['upper_bound_machine_pu']+tol:
            bad.append(d['bus'])
    state['pre_frozen_coi_hz']=pre
    state['initial_linear_branch_failed_buses']=bad
    state['fit_range_selection_timing']='Included in online FNP timing; no unmeasured offline speed credit'
    input_s=perf_counter()-start
    result={}
    base={'qualification':'KNOWN_P0_UNCONSTRAINED_LINEAR_GOVERNOR_APPROXIMATION',
          'pre_event_nominal_offset_hz':pre,'observation_end_s':window['observation_end_s']}
    blocked=('pre-event electrical-power proxy lies outside selected governor linear branch at buses '+str(bad)) if bad else ''
    if len(state['tripped_buses'])>1:blocked='multiple observed unit outages outside verified single-outage scope'
    for method,target in [('CLASSICAL_SFR','signed_max_abs_excursion'),('FNP','first_downward_nadir')]:
        result[method]={**base,'target_kind':target,'status':'NOT_APPLICABLE' if blocked else 'PENDING',
                        'reason':blocked,'delta_f_hat_hz':None,'t_extremum_hat_rel_s':None,
                        'compute_s':input_s,'native_details':{}}
    if blocked:return result,state
    start=perf_counter()
    try:
        reduced=physical_sfr_reduction(state['units'],config['model_system_base_mva'])
        p=Parameters(**{k:reduced[k] for k in Parameters.__annotations__})
        curve=pre+step_response(post_times,state['deficit_hat_mw']/config['model_system_base_mva'],p)
        peak=int(np.argmax(np.abs(curve)))
        if not np.all(np.isfinite(curve)):raise ValueError('Nonfinite SFR response')
        result['CLASSICAL_SFR'].update(status='OK',delta_f_hat_hz=float(curve[peak]),
            t_extremum_hat_rel_s=float(post_times[peak]),native_details={'parameters':vars(p),'last_recorded_s':float(post_times[-1])})
    except Exception as exc:
        result['CLASSICAL_SFR'].update(status='NUMERICAL_FAILURE',reason=type(exc).__name__+': '+str(exc))
    result['CLASSICAL_SFR']['compute_s']+=perf_counter()-start
    start=perf_counter()
    try:
        if state['deficit_hat_mw']<=0:
            result['FNP'].update(status='NOT_APPLICABLE',reason='native FNP requires positive estimated net generation deficit')
        else:
            keep=np.asarray(state['unit_keep_mask'],dtype=bool)
            caps=np.array([u['mbase_mva'] for u in state['units']])
            hs=np.array([u['H_base_s'] for u in state['units']])
            if list(convolution['unit_buses'])!=bank['unit_buses']:
                raise ValueError('FNP offline table unit identity mismatch')
            direct_q=(caps/config['model_system_base_mva'])@convolution['unit_q'][keep]
            target=2*state['physical_H_sys_s'];ix=np.flatnonzero(direct_q>=target)
            direct=None
            if len(ix) and ix[0]>0:
                k=int(ix[0]);direct=float(np.interp(target,direct_q[k-1:k+1],convolution['time_s'][k-1:k+1]))
            horizon=next((h for h in bank['config']['fit_horizons_s'] if direct is not None and direct<=bank['config']['range_interior_fraction']*h),None)
            if horizon is None:
                result['FNP'].update(status='NO_VALID_ROOT',reason='no parameter-convolution root inside frozen admissible fit range')
            else:
                coeff=np.asarray(bank['coefficients_by_horizon'][str(horizon)])[keep]
                native=predict_from_coefficients(coeff,caps,hs,state['deficit_hat_mw'],config['model_system_base_mva'],horizon_s=horizon)
                native.update(fit_horizon_s=horizon,fit_degree=bank['config']['degree'],parameter_only_reference_root_s=direct)
                result['FNP'].update(status=native['status'],reason=native.get('reason',''),native_details=native)
                if native['status']=='OK':
                    if abs(native['t_nadir_s']-direct)>max(.005,.005*direct):
                        result['FNP'].update(status='NUMERICAL_FAILURE',reason='polynomial root fails frozen parameter-convolution approximation tolerance')
                    elif native['t_nadir_s']>post_times[-1]:
                        result['FNP'].update(status='NO_VALID_ROOT',reason='native nadir falls beyond the stored evaluation horizon')
                    else:
                        result['FNP'].update(delta_f_hat_hz=pre+native['df_hz'],t_extremum_hat_rel_s=native['t_nadir_s'])
    except Exception as exc:
        result['FNP'].update(status='NUMERICAL_FAILURE',reason=type(exc).__name__+': '+str(exc))
    result['FNP']['compute_s']+=perf_counter()-start
    return result,state
