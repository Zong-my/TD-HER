"""Equipment-state-conditioned FNP fit range within the paper's 5-20 s band.

The direct convolution determines the fit interval from physical parameters
only. All eleven possible unit states can be prepared before an event. The
reported prediction still uses the frozen polynomial equation, not this
reference root. Offline preparation must be disclosed and timed separately.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
import numpy as np
from numpy.polynomial import Polynomial
from scipy import signal,optimize
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'reproduction_v1'))
from governor_models import from_dyr
from fnp_liu2020 import predict_from_coefficients


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def choose_horizon(direct_root,config):
    return next((h for h in config['fit_horizons_s'] if direct_root<=config['range_interior_fraction']*h),None)


def main():
    cfgp=(REPO_ROOT / 'configs/analytical/fnp_conditioned_prelock.json');cfg=json.loads(cfgp.read_text())
    cp=work_path('analytical/adaptation_v1/results_sav_v2/sav_bound_case_map.json');case=json.loads(cp.read_text())
    out=work_path('analytical/adaptation_v1/results_fnp_conditioned_v1')
    if out.exists():raise RuntimeError('Existing evidence retained')
    out.mkdir();(out/'pre_execution.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),
        'code_sha256':sha(Path(__file__)),'config_sha256':sha(cfgp),'case_sha256':sha(cp)},indent=2)+'\n')
    start=time.perf_counter();units=case['units'];dt=cfg['component_dt_s'];degree=cfg['degree']
    t=np.linspace(0,20,round(20/dt)+1);ramp=[];q=[]
    for u in units:
        if u['governor']=='NONE':ramp.append(np.zeros_like(t));q.append(np.zeros_like(t));continue
        a,b,c,d=signal.tf2ss(*from_dyr(u['governor'],u['governor_values']));n=len(a)
        aa=np.zeros((n+2,n+2));bb=np.zeros((n+2,1));aa[:n,:n]=a;bb[:n]=b;aa[n,:n]=c[0];bb[n]=d[0];aa[n+1,n]=1
        cc=np.zeros((2,n+2));cc[0,n]=1;cc[1,n+1]=1
        _,y,_=signal.lsim((aa,bb,cc,np.zeros((2,1))),np.ones_like(t),t)
        qq=np.zeros_like(t);qq[1:]=y[1:,0]-y[1:,1]/t[1:];ramp.append(y[:,0]);q.append(qq)
    ramp=np.asarray(ramp);q=np.asarray(q);banks={}
    for h in cfg['fit_horizons_s']:
        coef=np.zeros((len(units),degree+1));mask=t<=h+1e-12
        for i,r in enumerate(ramp):
            co=Polynomial.fit(t[mask],r[mask],degree).convert().coef;coef[i,:len(co)]=co
        banks[str(h)]=coef.tolist()
    cap=np.array([u['mbase_mva'] for u in units]);hi=np.array([u['H_base_s'] for u in units]);checks=[]
    for drop in [None]+[u['bus'] for u in units]:
        keep=np.array([u['bus']!=drop for u in units]);response=(cap[keep]/1000.)@q[keep]
        for scale in cfg['inertia_scales']:
            H=float((cap[keep]*hi[keep]*scale).sum()/1000.)
            ix=np.flatnonzero(response>=2*H)
            direct=float(optimize.brentq(lambda v:float(np.interp(v,t,response)-2*H),t[ix[0]-1],t[ix[0]])) if len(ix) else None
            horizon=choose_horizon(direct,cfg) if direct is not None else None
            pred=predict_from_coefficients(np.array(banks[str(horizon)])[keep],cap[keep],hi[keep]*scale,1.,1000.,horizon_s=horizon) if horizon else {}
            got=pred.get('t_nadir_s');err=abs(got-direct) if got is not None else None
            tol=max(.005,.005*direct) if direct is not None else None
            checks.append({'drop_bus':drop,'h_inertia':scale,'direct_root_s':direct,'fit_horizon_s':horizon,
                'polynomial_root_s':got,'abs_error_s':err,'tolerance_s':tol,'pass':err is not None and err<=tol})
    np.savez_compressed(out/'offline_convolution_table.npz',time_s=t,unit_q=q,unit_buses=np.array([u['bus'] for u in units]))
    bank={'config':cfg,'case_sha256':sha(cp),'unit_buses':[u['bus'] for u in units],'coefficients_by_horizon':banks,
          'qualification':'FNP_EQUIPMENT_CONDITIONED_FIT_RANGE','offline_seconds':time.perf_counter()-start}
    (out/'bank.json').write_text(json.dumps(bank,indent=2)+'\n');(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    from collections import Counter
    report={'checks':len(checks),'passed':sum(x['pass'] for x in checks),'all_pass':all(x['pass'] for x in checks),
            'max_abs_error_s':max(x['abs_error_s'] for x in checks if x['abs_error_s'] is not None),
            'horizon_counts':dict(Counter(x['fit_horizon_s'] for x in checks)),
            'failures':[x for x in checks if not x['pass']],'bank_sha256':sha(out/'bank.json'),
            'config_sha256':sha(cfgp),'offline_seconds':bank['offline_seconds'],
            'test_or_target_access':False,'new_psse_runs':0,
            'meaning':'Numerical approximation check only; no Fepfm prediction accuracy or physical-event equivalence claim.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
