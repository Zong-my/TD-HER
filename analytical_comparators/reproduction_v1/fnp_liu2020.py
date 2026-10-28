"""Liu 2020 FNP Eq.(1)-(18), offline unit construction and online scalar root.

Independent implementation: positive power deficit -> negative frequency
change. H is physical inertia on the stated common MVA base. Coefficients
come from independent governor models, never test-event future trajectories.
The native output is a FIRST DOWNWARD NADIR, not a signed global extremum.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import numpy as np
from numpy.polynomial import Polynomial
from scipy import integrate, signal


def characterize_governor(transfer_function, horizon_s=5., dt_s=.001, degree=2):
    """Unit frequency-drop step, integrated and least-squares polynomial fitted."""
    if horizon_s<=0 or dt_s<=0 or degree<1 or degree>=round(horizon_s/dt_s):
        raise ValueError('Invalid offline characterization grid or degree')
    times=np.linspace(0.,horizon_s,round(horizon_s/dt_s)+1)
    _,step,_=signal.lsim(signal.tf2ss(*transfer_function),np.ones_like(times),times)
    ramp=integrate.cumulative_trapezoid(step,times,initial=0.)
    if not np.all(np.isfinite(ramp)):
        raise ValueError('Nonfinite component response')
    return Polynomial.fit(times,ramp,degree).convert().coef


def predict_from_coefficients(unit_coefficients, unit_mva, inertia_s,
                              deficit_mw, system_base_mva, horizon_s=5., f0_hz=60.):
    """No event IO or fitting. Unavailable physics must be screened by caller.

    Include every remaining inertia-contributing unit; a unit with no primary
    frequency response has a zero coefficient row. A converter with no modeled
    inertial response has H=0; it is not a synchronous generator.
    """
    coef=np.asarray(unit_coefficients,dtype=float)
    cap=np.asarray(unit_mva,dtype=float)
    hs=np.asarray(inertia_s,dtype=float)
    if coef.ndim!=2 or len(coef)!=len(cap) or cap.shape!=hs.shape:
        raise ValueError('Unit arrays must have identical identities/order')
    if not all(np.all(np.isfinite(x)) for x in [coef,cap,hs,[deficit_mw,system_base_mva,horizon_s,f0_hz]]):
        raise ValueError('Nonfinite inputs')
    if min(system_base_mva,horizon_s,f0_hz)<=0 or np.any(cap<=0) or np.any(hs<0):
        raise ValueError('Invalid physical base or inertia')
    if deficit_mw<=0:
        return {'status':'NOT_APPLICABLE','reason':'native FNP requires a positive net power deficit','prediction_type':'first_downward_nadir','t_nadir_s':None,'df_hz':None}
    alpha=cap/system_base_mva
    h=float(alpha@hs)
    if h<=0:
        return {'status':'NOT_APPLICABLE','reason':'no positive synchronous/equivalent inertia','prediction_type':'first_downward_nadir','t_nadir_s':None,'df_hz':None}
    agg=alpha@coef
    order=np.arange(coef.shape[1])
    eq=agg*order/(order+1.)
    eq[0]=-2*h
    roots=Polynomial(eq).roots()
    eligible=sorted(float(z.real) for z in roots if abs(z.imag)<1e-8 and 0<z.real<=horizon_s)
    result={'status':'OK' if eligible else 'NO_VALID_ROOT','prediction_type':'first_downward_nadir',
            'H_sys_s':h,'system_base_mva':system_base_mva,'aggregate_coefficients':agg.tolist(),
            'eligible_roots_s':eligible,'t_nadir_s':eligible[0] if eligible else None,
            'df_hz':-f0_hz*(deficit_mw/system_base_mva)*eligible[0]/(4*h) if eligible else None,
            'root_rule':'earliest real positive root <= frozen fit horizon'}
    return result
