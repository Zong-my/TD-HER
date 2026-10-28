"""Unconstrained linear governor increment models from audited PSS/E families.

Positive input = frequency drop pu; positive output = mechanical power gain pu
on MACHINE base. Limits, rate limits, GAST temperature selection, and operating
point eligibility must be checked separately before operational use.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import numpy as np


def mul(a,b):
    return np.polymul(a[0],b[0]),np.polymul(a[1],b[1])


def add(a,b):
    return np.polyadd(np.polymul(a[0],b[1]),np.polymul(b[0],a[1])),np.polymul(a[1],b[1])


def lag(t):
    if t<0:raise ValueError('Negative time constant')
    return ([1.],[t,1.]) if t else ([1.],[1.])


def turbine_chain(taus,fractions):
    # Common denominator avoids artificial pole cancellations from summation.
    den=np.array([1.]);num=np.array([0.])
    for t in taus: den=np.polymul(den,lag(t)[1])
    for j,k in enumerate(fractions):
        term=np.array([float(k)])
        for t in taus[j+1:]:term=np.polymul(term,lag(t)[1])
        num=np.polyadd(num,term)
    return num,den


def from_dyr(model,values):
    v=list(map(float,values))
    if model=='GAST':
        R,T1,T2,T3,AT,KT,VMAX,VMIN,D=v
        tf=mul(([1/R],[1.]),mul(lag(T1),lag(T2)))
        return add(tf,([D],[1.])) if D else tf
    if model=='TGOV1':
        R,T1,VMAX,VMIN,T2,T3,D=v
        tf=mul(([1/R],[1.]),mul(lag(T1),([T2,1.],[T3,1.])))
        return add(tf,([D],[1.])) if D else tf
    if model=='IEESGO':
        T1,T2,T3,T4,T5,T6,K1,K2,K3,PMAX,PMIN=v
        regulator=mul(([K1],[1.]),mul(lag(T1),([T2,1.],lag(T3)[1])))
        turbine=turbine_chain([T4,T5,T6],[1-K2,K2*(1-K3),K2*K3])
        return mul(regulator,turbine)
    if model=='IEEEG1':
        if len(v)!=22 or v[:2]!=[0.,0.]:raise ValueError('Only the inspected single-machine IEEEG1 routing is implemented')
        K,T1,T2,T3,UO,UC,PMAX,PMIN,T4,K1,K2,T5,K3,K4,T6,K5,K6,T7,K7,K8=v[2:]
        if any(x!=0 for x in [K2,K4,K6,K8]):raise ValueError('Unmapped second-shaft fractions')
        regulator=mul(([K],[1.]),mul(([T2,1.],lag(T1)[1]),lag(T3)))
        return mul(regulator,turbine_chain([T4,T5,T6,T7],[K1,K3,K5,K7]))
    raise ValueError('Unsupported governor '+model)


def physical_sfr_reduction(units,system_base_mva=1000.):
    """Equipment-weighted reheat SFR approximation, with no label fitting.

    Regulator / valve / steam-chest lags are omitted in the classical reheat
    reduction; full lags remain in FNP. Slow turbine branch mean delays set Tr.
    This is a disclosed mixed-governor reduction, not an original-paper claim
    of exact equivalence for gas and multiple steam sections.
    """
    inertia=damping=gain=fast=slow_moment=0.
    for u in units:
        a=u['mbase_mva']/system_base_mva
        inertia+=a*u['H_base_s'];damping+=a*u['D_machine_pu']
        model=u['governor'];v=u['governor_values']
        if model=='NONE':continue
        if model=='GAST':
            k=1/v[0];fh=1.;moment=0.;damping+=a*v[8]
        elif model=='TGOV1':
            k=1/v[0];fh=v[4]/v[5];moment=(1-fh)*v[5];damping+=a*v[6]
        elif model=='IEESGO':
            T1,T2,T3,T4,T5,T6,k,K2,K3,_,_=v
            fh=1-K2;moment=K2*(1-K3)*T5+K2*K3*(T5+T6)
        elif model=='IEEEG1':
            k,T1,T2,T3,UO,UC,PMAX,PMIN,T4,K1,K2,T5,K3,K4,T6,K5,K6,T7,K7,K8=v[2:]
            fh=K1;moment=K3*T5+K5*(T5+T6)+K7*(T5+T6+T7)
        else:raise ValueError('Unknown physical reduction')
        gain+=a*k;fast+=a*k*fh;slow_moment+=a*k*moment
    if gain<=0:raise ValueError('No responsive governor gain')
    return {'H':inertia,'D':damping,'R':1/gain,'Km':1.,'Fh':fast/gain,
            'Tr':slow_moment/(gain-fast) if gain>fast else 0.,'f0':60.,
            'system_base_mva':system_base_mva,'status':'STATIC_TEMPLATE_ONLY_NOT_EVENT_BOUND'}
