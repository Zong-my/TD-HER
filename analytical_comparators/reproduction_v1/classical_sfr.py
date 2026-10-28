"""Classical Anderson SFR, independently derived two-state and closed-form step.

Positive deficit means loss of net generation; output is signed frequency
DEVIATION in Hz. All powers use one declared system MVA base. Initial state is
zero deviation. This module performs no file IO or PSS/E calls.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Parameters:
    H: float
    R: float
    Km: float
    Fh: float
    Tr: float
    D: float
    f0: float = 60.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in vars(self).values()):
            raise ValueError("Parameters must be finite")
        if self.H <= 0 or self.R <= 0 or self.f0 <= 0:
            raise ValueError("H, R and f0 must be positive")
        if min(self.Km, self.Tr, self.D) < 0 or not 0 <= self.Fh <= 1:
            raise ValueError("Unsupported nonphysical parameters")


def step_response(t, deficit_pu, p):
    """Exact zero-state step response, valid before/after event-relative t=0."""
    t = np.asarray(t, dtype=float)
    if not np.all(np.isfinite(t)) or not np.isfinite(deficit_pu):
        raise ValueError("Nonfinite time or deficit")
    tau = np.maximum(t, 0.)
    M = 2*p.H
    stiffness = p.D+p.Km/p.R
    d0 = -deficit_pu/M
    if stiffness == 0:
        x = d0*tau
    elif p.Tr == 0 or p.Fh == 1 or p.Km == 0:
        x = -deficit_pu/stiffness*(-np.expm1(-stiffness*tau/M))
    else:
        roots = np.roots([M*p.Tr, M+p.Tr*(p.D+p.Km*p.Fh/p.R), stiffness]).astype(complex)
        r1,r2 = roots
        xss = -deficit_pu/stiffness
        if abs(r1-r2) < 1e-10:
            r = (r1+r2)/2
            x = xss+(-xss+(d0+xss*r)*tau)*np.exp(r*tau)
        else:
            c1 = (d0+xss*r2)/(r1-r2)
            c2 = -xss-c1
            x = xss+c1*np.exp(r1*tau)+c2*np.exp(r2*tau)
        x = np.real_if_close(x, tol=1000).real
    return np.where(t > 0, p.f0*x, 0.)


def ode_response(t, deficit_pu, p):
    """Independent physical ODE for G2 verification, not the main predictor."""
    t = np.asarray(t, dtype=float)
    if np.any(t < 0) or np.any(np.diff(t) <= 0) or p.Tr <= 0:
        raise ValueError("ODE verification requires increasing nonnegative t and Tr>0")
    def rhs(_, state):
        x,z = state
        governor = -x/p.R
        pm = p.Km*(p.Fh*governor+(1-p.Fh)*z)
        return [(pm-deficit_pu-p.D*x)/(2*p.H), (governor-z)/p.Tr]
    sol = solve_ivp(rhs, (0., float(t[-1])), [0.,0.], t_eval=t,
                    method="DOP853", rtol=1e-12, atol=1e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    return p.f0*sol.y[0]
