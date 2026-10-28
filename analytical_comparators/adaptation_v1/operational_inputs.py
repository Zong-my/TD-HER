"""Causal, common input adapter for the classical SFR and native FNP.

Inputs contain measurements and declared P0 equipment state only. No filename,
event category, injected load change, target or future trajectory is accepted.
This is a linear-model replay under known pre-event inertia and equipment
assumptions, not a demonstration of field parameter estimation.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
from copy import deepcopy
import numpy as np


def assemble(window, h_inertia, case, config):
    units = deepcopy(case['units'])
    buses = [u['bus'] for u in units]
    if list(window['bus_order']) != buses:
        raise ValueError('Measurement and parameter identities differ')
    if not np.isfinite(h_inertia) or h_inertia <= 0:
        raise ValueError('Missing or nonpositive pre-event inertia scale')
    t = np.asarray(window['time_relative_s'], dtype=float)
    powers = np.asarray(window['power_channel_native_pu'], dtype=float)
    if powers.shape != (11, len(units)) or len(t) != 11:
        raise ValueError('The frozen eleven-row observation is required')
    if np.any(np.diff(t) < 0) or t[-1] > .100000001:
        raise ValueError('Observation violates causal cutoff')
    if not np.all(np.isfinite(powers)):
        raise ValueError('Missing observed electrical power')
    pe_mw = config['channel_system_base_mva'] * powers
    pre = pe_mw[0]
    last = pe_mw[-config['trip_persistence_rows']:]
    trip = ((np.abs(pre) >= config['trip_min_pre_power_mw']) &
            np.all(np.abs(last) <= config['trip_zero_power_mw'], axis=0))
    # Read the first three positive-time samples. A separate switching sample
    # at the clearing time cannot be included in this early estimate.
    early_idx = np.flatnonzero(t > 0)[:config['step_estimator_rows']]
    if len(early_idx) != config['step_estimator_rows']:
        raise ValueError('Incomplete early power observation')
    jumps = pe_mw[early_idx].sum(axis=1) - pre.sum()
    if np.any(trip):
        deficit = float(pre[trip].sum())
        route = 'observed_power_disappearance_times_pre_event_power'
    else:
        deficit = float(np.median(jumps))
        route = 'median_first_three_aggregate_electrical_power_increments'
    kept = []
    branch_records = []
    for i, u in enumerate(units):
        u['H_base_s'] *= float(h_inertia)
        u['PG_observed_pre_mw'] = float(pre[i])
        if trip[i]:
            continue
        kept.append(u)
        gov, v = u['governor'], u['governor_values']
        p0 = pre[i] / u['mbase_mva']
        if gov == 'NONE':
            continue
        if gov == 'GAST':
            high, low = v[6], v[7]
        elif gov == 'TGOV1':
            high, low = v[2], v[3]
        elif gov == 'IEESGO':
            high, low = v[9], v[10]
        elif gov == 'IEEEG1':
            high, low = v[8], v[9]
        else:
            raise ValueError('Unknown governor')
        # Electrical power approximates the initial mechanical output; machine
        # losses/internal state are not observed. This is a diagnostic, not an
        # exact nonlinear initial-condition or reserve certification.
        branch_records.append({
            'bus': u['bus'], 'governor': gov, 'p0_proxy_machine_pu': float(p0),
            'lower_bound_machine_pu': float(low), 'upper_bound_machine_pu': float(high),
            'initial_proxy_inside_limits': bool(low <= p0 <= high),
            'upward_headroom_proxy_mw': float((high-p0)*u['mbase_mva']),
            'gast_temperature_branch': 'UNOBSERVED' if gov == 'GAST' else 'NOT_APPLICABLE',
            'rate_limits': 'OMITTED_IN_LINEAR_MODEL' if gov == 'IEEEG1' else 'MODEL_SPECIFIC'
        })
    return {
        'units': kept, 'unit_keep_mask': (~trip).tolist(),
        'tripped_buses': [int(buses[i]) for i in np.flatnonzero(trip)],
        'deficit_hat_mw': deficit, 'disturbance_estimator_route': route,
        'early_power_jumps_mw': jumps.tolist(),
        'power_jumps_spread_mw': float(np.ptp(jumps)),
        'pre_power_mw': pre.tolist(), 'h_inertia': float(h_inertia),
        'branch_diagnostics': branch_records,
        'physical_H_sys_s': float(sum(u['mbase_mva']*u['H_base_s'] for u in kept)/config['model_system_base_mva']),
        'qualification': 'KNOWN_P0_LINEAR_GOVERNOR_REPLAY',
        'limitations': 'Governor saturation/rate/temperature response and voltage-dependent net-load change are not reconstructed.'
    }
