#!/usr/bin/env python
"""S1 -- R01 evaluation rule record, R02 security truth, R03 FNP target compatibility.

Release copy: the directories named below are those of the working tree in which this script ran. In this
copy the constants PROJECT, SHARED and MINE place the same files under data/ieee39_event_index/ and
results/ieee39/analytical_comparators/. The script needs the trajectory cache written by S0.

Order of operations (deliberate): the dated rule record is written and hashed BEFORE the evaluator-only
test cache is opened.  The fixed definitions are recorded without bundling historical diagnostics or timing measurements.

Outputs
  execution/shared/evaluation_rule_record.json                       (R01)
  execution/shared/eval_truth/truth_v1/security_truth.csv            (R02, interface schema)
  execution/shared/eval_truth/truth_v1/security_truth_extras.csv     (R02, evaluator extras)
  execution/shared/eval_truth/truth_v1/target_compatibility.csv      (R03, interface schema)
  execution/shared/eval_truth/truth_v1/target_compatibility_extras.csv
  execution/shared/eval_truth/truth_v1/truth_manifest.json
  execution/claude_security/results/s1_v1/{s1_report.json, run.log}
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PLAN = PROJECT
SHARED = str(work_path('analytical/shared'))
MINE = str(work_path('results/ieee39/analytical_comparators/claude_security'))
CACHE = f'{SHARED}/eval_truth/coi_cache_v1'
CALCACHE = f'{MINE}/results/calibration_cache_v1'
TRUTH_VERSION = 'truth_v1'
TOUT = f'{SHARED}/eval_truth/truth_v1'
OUT = f'{MINE}/results/s1_v1'
ACTIVE = str(data_path('ieee39'))
MANIFEST_ID = 'ieee39_phase1_event_index_v1'
LABEL_DEF_ID = 'ieee39_v8_extract_features_coi_v1'
F_BASE = 60.0
TRIGGER = 1.0
SPLIT = {'L1': 'test', 'L2': 'cross_cond_test', 'L3': 'cross_cond_topo_test'}

# ----------------------------------------------------------------------------- frozen rule parameters
SYM_THRESHOLD_HZ = 0.5          # E_sym: |dev| >= 0.5 Hz  (>= convention)
UF_ABS_HZ = 59.5                # E_U:   60 + dev <= 59.5 Hz (<= convention)  <=> dev <= -0.5
SENS_SYM = [0.3, 0.7]           # secondary sensitivities (reported in full, never primary)
SENS_UF = [59.3, 59.7]
SAMPLE_TOL_S = 0.0100001        # "within the stored resolution": same or adjacent stored sample (10 ms grid)
NADIR_FLOOR_HZ = 1e-4           # numerical-resolution floor (calibration micro-extrema population < 1e-5 Hz)
NADIR_RHO = 0.2                 # primary: prominence >= max(floor, rho*|y1|)
COMPAT_RULE_ID = 'first_nadir_relprom_rho0.2_floor1e-4_v1'
SENSITIVITY_RULES = {'abs_tau_0.005': dict(kind='abs', tau=0.005), 'rho_0.1': dict(kind='rel', rho=0.1), 'rho_0.3': dict(kind='rel', rho=0.3)}

log = logging.getLogger('s1')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_rule_record():
    """Record fixed evaluation definitions before opening test trajectories.

    This source-only release contains no historical timing or calibration
    diagnostics. Timing measurements, if needed, must be supplied by the user.
    """
    rec = dict(
        record_id='evaluation_rule_record_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        endpoints=dict(
            SYM_ABS_0P5_HZ=dict(threshold_hz=SYM_THRESHOLD_HZ, convention='>='),
            UF_59P5_HZ=dict(threshold_hz_abs=UF_ABS_HZ, convention='<='),
            secondary_sensitivities=dict(symmetric_hz=SENS_SYM, underfrequency_abs_hz=SENS_UF)),
        time_rules=dict(
            trigger_s=TRIGGER,
            first_crossing='Earliest stored post-trigger row satisfying the condition; keep stored order at duplicate timestamps.',
            bracket='Last earlier row with a strictly smaller timestamp to the crossing row; no sub-sample interpolation.',
            availability='Observation-window end plus independently measured compute time; no latency result is bundled.',
            finite_horizon='No crossing inside the stored record is not proof of safety.'),
        fnp_compatibility=dict(
            rule_id=COMPAT_RULE_ID,
            prominence_floor_hz=NADIR_FLOOR_HZ,
            relative_prominence=NADIR_RHO,
            definition='First interior downward minimum with prominence >= max(floor, rho*abs(y1)); compatible if y1 < 0 and that minimum matches the global extremum within one stored sample.',
            sensitivity_rules=SENSITIVITY_RULES,
            not_an_online_detector=True),
        forbidden=['no threshold chosen by test outcome', 'no retuning of frozen models',
                   'no predicted extremum time treated as a threshold-crossing time'])
    os.makedirs(SHARED, exist_ok=True)
    record_path = f'{SHARED}/evaluation_rule_record.json'
    with open(record_path, 'w') as fh:
        json.dump(rec, fh, indent=2, ensure_ascii=False)
    return record_path, sha256_file(record_path)


def first_nadir(dp, y1, kind, **kw):
    """Return (row_in_post, prominence) of the first accepted interior minimum or (None, None)."""
    idx, props = find_peaks(-dp, prominence=(NADIR_FLOOR_HZ, None), plateau_size=(1, None))
    le, pr = props['left_edges'], props['prominences']
    thr = max(NADIR_FLOOR_HZ, kw['tau']) if kind == 'abs' else max(NADIR_FLOOR_HZ, kw['rho'] * abs(y1))
    sel = np.where(pr >= thr)[0]
    if len(sel) == 0:
        return None, None
    j = sel[0]
    return int(le[j]), float(pr[j])


def compat_status(dp, tp, y1, g, kind, **kw):
    if y1 > 0:
        return 'GLOBAL_UPWARD', None, None
    if y1 == 0:
        return 'UNRESOLVED', None, None
    if g == len(dp) - 1:
        return 'BOUNDARY', None, None
    r, prom = first_nadir(dp, y1, kind, **kw)
    if r is None:
        return 'NO_INTERIOR_NADIR', None, None
    if abs(tp[r] - tp[g]) <= SAMPLE_TOL_S:
        return 'MATCHED_FIRST_DOWNWARD_NADIR', r, prom
    return 'LATER_DEEPER_NADIR', r, prom


def crossing(cond, t, s, tf0):
    """cond: boolean over all rows; s: first post-trigger row; returns dict of crossing facts."""
    pre = bool(cond[:tf0 + 1].any())
    post = np.where(cond[s:])[0]
    if len(post) == 0:
        return dict(pre=pre, k=None)
    k = s + int(post[0])
    j = k - 1
    while j >= 0 and t[j] >= t[k]:
        j -= 1
    lower = t[j] if j >= 0 else t[0]
    dup = (k + 1 < len(t) and t[k + 1] == t[k]) or (k - 1 >= 0 and t[k - 1] == t[k])
    amb = bool(dup or (t[k] - lower) > 0.0101)
    return dict(pre=pre, k=k, t_cross=t[k] - TRIGGER, lower=lower - TRIGGER, upper=t[k] - TRIGGER, ambiguous=amb)


def run():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TOUT, exist_ok=True)
    os.makedirs(TOUT, exist_ok=True)
    rule_path, rule_sha = write_rule_record()
    log.info('rule record written %s sha256=%s', rule_path, rule_sha[:16])
    idx = pd.read_csv(f'{SHARED}/event_index.csv', keep_default_na=False)
    truth_rows, extra_rows, comp_rows, comp_extra = [], [], [], []
    report = {}
    for sc, split in SPLIT.items():
        z = np.load(f'{CACHE}/{sc}_coi_cache.npz')
        ids = z['event_id'].astype(str); T = z['time_abs_s']; C = z['coi_pu']; nr = z['n_rows']
        tr = z['trans_row']; tf0 = z['tf0_row']
        y = np.load(f'{ACTIVE}/repA/ms10/y_{split}.npy').astype(float)
        sub = idx[idx.scenario == sc].sort_values('row_index')
        assert np.array_equal(sub.event_id.to_numpy(), ids), 'cache/index identity mismatch'
        n = len(ids)
        cnt = dict(n=n, sym_unsafe=0, uf_unsafe=0, sym_pre=0, uf_pre=0, sym_amb=0, uf_amb=0, sym_label_mismatch=0, uf_blind=0,
                   compat={}, compat_sens={k: {} for k in SENSITIVITY_RULES})
        for i in range(n):
            m = int(nr[i]); t = T[i, :m]; d = F_BASE * C[i, :m]; s = int(tr[i]); f0 = int(tf0[i])
            dp = d[s:]; tp = t[s:]
            g = int(np.argmax(np.abs(dp)))
            y1, y2 = float(y[i, 0]), float(y[i, 1])
            last_rel = float(t[-1] - TRIGGER)
            gmin = int(np.argmin(dp)); gmax = int(np.argmax(dp))
            base = dict(manifest_id=MANIFEST_ID, event_id=ids[i], truth_version=TRUTH_VERSION, label_definition_id=LABEL_DEF_ID)
            for ep, cond, thr_desc in [('SYM_ABS_0P5_HZ', np.abs(d) >= SYM_THRESHOLD_HZ, '|dev|>=0.5'),
                                       ('UF_59P5_HZ', d <= (UF_ABS_HZ - F_BASE), 'dev<=-0.5')]:
                c = crossing(cond, t, s, f0)
                unsafe = c['k'] is not None
                row = dict(base, endpoint_id=ep, y1_true_hz=y1, y2_true_rel_s=y2, unsafe=unsafe, preexisting_at_t0=c['pre'],
                           t_cross_observed_rel_s=f"{c['t_cross']:.7f}" if unsafe else '',
                           cross_bracket_lower_rel_s=f"{c['lower']:.7f}" if unsafe else '',
                           cross_bracket_upper_rel_s=f"{c['upper']:.7f}" if unsafe else '',
                           last_recorded_rel_s=f'{last_rel:.7f}',
                           truth_status='VALID' if unsafe else 'NO_CROSSING_WITHIN_HORIZON',
                           reason=('resolution_ambiguous' if (unsafe and c['ambiguous']) else ''))
                truth_rows.append(row)
                if ep == 'SYM_ABS_0P5_HZ':
                    cnt['sym_unsafe'] += unsafe; cnt['sym_pre'] += c['pre']; cnt['sym_amb'] += bool(unsafe and c['ambiguous'])
                    if unsafe != (abs(y1) >= SYM_THRESHOLD_HZ):
                        cnt['sym_label_mismatch'] += 1
                else:
                    cnt['uf_unsafe'] += unsafe; cnt['uf_pre'] += c['pre']; cnt['uf_amb'] += bool(unsafe and c['ambiguous'])
                    if unsafe and y1 > 0:
                        cnt['uf_blind'] += 1
            # secondary sensitivities (first crossing times only)
            sens = {}
            for thr in SENS_SYM:
                c = crossing(np.abs(d) >= thr, t, s, f0); sens[f'sym_{thr}_t_cross'] = f"{c['t_cross']:.7f}" if c['k'] is not None else ''
            for fa in SENS_UF:
                c = crossing(d <= (fa - F_BASE), t, s, f0); sens[f'uf_{fa}_t_cross'] = f"{c['t_cross']:.7f}" if c['k'] is not None else ''
            extra_rows.append(dict(base, scenario=sc, row_index=i, n_post_rows=len(dp), min_dev_hz=float(dp[gmin]), t_min_rel_s=float(tp[gmin] - TRIGGER),
                                   max_dev_hz=float(dp[gmax]), t_max_rel_s=float(tp[gmax] - TRIGGER), global_extremum_dev_hz=float(dp[g]),
                                   global_extremum_rel_s=float(tp[g] - TRIGGER), last_recorded_rel_s=last_rel, **sens))
            # compatibility
            st, r, prom = compat_status(dp, tp, y1, g, 'rel', rho=NADIR_RHO)
            cnt['compat'][st] = cnt['compat'].get(st, 0) + 1
            comp_rows.append(dict(base, compatibility_rule_id=COMPAT_RULE_ID, fnp_target_compatible=(st == 'MATCHED_FIRST_DOWNWARD_NADIR'), status=st,
                                  reason='' if st == 'MATCHED_FIRST_DOWNWARD_NADIR' else st.lower()))
            ce = dict(base, scenario=sc, row_index=i, primary_status=st, first_nadir_rel_s=(float(tp[r] - TRIGGER) if r is not None else ''),
                      first_nadir_dev_hz=(float(dp[r]) if r is not None else ''), first_nadir_prominence_hz=(prom if prom is not None else ''),
                      global_extremum_rel_s=float(tp[g] - TRIGGER))
            for k, rule in SENSITIVITY_RULES.items():
                st2, _, _ = compat_status(dp, tp, y1, g, rule['kind'], **{kk: v for kk, v in rule.items() if kk != 'kind'})
                ce[f'status_{k}'] = st2
                cnt['compat_sens'][k][st2] = cnt['compat_sens'][k].get(st2, 0) + 1
            comp_extra.append(ce)
        report[sc] = cnt
        log.info('%s: %s', sc, json.dumps(cnt))

    cols = ['manifest_id', 'event_id', 'truth_version', 'label_definition_id', 'endpoint_id', 'y1_true_hz', 'y2_true_rel_s', 'unsafe', 'preexisting_at_t0',
            't_cross_observed_rel_s', 'cross_bracket_lower_rel_s', 'cross_bracket_upper_rel_s', 'last_recorded_rel_s', 'truth_status', 'reason']
    pd.DataFrame(truth_rows)[cols].to_csv(f'{TOUT}/security_truth.csv', index=False)
    pd.DataFrame(extra_rows).to_csv(f'{TOUT}/security_truth_extras.csv', index=False)
    ccols = ['manifest_id', 'event_id', 'truth_version', 'compatibility_rule_id', 'fnp_target_compatible', 'status', 'reason']
    pd.DataFrame(comp_rows)[ccols].to_csv(f'{TOUT}/target_compatibility.csv', index=False)
    pd.DataFrame(comp_extra).to_csv(f'{TOUT}/target_compatibility_extras.csv', index=False)
    fld = json.load(open(f'{SHARED}/frozen_label_definition.json'))
    manifest = dict(
        truth_version=TRUTH_VERSION, manifest_id=MANIFEST_ID, label_definition_id=LABEL_DEF_ID,
        source_raw_hash_manifest=dict(event_index_csv=sha256_file(f'{SHARED}/event_index.csv'),
                                      coi_cache={sc: sha256_file(f'{CACHE}/{sc}_coi_cache.npz') for sc in SPLIT}),
        frozen_label_hashes=fld['source_label_hashes'],
        extraction_code_hash=sha256_file(os.path.abspath(__file__)),
        extraction_config_hash=rule_sha,
        truth_csv_sha256=sha256_file(f'{TOUT}/security_truth.csv'),
        truth_extras_sha256=sha256_file(f'{TOUT}/security_truth_extras.csv'),
        compatibility_rule_id=COMPAT_RULE_ID,
        compatibility_csv_sha256=sha256_file(f'{TOUT}/target_compatibility.csv'),
        compatibility_extras_sha256=sha256_file(f'{TOUT}/target_compatibility_extras.csv'),
        missing_counts=dict(missing_truth=0, workbook_errors=0),
        finite_horizon_and_time_policy='see evaluation_rule_record.json time_rules; stored record ends ~29.0006 s after trigger; duplicates kept in stored order',
        created_utc=datetime.now(timezone.utc).isoformat(),
        visibility='EVALUATOR_ONLY',
        counts=report,
    )
    with open(f'{TOUT}/truth_manifest.json', 'w') as fh:
        json.dump(manifest, fh, indent=2)
    with open(f'{OUT}/s1_report.json', 'w') as fh:
        json.dump(dict(rule_record_sha256=rule_sha, counts=report), fh, indent=2)
    log.info('S1 complete')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TOUT, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{OUT}/run.log')])
    run()
