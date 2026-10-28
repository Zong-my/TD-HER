#!/usr/bin/env python
"""S0 -- event identity index and frozen label definition (phase-1 security replay).

Release copy: the directories named below are those of the working tree in which this script ran. In this
copy the constants PROJECT, SHARED and MINE place the same files under data/ieee39_event_index/ and
results/ieee39/analytical_comparators/. The script needs the raw workbooks of the benchmark.

Owner: Claude (execution/claude_security).  Interface: coordination/interface_v1.json (1.0.0).

Reads
  * active CSVs           {ACTIVE}/csv/{split}_ms10.csv          (row order = authority)
  * frozen label arrays   {ACTIVE}/repA/ms10/y_{split}.npy
  * saved predictions     results/ieee39/exp_tdh_router/scenarios/*/  and
                          results/ieee39/resubmission_v3_mtl_capacity_matched/vanilla/seed_*/L*/preds.npy
  * raw workbooks         {RAW}/{raw_split}/{category}/{file_name}   (FREQ sheet + time axis only)

Writes
  * execution/shared/event_index.csv                 identity / window metadata only (no truth values)
  * execution/shared/frozen_label_definition.json    definition only
  * execution/shared/event_index_manifest.json       hashes of the two files above
  * execution/shared/eval_truth/coi_cache_v1/*.npz   EVALUATOR-ONLY cached COI trajectories (for S1)
  * execution/claude_security/results/s0_v1/         identity report, extras, MAE reproduction, manifests

Nothing here scores a security endpoint.  Labels are reconstructed from the workbooks with the frozen
extractor logic solely to produce a per-event PASS/FAIL identity flag; the reconstructed values are not
written to the shared index.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from fastexcel import read_excel

PROJECT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
PLAN = PROJECT
ACTIVE = str(data_path('ieee39'))
RAW = str(data_path('raw/ieee39'))
SHARED = str(work_path('analytical/shared'))
MINE = str(work_path('results/ieee39/analytical_comparators/claude_security'))
OUT = f'{MINE}/results/s0_v1'
CACHE = f'{SHARED}/eval_truth/coi_cache_v1'

INTERFACE_VERSION = '1.0.0'
MANIFEST_ID = 'ieee39_phase1_event_index_v1'
LABEL_DEF_ID = 'ieee39_v8_extract_features_coi_v1'
WINDOW_DEF_ID = 'ms10_T11_t0row_plus_tf0row_1to10'
SYSTEM = 'ieee39'
F_BASE = 60.0
TRIGGER = 1.0
MS = 10
CATEGORIES = ['circuit_short', 'cut_machine', 'load_change']
FREQ_BUSES = ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39']
# Frozen evaluation COI weights (data_proc/extract_features.py::IEEE39_CONFIG['freq_weights']).
FREQ_WEIGHTS = np.array([6.05, 3.41, 6.05, 0.0, 3.41, 5.016, 3.141, 3.141, 5.32, 500.0])

SCENARIOS = {
    'L1': dict(split='test', raw_dirs=['train', 'val', 'test'], router='L1_same_dist'),
    'L2': dict(split='cross_cond_test', raw_dirs=['cross_cond_test'], router='L2_cross_cond_fewshot'),
    'L3': dict(split='cross_cond_topo_test', raw_dirs=['cross_cond_topo_test'], router='L3_cross_topo_fewshot'),
}
ROUTER = str(work_path('results/ieee39/exp_tdh_router/scenarios'))
WIDE = str(work_path('results/ieee39/resubmission_v3_mtl_capacity_matched'))
PRED_FILES = {
    'TD-HER': 'admission_affine_convex_blend_nonnegative_y2_preds.npy',
    'BASE_REF_PER_TARGET': 'base_best_expert_preds.npy',
    'SHARED_ROUTE': 'admission_shared_affine_convex_blend_nonnegative_y2_preds.npy',
}
METRIC_KEYS = {
    'TD-HER': 'admission_affine_convex_blend_nonnegative_y2',
    'BASE_REF_PER_TARGET': 'base_best_expert',
    'SHARED_ROUTE': 'admission_shared_affine_convex_blend_nonnegative_y2',
}
WIDE_SEEDS = [42, 52, 62]

log = logging.getLogger('s0')


# ----------------------------------------------------------------------------- helpers
def sha256_file(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def event_key(system, active_split, category, file_name):
    """interface_v1: SHA256 of canonical UTF-8 JSON array, ensure_ascii=False, separators=(',',':')."""
    s = json.dumps([system, active_split, category, file_name], ensure_ascii=False, separators=(',', ':'))
    return sha256_bytes(s.encode('utf-8'))


def trigger_rows(tm, trigger_time=TRIGGER):
    """Verbatim re-implementation of the trigger-row logic in data_proc/extract_features.py."""
    abs_diff = np.abs(tm - trigger_time)
    closest = int(np.argmin(abs_diff))
    if tm[closest] >= trigger_time:
        closest -= 1
    t0_row = max(0, closest)
    trans_row = t0_row + 1
    while trans_row < len(tm) and tm[trans_row] <= trigger_time:
        trans_row += 1
    tf0_row = trans_row - 1
    return t0_row, trans_row, tf0_row


def process_workbook(task):
    """Worker: hash the workbook, read the FREQ sheet, rebuild COI trajectory and frozen labels."""
    idx, path, y1_csv, y2_csv = task
    out = dict(row_index=idx, status='OK', reason='')
    try:
        out['sha256'] = sha256_file(path)
        out['size_bytes'] = os.path.getsize(path)
        parser = read_excel(path)
        out['sheets'] = ';'.join(parser.sheet_names)
        df = parser.load_sheet_by_name('FREQ').to_polars()
        cols = df.columns
        unnamed = [c for c in cols if c.startswith('__UNNAMED__') or c.startswith('Unnamed')]
        if not unnamed:
            raise RuntimeError('no time column in FREQ sheet')
        tm = df[unnamed[0]].to_numpy().astype(np.float64)
        n = len(tm)
        if n == 0:
            raise RuntimeError('empty FREQ sheet')
        freq_system = np.zeros(n)
        w_sum = 0.0
        present = 0
        for i, b in enumerate(FREQ_BUSES):
            if b in cols:
                arr = df[b].to_numpy()
                if arr.dtype == object:
                    arr = np.array([float(x) if x is not None else 0.0 for x in arr])
                freq_system += arr.astype(np.float64) * FREQ_WEIGHTS[i]
                w_sum += FREQ_WEIGHTS[i]
                present += 1
        if w_sum > 1e-12:
            freq_system /= w_sum
        t0_row, trans_row, tf0_row = trigger_rows(tm)
        search_start = min(trans_row, n - 1)
        abs_freq = np.abs(freq_system[search_start:])
        max_row = search_start + int(np.argmax(abs_freq))
        y1_rec = float(freq_system[max_row]) * F_BASE
        y2_rec = float(tm[max_row]) - TRIGGER
        last_win_row = tf0_row + MS
        if last_win_row >= n:
            raise RuntimeError(f'window row {last_win_row} beyond record length {n}')
        dt = np.diff(tm)
        dup_rows = np.where(dt == 0)[0]            # row k such that tm[k] == tm[k+1]
        dup_rel = [float(tm[k] - TRIGGER) for k in dup_rows]
        dup_in_window = [round(v, 7) for v in dup_rel if 0.0 <= v <= (tm[last_win_row] - TRIGGER) + 1e-9]
        out.update(
            n_rows=n,
            n_freq_buses_present=present,
            time_first_abs=float(tm[0]),
            time_last_abs=float(tm[-1]),
            t0_row=int(t0_row),
            trans_row=int(trans_row),
            tf0_row=int(tf0_row),
            t0_row_time_abs=float(tm[t0_row]),
            tf0_row_time_abs=float(tm[tf0_row]),
            first_post_time_abs=float(tm[trans_row]),
            window_last_row=int(last_win_row),
            t_obs_end_abs=float(tm[last_win_row]),
            t_obs_end_rel=float(tm[last_win_row] - TRIGGER),
            n_dup_timestamps=int(len(dup_rows)),
            dup_timestamps_rel=';'.join(f'{v:.7f}' for v in dup_rel),
            dup_timestamps_in_window=';'.join(f'{v:.7f}' for v in dup_in_window),
            max_dt=float(dt.max()) if n > 1 else float('nan'),
            n_dt_gt_0p0101=int(np.sum(dt > 0.0101)),
            y1_diff=abs(y1_rec - y1_csv),
            y2_diff=abs(y2_rec - y2_csv),
            time=tm,
            coi_pu=freq_system,
        )
    except Exception as e:  # noqa: BLE001
        out['status'] = 'ERROR'
        out['reason'] = f'workbook_error: {type(e).__name__}: {str(e)[:160]}'
    return out


def mae(y, p):
    return float(np.mean(np.abs(y.astype(np.float64) - p.astype(np.float64))))


# ----------------------------------------------------------------------------- main
def run(scenarios, workers, limit):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CACHE, exist_ok=True)
    report = {}
    index_rows = []
    extras_rows = []
    mae_rows = []
    pred_manifest = []
    label_hashes = {}

    for sc in scenarios:
        cfg = SCENARIOS[sc]
        split = cfg['split']
        t_sc = time.time()
        log.info('=== %s (%s) ===', sc, split)

        # ---- active CSV (authority for row order)
        csv_path = f'{ACTIVE}/csv/{split}_ms{MS}.csv'
        df = pd.read_csv(csv_path, usecols=['distu_kind', 'file_name', 'fpu_deltamax', 't_delta'])
        n = len(df)
        csv_sha = sha256_file(csv_path)
        label_hashes[f'csv/{split}_ms{MS}.csv'] = csv_sha
        log.info('CSV rows=%d sha256=%s', n, csv_sha[:16])

        # ---- frozen y arrays vs CSV targets
        y_path = f'{ACTIVE}/repA/ms{MS}/y_{split}.npy'
        y = np.load(y_path)
        y_sha = sha256_file(y_path)
        label_hashes[f'repA/ms{MS}/y_{split}.npy'] = y_sha
        csv_targets = df[['fpu_deltamax', 't_delta']].to_numpy()
        y_equal = (y.shape == (n, 2)) and np.array_equal(y, csv_targets.astype(np.float32))
        y_maxdiff = float(np.max(np.abs(y.astype(np.float64) - csv_targets))) if y.shape == (n, 2) else float('nan')
        log.info('y array %s dtype=%s equal_to_csv_float32=%s max|y-csv|=%.3e', y.shape, y.dtype, y_equal, y_maxdiff)

        # ---- saved predictions: shapes, hashes, MAE reproduction
        metrics_path = f'{ROUTER}/{cfg["router"]}/metrics_summary.json'
        with open(metrics_path) as fh:
            stored = json.load(fh)
        for method, fname in PRED_FILES.items():
            p_path = f'{ROUTER}/{cfg["router"]}/{fname}'
            p = np.load(p_path)
            ok_shape = p.shape == (n, 2)
            m1, m2 = (mae(y[:, 0], p[:, 0]), mae(y[:, 1], p[:, 1])) if ok_shape else (float('nan'),) * 2
            key = METRIC_KEYS[method]
            s1 = stored.get(key, {}).get('metrics', {}).get('y1_MAE', float('nan'))
            s2 = stored.get(key, {}).get('metrics', {}).get('y2_MAE', float('nan'))
            rel1 = abs(m1 - s1) / max(abs(s1), 1e-12)
            rel2 = abs(m2 - s2) / max(abs(s2), 1e-12)
            mae_rows.append(dict(scenario=sc, method=method, source=os.path.relpath(p_path, PROJECT), dtype=str(p.dtype),
                                 shape=str(p.shape), y1_MAE_recomputed=m1, y1_MAE_stored=s1, y1_rel_diff=rel1,
                                 y2_MAE_recomputed=m2, y2_MAE_stored=s2, y2_rel_diff=rel2,
                                 match=bool(ok_shape and rel1 < 1e-9 and rel2 < 1e-9)))
            pred_manifest.append(dict(scenario=sc, method=method, path=os.path.relpath(p_path, PROJECT),
                                      sha256=sha256_file(p_path), shape=list(p.shape), dtype=str(p.dtype)))
            log.info('%-20s y1 %.9f (stored %.9f) y2 %.9f (stored %.9f)', method, m1, s1, m2, s2)
        runs = pd.read_csv(f'{WIDE}/all_runs.csv')
        for seed in WIDE_SEEDS:
            p_path = f'{WIDE}/vanilla/seed_{seed}/{sc}/preds.npy'
            p = np.load(p_path)
            ok_shape = p.shape == (n, 2)
            m1, m2 = (mae(y[:, 0], p[:, 0]), mae(y[:, 1], p[:, 1])) if ok_shape else (float('nan'),) * 2
            r = runs[(runs.scenario == sc) & (runs.seed == seed) & (runs.strategy == 'vanilla')]
            s1 = float(r.y1_MAE.iloc[0]) if len(r) else float('nan')
            s2 = float(r.y2_MAE.iloc[0]) if len(r) else float('nan')
            rel1 = abs(m1 - s1) / max(abs(s1), 1e-12)
            rel2 = abs(m2 - s2) / max(abs(s2), 1e-12)
            mae_rows.append(dict(scenario=sc, method=f'WIDE_MTL_seed{seed}', source=os.path.relpath(p_path, PROJECT),
                                 dtype=str(p.dtype), shape=str(p.shape), y1_MAE_recomputed=m1, y1_MAE_stored=s1,
                                 y1_rel_diff=rel1, y2_MAE_recomputed=m2, y2_MAE_stored=s2, y2_rel_diff=rel2,
                                 match=bool(ok_shape and rel1 < 1e-5 and rel2 < 1e-5)))
            pred_manifest.append(dict(scenario=sc, method=f'WIDE_MTL_seed{seed}', path=os.path.relpath(p_path, PROJECT),
                                      sha256=sha256_file(p_path), shape=list(p.shape), dtype=str(p.dtype)))
            log.info('WIDE seed %d          y1 %.9f (stored %.9f) y2 %.9f (stored %.9f)', seed, m1, s1, m2, s2)

        # ---- resolve raw workbook paths
        lookup = {}
        for rd in cfg['raw_dirs']:
            for cat in CATEGORIES:
                d = f'{RAW}/{rd}/{cat}'
                if not os.path.isdir(d):
                    continue
                for f in os.listdir(d):
                    if f.endswith('.xlsx'):
                        lookup.setdefault((cat, f), []).append(f'{d}/{f}')
        # names that also exist in raw folders outside this scenario's pool (provenance note only)
        others = {}
        for rd in [d for d in os.listdir(RAW) if os.path.isdir(f'{RAW}/{d}') and d not in cfg['raw_dirs']]:
            for cat in CATEGORIES:
                d = f'{RAW}/{rd}/{cat}'
                if os.path.isdir(d):
                    for f in os.listdir(d):
                        if f.endswith('.xlsx'):
                            others.setdefault((cat, f), []).append(rd)
        tasks, resolved, status, reason = [], [], [], []
        for i, (cat, fn) in enumerate(zip(df.distu_kind, df.file_name)):
            hits = lookup.get((cat, fn), [])
            if len(hits) == 1:
                resolved.append(hits[0]); status.append('OK'); reason.append('')
            elif len(hits) == 0:
                resolved.append(''); status.append('UNRESOLVED'); reason.append('raw_workbook_not_found')
            else:
                resolved.append(hits[0]); status.append('UNRESOLVED'); reason.append('raw_workbook_ambiguous:' + '|'.join(hits))
        n_unres = sum(s != 'OK' for s in status)
        log.info('raw paths resolved: %d / %d (unresolved %d)', n - n_unres, n, n_unres)

        ids = [event_key(SYSTEM, split, c, f) for c, f in zip(df.distu_kind, df.file_name)]
        assert len(set(ids)) == n, 'event_id collision inside scenario'

        # ---- workbook pass
        sel = range(n) if limit is None else range(min(n, limit))
        tasks = [(i, resolved[i], float(csv_targets[i, 0]), float(csv_targets[i, 1])) for i in sel if status[i] == 'OK']
        log.info('workbook pass: %d tasks, %d workers', len(tasks), workers)
        results = {}
        t_wb = time.time()
        with Pool(processes=workers) as pool:
            for k, r in enumerate(pool.imap_unordered(process_workbook, tasks, chunksize=4)):
                results[r['row_index']] = r
                if (k + 1) % 1000 == 0 or (k + 1) == len(tasks):
                    log.info('  %d / %d workbooks (%.1f s)', k + 1, len(tasks), time.time() - t_wb)

        # ---- assemble
        max_rows = max((r['n_rows'] for r in results.values() if r['status'] == 'OK'), default=0)
        n_sel = len(sel)
        time_arr = np.full((n_sel, max_rows), np.nan)
        coi_arr = np.full((n_sel, max_rows), np.nan)
        n_rows_arr = np.zeros(n_sel, dtype=np.int32)
        t0_rows = np.full(n_sel, -1, dtype=np.int32)
        trans_rows = np.full(n_sel, -1, dtype=np.int32)
        tf0_rows = np.full(n_sel, -1, dtype=np.int32)
        win_last = np.full(n_sel, -1, dtype=np.int32)
        counters = dict(verified=0, unresolved=0, label_mismatch=0, workbook_error=0, y2_negative_in_csv=0)
        dup_hist = {}
        tobs_hist = {}
        for i in sel:
            cat, fn = df.distu_kind.iloc[i], df.file_name.iloc[i]
            r = results.get(i)
            row = dict(manifest_id=MANIFEST_ID, event_id=ids[i], system=SYSTEM, scenario=sc, active_split=split,
                       disturbance_category=cat, source_filename=fn, row_index=i, raw_path=(os.path.relpath(resolved[i], PROJECT) if resolved[i] else ''), raw_sha256='',
                       t0_abs_s=TRIGGER, t_obs_end_rel_s='', window_definition_id=WINDOW_DEF_ID,
                       label_definition_id=LABEL_DEF_ID, identity_status='UNRESOLVED', reason=reason[i])
            ex = dict(manifest_id=MANIFEST_ID, event_id=ids[i], scenario=sc, row_index=i,
                      y_array_equals_csv=bool(y_equal), same_name_in_other_raw_dirs=';'.join(others.get((cat, fn), [])))
            if csv_targets[i, 1] < 0:
                counters['y2_negative_in_csv'] += 1
            if r is None:
                counters['unresolved'] += 1
            elif r['status'] != 'OK':
                row['reason'] = r['reason']; counters['workbook_error'] += 1
            else:
                row['raw_sha256'] = r['sha256']
                row['t_obs_end_rel_s'] = f"{r['t_obs_end_rel']:.7f}"
                label_ok = (r['y1_diff'] <= 1e-9) and (r['y2_diff'] <= 1e-9)
                if label_ok and y_equal:
                    row['identity_status'] = 'VERIFIED'; counters['verified'] += 1
                else:
                    row['identity_status'] = 'UNRESOLVED'
                    row['reason'] = f"label_reconstruction_mismatch: y1_diff={r['y1_diff']:.3e} y2_diff={r['y2_diff']:.3e} y_equal={y_equal}"
                    counters['label_mismatch'] += 1
                ex.update({k: r[k] for k in ['size_bytes', 'sheets', 'n_rows', 'n_freq_buses_present', 'time_first_abs', 'time_last_abs',
                                              't0_row', 'trans_row', 'tf0_row', 't0_row_time_abs', 'tf0_row_time_abs',
                                              'first_post_time_abs', 'window_last_row', 't_obs_end_abs', 't_obs_end_rel',
                                              'n_dup_timestamps', 'dup_timestamps_rel', 'dup_timestamps_in_window', 'max_dt',
                                              'n_dt_gt_0p0101', 'y1_diff', 'y2_diff']})
                dup_hist[r['n_dup_timestamps']] = dup_hist.get(r['n_dup_timestamps'], 0) + 1
                tk = round(r['t_obs_end_rel'], 3)
                tobs_hist[tk] = tobs_hist.get(tk, 0) + 1
                m = r['n_rows']
                time_arr[i, :m] = r['time']; coi_arr[i, :m] = r['coi_pu']; n_rows_arr[i] = m
                t0_rows[i] = r['t0_row']; trans_rows[i] = r['trans_row']; tf0_rows[i] = r['tf0_row']; win_last[i] = r['window_last_row']
            index_rows.append(row); extras_rows.append(ex)

        # ---- evaluator-only COI cache
        cache_path = f'{CACHE}/{sc}_coi_cache.npz'
        np.savez(cache_path, event_id=np.array(ids[:n_sel]), row_index=np.arange(n_sel), time_abs_s=time_arr, coi_pu=coi_arr,
                 n_rows=n_rows_arr, t0_row=t0_rows, trans_row=trans_rows, tf0_row=tf0_rows, window_last_row=win_last,
                 f_base_hz=F_BASE, trigger_abs_s=TRIGGER, label_definition_id=LABEL_DEF_ID, manifest_id=MANIFEST_ID)
        report[sc] = dict(split=split, n_csv_rows=n, n_processed=n_sel, csv_sha256=csv_sha, y_sha256=y_sha, y_shape=list(y.shape),
                          y_dtype=str(y.dtype), y_equals_csv_float32=bool(y_equal), y_max_abs_diff_vs_csv=y_maxdiff,
                          counters=counters, dup_timestamp_histogram={str(k): v for k, v in sorted(dup_hist.items())},
                          t_obs_end_rel_histogram={f'{k:.3f}': v for k, v in sorted(tobs_hist.items())},
                          coi_cache=os.path.relpath(cache_path, PLAN), coi_cache_sha256=sha256_file(cache_path),
                          elapsed_s=round(time.time() - t_sc, 1))
        log.info('%s done in %.1f s: %s', sc, time.time() - t_sc, counters)

    # ---- shared deliverables
    cols = ['manifest_id', 'event_id', 'system', 'scenario', 'active_split', 'disturbance_category', 'source_filename', 'row_index',
            'raw_path', 'raw_sha256', 't0_abs_s', 't_obs_end_rel_s', 'window_definition_id', 'label_definition_id', 'identity_status', 'reason']
    idx_path = f'{SHARED}/event_index.csv'
    pd.DataFrame(index_rows)[cols].to_csv(idx_path, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(extras_rows).to_csv(f'{OUT}/event_index_extras.csv', index=False, encoding='utf-8')
    pd.DataFrame(mae_rows).to_csv(f'{OUT}/mae_reproduction.csv', index=False)
    with open(f'{OUT}/saved_prediction_manifest.json', 'w') as fh:
        json.dump(pred_manifest, fh, indent=2)

    extractor = str(REPO_ROOT / 'data_proc/extract_features.py')
    label_def = dict(
        label_definition_id=LABEL_DEF_ID,
        interface_version=INTERFACE_VERSION,
        manifest_id=MANIFEST_ID,
        source_code='data_proc/extract_features.py (IEEE39_CONFIG + extract_single_file); representations via data_proc/build_representations.py',
        source_code_hash=dict(extract_features_py=sha256_file(extractor),
                              build_representations_py=sha256_file(str(REPO_ROOT / 'data_proc/build_representations.py')),
                              s0_event_index_py=sha256_file(os.path.abspath(__file__))),
        source_label_hashes=label_hashes,
        f_base_hz=F_BASE,
        FREQ_conversion='Workbook sheet FREQ holds per-unit bus frequency deviation. Hz deviation = FREQ * 60; absolute frequency = 60 + 60*FREQ. '
                        'y1 is stored as Hz deviation (fpu_deltamax = coi_pu[max_row] * 60).',
        generator_bus_order=[int(b) for b in FREQ_BUSES],
        frozen_evaluation_coi_weights={b: float(w) for b, w in zip(FREQ_BUSES, FREQ_WEIGHTS)},
        coi_rule='coi_pu[t] = sum_i w_i * FREQ_i[t] / sum_i w_i over the generator buses present in the sheet (bus 33 weight 0). '
                 'These are the frozen historical label weights; they are not the physical H_sys used inside analytical models.',
        y1_definition='fpu_deltamax: signed COI frequency deviation (Hz) at the post-trigger row with the largest |coi_pu|, '
                      'searched from the first row whose time exceeds trigger_time (trans_row) to the end of the record.',
        y2_definition='t_delta: time_abs[max_row] - trigger_time (s); the arrival time of the extremum, NOT a threshold-crossing time.',
        trigger_rule='trigger_time = 1.0 s (configs/experiment_config.yaml). closest = argmin|t-1.0|; if t[closest] >= 1.0 step back one row; '
                     't0_row = that row (pre-disturbance state, recorded as 0.9999993 s). trans_row = first row with t > 1.0; tf0_row = trans_row-1 '
                     '(the duplicate switch-time row). Relative time = absolute time - 1.0 s.',
        finite_horizon_rule='Each record ends at its last stored row (about 30.0006 s absolute, about 29.0006 s after trigger). '
                            'No crossing inside the record means no crossing within the stored horizon only.',
        duplicate_timestamp_rule='PSS/E writes a duplicate timestamp at every switching instant (trigger, fault clearing, ...). '
                                 'Rows are kept in stored order; duplicates are never averaged or dropped. Sample index is never converted '
                                 'to time by assumption; the stored time column governs.',
        window_definition=dict(id=WINDOW_DEF_ID, ms=MS, T=MS + 1,
                               rows='[t0_row] + [tf0_row + k for k in 1..10]',
                               note='Timestep 0 is the pre-disturbance row at 0.9999993 s; timesteps 1..10 are the ten stored rows after the '
                                    'switch-time duplicate. When a second switching duplicate (e.g. fault clearing at 0.08 s) falls inside the '
                                    'window, the tenth row is at 0.09 s, so t_obs_end_rel_s is per event and is read from the stored time column.'),
        endpoint_definitions_frozen_for_s1=dict(
            SYM_ABS_0P5_HZ='unsafe = 1[|y1| >= 0.5 Hz]; alarm = 1[|y1_hat| >= 0.5 Hz]',
            UF_59P5_HZ='unsafe = 1[min_{t > trigger} (60 + 60*coi_pu[t]) <= 59.5 Hz]  (from the full stored trajectory); '
                       'alarm = 1[60 + y1_hat <= 59.5 Hz] (signed-extremum proxy; its blind spot is counted, not removed)',
            first_crossing='earliest stored post-trigger row (t > trigger) meeting the endpoint condition; bracket = [previous distinct-time row, that row]',
            lead_time='t_cross_rel - d, d = t_obs_end_rel_s + declared compute time; strict early requires d < t_cross',
            near_threshold_stratum_halfwidth_hz=0.05,
            ambiguous_band='per-method P95 |y1_hat - y1| taken from validation/calibration residuals, fixed before test scoring; diagnostic only'),
        created='2026-09-06',
        owner='Claude',
    )
    ld_path = f'{SHARED}/frozen_label_definition.json'
    with open(ld_path, 'w') as fh:
        json.dump(label_def, fh, indent=2, ensure_ascii=False)

    manifest = dict(interface_version=INTERFACE_VERSION, manifest_id=MANIFEST_ID, owner='Claude', created='2026-09-06',
                    files={'event_index.csv': sha256_file(idx_path), 'frozen_label_definition.json': sha256_file(ld_path)},
                    scenarios=report, note='event_index.csv contains identity and window metadata only; COI cache under eval_truth is evaluator-only.')
    with open(f'{SHARED}/event_index_manifest.json', 'w') as fh:
        json.dump(manifest, fh, indent=2)
    with open(f'{OUT}/s0_report.json', 'w') as fh:
        json.dump(dict(report=report, mae_reproduction=mae_rows), fh, indent=2)
    with open(f'{CACHE}/README.md', 'w') as fh:
        fh.write('# EVALUATOR-ONLY COI trajectory cache (coi_cache_v1)\n\n'
                 'Written by execution/claude_security/code/s0_event_index.py. Contains the full stored COI trajectory of every '
                 'test event (time_abs_s, coi_pu, padded with NaN) plus the frozen trigger/window row indices. This is test-event '
                 'future truth: it must not be read by any model, parameter fit, root selection, threshold choice or event estimator. '
                 'Consumers: the S1 truth extractor and the security scorer only.\n')
    log.info('S0 complete.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', nargs='+', default=['L1', 'L2', 'L3'])
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--limit', type=int, default=None, help='debug: process only the first N rows per scenario')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(f'{OUT}/run.log')])
    run(a.scenarios, a.workers, a.limit)
