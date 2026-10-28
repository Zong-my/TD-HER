# -*- coding: utf-8 -*-
"""TD-HER prospective families F1/F2 -- Windows runner (Python 2.7, 32-bit, PSS/E 33.4).

Executes a fixed, seeded scenario plan with the UNCHANGED event functions of
generate_freq_datav7_3.py (circuit_short, load_change, cut_machine) on the unchanged SAV/DYR.
The generator source is loaded with execfile; its __main__ batch loop never runs. Nothing in
the generator, the case or the dynamic data is edited.

Runner version 7. Version 2 (after Codex's line review of version 1): the outaged branch of an F2
short circuit is excluded as an undirected line; a resume compares every field of the plan and
keeps the first meta file; the ledger says "generated", never "ok" -- acceptance is decided
elsewhere. Version 3 (after the plan made on Windows with version 1): the case has 28 loads, some
buses carrying more than one, and the file name carries the bus only, so load changes are sampled
over distinct load buses (the first listed load of a bus); the plan is checked for repeated file
names and for outages on the faulted line before anything is written. Version 4 (after Codex's
review of version 3): the load-change value is a PERCENTAGE of the selected bus load (generator:
load = (1 + ld/100) x load), so -300 would turn the load into an injection of twice its size; the
new families use -100, -50, -10, 10, 100, 300 per cent, i.e. the set of the existing shifted sets
with -300 replaced by -50. The generator changes the ACTIVE part only: -100 sets the active power
of the load to zero and leaves its reactive part as it is (active-load reduction, not a
disconnection). dist_clear_fault is a core step of a short circuit and a clean-up step only of a
generator trip, and is recorded accordingly. Version 5 (after the check of the version-4 plan on
serverA, before any event of the families was simulated): in the version-4 plan 310 of the 1,104
F2 events had their outage on one of the 15 lines whose removal splits the network (unit step-up
connections, load-coupler branches, lines 16-19 and 19-20), which takes a unit or a load out
before the power flow instead of changing the paths. The outage of F2 is now drawn among the lines
whose removal leaves every bus connected; the plan is checked for it. The F1 part of the plan is
the same as in version 4. The convergence flag of every power flow (psspy.solved(), a query) and
the live element lists are recorded. The load-change values are those of version 4: the version-4
pilot showed that the generator's own rule (more than two "Network not converged" messages in the
last 20 log lines -> no export) gives no output for large active-load reductions, as it did when
the benchmark was generated; that rule decides which planned events are exported, the plan is not
adapted to it. The pilot is a check of the chain and is chosen accordingly (see --pilot).
(Correction of this paragraph as it stood in version 5: six cases of plan v4 had been simulated in
the version-4 pilot when version 5 was written; "before any event was simulated" was wrong.)
Version 6 (after Codex's two-process probe of version 5): the generator takes the F2 line out
with psspy.dist_branch_trip before the power flow. That API belongs to the dynamic simulation; in
a process that has not yet run a dynamic case it returns 5 ("No dynamics data in memory") and the
line stays in service, so the first F2 event of a fresh process was simulated on the original
topology. In the ten-case pilot the F1 cases had run first, which hid this. The runner now
supplies the planned outage itself, at the point where the generator issues that call and before
the power flow: status 0 through the power-flow data API (branch_chng, or two_winding_chng_4 for
a two-winding transformer), read back, and recorded. The generator's dynamic call is not used for
this step; its later use of the same API during the dynamic run (fault clearing) is untouched.
If the outage cannot be confirmed, the case stops with an exception and no file is produced.
The plan is the one of version 5, unchanged (its hash is checked), and so are the inputs.
--case-ids runs named cases of the plan (diagnostic use).
Version 7 (after Codex's check of version 6 against the installed PSS/E 33.4 API manual, before
version 6 ran any case): version 6 asked xfrint for 'STATUS', which is not a string of that API
(it would return 4, invalid string, and a transformer would have been taken for a line). A
two-winding transformer is now recognized with xfrint(..., 'CW') (0: transformer; 3: the branch is
not a transformer; anything else stops the case), and the status of either kind of branch is read
with brnint(..., 'STATUS'). Nothing else differs from version 6.

Plan F1F2-v5-20261005 (protocol: claude_pssE_F1F2_protocol_v5_20261005.md)
  F1  new operating conditions, original topology                    (random_cut_bus = False)
  F2  the same conditions with one line out of service beforehand, the network staying connected
                                                                     (random_cut_bus = True)
  48 conditions = load level {87.5, 97.5, 107.5} x ZIP {45-25-30, 25-45-30, 60-20-20, 30-30-40}
                  x reserve {1, 3} x inertia scale {0.6, 1.4}
  per condition: 10 short circuits (sampled branches), 12 load changes (2 distinct sampled load
                 buses x 6 values in per cent of the active bus load), 1 trip of the bus-33 unit
                 ->  23 events; 1,104 per family.
  Calibration/test assignment is part of the plan (5/5, 6/6, trip alternating by condition).

Usage (from the directory of this file):
  C:\\Python27\\python.exe tdher_f1f2_runner_v7.py --plan-only --out <new directory>
  C:\\Python27\\python.exe tdher_f1f2_runner_v7.py --case-ids F2-1126 --out <new directory>
  C:\\Python27\\python.exe tdher_f1f2_runner_v7.py --out <directory> [--family F1] [--kind circuit_short]
Every process needs its own --out directory (the export rule reads that directory's progress log).
A rerun with the same --out resumes: finished cases are skipped. No file is deleted.
"""
from __future__ import print_function
import os, sys, time, json, hashlib, traceback, random, csv, argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(ROOT, 'generate_freq_datav7_3.py')
SAV = os.path.normpath(os.path.join(ROOT, '..', '..', 'data', 'psse_cases', 'benchmark_ieee39', 'IEEE39.sav'))
DYR = os.path.normpath(os.path.join(ROOT, '..', '..', 'data', 'psse_cases', 'benchmark_ieee39', 'IEEE39bus_rew_10_v33.dyr'))
# inputs of the pilot of 2026-10-05, which reproduced an existing benchmark file exactly
EXPECTED_SHA256 = {
    SOURCE: '7421e3a73d6a2341e0dae30c28d4c5d4b0a8b75648bee073c90ba3195f2b861a',
    SAV: 'd982ed3c38b6fdd1bb47915e7a1f73f5f2688e6bd15dea30bf83e6e7eb910364',
    DYR: '36bdea815b9cd7f574f8473e6e0e4ed0b14c191a54d6477339fc257edcd37428',
}
PSSBIN = r'C:\Program Files (x86)\PTI\PSSE33\PSSBIN'
PSSLIB = r'C:\Program Files (x86)\PTI\PSSE33\PSSLIB'

RUNNER_VERSION = 7
PLAN_VERSION = 'F1F2-v5-20261005'      # the plan is unchanged in runner version 6
EXPECTED_PLAN_SHA256 = '51682d108ec82ff59788c991fbd733ef2339232b39f06e91029fb48f62f979de'
MASTER_SEED = 20261005
LOAD_LEVELS = [87.5, 97.5, 107.5]
ZIPS = [[45.0, 25.0, 30.0], [25.0, 45.0, 30.0], [60.0, 20.0, 20.0], [30.0, 30.0, 40.0]]
RESERVES = [1.0, 3.0]
INERTIAS = [0.6, 1.4]
LOAD_DELTAS = [-100.0, -50.0, -10.0, 10.0, 100.0, 300.0]  # per cent of the active power of the selected load
PILOT_LD = [-50.0, -10.0, 10.0]   # load changes of the pilot: first sampled load bus of condition 0
N_BRANCH = 10
N_LOADBUS = 2
TRIP_BUS = 33
SAV_NAME = 'IEEE39'
CHANNEL_OPTION = 'All'
TST, TET, RT, REF_MACHINE = 1.0, 1.17, 30, 30
FAMILY_OFFSET = {'F1': 1, 'F2': 2}
KINDS = ['circuit_short', 'load_change', 'cut_machine']
PLAN_FIELDS = ['case_id', 'family', 'split', 'cond_index', 'le', 'zip_z', 'zip_i', 'zip_p', 'rr', 'hi',
               'kind', 'frmbus', 'tobus', 'cktid', 'lodbus', 'lodid', 'ld', 'gbus', 'gid',
               'pre_frm', 'pre_to', 'pre_ckt', 'out_name']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def line_key(branch):
    """Undirected line: the same circuit listed from either end is one line."""
    f, t, ck = branch[0], branch[1], branch[2]
    return (min(int(f), int(t)), max(int(f), int(t)), str(ck).strip())


def splitting_lines(branch3):
    """Lines whose removal leaves some bus of the branch list unconnected to the rest (undirected keys)."""
    keys = sorted(set(line_key(b) for b in branch3))
    buses = sorted(set(k[0] for k in keys) | set(k[1] for k in keys))
    out = []
    for drop in keys:
        adj = {}
        for k in keys:
            if k == drop:
                continue
            adj.setdefault(k[0], set()).add(k[1])
            adj.setdefault(k[1], set()).add(k[0])
        seen, stack = set([buses[0]]), [buses[0]]
        while stack:
            u = stack.pop()
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != len(buses):
            out.append(drop)
    return out


def ierr_of(result):
    """Error code of a psspy call: the value itself, or the first item of a returned tuple."""
    if isinstance(result, (int, long)):
        return int(result)
    if isinstance(result, tuple) and result and isinstance(result[0], (int, long)):
        return int(result[0])
    return None


def conditions():
    out = []
    for le in LOAD_LEVELS:
        for lz in ZIPS:
            for rr in RESERVES:
                for hi in INERTIAS:
                    out.append({'le': le, 'lz': list(lz), 'rr': rr, 'hi': hi})
    return out


def stem(sp):
    return '%s_%s%s_%s%s-%s-%s_%s%s_%s%s' % (SAV_NAME, 'le', sp['le'], 'zip', sp['lz'][0], sp['lz'][1],
                                             sp['lz'][2], 'rr', sp['rr'], 'hi', sp['hi'])


def build_plan(branches, loads, generators):
    """Deterministic plan from the live element lists of the unchanged SAV (order as returned by PSS/E)."""
    branch3 = [tuple(b[:3]) for b in branches]
    trip = [g for g in generators if int(g[0]) == TRIP_BUS]
    if len(trip) != 1:
        raise RuntimeError('expected exactly one machine at bus %d, found %d' % (TRIP_BUS, len(trip)))
    gbus, gid = trip[0][0], trip[0][1]
    # one entry per load bus: the first listed load of the bus (file names carry the bus only)
    load_buses, seen = [], set()
    for lb in loads:
        if lb[0] not in seen:
            seen.add(lb[0])
            load_buses.append(lb)
    # F2: the line taken out beforehand must leave every bus connected
    split_keys = set(splitting_lines(branch3))
    outage_pool = [b for b in branch3 if line_key(b) not in split_keys]
    if not outage_pool:
        raise RuntimeError('no line can be taken out without splitting the network')
    plan = []
    for family in ['F1', 'F2']:
        for c, sp in enumerate(conditions()):
            rng = random.Random(MASTER_SEED * 1000 + FAMILY_OFFSET[family] * 100 + c)
            base = {'family': family, 'cond_index': c, 'le': sp['le'], 'zip_z': sp['lz'][0], 'zip_i': sp['lz'][1],
                    'zip_p': sp['lz'][2], 'rr': sp['rr'], 'hi': sp['hi']}

            def pre_trip(exclude=None):
                if family != 'F2':
                    return ('', '', '')
                choice = rng.choice(outage_pool)
                while exclude is not None and line_key(choice) == line_key(exclude):
                    choice = rng.choice(outage_pool)
                return choice

            shorts = rng.sample(branch3, min(N_BRANCH, len(branch3)))
            order = list(range(len(shorts)))
            rng.shuffle(order)
            cal_idx = set(order[:len(shorts) // 2])
            for i, (f, t, ck) in enumerate(shorts):
                pre = pre_trip(exclude=(f, t, ck))
                row = dict(base)
                row.update({'kind': 'circuit_short', 'split': 'cal' if i in cal_idx else 'test',
                            'frmbus': f, 'tobus': t, 'cktid': ck, 'pre_frm': pre[0], 'pre_to': pre[1], 'pre_ckt': pre[2],
                            'out_name': 'psse3304_%s-circuit_short-%s%s_%s%s-%s_%ss' % (
                                stem(sp), 'frmbus', f, 'tobus', t, CHANNEL_OPTION, RT)})
                plan.append(row)

            picked = rng.sample(load_buses, min(N_LOADBUS, len(load_buses)))
            lc = [(ld, lb[0], lb[1]) for lb in picked for ld in LOAD_DELTAS]
            order = list(range(len(lc)))
            rng.shuffle(order)
            cal_idx = set(order[:len(lc) // 2])
            for i, (ld, lodbus, lodid) in enumerate(lc):
                pre = pre_trip()
                row = dict(base)
                row.update({'kind': 'load_change', 'split': 'cal' if i in cal_idx else 'test',
                            'lodbus': lodbus, 'lodid': lodid, 'ld': ld, 'pre_frm': pre[0], 'pre_to': pre[1], 'pre_ckt': pre[2],
                            'out_name': 'psse3304_%s_%s_%s-load_change-%s_%s-%s_%ss' % (
                                stem(sp), 'ld', ld, 'lodbus', lodbus, CHANNEL_OPTION, RT)})
                plan.append(row)

            pre = pre_trip()
            row = dict(base)
            row.update({'kind': 'cut_machine', 'split': 'cal' if c % 2 == 0 else 'test', 'gbus': gbus, 'gid': gid,
                        'pre_frm': pre[0], 'pre_to': pre[1], 'pre_ckt': pre[2],
                        'out_name': 'psse3304_%s-cut_machine-%s_%s-%s_%ss' % (stem(sp), 'gbus', gbus, CHANNEL_OPTION, RT)})
            plan.append(row)
    for n, row in enumerate(plan):
        row['case_id'] = '%s-%04d' % (row['family'], n)
        for k in PLAN_FIELDS:
            row.setdefault(k, '')
    # self-checks: a plan that fails them is never written or run
    names = set()
    for row in plan:
        key = (row['family'], row['split'], row['kind'], row['out_name'])
        if key in names:
            raise RuntimeError('plan has a repeated file name: %r' % (key,))
        names.add(key)
        if row['family'] == 'F2' and row['kind'] == 'circuit_short':
            if line_key((row['pre_frm'], row['pre_to'], row['pre_ckt'])) == line_key((row['frmbus'], row['tobus'], row['cktid'])):
                raise RuntimeError('plan has an outage on the faulted line: %s' % row['case_id'])
        if (row['family'] == 'F2') != (row['pre_frm'] != ''):
            raise RuntimeError('planned outage inconsistent with the family: %s' % row['case_id'])
        if row['family'] == 'F2' and line_key((row['pre_frm'], row['pre_to'], row['pre_ckt'])) in split_keys:
            raise RuntimeError('planned outage splits the network: %s' % row['case_id'])
    return plan


def write_plan(plan, path):
    with open(path, 'wb') as f:
        w = csv.DictWriter(f, fieldnames=PLAN_FIELDS)
        w.writeheader()
        for row in plan:
            w.writerow(dict((k, row[k]) for k in PLAN_FIELDS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--pilot', action='store_true',
                    help='condition 0 of each family: first short circuit, the trip, and the load changes of the '
                         'first sampled load bus at -50, -10 and +10 per cent (10 cases)')
    ap.add_argument('--family', choices=['F1', 'F2'])
    ap.add_argument('--kind', choices=KINDS)
    ap.add_argument('--case-ids', help='comma-separated case IDs of the plan; only these are run (diagnostic use)')
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    if not os.path.isdir(out_root):
        os.makedirs(out_root)
    sys.dont_write_bytecode = True
    os.environ['PATH'] = PSSBIN + ';' + PSSLIB + ';' + os.environ.get('PATH', '')
    sys.path.insert(0, PSSBIN)

    hashes = dict((p, sha256(p)) for p in [SOURCE, SAV, DYR])
    for p, h in hashes.items():
        if h != EXPECTED_SHA256[p]:
            raise RuntimeError('input changed: %s has %s, expected %s' % (p, h, EXPECTED_SHA256[p]))

    import psspy, redirect
    redirect.psse2py()
    calls = []
    # per case: the planned outage of F2, whether a power flow has been solved, and the outage record
    ctx = {'pre': None, 'fnsl_seen': False, 'record': None}
    _i, _f, _s = psspy.getdefaultint(), psspy.getdefaultreal(), psspy.getdefaultchar()

    def static_outage(branch):
        """Take the planned line out of service in the working case with the power-flow data API and read
        the status back. Queries (xfrint, brnint) change nothing."""
        i, j, ck = int(branch[0]), int(branch[1]), str(branch[2])
        rec = {'branch': [i, j, ck]}
        # PSS/E 33.4 API manual: xfrint returns 0 for a two-winding transformer and 3 if the branch is not
        # a transformer; 'CW' is one of its documented strings
        probe = psspy.xfrint(i, j, ck, 'CW')
        code = probe[0] if isinstance(probe, tuple) else None
        rec['transformer_query'] = repr(probe)
        if code not in (0, 3):
            rec['ok'] = False
            rec['reason'] = 'xfrint returned %r; expected error code 0 or 3' % (probe,)
            return rec
        is_trf = (code == 0)
        rec['two_winding_transformer'] = bool(is_trf)

        def status():
            r = psspy.brnint(i, j, ck, 'STATUS')
            return [r[0], r[1]] if isinstance(r, tuple) else [None, repr(r)]
        rec['status_before'] = status()
        if is_trf:
            if not hasattr(psspy, 'two_winding_chng_4'):
                raise RuntimeError('planned outage not established: psspy.two_winding_chng_4 is not available')
            result = psspy.two_winding_chng_4(i, j, ck, [0] + [_i] * 14, [_f] * 24, [_s] * 2)
            rec['api'] = 'two_winding_chng_4'
        else:
            result = psspy.branch_chng(i, j, ck, [0, _i, _i, _i, _i, _i], [_f] * 15)
            rec['api'] = 'branch_chng'
        rec['return'] = repr(result)
        rec['ierr'] = ierr_of(result)
        rec['status_after'] = status()
        rec['before_power_flow'] = not ctx['fnsl_seen']
        rec['ok'] = bool(rec['ierr'] == 0 and rec['status_before'] == [0, 1] and rec['status_after'] == [0, 0]
                         and rec['before_power_flow'])
        return rec

    def track(name, original):
        def call(*a, **k):
            if name == 'dist_branch_trip' and ctx['pre'] is not None and ctx['record'] is None and not ctx['fnsl_seen']:
                # the generator's pre-disturbance trip of an F2 event: the planned outage is applied here,
                # through the power-flow data API (see the header, version 6)
                want = ctx['pre']
                same = (len(a) == 3 and int(a[0]) == int(want[0]) and int(a[1]) == int(want[1])
                        and str(a[2]).strip() == str(want[2]).strip())
                if not same:
                    ctx['record'] = {'ok': False, 'reason': 'generator asked for %r, plan has %r' % (a, tuple(want))}
                    raise RuntimeError('planned outage not established: ' + ctx['record']['reason'])
                rec = static_outage(want)
                ctx['record'] = rec
                if 'api' in rec:
                    calls.append({'api': rec['api'], 'args': repr(tuple(rec['branch'])), 'return': rec['return'],
                                  'ierr': rec['ierr'], 'role': 'planned outage (power-flow data), in place of the '
                                                               "generator's dist_branch_trip before the power flow"})
                if not rec['ok']:
                    raise RuntimeError('planned outage not established: %r' % (rec,))
                return 0
            result = original(*a, **k)
            entry = {'api': name, 'args': repr(a), 'return': repr(result), 'ierr': ierr_of(result)}
            if name == 'fnsl':
                ctx['fnsl_seen'] = True
                # convergence flag of the power flow just solved (a query; 0 = tolerance met)
                try:
                    entry['solved'] = int(psspy.solved())
                except Exception:
                    entry['solved'] = None
            calls.append(entry)
            return result
        return call
    for name in ['case', 'fnsl', 'cong', 'conl', 'ordr', 'fact', 'tysl', 'dyre_new', 'strt', 'run',
                 'dist_bus_fault', 'dist_branch_trip', 'dist_clear_fault', 'dist_branch_close',
                 'dist_machine_trip', 'load_chng_4', 'load_data_4', 'machine_chng_2']:
        if hasattr(psspy, name):
            setattr(psspy, name, track(name, getattr(psspy, name)))

    rc = psspy.psseinit(200000)
    if rc != 0:
        raise RuntimeError('PSS/E initialization failed: %s' % rc)
    log_path = os.path.join(out_root, 'progress.log')
    for kind in ['progress', 'alert', 'report', 'prompt']:
        getattr(psspy, kind + '_output')(2, os.path.join(out_root, kind + '.log'))

    scope = {'__name__': 'tdher_original_functions', '__file__': SOURCE}
    execfile(SOURCE, scope)
    scope['log_path'] = log_path
    scope['random_cut_bus'] = False

    info = scope['get_system_info'](SAV)
    generators, loads, branches = list(info[2]), list(info[4]), list(info[5])
    plan = build_plan(branches, loads, generators)
    plan_path = os.path.join(out_root, 'plan_%s.csv' % PLAN_VERSION)
    if os.path.exists(plan_path):
        with open(plan_path, 'rb') as f:
            old = list(csv.DictReader(f))
        if len(old) != len(plan):
            raise RuntimeError('existing plan in --out has %d rows, the plan computed now %d; use a new directory'
                               % (len(old), len(plan)))
        for o, p in zip(old, plan):
            for k in PLAN_FIELDS:
                if str(o[k]) != str(p[k]):
                    raise RuntimeError('existing plan in --out differs from the plan computed now at %s, field %s '
                                       '(%r against %r); use a new directory' % (p['case_id'], k, o[k], p[k]))
    else:
        write_plan(plan, plan_path)
    if sha256(plan_path) != EXPECTED_PLAN_SHA256:
        raise RuntimeError('the plan in --out is not the reviewed plan v5 (%s); nothing is run' % sha256(plan_path))
    meta_path = os.path.join(out_root, 'runner_meta.json')
    meta_now = {'runner_version': RUNNER_VERSION, 'plan_version': PLAN_VERSION, 'master_seed': MASTER_SEED,
                'inputs_sha256': hashes, 'runner_sha256': sha256(os.path.abspath(__file__)), 'python': sys.version,
                'n_cases': len(plan), 'n_branches': len(branches), 'n_loads': len(loads),
                'n_generators': len(generators), 'plan_csv_sha256': sha256(plan_path),
                'splitting_lines': [list(k) for k in splitting_lines([tuple(b[:3]) for b in branches])],
                'elements': {'branches': [[int(b[0]), int(b[1]), str(b[2])] for b in branches],
                             'loads': [[int(l[0]), str(l[1])] for l in loads],
                             'generators': [[int(g[0]), str(g[1])] for g in generators]}}
    if os.path.exists(meta_path):
        # the first meta file is kept; a resume must agree with it and is logged separately
        with open(meta_path) as f:
            first = json.load(f)
        for k in ['plan_version', 'master_seed', 'inputs_sha256', 'plan_csv_sha256', 'n_cases', 'runner_sha256']:
            if first.get(k) != meta_now[k]:
                raise RuntimeError('this run disagrees with the first run in --out on %s; use a new directory' % k)
    else:
        with open(meta_path, 'w') as f:
            json.dump(meta_now, f, indent=2)
    with open(os.path.join(out_root, 'runner_sessions.jsonl'), 'a') as f:
        f.write(json.dumps({'started': time.time(), 'argv': sys.argv[1:], 'runner_sha256': meta_now['runner_sha256']}) + '\n')
    print('plan: %d cases -> %s' % (len(plan), plan_path))
    if args.plan_only:
        return 0

    todo = plan
    if args.family:
        todo = [r for r in todo if r['family'] == args.family]
    if args.kind:
        todo = [r for r in todo if r['kind'] == args.kind]
    if args.case_ids:
        wanted = [c.strip() for c in args.case_ids.split(',') if c.strip()]
        known = set(r['case_id'] for r in plan)
        missing = [c for c in wanted if c not in known]
        if missing:
            raise RuntimeError('not in the plan: %s' % ', '.join(missing))
        todo = [r for r in todo if r['case_id'] in set(wanted)]
    if args.pilot:
        sel = []
        for fam in ['F1', 'F2']:
            rows = [r for r in todo if r['family'] == fam and int(r['cond_index']) == 0]
            lcs = [r for r in rows if r['kind'] == 'load_change']
            first_bus = lcs[0]['lodbus'] if lcs else None
            sel += [r for r in rows if r['kind'] == 'circuit_short'][:1]
            sel += [r for r in lcs if r['lodbus'] == first_bus and float(r['ld']) in PILOT_LD]
            sel += [r for r in rows if r['kind'] == 'cut_machine'][:1]
        todo = sel

    ledger_path = os.path.join(out_root, 'ledger.jsonl')
    done = set()
    if os.path.exists(ledger_path):
        with open(ledger_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get('status') in ('generated', 'no_output'):
                    done.add(rec['case_id'])

    for n, r in enumerate(todo):
        if r['case_id'] in done:
            continue
        path_base = os.path.join(out_root, r['family'], r['split'], r['kind'])
        if not os.path.isdir(path_base):
            os.makedirs(path_base)
        out_file = r['out_name'] + '.out'
        xlsx = os.path.join(path_base, r['out_name'] + '.xlsx')
        sp = {'le': float(r['le']), 'lz': [float(r['zip_z']), float(r['zip_i']), float(r['zip_p'])],
              'rr': float(r['rr']), 'hi': float(r['hi'])}
        is_f2 = r['family'] == 'F2'
        pre = (r['pre_frm'], r['pre_to'], r['pre_ckt']) if is_f2 else None
        scope['random_cut_bus'] = bool(is_f2)
        scope['out_file'] = out_file
        del calls[:]
        ctx['pre'], ctx['fnsl_seen'], ctx['record'] = (tuple(pre) if is_f2 else None), False, None
        log_start = os.path.getsize(log_path) if os.path.exists(log_path) else 0
        rec = {'case_id': r['case_id'], 'family': r['family'], 'split': r['split'], 'kind': r['kind'],
               'out_name': r['out_name'], 'pre_trip_planned': list(pre) if pre else None, 'started': time.time()}
        original_choice = random.choice
        try:
            if r['kind'] == 'circuit_short':
                if is_f2:
                    # the generator draws the outaged branch with random.choice inside the function;
                    # the planned branch is supplied through that call, the function itself is unchanged
                    random.choice = lambda seq, _pre=pre: _pre
                scope['circuit_short'](path_base, out_file, TST, TET, r['frmbus'], r['tobus'], r['cktid'], RT,
                                       SAV, DYR, sp, CHANNEL_OPTION, REF_MACHINE)
            elif r['kind'] == 'load_change':
                extra = list(pre) if is_f2 else []
                scope['load_change'](path_base, out_file, TST, TET, float(r['ld']), r['lodbus'], r['lodid'], RT,
                                     SAV, DYR, sp, CHANNEL_OPTION, REF_MACHINE, *extra)
            else:
                extra = list(pre) if is_f2 else []
                scope['cut_machine'](path_base, out_file, TST, TET, r['gbus'], r['gid'], RT,
                                     SAV, DYR, sp, CHANNEL_OPTION, REF_MACHINE, *extra)
            # 'generated' means only that the output file exists; it is not an acceptance
            rec['status'] = 'generated' if os.path.exists(xlsx) else 'no_output'
        except Exception:
            rec['status'] = 'exception'
            rec['exception'] = traceback.format_exc()
        finally:
            random.choice = original_choice
        rec['elapsed_seconds'] = time.time() - rec['started']
        rec['api_calls'] = list(calls)
        trips = [c for c in calls if c['api'] == 'dist_branch_trip']
        rec['branch_trips'] = [c['args'] for c in trips]
        # dist_clear_fault clears the fault of a short circuit (core step). In cut_machine the generator
        # calls it although no fault was applied: there it is a clean-up step, kept apart from the core.
        cleanup_here = r['kind'] == 'cut_machine'
        core = [c for c in calls if not (cleanup_here and c['api'] == 'dist_clear_fault')]
        rec['core_api_nonzero'] = [{'api': c['api'], 'args': c['args'], 'return': c['return']}
                                   for c in core if c['ierr'] not in (0,)]
        rec['core_api_all_zero'] = len(rec['core_api_nonzero']) == 0 and len(core) > 0
        rec['cleanup_api'] = [{'api': c['api'], 'return': c['return']} for c in calls
                              if cleanup_here and c['api'] == 'dist_clear_fault']
        if is_f2:
            # the outage as applied and read back; the event counts only if 'ok' is true
            rec['pre_outage'] = ctx['record']
            rec['pre_outage_ok'] = bool(ctx['record'] and ctx['record'].get('ok'))
        rec['n_run_segments'] = len([c for c in calls if c['api'] == 'run'])
        rec['power_flow_solved'] = [c.get('solved') for c in calls if c['api'] == 'fnsl']
        tail_text = ''
        if os.path.exists(log_path):
            with open(log_path, 'rb') as f:
                f.seek(log_start)
                tail_text = f.read()
        rec['log_flags'] = {'initial_conditions_suspect': tail_text.count('INITIAL CONDITIONS SUSPECT'),
                            'initial_conditions_ok': tail_text.count('INITIAL CONDITIONS CHECK O.K.'),
                            'network_not_converged': tail_text.count('Network not converged'),
                            'nan': tail_text.count('NaN')}
        rec['xlsx_bytes'] = os.path.getsize(xlsx) if os.path.exists(xlsx) else 0
        with open(ledger_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print('[%d/%d] %s %s %s -> %s, core API all zero: %s%s (%.1f s)' % (
            n + 1, len(todo), r['case_id'], r['split'], r['kind'], rec['status'],
            rec.get('core_api_all_zero'), (', outage ok: %s' % rec['pre_outage_ok']) if is_f2 else '',
            rec['elapsed_seconds']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
