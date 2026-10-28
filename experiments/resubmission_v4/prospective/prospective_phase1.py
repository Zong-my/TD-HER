#!/usr/bin/env python3
"""Prospective families F1/F2, phase 1: intake, eligibility, extraction, representations, expert inference.

Protocol: claude_pssE_F1F2_protocol_v5_20261005.md with addenda a and b.

What this script does, in this order, for the instance directories copied from Windows:
  1. identity: plan hash, runner hash, input hashes of every instance; coverage of the plan by the ledgers
     and by the records of technical timeouts (addendum d: a case whose worker made no progress for 600 s was
     terminated, is not retried and has the status technical_timeout; the remaining cases of that worker were
     continued in a further instance directory);
  2. E2 (simulation): exported by the generator's rule, every core API return zero, power flow met its
     tolerance, and for F2 the planned outage confirmed by the runner's read-back;
  3. extraction with the unchanged extractor (additional sheets of the current generator dropped by name);
  4. E3 (interface): benchmark columns, all finite, two rows at the trigger, 10 ms sampling; and, from the
     file itself, the flow channels of the POWR sheet: equal to the case for F1, fewer by one at each end of
     the planned line for F2;
  5. E4 (exact rows): a new row equal to an existing benchmark row, or to an earlier new row of the same
     family, in all 550 measured inputs and both labels is dropped (the name-derived descriptor columns are
     not part of the comparison: in this generator, events that differ only in ZIP composition or reserve
     setting can be identical simulations);
  6. tables and representations with the unchanged writers, in a new directory;
  7. predictions of the seven frozen experts, one call per expert and split, stored.

It computes NO error: no prediction of a new event is compared with its label, here or in any output of
this script. Phase 2 (calibration stage, endpoints) is a separate script that runs on the frozen output.
Nothing existing is written to; both output directories must be new.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import argparse, csv, hashlib, json, os, re, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(str(REPO_ROOT))
sys.path.insert(0, str(ROOT))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_ROOT)

from data_proc import extract_features as EX            # noqa: E402
from data_proc import build_representations as BR       # noqa: E402

EXPECTED_PLAN_SHA256 = "51682d108ec82ff59788c991fbd733ef2339232b39f06e91029fb48f62f979de"
RUNNER_SHA256 = "33facb08cb9f038cda663703a193f5e04b6783f7ffafbf9555b6972dd6e0e862"      # runner version 7
EXPECTED_INPUT_SHA256 = {"generate_freq_datav7_3.py": "7421e3a73d6a2341e0dae30c28d4c5d4b0a8b75648bee073c90ba3195f2b861a",
                         "IEEE39.sav": "d982ed3c38b6fdd1bb47915e7a1f73f5f2688e6bd15dea30bf83e6e7eb910364",
                         "IEEE39bus_rew_10_v33.dyr": "36bdea815b9cd7f574f8473e6e0e4ed0b14c191a54d6477339fc257edcd37428"}
BENCH = Path(str(data_path('ieee39_v8_80_10_10')))
BENCH_SPLITS = ["train", "val", "test", "cross_cond_finetune", "cross_cond_test",
                "cross_cond_topo_finetune", "cross_cond_topo_test"]
BENCH_SHEETS = ["ANGL", "FREQ", "POWR", "SPD", "VOLT"]
KNOWN_EXTRA_SHEETS = {"PMEC", "IFD", "GREF"}
META = ["distu_kind", "file_name"]
STATIC = ["load_level", "load_zip_z", "load_zip_i", "load_zip_p", "reserve_ratio", "h_inertia", "load_delta"]
TARGETS = ["fpu_deltamax", "t_delta"]
CFG = EX.IEEE39_CONFIG
MS = 10
PILOT_LD = (-50.0, -10.0, 10.0)
FAMILIES, SPLITS = ["F1", "F2"], ["cal", "test"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def line_key(f, t, ck):
    return (min(int(f), int(t)), max(int(f), int(t)), str(ck).strip())


# ----------------------------------------------------------------------------- ledger and E2
TIMEOUT_FILE = "technical_timeouts.jsonl"
TIMEOUT_REASON = "technical timeout: no progress for 600 s, process terminated, not retried"


def read_instances(dirs, runner_sha, smoke_v5):
    plan, inst, records, timeouts = None, [], defaultdict(list), {}
    for d in dirs:
        d = Path(d)
        plans = sorted(d.glob("plan_F1F2-*.csv"))
        if len(plans) != 1:
            sys.exit(f"{d}: expected one plan file, found {len(plans)}")
        h = sha(plans[0])
        if h != EXPECTED_PLAN_SHA256:
            sys.exit(f"{d}: plan hash {h[:16]} is not the reviewed plan")
        meta = json.load(open(d / "runner_meta.json"))
        if meta["runner_sha256"] != runner_sha:
            sys.exit(f"{d}: runner hash {meta['runner_sha256'][:16]} differs from the expected {runner_sha[:16]}")
        for p, v in meta["inputs_sha256"].items():
            name = p.replace("\\", "/").split("/")[-1]
            if EXPECTED_INPUT_SHA256.get(name) != v:
                sys.exit(f"{d}: input {name} has an unexpected hash")
        if plan is None:
            plan = list(csv.DictReader(open(plans[0], newline="")))
            elements = meta["elements"]
        n = 0
        with open(d / "ledger.jsonl") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_instance"] = d.name
                rec["_dir"] = str(d)
                records[rec["case_id"]].append(rec)
                n += 1
        n_to = 0
        if (d / TIMEOUT_FILE).exists():
            with open(d / TIMEOUT_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line)
                    if t["case_id"] in timeouts:
                        sys.exit(f"{t['case_id']}: more than one timeout record")
                    t["_instance"] = d.name
                    timeouts[t["case_id"]] = t
                    n_to += 1
        inst.append({"dir": str(d), "name": d.name, "ledger_lines": n, "timeout_records": n_to,
                     "runner_sha256": meta["runner_sha256"], "runner_version": meta.get("runner_version")})
    return plan, elements, inst, records, timeouts


def load_manifests(instance_dirs, extra):
    """Hashes computed on Windows before copying, keyed by the absolute path of the copy on this machine.
    Formats: a list of entries with a path (RelativePath, File or Path) relative to the manifest's own directory
    and a hash taken at the source (SourceSHA256 or SHA256). Key names are matched without regard to case or
    underscores (the batch supervisor writes file / source_sha256)."""
    files = [Path(e) for e in (extra or [])]
    for d in instance_dirs:
        d = Path(d).resolve()
        for up in (d, d.parent, d.parent.parent):
            for name in ("copy-hash-manifest.json", "copy-sha256.json"):
                if (up / name).exists() and (up / name) not in files:
                    files.append(up / name)
    by_path, used = {}, []
    for mf in files:
        entries = json.load(open(mf))
        n = 0
        for e in entries if isinstance(entries, list) else []:
            e = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in e.items()}
            rel = e.get("relativepath") or e.get("file") or e.get("path")
            h = e.get("sourcesha256") or e.get("sha256")
            if not rel or not h:
                continue
            by_path[str((Path(mf).resolve().parent / rel.replace("\\", "/")).resolve())] = h.lower()
            n += 1
        used.append({"manifest": str(mf), "entries": n})
    return by_path, used


def pilot_case_ids(plan):
    out = []
    for fam in FAMILIES:
        rows = [r for r in plan if r["family"] == fam and int(r["cond_index"]) == 0]
        lcs = [r for r in rows if r["kind"] == "load_change"]
        first_bus = lcs[0]["lodbus"]
        out += [r["case_id"] for r in rows if r["kind"] == "circuit_short"][:1]
        out += [r["case_id"] for r in lcs if r["lodbus"] == first_bus and float(r["ld"]) in PILOT_LD]
        out += [r["case_id"] for r in rows if r["kind"] == "cut_machine"][:1]
    return out


def judge_e2(p, recs, smoke_v5):
    """Independent reading of the ledger records of one planned case. Returns a dict for the eligibility table."""
    out = {"n_ledger_records": len(recs), "earlier_statuses": ";".join(r["status"] for r in recs[:-1])}
    if not recs:
        out.update({"status": "not_run", "e2_pass": False, "e2_reason": "no ledger record"})
        return out, None
    rec = recs[-1]
    calls = rec.get("api_calls", [])
    cleanup = lambda c: p["kind"] == "cut_machine" and c["api"] == "dist_clear_fault"
    core = [c for c in calls if not cleanup(c)]
    nonzero = [(c["api"], c["ierr"]) for c in core if c["ierr"] != 0]
    fnsl = [c for c in calls if c["api"] == "fnsl"]
    solved = [c.get("solved") for c in fnsl]
    flags = rec.get("log_flags", {})
    out.update({"status": rec["status"], "instance": rec["_instance"], "elapsed_seconds": rec.get("elapsed_seconds"),
                "core_api_nonzero": ";".join(f"{a}={i}" for a, i in nonzero), "n_core_calls": len(core),
                "power_flow_solved": ";".join(str(s) for s in solved),
                "cleanup_return": ";".join(str(c["ierr"]) for c in calls if cleanup(c)),
                "not_converged_messages": flags.get("network_not_converged"), "nan_messages": flags.get("nan"),
                "initial_conditions_suspect": flags.get("initial_conditions_suspect")})
    if rec["out_name"] != p["out_name"] or rec["family"] != p["family"] or rec["split"] != p["split"] or rec["kind"] != p["kind"]:
        sys.exit(f"{p['case_id']}: ledger record does not match the plan row")
    reason = []
    if rec["status"] == "exception":
        reason.append("exception")
    elif rec["status"] == "no_output":
        fn = [c for c in fnsl if c["ierr"] != 0]
        if fn:
            reason.append(f"no output: power flow refused (return {fn[0]['ierr']})")
        elif (flags.get("nan") or 0) > 0:
            reason.append("no output: NaN in the simulation log")
        elif (flags.get("network_not_converged") or 0) > 2:
            reason.append("no output: generator's convergence rule")
        else:
            reason.append("no output: other (to be examined)")
    elif rec["status"] != "generated":
        reason.append(f"status {rec['status']}")
    if rec["status"] == "generated":
        if nonzero or not core:
            reason.append("core API return not zero")
        if not solved or any(s != 0 for s in solved):
            reason.append("power flow did not meet its tolerance")
        if p["family"] == "F2":
            if smoke_v5:
                ok = rec.get("pre_trip_issued_first") is True and not nonzero
            else:
                po = rec.get("pre_outage") or {}
                ok = (rec.get("pre_outage_ok") is True and po.get("ierr") == 0 and po.get("status_before") == [0, 1]
                      and po.get("status_after") == [0, 0] and po.get("before_power_flow") is True
                      and line_key(*po.get("branch", [0, 0, ""])) == line_key(p["pre_frm"], p["pre_to"], p["pre_ckt"]))
                out["pre_outage_api"] = po.get("api")
            if not ok:
                reason.append("planned outage not confirmed in the ledger")
    out["e2_pass"] = (rec["status"] == "generated" and not reason)
    out["e2_reason"] = "; ".join(reason)
    return out, rec


# ----------------------------------------------------------------------------- extraction worker
def _work(job):
    """One exported file: unchanged extractor, plus quantities read from the raw file for E3 and monitoring."""
    path, cat = job
    res = {"extracted": False}
    try:
        gen = set(str(b) for b in CFG["gen_buses"])
        r = EX.extract_single_file(path, cat, gen, CFG["freq_bus_strs"], CFG["freq_weights"], CFG["base_freq"],
                                   CFG["trigger_time"], max_ms=25)
        res["row"] = r
        res["extracted"] = r is not None
        from fastexcel import read_excel
        x = read_excel(path)
        sheets = list(x.sheet_names)
        res["sheets"] = sheets
        f = x.load_sheet_by_name("FREQ").to_polars()
        t = f[f.columns[0]].to_numpy().astype(float)
        w = CFG["freq_weights"]
        coi = sum(f[b].to_numpy().astype(float) * w[i] for i, b in enumerate(CFG["freq_bus_strs"])) / w.sum()
        i0, i1 = int(np.argmin(np.abs(t - 0.0))), int(np.argmin(np.abs(t - 0.99)))
        dt = np.diff(t)
        res.update({"rows": int(len(t)), "t_first": float(t[0]), "t_last": float(t[-1]),
                    "median_step": float(np.median(dt[dt > 0])),
                    "rows_at_trigger": int(np.sum(np.abs(t - CFG["trigger_time"]) < 1e-6)),
                    "drift_mHz": float((coi[i1] - coi[i0]) * CFG["base_freq"] * 1000.0)})
        p = x.load_sheet_by_name("POWR").to_polars()
        cols = p.columns[1:]
        base = Counter()
        for c in cols:
            m = re.match(r"^(\d+)(?:[._]\d+)?$", str(c))
            base[m.group(1) if m else f"?{c}"] += 1
        for b in CFG["freq_bus_strs"]:
            base[b] -= 1                                   # the first column of a generator bus is the unit itself
        res["flow_channels"] = {k: v for k, v in base.items() if v != 0}
        res["n_powr_columns"] = len(cols)
        res["unit_power_first_row"] = {b: float(p[b][0]) for b in CFG["freq_bus_strs"]}
        res["gen_columns"] = {s: int(sum(b in x.load_sheet_by_name(s).to_polars().columns for b in CFG["freq_bus_strs"]))
                              for s in BENCH_SHEETS if s in sheets}
    except Exception as e:                                  # reported per file, never silently dropped
        res["error"] = repr(e)[:300]
    return res


def row_digests(values):
    return [hashlib.blake2b(np.ascontiguousarray(v, dtype=np.float64).tobytes(), digest_size=16).digest() for v in values]


def e4_decisions(order, family_of, digest_of, bench_digest):
    """Rule E4, applied in plan order. `order`: case ids in plan order; `digest_of`: key of every row that
    reached this step. A row is dropped if its key equals an existing benchmark row or an earlier row of the
    same family, whichever split either row is in. Returns ({case id: reason, '' if kept}, rows equal to a
    kept row of the other family, which are reported and kept)."""
    seen, out, cross = {}, {}, []
    for cid in order:
        if cid not in digest_of:
            continue
        d, fam = digest_of[cid], family_of[cid]
        mine = seen.setdefault(fam, {})
        if d in bench_digest:
            out[cid] = f"equals an existing row ({bench_digest[d]})"
        elif d in mine:
            out[cid] = f"equals the new row {mine[d]}"
        else:
            mine[d] = cid
            out[cid] = ""
            other = [v[d] for f, v in seen.items() if f != fam and d in v]
            if other:
                cross.append([cid, other[0]])
    return out, cross


def selftest_e4():
    """Synthetic rows: the earliest row in plan order is kept, across the calibration/test split."""
    order = ["F1-0001", "F1-0002", "F1-0003", "F1-0004", "F1-0005", "F2-1104", "F2-1105"]
    split = {"F1-0001": "test", "F1-0002": "cal", "F1-0003": "cal", "F1-0004": "test", "F1-0005": "cal",
             "F2-1104": "cal", "F2-1105": "test"}
    dig = {"F1-0001": b"a", "F1-0002": b"a", "F1-0003": b"b", "F1-0004": b"b", "F1-0005": b"old",
           "F2-1104": b"a", "F2-1105": b"a"}
    fam = {c: c[:2] for c in order}
    got, cross = e4_decisions(order, fam, dig, {b"old": "train"})
    want = {"F1-0001": "", "F1-0002": "equals the new row F1-0001", "F1-0003": "", "F1-0004": "equals the new row F1-0003",
            "F1-0005": "equals an existing row (train)", "F2-1104": "", "F2-1105": "equals the new row F2-1104"}
    ok = got == want and cross == [["F2-1104", "F1-0001"]]
    for c in order:
        print(f"  {c} ({split[c]:4s}) -> {got[c] or 'kept'}")
    print("  equal across families, kept and reported:", cross)
    # the order used by the script before this correction: all calibration rows of a family, then its test rows
    old = [c for f in FAMILIES for sp in SPLITS for c in order if fam[c] == f and split[c] == sp]
    got_old, _ = e4_decisions(old, fam, dig, {b"old": "train"})
    print("  split-first order (before the correction) would have kept",
          [c for c in order if got_old[c] == "" and got[c] != ""], "and dropped", [c for c in order if got_old[c] != "" and got[c] == ""])
    print("E4 self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-e4", action="store_true", help="synthetic check of rule E4; reads and writes nothing")
    ap.add_argument("--instances", nargs="+")
    ap.add_argument("--mode", choices=["batch", "pilot", "cases"])
    ap.add_argument("--tables-out")
    ap.add_argument("--results-out")
    ap.add_argument("--runner-sha", default=RUNNER_SHA256)
    ap.add_argument("--smoke-v5", action="store_true",
                    help="smoke test on the version-5 pilot directory only: its ledger has no read-back of the outage")
    ap.add_argument("--manifest", nargs="*", help="hash manifests written on Windows, in addition to those found "
                                                    "in an instance directory or up to two levels above it")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--skip-inference", action="store_true")
    args = ap.parse_args()
    if args.selftest_e4:
        sys.exit(selftest_e4())
    if not (args.instances and args.mode and args.tables_out and args.results_out):
        ap.error("--instances, --mode, --tables-out and --results-out are required")
    if args.smoke_v5 and args.mode == "batch":
        sys.exit("--smoke-v5 is not allowed for the batch")
    tables, results = Path(args.tables_out), Path(args.results_out)
    if tables.exists() or results.exists():
        sys.exit("both output directories must be new")
    t0 = time.time()

    # ---------------- 1. identity and coverage (before anything is written)
    plan, elements, inst, records, timeouts = read_instances(args.instances, args.runner_sha, args.smoke_v5)
    by_id = {r["case_id"]: r for r in plan}
    unknown = [c for c in list(records) + list(timeouts) if c not in by_id]
    if unknown:
        sys.exit(f"ledger cases not in the plan: {unknown[:5]}")
    multi = {c: sorted({r["_instance"] for r in rs}) for c, rs in records.items() if len({r["_instance"] for r in rs}) > 1}
    if multi and args.mode == "batch":
        sys.exit(f"{len(multi)} cases occur in more than one instance, e.g. {list(multi.items())[:3]}")
    # a timed-out case has no completed ledger record anywhere: the process was terminated in mid-case
    clash = [c for c in timeouts if any(r["status"] in ("generated", "no_output") for r in records.get(c, []))]
    if clash:
        sys.exit(f"cases with both a timeout record and a completed ledger record: {clash[:5]}")
    if args.mode == "batch":
        expected = [r["case_id"] for r in plan]
    elif args.mode == "pilot":
        expected = pilot_case_ids(plan)
    else:
        expected = [r["case_id"] for r in plan if r["case_id"] in records or r["case_id"] in timeouts]
    missing = [c for c in expected if c not in records and c not in timeouts]
    extra = [c for c in list(records) + list(timeouts) if c not in set(expected)]
    if missing or (extra and args.mode != "cases"):
        sys.exit(f"coverage: {len(missing)} planned cases without a ledger record (e.g. {missing[:5]}), "
                 f"{len(extra)} records outside the expected set (e.g. {extra[:5]})")
    tables.mkdir(parents=True, exist_ok=False)
    results.mkdir(parents=True, exist_ok=False)
    log = open(results / "phase1.log", "w")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    say(f"plan {EXPECTED_PLAN_SHA256[:16]}, {len(inst)} instance(s), mode {args.mode}: {len(expected)} cases covered")

    manifest, manifests_used = load_manifests(args.instances, args.manifest)
    say(f"hash manifests: {[(m['manifest'].split('/')[-2] + '/' + m['manifest'].split('/')[-1], m['entries']) for m in manifests_used]}")

    # reference flow channels of the case: one per directed in-service branch, named by its from bus
    ref_flow = Counter(str(int(b[0])) for b in elements["branches"])

    # ---------------- 2. E2
    table, jobs = {}, []
    for cid in expected:
        p = by_id[cid]
        if cid in timeouts:
            prior = records.get(cid, [])
            e2, rec = {"n_ledger_records": len(prior), "earlier_statuses": ";".join(r["status"] for r in prior),
                       "status": "technical_timeout", "instance": timeouts[cid]["_instance"],
                       "e2_pass": False, "e2_reason": TIMEOUT_REASON}, None
        else:
            e2, rec = judge_e2(p, records.get(cid, []), args.smoke_v5)
        row = {"case_id": cid, "family": p["family"], "split": p["split"], "kind": p["kind"], "cond_index": int(p["cond_index"]),
               "le": p["le"], "zip": f"{p['zip_z']}-{p['zip_i']}-{p['zip_p']}", "rr": p["rr"], "hi": p["hi"],
               "ld": p["ld"], "lodbus": p["lodbus"], "frmbus": p["frmbus"], "tobus": p["tobus"],
               "pre_line": f"{p['pre_frm']}-{p['pre_to']}" if p["pre_frm"] else "", "out_name": p["out_name"]}
        row.update(e2)
        row.update({"file_found": "", "e3_pass": "", "e3_reason": "", "e4_dropped": "", "analysed": False, "table_row": ""})
        if e2["e2_pass"]:
            path = Path(rec["_dir"]) / p["family"] / p["split"] / p["kind"] / (p["out_name"] + ".xlsx")
            row["file_found"] = path.exists() and path.stat().st_size > 0
            if not row["file_found"]:
                row["e2_pass"], row["e2_reason"] = False, "exported according to the ledger, file missing"
            else:
                row["_path"] = str(path)
                row["_manifest_sha"] = manifest.get(str(path.resolve()))
                jobs.append(cid)
        table[cid] = row
    say(f"E2: {sum(r['e2_pass'] for r in table.values())} of {len(table)} cases pass; "
        f"statuses {dict(Counter(r['status'] for r in table.values()))}")
    # every file that can enter the analysis must equal the hash taken on Windows before it was copied
    for cid in jobs:
        row = table[cid]
        row["file_sha256"] = sha(row["_path"])
        row["file_sha_equals_manifest"] = ("no manifest entry" if row["_manifest_sha"] is None
                                           else row["file_sha256"] == row["_manifest_sha"])
    unverified = [c for c in jobs if table[c]["file_sha_equals_manifest"] is not True]
    say(f"copies verified against the Windows manifests: {len(jobs) - len(unverified)} of {len(jobs)}")
    if unverified and args.mode == "batch":
        sys.exit(f"{len(unverified)} exported files without a verified hash, e.g. "
                 f"{[(c, table[c]['file_sha_equals_manifest']) for c in unverified[:5]]}; nothing is analysed")

    # ---------------- 3. extraction (unchanged extractor), 4. E3
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, len(jobs) or 1))) as ex:
        outs = list(ex.map(_work, [(table[c]["_path"], table[c]["kind"]) for c in jobs], chunksize=4))
    bench_cols = list(pd.read_csv(BENCH / "csv" / f"test_ms{MS}.csv", nrows=0).columns)
    extracted = {}
    unknown_sheets = set()
    for cid, o in zip(jobs, outs):
        row, p = table[cid], by_id[cid]
        reasons = []
        if o.get("error"):
            reasons.append("read error: " + o["error"])
        if not o.get("extracted"):
            reasons.append("extractor returned nothing")
        unknown_sheets |= {s for s in o.get("sheets", []) if s not in BENCH_SHEETS and s not in KNOWN_EXTRA_SHEETS and s != "PLOD"}
        for k in ["rows", "rows_at_trigger", "median_step", "drift_mHz", "n_powr_columns", "t_first", "t_last"]:
            row[k] = o.get(k)
        row["unit31_power_first_row"] = (o.get("unit_power_first_row") or {}).get("31")
        row["extra_sheets"] = ";".join(sorted(s for s in o.get("sheets", []) if s in KNOWN_EXTRA_SHEETS))
        if o.get("rows_at_trigger") != 2:
            reasons.append(f"{o.get('rows_at_trigger')} rows at the trigger")
        if o.get("median_step") is None or abs(o["median_step"] - 0.01) > 1e-4:
            reasons.append("sampling is not 10 ms")
        if any(v != 10 for v in (o.get("gen_columns") or {"-": 0}).values()) or len(o.get("gen_columns") or {}) != 5:
            reasons.append("generator-bus columns incomplete")
        want = Counter(ref_flow)
        if p["family"] == "F2":
            want[str(int(p["pre_frm"]))] -= 1
            want[str(int(p["pre_to"]))] -= 1
        got = Counter(o.get("flow_channels") or {})
        row["flow_channels_as_planned"] = (+want == +got)
        if not row["flow_channels_as_planned"]:
            diff = {k: got.get(k, 0) - want.get(k, 0) for k in set(want) | set(got) if got.get(k, 0) != want.get(k, 0)}
            reasons.append(f"flow channels differ from the planned topology: {diff}")
        if o.get("extracted"):
            r = o["row"]
            vals = list(r["temporal"].values()) + list(r["targets"].values()) + list(r["static"].values())
            if not np.all(np.isfinite(np.array(vals, dtype=float))):
                reasons.append("non-finite value")
            extracted[cid] = r
        if row["file_sha_equals_manifest"] is False:
            reasons.append("file differs from the Windows hash manifest")
        row["e3_pass"], row["e3_reason"] = (not reasons), "; ".join(reasons)
    if unknown_sheets:
        sys.exit(f"sheets neither in the benchmark nor in the known extra set: {sorted(unknown_sheets)}")
    say(f"E3: {sum(r['e3_pass'] is True for r in table.values())} of {len(jobs)} exported files pass")

    # ---------------- 5. E4: exact rows against the benchmark and among the new rows (plan order)
    key_cols = [c for c in bench_cols if c not in META + STATIC]            # 550 measured inputs and both labels
    bench_digest = {}
    for sp in BENCH_SPLITS:
        df = pd.read_csv(BENCH / "csv" / f"{sp}_ms{MS}.csv", usecols=key_cols)[key_cols]
        for d in row_digests(df.values):
            bench_digest.setdefault(d, sp)
        del df
    say(f"E4: {len(bench_digest)} distinct existing rows indexed over {len(BENCH_SPLITS)} benchmark tables")
    frames = {}
    fixed = set(META + STATIC + TARGETS)
    for fam in FAMILIES:
        for sp in SPLITS:
            ids = [c for c in expected if table[c]["family"] == fam and table[c]["split"] == sp and table[c]["e3_pass"] is True]
            if not ids:
                continue
            df = EX.results_to_dataframe([extracted[c] for c in ids])
            keep = [c for c in df.columns if c in fixed or c.split("_")[0] in BENCH_SHEETS]
            df = df[keep]
            df.insert(0, "_case_id", ids)
            frames[(fam, sp)] = df
    # key of every new row (benchmark column order); decisions in plan order, whichever split a row is in
    digest_of = {}
    for df in frames.values():
        for cid, d in zip(df["_case_id"], row_digests(df[key_cols].values.astype(np.float64))):
            digest_of[cid] = d
    decisions, cross_family = e4_decisions(expected, {c: table[c]["family"] for c in expected}, digest_of, bench_digest)
    for cid, reason in decisions.items():
        table[cid]["e4_dropped"] = reason
    for key, df in list(frames.items()):
        frames[key] = df[[decisions[c] == "" for c in df["_case_id"]]].reset_index(drop=True)
    say(f"E4: {sum(bool(r['e4_dropped']) for r in table.values())} new rows dropped; "
        f"{len(cross_family)} rows equal a row of the other family (kept, reported)")

    # ---------------- 6. tables and representations (unchanged writers)
    csv_dir = tables / "csv"
    names = []
    for (fam, sp), df in sorted(frames.items()):
        if len(df) == 0:
            continue
        name = f"{fam}_{sp}"
        ids = list(df["_case_id"])
        body = df.drop(columns=["_case_id"])
        csv_dir.mkdir(parents=True, exist_ok=True)
        body.to_csv(csv_dir / f"{name}_full.csv", index=False)
        EX.save_per_ms(body, str(csv_dir), name, [MS])
        back = pd.read_csv(csv_dir / f"{name}_ms{MS}.csv")
        if list(back.columns) != bench_cols:
            sys.exit(f"{name}: table columns differ from the benchmark's")
        with open(csv_dir / f"{name}_case_ids.json", "w") as f:
            json.dump(ids, f)
        for i, cid in enumerate(ids):
            table[cid]["analysed"], table[cid]["table_row"] = True, i
        names.append(name)
    if names:
        BR.process_csv_to_reps(str(csv_dir), str(tables), [MS], names)
    per_table = Counter(r["family"] + "_" + r["split"] for r in table.values() if r["analysed"])
    say("tables: " + ", ".join("%s (%d)" % (n, per_table[n]) for n in names))

    # ---------------- 7. frozen experts, one call per expert and split; shape and finiteness only
    inference = {}
    if names and not args.skip_inference:
        import experiments.exp4_generalization as G4
        import experiments.exp_tdh_router as R
        scalers = G4.load_train_scalers()

        def level_data(d, split):
            """The lines of exp4_generalization.load_level_data, reading from the directory given."""
            X_A = np.load(d / f"repA/ms{MS}/X_{split}.npy"); y = np.load(d / f"repA/ms{MS}/y_{split}.npy")
            Xt = np.load(d / f"repB/ms{MS}/X_temporal_{split}.npy"); Xs = np.load(d / f"repB/ms{MS}/X_static_{split}.npy")
            X_n = scalers["scaler_A"].transform(X_A[:, scalers["feat_idx"]]).astype(np.float32)
            B, T, N, C = Xt.shape
            Xt_n = scalers["sc_t"].transform(Xt.reshape(-1, N * C)).reshape(B, T, N, C).astype(np.float32)
            Xs_n = scalers["sc_s"].transform(Xs[:, scalers["static_idx"]]).astype(np.float32)
            Xn_n = np.transpose(Xt_n, (0, 2, 1, 3))
            return {"X_A": X_A, "X_n": X_n, "y": y, "Xt_n": Xt_n, "Xs_n": Xs_n, "Xn_n": Xn_n}

        # the loader above against the production loader, on a stored split (arrays must be identical)
        mine, prod = level_data(BENCH, "cross_cond_test"), G4.load_level_data("cross_cond_test", scalers)
        same = {k: bool(np.array_equal(mine[k], prod[k])) for k in prod}
        inference["loader_identical_to_production_on_cross_cond_test"] = same
        if not all(same.values()):
            sys.exit(f"loader differs from the production loader: {same}")
        feat_ok = json.load(open(tables / f"repA/ms{MS}/feature_names.json")) == scalers["feature_names"]
        inference["feature_names_equal_benchmark"] = bool(feat_ok)
        if not feat_ok:
            sys.exit("feature names of the new representations differ from the benchmark's")
        for name in names:
            data = level_data(tables, name)
            np.save(results / f"X_n_{name}.npy", data["X_n"])           # scaled tabular inputs, used by the adapter in phase 2
            for m in R.ROUTER_MODELS:
                pred = R.predict_expert(m, data, scalers)
                (results / "experts" / m).mkdir(parents=True, exist_ok=True)
                np.save(results / "experts" / m / f"{name}_preds.npy", pred)
                inference[f"{m}/{name}"] = {"shape": list(pred.shape), "finite": bool(np.isfinite(pred).all())}
        bad = [k for k, v in inference.items() if isinstance(v, dict) and "finite" in v and not v["finite"]]
        if bad:
            sys.exit(f"non-finite expert output: {bad}")
        say(f"inference: {len(R.ROUTER_MODELS)} experts x {len(names)} splits stored")

    # ---------------- outputs
    cols = [c for c in next(iter(table.values())).keys() if not c.startswith("_")]
    allcols = []
    for r in table.values():
        for c in r:
            if not c.startswith("_") and c not in allcols:
                allcols.append(c)
    with open(results / "eligibility.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allcols, extrasaction="ignore")
        w.writeheader()
        for cid in expected:
            w.writerow(table[cid])

    def count(rows):
        rows = list(rows)
        return {"planned": len(rows), "exported": sum(r["status"] == "generated" for r in rows),
                "technical_timeout": sum(r["status"] == "technical_timeout" for r in rows),
                "E2": sum(bool(r["e2_pass"]) for r in rows), "E3": sum(r["e3_pass"] is True for r in rows),
                "analysed": sum(bool(r["analysed"]) for r in rows)}
    rows = list(table.values())
    summary = {"by_family_split_kind": {}, "load_changes_by_family_and_value": {}, "reasons": {}}
    for fam in FAMILIES:
        for sp in SPLITS:
            for kind in ["circuit_short", "load_change", "cut_machine", "all"]:
                sel = [r for r in rows if r["family"] == fam and r["split"] == sp and (kind == "all" or r["kind"] == kind)]
                if sel:
                    summary["by_family_split_kind"][f"{fam}/{sp}/{kind}"] = count(sel)
        for ld in sorted({float(r["ld"]) for r in rows if r["ld"] != ""}):
            sel = [r for r in rows if r["family"] == fam and r["ld"] != "" and float(r["ld"]) == ld]
            if sel:
                summary["load_changes_by_family_and_value"][f"{fam}/{ld:g}"] = count(sel)
    summary["reasons"] = dict(Counter((r["e2_reason"] or r["e3_reason"] or r["e4_dropped"] or "analysed") for r in rows))
    drift = [r["drift_mHz"] for r in rows if r.get("drift_mHz") is not None]
    summary["monitored"] = {"drift_mHz": {"min": min(drift), "median": float(np.median(drift)), "max": max(drift)} if drift else None,
                            "unit31_first_row_pu": [min(r["unit31_power_first_row"] for r in rows if r.get("unit31_power_first_row") is not None),
                                                    max(r["unit31_power_first_row"] for r in rows if r.get("unit31_power_first_row") is not None)] if drift else None,
                            "cases_with_exception_records": sorted(c for c, rs in records.items() if any(r["status"] == "exception" for r in rs))}
    report = {"created_unix": time.time(), "argv": sys.argv[1:], "mode": args.mode, "smoke_v5": args.smoke_v5,
              "plan_sha256": EXPECTED_PLAN_SHA256, "runner_sha256": args.runner_sha,
              "instances": [{k: v for k, v in i.items() if not k.startswith("_")} for i in inst],
              "cases_in_more_than_one_instance": multi, "summary": summary, "inference": inference,
              "technical_timeouts": {c: {k: v for k, v in t.items() if not k.startswith("_")} for c, t in timeouts.items()},
              "hash_manifests": manifests_used, "exported_files_verified": len(jobs) - len(unverified),
              "exported_files_unverified": [[c, table[c]["file_sha_equals_manifest"]] for c in unverified],
              "rows_equal_across_families": cross_family,
              "e4_dropped": {c: r["e4_dropped"] for c, r in table.items() if r["e4_dropped"]},
              "statement": "No prediction of a new event was compared with its label; no error was computed.",
              "code_sha256": {p: sha(ROOT / p) for p in ["data_proc/extract_features.py", "data_proc/build_representations.py",
                                                         "experiments/exp4_generalization.py", "experiments/exp_tdh_router.py"]},
              "script_sha256": sha(os.path.abspath(__file__)), "seconds": time.time() - t0}
    with open(results / "phase1_report.json", "w") as f:
        json.dump(report, f, indent=1)
    # manifest of everything written, so that phase 2 can refuse changed inputs
    man = {}
    for base in (tables, results):
        for pth in sorted(base.rglob("*")):
            if pth.is_file() and pth.name not in ("phase1_manifest.json", "phase1.log"):
                man[str(pth)] = sha(pth)
    with open(results / "phase1_manifest.json", "w") as f:
        json.dump({"files": man, "n": len(man)}, f, indent=1)
    say(json.dumps(summary["by_family_split_kind"], indent=1))
    say(f"written {results} ({len(man)} files in the manifest) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
