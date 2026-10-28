#!/usr/bin/env python3
"""FNP-BPNN integrated nadir predictor (Wang et al., iSPEC 2022) on the TD-HER
benchmark. Additive research phase 2026-10-14, task 1 (secondary comparator).

The analytical FNP inputs of NEW training/validation/calibration events are
produced with the frozen analytical interface (read-only harness modules); the
stored test FNP predictions are reused as fixed inputs. Every BPNN setting is a
benchmark setting selected on the validation split (the paper fixes none).

Phases
  describe  raw-pool resolution counts, primary-category counts, stored test
            inputs and frozen common-cohort sizes (no workbook reads)
  extract   NEW per-event rows for one split: identity-verified workbook,
            COI trajectory and the frozen first-nadir compatibility status,
            FNP/SFR analytical outputs, timings (--limit for smokes)
  search    BPNN benchmark-setting grid on the extracted train/val rows
  final     (after the gate) common-cohort evaluation with stored test inputs

Usage (repository root, fepfm interpreter):
  python experiments/resubmission_v4/exp_v4_published_predictor_fnp_bpnn.py --phase describe
  python experiments/resubmission_v4/exp_v4_published_predictor_fnp_bpnn.py --phase extract --split train --limit 48 --n-jobs 8 --smoke
"""
from __future__ import annotations

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import os
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
import sys
sys.dont_write_bytecode = True  # the frozen harness directories must receive no byte-code writes

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = WORK_ROOT
DEFAULT_PROTOCOL = ARTIFACT_ROOT / "protocols/comparator_fnp_bpnn.json"
DEFAULT_OUT = work_path('results/ieee39/resubmission_v4_published_predictor_fnp_bpnn_v1')
ACTIVE = Path(str(data_path('ieee39_v8_80_10_10')))
RAW = Path(str(data_path('ieee39_v8')))
PLAN = work_path("analytical")
SHARED = PLAN / "shared"
ADAPT = REPO_ROOT / "analytical_comparators/adaptation_v1"
REPRO = REPO_ROOT / "analytical_comparators/reproduction_v1"
DRIVER_PATH = Path(__file__).resolve()
F_BASE = 60.0
TRIGGER = 1.0
MS = 10
FREQ_BUSES = ["30", "31", "32", "33", "34", "35", "36", "37", "38", "39"]
FREQ_WEIGHTS = np.array([6.05, 3.41, 6.05, 0.0, 3.41, 5.016, 3.141, 3.141, 5.32, 500.0])
SAMPLE_TOL_S = 0.0100001
NADIR_FLOOR_HZ = 1e-4
NADIR_RHO = 0.2
COMPAT_RULE_ID = "first_nadir_relprom_rho0.2_floor1e-4_v1"
LABEL_TOL_HZ = 1e-5
LABEL_TOL_S = 1e-5
REPB_TOL = 1e-6          # stored repB window is float32; the workbook window must reproduce it exactly
CATEGORIES = ["circuit_short", "cut_machine", "load_change"]
PRIMARY = ("cut_machine", "load_change")
SCENARIO_OF_CAL = {"cross_cond_finetune": "L2", "cross_cond_topo_finetune": "L3"}
HARNESS_FILES = [
    ADAPT / "analytical_predictor_v1.py", ADAPT / "operational_inputs.py", ADAPT / "governor_models.py",
    ADAPT / "causal_inputs.py", ADAPT / "causal_inputs_v2.py", REPRO / "classical_sfr.py",
    REPRO / "fnp_liu2020.py", (REPO_ROOT / 'configs/analytical/analytical_operational_config_v1.json'),
    work_path('analytical/adaptation_v1/results_sav_v2/sav_bound_case_map.json'), work_path('analytical/adaptation_v1/results_fnp_conditioned_v1/bank.json'),
    work_path('analytical/adaptation_v1/results_fnp_conditioned_v1/offline_convolution_table.npz'),
]

# --------------------------------------------------------------------------
# provenance helpers and the no-test-array guard
# --------------------------------------------------------------------------
_FILES_READ: list[str] = []
_NP_LOAD = np.load


def _tracked_np_load(file, *args, **kwargs):
    _FILES_READ.append(str(file))
    return _NP_LOAD(file, *args, **kwargs)


np.load = _tracked_np_load


def assert_no_test_reads() -> None:
    offenders = [p for p in _FILES_READ if "test" in Path(p).name]
    if offenders:
        raise RuntimeError(f"test arrays were read in a train/val-only phase: {offenders}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def runtime_provenance(protocol: dict) -> dict:
    import sklearn
    try:
        import fastexcel, polars
        fx, pl = fastexcel.__version__, polars.__version__
    except Exception:
        fx = pl = None
    return {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "fastexcel": fx, "polars": pl,
        "git_head": git_head(),
        "driver_path": str(DRIVER_PATH.relative_to(PROJECT_ROOT)), "driver_sha256": sha256_file(DRIVER_PATH),
        "protocol_id": protocol.get("protocol_id"),
        "harness_sha256": {str(p): sha256_file(p) for p in HARNESS_FILES},
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, default=str)


def load_protocol(path: Path) -> dict:
    with open(path) as handle:
        protocol = json.load(handle)
    for key in ("analytical_inputs", "extraction", "model", "final", "evaluation", "never_write"):
        if key not in protocol:
            raise KeyError(f"protocol lacks '{key}'")
    return protocol


# --------------------------------------------------------------------------
# frozen rules re-implemented verbatim (s0_event_index.py / s1_security_truth.py)
# --------------------------------------------------------------------------
def trigger_rows(tm: np.ndarray, trigger_time: float = TRIGGER):
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


def first_nadir(dp, y1, rho):
    from scipy.signal import find_peaks
    idx, props = find_peaks(-dp, prominence=(NADIR_FLOOR_HZ, None), plateau_size=(1, None))
    le, pr = props["left_edges"], props["prominences"]
    thr = max(NADIR_FLOOR_HZ, rho * abs(y1))
    sel = np.where(pr >= thr)[0]
    if len(sel) == 0:
        return None, None
    j = sel[0]
    return int(le[j]), float(pr[j])


def compat_status(dp, tp, y1, g, rho=NADIR_RHO):
    if y1 > 0:
        return "GLOBAL_UPWARD", None, None
    if y1 == 0:
        return "UNRESOLVED", None, None
    if g == len(dp) - 1:
        return "BOUNDARY", None, None
    r, prom = first_nadir(dp, y1, rho)
    if r is None:
        return "NO_INTERIOR_NADIR", None, None
    if abs(tp[r] - tp[g]) <= SAMPLE_TOL_S:
        return "MATCHED_FIRST_DOWNWARD_NADIR", r, prom
    return "LATER_DEEPER_NADIR", r, prom


def coi_from_workbook(path: str):
    """Full FREQ sheet -> (time_abs, coi_pu, n_buses_present); verbatim of s0 process_workbook."""
    import fastexcel
    parser = fastexcel.read_excel(path)
    df = parser.load_sheet_by_name("FREQ").to_polars()
    cols = df.columns
    unnamed = [c for c in cols if c.startswith("__UNNAMED__") or c.startswith("Unnamed")]
    if not unnamed:
        raise RuntimeError("no time column in FREQ sheet")
    tm = df[unnamed[0]].to_numpy().astype(np.float64)
    if len(tm) == 0:
        raise RuntimeError("empty FREQ sheet")
    freq_system = np.zeros(len(tm))
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
    return tm, freq_system, present


def reconstruct_labels(tm, coi):
    t0_row, trans_row, tf0_row = trigger_rows(tm)
    search_start = min(trans_row, len(tm) - 1)
    max_row = search_start + int(np.argmax(np.abs(coi[search_start:])))
    return float(coi[max_row]) * F_BASE, float(tm[max_row]) - TRIGGER, trans_row


# --------------------------------------------------------------------------
# raw pools and event lists
# --------------------------------------------------------------------------
def build_lookup(pool: list[str]) -> dict:
    lookup = {}
    for rd in pool:
        for cat in CATEGORIES:
            d = RAW / rd / cat
            if not d.is_dir():
                continue
            for f in os.listdir(d):
                if f.endswith(".xlsx"):
                    lookup.setdefault((cat, f), []).append(str(d / f))
    return lookup


def load_events(split: str, protocol: dict) -> pd.DataFrame:
    if split not in protocol["extraction"]["splits"]:
        raise ValueError(f"split {split} is not an extraction split (test splits are never extracted)")
    df = pd.read_csv(ACTIVE / "csv" / f"{split}_ms{MS}.csv",
                     usecols=["distu_kind", "file_name", "h_inertia", "fpu_deltamax", "t_delta"])
    df = df.reset_index().rename(columns={"index": "row_index"})
    y = np.load(ACTIVE / "repA" / f"ms{MS}" / f"y_{split}.npy").astype(np.float64)
    if y.shape != (len(df), 2) or np.max(np.abs(y - df[["fpu_deltamax", "t_delta"]].to_numpy())) > 1e-5:
        raise RuntimeError(f"stored y_{split}.npy differs from the CSV labels")
    lookup = build_lookup(protocol["extraction"]["raw_pools"][split])
    cands = [lookup.get((c, f), []) for c, f in zip(df["distu_kind"], df["file_name"])]
    df["candidates"] = cands
    df["n_candidates"] = [len(c) for c in cands]
    df["split"] = split
    return df


# --------------------------------------------------------------------------
# extraction workers
# --------------------------------------------------------------------------
G: dict = {}


def init_worker(split: str = ""):
    sys.dont_write_bytecode = True
    if split:
        # memory-mapped stored window of the same split (guard 1: new input identity vs repB rows)
        G["repb"] = _NP_LOAD(ACTIVE / "repB" / f"ms{MS}" / f"X_temporal_{split}.npy", mmap_mode="r")
        with open(ACTIVE / "repB" / f"ms{MS}" / "meta.json") as h:
            sheets = json.load(h)["sheets"]
        G["freq_idx"], G["powr_idx"] = sheets.index("FREQ"), sheets.index("POWR")
    for p in (str(ADAPT), str(REPRO)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import analytical_predictor_v1 as ap  # noqa: E402  (frozen harness module, read-only)
    import causal_inputs_v2 as ci  # noqa: E402
    G["predict"] = ap.predict
    G["read_window"] = ci.read_frozen_window
    G["read_times"] = ci.read_post_event_times_metadata
    with open((REPO_ROOT / 'configs/analytical/analytical_operational_config_v1.json')) as h:
        G["cfg"] = json.load(h)
    with open(work_path('analytical/adaptation_v1/results_sav_v2/sav_bound_case_map.json')) as h:
        G["case"] = json.load(h)
    with open(work_path('analytical/adaptation_v1/results_fnp_conditioned_v1/bank.json')) as h:
        G["bank"] = json.load(h)
    G["conv"] = dict(_NP_LOAD(work_path('analytical/adaptation_v1/results_fnp_conditioned_v1/offline_convolution_table.npz')))


EMPTY = {
    "workbook_path": "", "workbook_sha256": "", "identity_status": "", "identity_reason": "",
    "n_candidates": 0, "y1_rec_diff_hz": "", "y2_rec_diff_s": "", "n_freq_buses_present": "",
    "compat_status": "", "fnp_target_compatible": "", "first_nadir_rel_s": "", "first_nadir_dev_hz": "",
    "first_nadir_prominence_hz": "", "global_extremum_rel_s": "", "global_extremum_dev_hz": "",
    "observation_end_s": "", "repb_freq_maxdiff": "", "repb_powr_maxdiff": "", "deficit_hat_mw": "", "tripped_buses": "",
    "SFR_status": "", "SFR_reason": "", "SFR_delta_f_hat_hz": "", "SFR_t_extremum_hat_rel_s": "",
    "FNP_status": "", "FNP_reason": "", "FNP_delta_f_hat_hz": "", "FNP_t_extremum_hat_rel_s": "",
    "coi_read_s": "", "window_read_s": "", "predict_s": "", "total_s": "",
}
FIELDS = ["split", "row_index", "file_name", "distu_kind", "h_inertia", "y1", "y2"] + list(EMPTY)


def worker(task: dict) -> dict:
    out = {k: task[k] for k in ("split", "row_index", "file_name", "distu_kind", "h_inertia", "y1", "y2")}
    out.update(EMPTY)
    out["n_candidates"] = len(task["candidates"])
    t_all = time.perf_counter()
    if task["distu_kind"] not in PRIMARY:
        out.update(identity_status="NOT_ATTEMPTED", identity_reason="short-circuit event outside frozen power-imbalance cohort",
                   SFR_status="NOT_APPLICABLE", FNP_status="NOT_APPLICABLE",
                   SFR_reason="short-circuit event outside frozen power-imbalance cohort",
                   FNP_reason="short-circuit event outside frozen power-imbalance cohort")
        out["total_s"] = time.perf_counter() - t_all
        return out
    if not task["candidates"]:
        out.update(identity_status="UNRESOLVED", identity_reason="raw_workbook_not_found")
        out["total_s"] = time.perf_counter() - t_all
        return out
    matches = []
    t_coi = time.perf_counter()
    for path in task["candidates"]:
        try:
            tm, coi, present = coi_from_workbook(path)
            y1_rec, y2_rec, trans_row = reconstruct_labels(tm, coi)
            d1, d2 = abs(y1_rec - task["y1"]), abs(y2_rec - task["y2"])
            if d1 <= LABEL_TOL_HZ and d2 <= LABEL_TOL_S:
                matches.append((path, tm, coi, present, trans_row, d1, d2))
            else:
                out["identity_reason"] += f"{Path(path).parent.parent.name}:dy1={d1:.2e},dy2={d2:.2e};"
        except Exception as exc:
            out["identity_reason"] += f"{Path(path).parent.parent.name}:{type(exc).__name__};"
    out["coi_read_s"] = time.perf_counter() - t_coi
    if len(matches) != 1:
        out["identity_status"] = "AMBIGUOUS" if len(matches) > 1 else "IDENTITY_MISMATCH"
        out["total_s"] = time.perf_counter() - t_all
        return out
    path, tm, coi, present, trans_row, d1, d2 = matches[0]
    out.update(workbook_path=path, workbook_sha256=sha256_file(Path(path)), identity_status="VERIFIED",
               identity_reason="", y1_rec_diff_hz=d1, y2_rec_diff_s=d2, n_freq_buses_present=present)
    d = F_BASE * coi
    dp, tp = d[trans_row:], tm[trans_row:]
    g = int(np.argmax(np.abs(dp)))
    st, r, prom = compat_status(dp, tp, float(task["y1"]), g)
    out.update(compat_status=st, fnp_target_compatible=(st == "MATCHED_FIRST_DOWNWARD_NADIR"),
               first_nadir_rel_s=(float(tp[r] - TRIGGER) if r is not None else ""),
               first_nadir_dev_hz=(float(dp[r]) if r is not None else ""),
               first_nadir_prominence_hz=(prom if prom is not None else ""),
               global_extremum_rel_s=float(tp[g] - TRIGGER), global_extremum_dev_hz=float(dp[g]))
    try:
        t_w = time.perf_counter()
        w = G["read_window"](path, trigger_s=TRIGGER)
        ts = G["read_times"](path, trigger_s=TRIGGER)
        out["window_read_s"] = time.perf_counter() - t_w
        if "repb" in G:
            stored = np.asarray(G["repb"][int(task["row_index"])], dtype=np.float32)
            freq_wb = (np.asarray(w["frequency_deviation_hz"], dtype=np.float64) / F_BASE).astype(np.float32)
            powr_wb = np.asarray(w["power_channel_native_pu"], dtype=np.float64).astype(np.float32)
            d_f = float(np.max(np.abs(freq_wb - stored[:, :, G["freq_idx"]])))
            d_p = float(np.max(np.abs(powr_wb - stored[:, :, G["powr_idx"]])))
            out.update(repb_freq_maxdiff=d_f, repb_powr_maxdiff=d_p)
            if not (d_f <= REPB_TOL and d_p <= REPB_TOL):
                out["identity_status"] = "REPB_WINDOW_MISMATCH"
                raise RuntimeError(f"stored repB window mismatch: FREQ {d_f:.3e} pu, POWR {d_p:.3e} pu")
        t_p = time.perf_counter()
        res, state = G["predict"](w, float(task["h_inertia"]), G["case"], G["bank"], G["conv"], ts, G["cfg"])
        out["predict_s"] = time.perf_counter() - t_p
        out.update(observation_end_s=float(w["observation_end_s"]), deficit_hat_mw=state.get("deficit_hat_mw", ""),
                   tripped_buses=";".join(str(b) for b in state.get("tripped_buses", [])))
        for method in ("SFR", "FNP"):
            x = res["CLASSICAL_SFR" if method == "SFR" else "FNP"]
            out[f"{method}_status"] = x["status"]
            out[f"{method}_reason"] = x["reason"]
            if x["status"] == "OK":
                out[f"{method}_delta_f_hat_hz"] = x["delta_f_hat_hz"]
                out[f"{method}_t_extremum_hat_rel_s"] = x["t_extremum_hat_rel_s"]
    except Exception as exc:
        out.update(SFR_status="INPUT_UNAVAILABLE", FNP_status="INPUT_UNAVAILABLE",
                   SFR_reason=f"{type(exc).__name__}: {exc}", FNP_reason=f"{type(exc).__name__}: {exc}")
    out["total_s"] = time.perf_counter() - t_all
    return out


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------
def load_common_cohort(scenario: str, protocol: dict) -> np.ndarray:
    rule = protocol["evaluation"]["common_cohort"]
    ap = pd.read_csv(SHARED / "analytical/physical_causal_sfr_fnp_v2/analytical_predictions.csv")
    ap = ap[ap["scenario"] == scenario]
    keep = None
    tc = None
    for cond in rule["conditions"]:
        if cond["file"] == "analytical_predictions":
            ids = set(ap[(ap["method"] == cond["method"]) & (ap["prediction_status"] == cond["status"])]["event_id"])
        else:
            if tc is None:
                tc = pd.read_csv(SHARED / "eval_truth/truth_v1/target_compatibility.csv")
            col = tc[cond["column"]].astype(str).str.lower()
            ids = set(tc[col == str(cond["value"]).lower()]["event_id"])
        keep = ids if keep is None else keep & ids
    ei = pd.read_csv(SHARED / "event_index.csv")
    ei = ei[ei["scenario"] == scenario]
    return np.sort(ei[ei["event_id"].isin(keep)]["row_index"].to_numpy())


def phase_describe(args, protocol: dict) -> None:
    out = {"protocol_id": protocol["protocol_id"], "splits": {}}
    for split in protocol["extraction"]["splits"]:
        df = load_events(split, protocol)
        prim = df[df["distu_kind"].isin(PRIMARY)]
        out["splits"][split] = {
            "n_events": int(len(df)), "n_primary": int(len(prim)),
            "by_kind": df["distu_kind"].value_counts().to_dict(),
            "primary_resolved_unique": int((prim["n_candidates"] == 1).sum()),
            "primary_unresolved": int((prim["n_candidates"] == 0).sum()),
            "primary_multiple_candidates": int((prim["n_candidates"] > 1).sum()),
            "raw_pool": protocol["extraction"]["raw_pools"][split],
        }
    ap = pd.read_csv(SHARED / "analytical/physical_causal_sfr_fnp_v2/analytical_predictions.csv")
    out["stored_test_fnp_ok_rows"] = ap[(ap["method"] == "FNP") & (ap["prediction_status"] == "OK")].groupby("scenario").size().to_dict()
    out["common_cohorts"] = {}
    for sc, exp in protocol["evaluation"]["common_cohort"]["expected_counts"].items():
        rows = load_common_cohort(sc, protocol)
        out["common_cohorts"][sc] = {"n": int(len(rows)), "expected": int(exp), "ok": bool(len(rows) == int(exp))}
    assert_no_test_reads()
    out["files_read"] = _FILES_READ
    out["provenance"] = runtime_provenance(protocol)
    print(json.dumps(out, indent=2, default=str))


def phase_extract(args, protocol: dict) -> None:
    split = args.split
    out_dir = args.out_dir / "extract" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_events(split, protocol)
    assert_no_test_reads()
    prim_idx = df.index[df["distu_kind"].isin(PRIMARY)].to_numpy()
    if args.limit:
        prim_idx = prim_idx[args.start::max(1, args.stride)][: args.limit]
        selected = df.loc[prim_idx]
    else:
        selected = df
    features_path = out_dir / "features.csv"
    done = set()
    if features_path.exists():
        done = set(pd.read_csv(features_path, usecols=["file_name"])["file_name"])
    tasks = [
        {"split": split, "row_index": int(r.row_index), "file_name": r.file_name, "distu_kind": r.distu_kind,
         "h_inertia": float(r.h_inertia), "y1": float(r.fpu_deltamax), "y2": float(r.t_delta),
         "candidates": list(r.candidates)}
        for r in selected.itertuples() if r.file_name not in done
    ]
    logger.info(f"{split}: {len(tasks)} events to extract ({len(done)} already done), n_jobs={args.n_jobs}")
    provenance = runtime_provenance(protocol)
    provenance.update({"phase": "extract", "split": split, "limit": args.limit, "start": args.start,
                       "stride": args.stride, "n_jobs": args.n_jobs,
                       "smoke": bool(args.smoke), "files_read": list(_FILES_READ)})
    write_json(out_dir / "provenance.json", provenance)
    counts, timings = {}, {"coi_read_s": [], "window_read_s": [], "predict_s": [], "total_s": []}
    start = time.perf_counter()
    new_file = not features_path.exists()
    with open(features_path, "a", newline="") as fh, open(out_dir / "diagnostics.jsonl", "a") as dg:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=init_worker, initargs=(split,)) as pool:
            for n, row in enumerate(pool.map(worker, tasks, chunksize=4), 1):
                writer.writerow(row)
                key = (row["identity_status"], row["compat_status"], row["FNP_status"])
                counts[str(key)] = counts.get(str(key), 0) + 1
                for k in timings:
                    if row[k] != "":
                        timings[k].append(float(row[k]))
                dg.write(json.dumps({k: row[k] for k in ("row_index", "file_name", "identity_status",
                                                         "identity_reason", "compat_status", "FNP_status",
                                                         "FNP_reason", "total_s")}, default=str) + "\n")
                if n % 500 == 0 or n == len(tasks):
                    fh.flush()
                    logger.info(f"{split}: {n}/{len(tasks)} events, {time.perf_counter() - start:.1f}s")
    wall = time.perf_counter() - start
    # cumulative statistics over every row in features.csv (resumed runs must not discard earlier ones)
    allrows = pd.read_csv(features_path, keep_default_na=False)
    cum_counts = allrows.groupby(["identity_status", "compat_status", "FNP_status"]).size()
    cum_timing = {}
    for k in ("coi_read_s", "window_read_s", "predict_s", "total_s"):
        v = pd.to_numeric(allrows[k], errors="coerce").dropna()
        cum_timing[k] = {"mean": float(v.mean()) if len(v) else None,
                         "median": float(v.median()) if len(v) else None, "n": int(len(v))}
    manifest = {
        "split": split, "this_invocation": {
            "n_tasks": len(tasks), "n_previously_done": len(done), "wall_seconds": wall,
            "n_jobs": args.n_jobs, "limit": args.limit, "start": args.start, "stride": args.stride,
            "smoke": bool(args.smoke), "counts_by_(identity,compat,FNP)": counts,
            "throughput_events_per_second": (len(tasks) / wall) if (wall > 0 and tasks) else None,
            "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        },
        "cumulative": {
            "n_rows": int(len(allrows)),
            "counts_by_(identity,compat,FNP)": {str(k): int(v) for k, v in cum_counts.items()},
            "per_event_seconds": cum_timing,
        },
        "features_sha256": sha256_file(features_path),
        "policy": protocol["extraction"]["missing_policy"],
        "training_cohort_rule": protocol["extraction"]["training_cohort_rule"]["id"],
    }
    history = out_dir / "manifest_history.jsonl"
    with open(history, "a") as hh:
        hh.write(json.dumps(manifest["this_invocation"], default=str) + "\n")
    write_json(out_dir / "manifest.json", manifest)
    assert_no_test_reads()
    logger.info(json.dumps(manifest, indent=2, default=str))


def verify_extraction(out_dir: Path, split: str) -> dict:
    """Guard 2: every CSV row present exactly once, manifest agrees, eligible features/targets finite."""
    n_csv = len(pd.read_csv(ACTIVE / "csv" / f"{split}_ms{MS}.csv", usecols=["file_name"]))
    feat = pd.read_csv(out_dir / "extract" / split / "features.csv", keep_default_na=False)
    with open(out_dir / "extract" / split / "manifest.json") as h:
        manifest = json.load(h)
    rows = feat["row_index"].to_numpy()
    expected = np.arange(n_csv)
    missing = np.setdiff1d(expected, rows)
    extra = np.setdiff1d(rows, expected)
    duplicates = int(len(rows) - len(np.unique(rows)))
    elig = feat[(feat["identity_status"] == "VERIFIED") & (feat["FNP_status"] == "OK")
                & (feat["compat_status"] == "MATCHED_FIRST_DOWNWARD_NADIR")]
    values = elig[["FNP_delta_f_hat_hz", "FNP_t_extremum_hat_rel_s", "y1", "y2"]].apply(pd.to_numeric, errors="coerce")
    finite = bool(np.isfinite(values.to_numpy(dtype=float)).all()) if len(elig) else True
    report = {
        "split": split, "n_csv_rows": int(n_csv), "n_feature_rows": int(len(feat)),
        "manifest_cumulative_rows": int(manifest["cumulative"]["n_rows"]),
        "missing_row_indices": int(len(missing)), "extra_row_indices": int(len(extra)), "duplicate_rows": duplicates,
        "identity_counts": feat["identity_status"].value_counts().to_dict(),
        "fnp_status_counts": feat["FNP_status"].value_counts().to_dict(),
        "compat_counts": feat["compat_status"].value_counts().to_dict(),
        "n_eligible": int(len(elig)), "eligible_finite": finite,
        "repb_maxdiff": {k: float(pd.to_numeric(feat[k], errors="coerce").max()) for k in ("repb_freq_maxdiff", "repb_powr_maxdiff")
                         if k in feat.columns},
    }
    report["ok"] = (len(missing) == 0 and len(extra) == 0 and duplicates == 0
                    and len(feat) == n_csv == report["manifest_cumulative_rows"] and finite)
    if not report["ok"]:
        raise RuntimeError(f"extraction of {split} is incomplete or inconsistent: {report}")
    return report


def phase_verify(args, protocol: dict) -> None:
    out = {s: verify_extraction(args.out_dir, s) for s in args.verify_splits}
    write_json(args.out_dir / "extraction_verification.json", out)
    print(json.dumps(out, indent=2, default=str))


def load_training_rows(out_dir: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(out_dir / "extract" / split / "features.csv")
    ok = (df["identity_status"] == "VERIFIED") & (df["FNP_status"] == "OK") & (df["compat_status"] == "MATCHED_FIRST_DOWNWARD_NADIR")
    rows = df[ok].copy()
    rows["x1"] = rows["FNP_delta_f_hat_hz"].astype(float)
    rows["x2"] = rows["FNP_t_extremum_hat_rel_s"].astype(float)
    return rows


def fit_bpnn(Xtr, Ytr, hidden: int, lr: float, max_iter: int, seed: int):
    from sklearn.neural_network import MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(hidden,), activation="logistic", solver="sgd",
                         learning_rate_init=lr, momentum=0.9, max_iter=max_iter, tol=1e-6,
                         n_iter_no_change=50, random_state=seed)
    model.fit(Xtr, Ytr)
    return model


def phase_search(args, protocol: dict) -> None:
    out_dir = args.out_dir / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        verification = {s: verify_extraction(args.out_dir, s) for s in ("train", "val")}
        write_json(out_dir / "extraction_verification.json", verification)
    tr = load_training_rows(args.out_dir, "train")
    va = load_training_rows(args.out_dir, "val")
    if len(tr) < 4 or len(va) < 2:
        raise RuntimeError(f"too few compatible rows: train {len(tr)}, val {len(va)}")
    Xtr, Ytr = tr[["x1", "x2"]].to_numpy(), tr[["y1", "y2"]].to_numpy()
    Xva, Yva = va[["x1", "x2"]].to_numpy(), va[["y1", "y2"]].to_numpy()
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-12
    my, sy = Ytr.mean(0), Ytr.std(0) + 1e-12
    grid = protocol["model"]["candidate_grid"]
    max_iter = int(grid["max_iter_smoke"] if args.smoke else grid["max_iter_full"])
    rows = []
    ref = np.mean(np.abs(Xva - Yva), axis=0)      # FNP inputs used directly (identity mapping)
    rows.append({"candidate": "FNP_identity_reference", "hidden": "", "lr": "", "seed": "",
                 "val_mae_y1_hz": float(ref[0]), "val_mae_y2_s": float(ref[1]),
                 "val_objective": float(np.mean(ref / sy)), "fit_seconds": 0.0, "n_iter": ""})
    for hidden in grid["hidden_units"]:
        for lr in grid["learning_rate_init"]:
            t0 = time.time()
            model = fit_bpnn((Xtr - mx) / sx, (Ytr - my) / sy, int(hidden), float(lr), max_iter, args.seed)
            pred = model.predict((Xva - mx) / sx) * sy + my
            mae = np.mean(np.abs(pred - Yva), axis=0)
            rows.append({"candidate": f"h{hidden}_lr{lr}", "hidden": hidden, "lr": lr, "seed": args.seed,
                         "val_mae_y1_hz": float(mae[0]), "val_mae_y2_s": float(mae[1]),
                         "val_objective": float(np.mean(mae / sy)), "fit_seconds": time.time() - t0,
                         "n_iter": int(model.n_iter_)})
            logger.info(json.dumps(rows[-1]))
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "candidates.csv", index=False)
    cand = frame[frame["candidate"] != "FNP_identity_reference"]
    best = cand.sort_values("val_objective").iloc[0].to_dict()
    write_json(out_dir / "best_settings.json", {
        **best, "label": "benchmark settings selected on the validation split; the paper fixes none",
        "n_train_rows": int(len(tr)), "n_val_rows": int(len(va)), "max_iter": max_iter,
        "standardization": {"x_mean": mx.tolist(), "x_std": sx.tolist(), "y_mean": my.tolist(), "y_std": sy.tolist()},
    })
    write_json(out_dir / "provenance.json", {**runtime_provenance(protocol), "phase": "search",
                                             "smoke": bool(args.smoke), "files_read": list(_FILES_READ)})
    assert_no_test_reads()
    logger.info(f"best: {best}")


def phase_final(args, protocol: dict) -> None:
    raise NotImplementedError("final phase is implemented after the search gate (reads stored test inputs)")


# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("describe", "extract", "verify", "search", "final"), required=True)
    parser.add_argument("--verify-splits", nargs="+", default=["train", "val"])
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="extract: N primary-category events (0 = all)")
    parser.add_argument("--start", type=int, default=0, help="extract: offset into the primary-category list")
    parser.add_argument("--stride", type=int, default=1, help="extract: take every k-th primary event (smokes)")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    if args.smoke and not str(args.out_dir).endswith("_smoke"):
        args.out_dir = Path(str(args.out_dir) + "_smoke")
    for never in protocol["never_write"]:
        never_path = (PROJECT_ROOT / never).resolve()
        if str(args.out_dir.resolve()).startswith(str(never_path)):
            raise RuntimeError(f"output directory {args.out_dir} lies inside a frozen path {never}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(args.out_dir / f"{args.phase}.log", enqueue=True)
    logger.info(f"phase={args.phase} out_dir={args.out_dir} protocol={args.protocol}")
    {"describe": phase_describe, "extract": phase_extract, "verify": phase_verify,
     "search": phase_search, "final": phase_final}[args.phase](args, protocol)


if __name__ == "__main__":
    main()
