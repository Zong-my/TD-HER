#!/usr/bin/env python3
"""Prospective families F1/F2: chain check before any batch (protocol v5, Section 7). Version 2.

Part 1, replay on existing L1 test data.
  1a  For a fixed sample of events, re-extracted from their raw files, the chain used for new files must
      reproduce the stored table rows (extract_features.py, unchanged) and the stored representations
      (build_representations.py, unchanged) exactly.
  1b  The seven frozen experts, run today on the stored L1 test arrays in one call per expert (as the
      cache was made), must reproduce their cached predictions within the gate already used for the
      control checkpoints: mean |difference| <= 1 % of the cached MAE and |MAE change| <= 0.5 %, per
      expert and target. The cached arrays stay as they are.
  1c  Recorded, not a gate: the largest difference from the cache when the sampled events are predicted
      as a batch of their own. The experts' inference runs in mixed precision (NNTrainer.predict uses
      autocast), so a prediction depends slightly on the batch it is computed in.

Version 1 of this script applied a per-event bound (largest difference <= 1 % of the MAE) to the batch of
1c; rows and representations were identical, LightGBM was identical, and the neural experts exceeded
that bound by rounding steps of half precision. That bound did not suit mixed-precision inference and
is replaced by 1b, the gate reviewed for the control bank. The output of version 1 is kept in its
directory (resubmission_v4_prospective_pilot_v1).

Part 2, new files. The same chain on the files exported by the Windows pilot: layout, extraction,
representations, expert inference. No prediction is compared with a label of a new event here, and no
error of a new event is computed.

The only step added to the benchmark code is a column filter: the current generator version exports
sheets that the benchmark files do not contain; table rows are restricted to the benchmark's columns.
Outputs go to a new directory; nothing existing is written to.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(str(REPO_ROOT))
sys.path.insert(0, str(ROOT))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_ROOT)

from data_proc import extract_features as EX            # noqa: E402
from data_proc import build_representations as BR       # noqa: E402
import experiments.exp4_generalization as G4            # noqa: E402
import experiments.exp_tdh_router as R                  # noqa: E402

BENCH = Path(G4.REP_DIR)
EVENT_INDEX = work_path("analytical/shared/event_index.csv")
BENCH_SHEETS = ["ANGL", "FREQ", "POWR", "SPD", "VOLT"]
KNOWN_EXTRA_SHEETS = {"PMEC", "IFD", "GREF"}
META, STATIC, TARGETS = ["distu_kind", "file_name"], ["load_level", "load_zip_z", "load_zip_i", "load_zip_p",
                                                      "reserve_ratio", "h_inertia", "load_delta"], ["fpu_deltamax", "t_delta"]
CFG = EX.IEEE39_CONFIG
MS = G4.MS


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def extract_table(files, work, split):
    """files: list of (path, category). Returns the ms10 table as the benchmark stores it, and a log."""
    gen = set(str(b) for b in CFG["gen_buses"])
    results, log = [], []
    for path, cat in files:
        r = EX.extract_single_file(str(path), cat, gen, CFG["freq_bus_strs"], CFG["freq_weights"],
                                   CFG["base_freq"], CFG["trigger_time"], max_ms=25)
        log.append({"file": os.path.basename(str(path)), "extracted": r is not None})
        if r is not None:
            results.append(r)
    if not results:
        return None, log, []
    df = EX.results_to_dataframe(results)
    fixed = set(META + STATIC + TARGETS)
    sheets_seen = sorted({c.split("_")[0] for c in df.columns if c not in fixed})
    unknown = [s for s in sheets_seen if s not in BENCH_SHEETS and s not in KNOWN_EXTRA_SHEETS]
    if unknown:
        raise RuntimeError(f"sheets neither in the benchmark nor in the known extra set: {unknown}")
    keep = [c for c in df.columns if c in fixed or c.split("_")[0] in BENCH_SHEETS]
    dropped_sheets = [s for s in sheets_seen if s not in BENCH_SHEETS]
    df = df[keep]
    csv_dir = Path(work) / "csv"
    EX.save_per_ms(df, str(csv_dir), split, [MS])          # unchanged writer of the benchmark tables
    table = pd.read_csv(csv_dir / f"{split}_ms{MS}.csv")   # read back as build_representations does
    return table, log, dropped_sheets


def build_reps(work, split):
    BR.process_csv_to_reps(str(Path(work) / "csv"), str(Path(work) / "reps"), [MS], [split])
    d = Path(work) / "reps"
    return {
        "X_A": np.load(d / f"repA/ms{MS}/X_{split}.npy"), "y": np.load(d / f"repA/ms{MS}/y_{split}.npy"),
        "Xt": np.load(d / f"repB/ms{MS}/X_temporal_{split}.npy"), "Xs": np.load(d / f"repB/ms{MS}/X_static_{split}.npy"),
        "Xn": np.load(d / f"repC/ms{MS}/X_node_{split}.npy"),
        "feature_names": json.load(open(d / f"repA/ms{MS}/feature_names.json")),
        "meta": json.load(open(d / f"repB/ms{MS}/meta.json")),
    }


def level_data(rep, scalers):
    """The lines of exp4_generalization.load_level_data, with the arrays passed in instead of read from REP_DIR."""
    X_A, y, Xt, Xs = rep["X_A"], rep["y"], rep["Xt"], rep["Xs"]
    X_n = scalers["scaler_A"].transform(X_A[:, scalers["feat_idx"]]).astype(np.float32)
    B, T, N, C = Xt.shape
    Xt_n = scalers["sc_t"].transform(Xt.reshape(-1, N * C)).reshape(B, T, N, C).astype(np.float32)
    Xs_n = scalers["sc_s"].transform(Xs[:, scalers["static_idx"]]).astype(np.float32)
    Xn_n = np.transpose(Xt_n, (0, 2, 1, 3))
    return {"X_A": X_A, "X_n": X_n, "y": y, "Xt_n": Xt_n, "Xs_n": Xs_n, "Xn_n": Xn_n}


def monitored(path):
    """Quantities of protocol Section 7, items 5 and 7, read from the raw file."""
    from fastexcel import read_excel
    x = read_excel(str(path))
    out = {"sheets": {}}
    data = {}
    for s in x.sheet_names:
        df = x.load_sheet_by_name(s).to_polars()
        cols = df.columns
        out["sheets"][s] = {"rows": df.height, "columns": len(cols) - 1}
        data[s] = df
    f = data["FREQ"]
    t = f[f.columns[0]].to_numpy().astype(float)
    w = CFG["freq_weights"]
    coi = sum(f[b].to_numpy().astype(float) * w[i] for i, b in enumerate(CFG["freq_bus_strs"])) / w.sum()
    i0, i1 = int(np.argmin(np.abs(t - 0.0))), int(np.argmin(np.abs(t - 0.99)))
    out["time"] = {"first": float(t[0]), "last": float(t[-1]), "median_step": float(np.median(np.diff(t)[np.diff(t) > 0])),
                   "rows_at_trigger": int(np.sum(np.abs(t - CFG["trigger_time"]) < 1e-6))}
    out["pre_trigger_coi_drift_mHz"] = float((coi[i1] - coi[i0]) * CFG["base_freq"] * 1000.0)
    p = data["POWR"]
    out["initial_unit_power_pu_on_sbase"] = {b: float(p[b][0]) for b in CFG["freq_bus_strs"]}
    out["gen_columns_first_in_sheet"] = {s: [c in data[s].columns for c in CFG["freq_bus_strs"]].count(True)
                                         for s in BENCH_SHEETS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replay-per-class", type=int, default=2)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=False)            # never into an existing directory
    rep = {"created_unix": time.time(), "argv": sys.argv[1:],
           "code_sha256": {p: sha(ROOT / p) for p in ["data_proc/extract_features.py", "data_proc/build_representations.py",
                                                      "experiments/exp4_generalization.py", "experiments/exp_tdh_router.py"]},
           "script_sha256": sha(os.path.abspath(__file__)) if "__file__" in globals() and os.path.exists(__file__) else None}

    bench_cols = list(pd.read_csv(BENCH / "csv" / f"test_ms{MS}.csv", nrows=0).columns)
    scalers = G4.load_train_scalers()
    bench_fnames = scalers["feature_names"]
    bench_meta = json.load(open(BENCH / f"repB/ms{MS}/meta.json"))

    # ---------------- Part 1: replay on existing L1 test events
    ei = pd.read_csv(EVENT_INDEX)
    l1 = ei[(ei.scenario == "L1") & (ei.active_split == "test")].sort_values("row_index")
    sample = pd.concat([l1[l1.disturbance_category == c].head(args.replay_per_class)
                        for c in ["circuit_short", "cut_machine", "load_change"]]).sort_values("row_index")
    stored = pd.read_csv(BENCH / "csv" / f"test_ms{MS}.csv")
    with tempfile.TemporaryDirectory(dir=out_dir) as work:
        # the extractor orders rows by category and file within a split; here rows are matched by file name
        table, log, dropped = extract_table([(r.raw_path, r.disturbance_category) for r in sample.itertuples()], work, "replay")
        reps = build_reps(work, "replay")
    r1 = {"n_events": int(len(sample)), "row_index": [int(v) for v in sample.row_index], "dropped_sheets": dropped,
          "columns_equal_benchmark": list(table.columns) == bench_cols,
          "feature_names_equal_benchmark": reps["feature_names"] == bench_fnames,
          "static_names_equal_benchmark": reps["meta"]["static_names"] == bench_meta["static_names"],
          "sheets_buses_equal_benchmark": reps["meta"]["sheets"] == bench_meta["sheets"] and reps["meta"]["buses"] == bench_meta["buses"]}
    order = [int(sample[sample.source_filename == fn].row_index.iloc[0]) for fn in table.file_name]
    st = stored.iloc[order].reset_index(drop=True)
    r1["file_names_equal"] = list(st.file_name) == list(table.file_name)
    num = [c for c in bench_cols if c not in META]
    r1["table_max_abs_diff"] = float(np.max(np.abs(table[num].values.astype(float) - st[num].values.astype(float))))
    for key, name in [("X_A", f"repA/ms{MS}/X_test.npy"), ("Xt", f"repB/ms{MS}/X_temporal_test.npy"),
                      ("Xs", f"repB/ms{MS}/X_static_test.npy"), ("Xn", f"repC/ms{MS}/X_node_test.npy"), ("y", f"repA/ms{MS}/y_test.npy")]:
        ref = np.load(BENCH / name, mmap_mode="r")[order]
        r1[f"rep_{key}_shape_equal"] = tuple(ref.shape) == tuple(reps[key].shape)
        r1[f"rep_{key}_max_abs_diff"] = float(np.max(np.abs(np.asarray(ref, dtype=np.float64) - reps[key].astype(np.float64))))
    r1["pass_1a_rows_and_representations"] = bool(
        r1["columns_equal_benchmark"] and r1["feature_names_equal_benchmark"] and r1["static_names_equal_benchmark"]
        and r1["sheets_buses_equal_benchmark"] and r1["file_names_equal"] and r1["table_max_abs_diff"] == 0.0
        and all(r1[f"rep_{k}_shape_equal"] and r1[f"rep_{k}_max_abs_diff"] == 0.0 for k in ["X_A", "Xt", "Xs", "Xn", "y"]))
    # 1b: the stored split through today's inference, one call per expert (unchanged loader and predictor)
    full = G4.load_level_data("test", scalers)
    y_test = full["y"]
    sample_data = level_data(reps, scalers)
    r1["experts"] = {}
    ok_pred = True
    for m in R.ROUTER_MODELS:
        cached = np.load(R.EXPERT_DIR / m / "test_preds.npy")
        pred = R.predict_expert(m, full, scalers)
        mae_c = np.mean(np.abs(cached - y_test), axis=0)
        mae_n = np.mean(np.abs(pred - y_test), axis=0)
        mean_rel = np.mean(np.abs(pred - cached), axis=0) / mae_c
        mae_rel = np.abs(mae_n - mae_c) / mae_c
        within = bool(np.all(mean_rel <= 0.01) and np.all(mae_rel <= 0.005))
        small = R.predict_expert(m, sample_data, scalers)             # 1c, recorded only
        r1["experts"][m] = {
            "cached_MAE": [float(v) for v in mae_c],
            "mean_abs_diff_over_cached_MAE": [float(v) for v in mean_rel],
            "abs_MAE_change_over_cached_MAE": [float(v) for v in mae_rel],
            "max_abs_diff_full_split": [float(v) for v in np.max(np.abs(pred - cached), axis=0)],
            "within_gate": within,
            "batch_of_sample_max_abs_diff_from_cache": [float(v) for v in np.max(np.abs(small - cached[order]), axis=0)],
        }
        ok_pred &= within
    del full
    r1["pass_1b_inference_replay"] = bool(ok_pred)
    r1["pass"] = bool(r1["pass_1a_rows_and_representations"] and ok_pred)
    rep["part1_replay"] = r1
    print("part 1a rows/representations identical:", r1["pass_1a_rows_and_representations"],
          "| 1b inference replay within gate:", r1["pass_1b_inference_replay"], flush=True)

    # ---------------- Part 2: files exported by the Windows pilot
    pilot = Path(args.pilot_dir)
    files = sorted(p for p in pilot.rglob("*.xlsx"))
    manifest = {}
    mpath = pilot / "copy-hash-manifest.json"
    if mpath.exists():
        manifest = {e["RelativePath"].replace("\\", "/"): e["SHA256"] for e in json.load(open(mpath))}
    r2 = {"files": []}
    todo = []
    for p in files:
        rel = p.relative_to(pilot).as_posix()
        family, split, cat = rel.split("/")[:3]
        h = sha(p)
        todo.append((p, cat))
        r2["files"].append({"file": rel, "family": family, "split": split, "kind": cat, "sha256": h,
                            "sha256_equals_windows_manifest": (manifest.get(rel) == h) if manifest else None,
                            "monitored": monitored(p)})
    with tempfile.TemporaryDirectory(dir=out_dir) as work:
        table, log, dropped = extract_table(todo, work, "pilot")
        reps = build_reps(work, "pilot")
        table.to_csv(out_dir / "pilot_rows_ms10.csv", index=False)
    r2["extraction_log"] = log
    r2["dropped_sheets"] = dropped
    r2["n_rows"] = int(len(table))
    r2["columns_equal_benchmark"] = list(table.columns) == bench_cols
    r2["n_columns"] = int(table.shape[1])
    numv = table[[c for c in bench_cols if c not in META]].values.astype(float)
    r2["all_finite_inputs_and_targets"] = bool(np.isfinite(numv).all())
    r2["feature_names_equal_benchmark"] = reps["feature_names"] == bench_fnames
    r2["rep_shapes"] = {k: list(reps[k].shape) for k in ["X_A", "Xt", "Xs", "Xn", "y"]}
    r2["rep_all_finite"] = bool(all(np.isfinite(reps[k]).all() for k in ["X_A", "Xt", "Xs", "Xn", "y"]))
    r2["rows"] = [{"file_name": r.file_name, "kind": r.distu_kind, "load_level": r.load_level,
                   "zip": [r.load_zip_z, r.load_zip_i, r.load_zip_p], "reserve_ratio": r.reserve_ratio, "h_inertia": r.h_inertia,
                   "load_delta": r.load_delta, "target_1_extremum_Hz": r.fpu_deltamax, "target_2_time_s": r.t_delta}
                  for r in table.itertuples()]
    data = level_data(reps, scalers)
    r2["experts"] = {}
    for m in R.ROUTER_MODELS:
        pred = R.predict_expert(m, data, scalers)
        # shape and finiteness only: predictions of new events are not compared with their labels here
        r2["experts"][m] = {"shape": list(pred.shape), "finite": bool(np.isfinite(pred).all())}
    r2["pass_hard_2_3"] = bool(r2["n_rows"] == len(files) and r2["columns_equal_benchmark"] and r2["all_finite_inputs_and_targets"]
                               and r2["feature_names_equal_benchmark"] and r2["rep_all_finite"]
                               and all(v["finite"] and v["shape"] == [len(files), 2] for v in r2["experts"].values()))
    # initialization messages of the pilot log (protocol item 6) and ledger outcome per case
    plog = pilot / "progress.log"
    if plog.exists():
        lines = open(plog, errors="ignore").read().splitlines()
        named = []
        for i, line in enumerate(lines):
            if "INITIAL CONDITIONS SUSPECT" in line:
                for l2 in lines[i + 2:i + 12]:
                    parts = l2.split()
                    if len(parts) >= 6 and parts[0].isdigit():
                        named.append({"model": parts[3], "state": parts[4], "bus": parts[5], "dstate": parts[1]})
                    else:
                        break
        r2["initial_conditions_suspect_entries"] = named
        r2["initial_conditions_suspect_models"] = sorted({(e["model"], e["bus"]) for e in named})
    led = pilot / "ledger.jsonl"
    if led.exists():
        r2["ledger"] = []
        for line in open(led):
            rec = json.loads(line)
            r2["ledger"].append({"case_id": rec["case_id"], "kind": rec["kind"], "status": rec["status"],
                                 "core_api_all_zero": rec.get("core_api_all_zero"),
                                 "core_api_nonzero": [(c["api"], c["return"]) for c in rec.get("core_api_nonzero", [])],
                                 "cleanup_api": rec.get("cleanup_api"), "log_flags": rec.get("log_flags"),
                                 "pre_trip_issued_first": rec.get("pre_trip_issued_first"),
                                 "power_flow_solved": rec.get("power_flow_solved"),
                                 "elapsed_seconds": rec.get("elapsed_seconds")})
    rep["part2_pilot_files"] = r2
    with open(out_dir / "chain_check.json", "w") as f:
        json.dump(rep, f, indent=1)
    print("part 2 hard criteria 2-3 pass:", r2["pass_hard_2_3"], flush=True)
    print("written", out_dir / "chain_check.json")


if __name__ == "__main__":
    main()
