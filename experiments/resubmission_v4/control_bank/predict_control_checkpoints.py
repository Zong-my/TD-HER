#!/usr/bin/env python3
"""Protocol B (v2), step 1: replay gate and frozen inference of the existing control checkpoints.

No training. The L1 checkpoints of the multi-task controls are rebuilt with the constructors
of the original scripts (imported unchanged), loaded strictly, and evaluated on six splits
with the precision path they had when their predictions were saved. Before anything else is
written, each checkpoint must reproduce its own saved L1 test predictions within a
predeclared tolerance.

  --smoke   replay one checkpoint (fixed loss, seed 42) and stop
  (default) identity checks, replay gate for all 18 checkpoints, predictions on six splits
"""
from __future__ import annotations

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
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = PROJECT_ROOT / "experiments"
sys.path.insert(0, str(PROJECT_ROOT))
# exp_r2_mmoe_baseline imports exp_r2_mtl_baselines by its bare name from the experiments
# directory; that directory must come first so that the stray copy in the project root is not used.
sys.path.insert(0, str(EXP_DIR))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_ROOT)

import exp_r2_mtl_baselines as M  # noqa: E402
import exp_r2_mmoe_baseline as MM  # noqa: E402
from data_proc.datasets import TensorDataset  # noqa: E402

OUT_BASE = work_path('results/ieee39/resubmission_v4_control_bank')
EVENT_INDEX = (WORK_ROOT / "analytical"
               / "execution/shared/event_index.csv")
SPLITS = ("val", "test", "cross_cond_finetune", "cross_cond_test",
          "cross_cond_topo_finetune", "cross_cond_topo_test")
SCEN_OF_TEST = {"test": "L1", "cross_cond_test": "L2", "cross_cond_topo_test": "L3"}
TOL_MEAN_ABS_REL = 0.01   # mean |new - saved| <= 1 % of the checkpoint's saved test MAE
TOL_MAE_REL = 0.005       # |MAE_new - MAE_saved| <= 0.5 % of the saved MAE
SEEDS = (42, 52, 62)


def checkpoints() -> list[dict]:
    out = []
    for strategy in ("vanilla", "uncertainty", "gradnorm", "pcgrad"):
        for seed in SEEDS:
            out.append({"name": f"{strategy}_s{seed}", "family": "patchtst", "strategy": strategy, "seed": seed,
                        "dir": PROJECT_ROOT / f"results/ieee39/revision_r2_mtl_exact/{strategy}/seed_{seed}/L1"})
    for seed in SEEDS:
        out.append({"name": f"mmoe_s{seed}", "family": "mmoe", "strategy": "vanilla", "seed": seed,
                    "dir": PROJECT_ROOT / f"results/ieee39/revision_r2_mmoe_exact/mmoe/seed_{seed}/L1"})
    for seed in SEEDS:
        out.append({"name": f"wide_s{seed}", "family": "wide", "strategy": "vanilla", "seed": seed,
                    "dir": PROJECT_ROOT / f"results/ieee39/resubmission_v3_mtl_capacity_matched/vanilla/seed_{seed}/L1"})
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh_dir(base: Path) -> Path:
    v = 1
    while (base.parent / f"{base.name}_v{v}").exists():
        v += 1
    out = base.parent / f"{base.name}_v{v}"
    out.mkdir(parents=True)
    return out


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def build_model(entry: dict, data: dict, ckpt: dict):
    cfg = {"model": dict(ckpt["run_config"]["model"])}
    if entry["family"] == "mmoe":
        if ckpt["run_config"].get("architecture") != MM.ARCHITECTURE:
            raise SystemExit(f"{entry['name']}: saved architecture differs from the script's")
        model = MM.make_model(data, cfg)
    else:
        model = M.make_model(data, cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model


def make_trainer(entry: dict, model, ckpt: dict, device: str):
    rc = ckpt["run_config"]
    # precision path of the original run: AMP for fixed loss (also MMoE-head and the wide control),
    # FP32 for uncertainty weighting, GradNorm and PCGrad -- decided by the trainer from the strategy
    return M.ConflictAwareTrainer(
        model, entry["strategy"], device, rc["lr"], rc["weight_decay"], rc["batch_size"],
        1, 1, 8, base_task_weights=(rc["y1_weight"], rc["y2_weight"]))


def identity_checks(log, data: dict, shift: dict, rep_dir: Path) -> dict:
    """Row order of the six splits across the two representations and the common source."""
    report = {"rep_dir_controls": M.REP_DIR}
    audit = json.loads((work_path('results/ieee39/exp_tdh_router/audit_report.json')).read_text())
    report["rep_dir_router"] = audit["rep_dir"]
    if str(Path(audit["rep_dir"])) != str(Path(M.REP_DIR)):
        raise SystemExit("router and controls read different representation directories")
    names = json.loads((rep_dir / "repA/ms10/feature_names.json").read_text())
    meta = json.loads((rep_dir / "repB/ms10/meta.json").read_text())
    sheets, buses, steps = meta["sheets"], meta["buses"], meta["timesteps"]
    col = {n: i for i, n in enumerate(names)}
    import pandas as pd
    ev = {}
    with open(EVENT_INDEX) as f:
        for r in csv.DictReader(f):
            ev[(r["scenario"], int(r["row_index"]))] = r["source_filename"]
    report["splits"] = {}
    for sp in SPLITS:
        ya = np.load(rep_dir / f"repA/ms10/y_{sp}.npy")
        yb = np.load(rep_dir / f"repB/ms10/y_{sp}.npy")
        xa = np.load(rep_dir / f"repA/ms10/X_{sp}.npy", mmap_mode="r")
        xb = np.load(rep_dir / f"repB/ms10/X_temporal_{sp}.npy", mmap_mode="r")
        labels_equal = bool(ya.shape == yb.shape and np.array_equal(ya, yb))
        # every raw temporal column of the tabular representation against the tensor, row by row
        max_diff, n_cols = 0.0, 0
        for ti, t in enumerate(steps):
            for ni, b in enumerate(buses):
                for ci, s in enumerate(sheets):
                    j = col.get(f"{s}_{b}_{t}")
                    if j is None:
                        raise SystemExit(f"column {s}_{b}_{t} missing in the tabular representation")
                    d = float(np.max(np.abs(np.asarray(xa[:, j], dtype=np.float64) - np.asarray(xb[:, ti, ni, ci], dtype=np.float64))))
                    max_diff = max(max_diff, d); n_cols += 1
        # common source: per-split CSV (file name per row)
        df = pd.read_csv(rep_dir / f"csv/{sp}_ms10.csv", usecols=["file_name", "distu_kind", "fpu_deltamax", "t_delta"])
        csv_rows_equal = bool(len(df) == len(ya))
        csv_targets_max = float(np.max(np.abs(df[["fpu_deltamax", "t_delta"]].values.astype(np.float32) - ya))) if csv_rows_equal else None
        entry = {"n": int(len(ya)), "labels_equal_across_representations": labels_equal,
                 "unique_label_rows": int(len(np.unique(ya, axis=0))),
                 "temporal_columns_compared": n_cols, "max_abs_diff_tabular_vs_tensor": max_diff,
                 "csv_rows": int(len(df)), "csv_targets_max_abs_diff_vs_labels": csv_targets_max,
                 "sha256_y": sha(rep_dir / f"repA/ms10/y_{sp}.npy")}
        if sp in SCEN_OF_TEST:
            sc = SCEN_OF_TEST[sp]
            ev_names = [ev.get((sc, i)) for i in range(len(df))]
            entry["csv_file_names_equal_event_index"] = bool(ev_names == [os.path.basename(str(x)) for x in df["file_name"].tolist()]
                                                             or ev_names == df["file_name"].astype(str).tolist())
        # the arrays the control loader actually feeds: labels of this split
        y_loader = data[f"y_{sp}"] if sp in ("val", "test") else shift[sp]["y"]
        entry["control_loader_labels_equal"] = bool(np.array_equal(y_loader, ya))
        report["splits"][sp] = entry
        log(f"  identity {sp:26s} n={entry['n']:6d} labels A=B {labels_equal}; tabular=tensor max|d| {max_diff:.1e} over {n_cols} columns; "
            f"csv rows {entry['csv_rows']}, targets max|d| {csv_targets_max}; file names = event index: {entry.get('csv_file_names_equal_event_index', 'n/a')}")
        ok = labels_equal and max_diff == 0.0 and csv_rows_equal and entry["control_loader_labels_equal"] \
            and entry.get("csv_file_names_equal_event_index", True)
        if not ok:
            raise SystemExit(f"identity check failed for split {sp}")
    report["note"] = ("Both representations of a split are written from one data frame per split "
                      "(data_proc/build_representations.py, process_csv_to_reps): same rows, same order. "
                      "Checked here: every raw temporal column of the tabular array equals the tensor entry row by row; "
                      "the per-split CSV, the common source, has the same number of rows; for the three test splits its "
                      "file names equal the event index of 2026-09-06 row by row.")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if Path(M.__file__).resolve() != (EXP_DIR / "exp_r2_mtl_baselines.py").resolve():
        raise SystemExit(f"wrong control module imported: {M.__file__}")

    out_dir = fresh_dir(OUT_BASE.parent / (OUT_BASE.name + ("_smoke" if args.smoke else "")))
    log = Tee(out_dir / "replay_and_predict.log")
    t0 = time.time()
    log("Protocol B v2, step 1:", "SMOKE (one checkpoint)" if args.smoke else "full")
    log("output directory:", out_dir)
    log("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    M.set_seed(0)
    data = M.load_fixed_data()
    shift = {sp: M.load_shift_split(sp, data) for sp in SPLITS if sp not in ("val", "test")}
    rep_dir = Path(M.REP_DIR)

    datasets = {
        "val": TensorDataset(data["Xt_val_n"], data["Xs_val_n"], data["y_val_n"]),
        "test": TensorDataset(data["Xt_test_n"], data["Xs_test_n"], data["y_test"]),
    }
    for sp, d in shift.items():
        datasets[sp] = TensorDataset(d["Xt_n"], d["Xs_n"], d["y"])
    y_true = {"val": data["y_val"], "test": data["y_test"], **{sp: d["y"] for sp, d in shift.items()}}

    identity = None
    if not args.smoke:
        log("identity checks")
        identity = identity_checks(log, data, shift, rep_dir)

    entries = checkpoints()
    if args.smoke:
        entries = [e for e in entries if e["name"] == "vanilla_s42"]
    report, failed = {}, []
    for e in entries:
        ckpt_path = e["dir"] / "model.pth"
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model = build_model(e, data, ckpt)
        n_params = int(sum(p.numel() for p in model.parameters()))
        trainer = make_trainer(e, model, ckpt, args.device)
        saved = np.load(e["dir"] / "preds.npy").astype(np.float64)
        saved_metrics = json.loads((e["dir"] / "metrics.json").read_text())
        new_test = data["scaler_y"].inverse_transform(trainer.predict(datasets["test"])).astype(np.float64)
        y = y_true["test"].astype(np.float64)
        rep = {"checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)), "sha256": sha(ckpt_path),
               "family": e["family"], "strategy": e["strategy"], "seed": e["seed"],
               "parameters": n_params, "amp": bool(trainer.use_amp), "targets": {}}
        ok = saved.shape == new_test.shape == y.shape
        for t, tn in enumerate(("y1", "y2")):
            mae_saved = float(np.mean(np.abs(saved[:, t] - y[:, t])))
            mae_new = float(np.mean(np.abs(new_test[:, t] - y[:, t])))
            mean_abs = float(np.mean(np.abs(new_test[:, t] - saved[:, t])))
            max_abs = float(np.max(np.abs(new_test[:, t] - saved[:, t])))
            passed = (mean_abs <= TOL_MEAN_ABS_REL * mae_saved) and (abs(mae_new - mae_saved) <= TOL_MAE_REL * mae_saved)
            consistent = abs(mae_saved - saved_metrics[f"{tn}_MAE"]) <= 1e-6 * max(1.0, mae_saved)
            rep["targets"][tn] = {"MAE_saved": mae_saved, "MAE_saved_metrics_json": saved_metrics[f"{tn}_MAE"],
                                  "MAE_replayed": mae_new, "mean_abs_diff": mean_abs, "max_abs_diff": max_abs,
                                  "mean_abs_diff_over_MAE": mean_abs / mae_saved,
                                  "MAE_rel_change": (mae_new - mae_saved) / mae_saved,
                                  "passed": bool(passed), "saved_metrics_consistent": bool(consistent)}
            ok = ok and passed and consistent
        rep["passed"] = bool(ok)
        report[e["name"]] = rep
        log(f"  replay {e['name']:16s} params {n_params:9,d} amp={trainer.use_amp!s:5s} "
            + "; ".join(f"{tn}: MAE {r['MAE_saved']:.6f}->{r['MAE_replayed']:.6f}, mean|d|/MAE {r['mean_abs_diff_over_MAE']:.2e}, max|d| {r['max_abs_diff']:.2e}"
                        for tn, r in rep["targets"].items())
            + ("  PASS" if ok else "  FAIL"))
        if not ok:
            failed.append(e["name"])
        elif not args.smoke:
            pdir = out_dir / "preds" / e["name"]
            pdir.mkdir(parents=True)
            np.save(pdir / "test_preds.npy", new_test)
            for sp in SPLITS:
                if sp == "test":
                    continue
                p = data["scaler_y"].inverse_transform(trainer.predict(datasets[sp])).astype(np.float64)
                if p.shape != y_true[sp].shape or not np.isfinite(p).all():
                    raise SystemExit(f"{e['name']}: bad prediction array on {sp}")
                np.save(pdir / f"{sp}_preds.npy", p)
            rep["frozen_MAE_by_split"] = {sp: [float(v) for v in np.mean(np.abs(np.load(pdir / f"{sp}_preds.npy") - y_true[sp]), axis=0)]
                                          for sp in SPLITS}
        del trainer, model, ckpt
        torch.cuda.empty_cache()

    summary = {
        "protocol": "B v2 step 1 (claude_protocols_stage2_v2_20261005.md)",
        "smoke": bool(args.smoke),
        "tolerance": {"mean_abs_diff_over_saved_MAE": TOL_MEAN_ABS_REL, "MAE_rel_change": TOL_MAE_REL,
                      "bitwise_identity_required": False},
        "all_passed": not failed, "failed": failed,
        "identity": identity, "checkpoints": report,
        "module_sha256": {"experiments/exp_r2_mtl_baselines.py": sha(EXP_DIR / "exp_r2_mtl_baselines.py"),
                          "experiments/exp_r2_mmoe_baseline.py": sha(EXP_DIR / "exp_r2_mmoe_baseline.py"),
                          "models/patchtst_mtl_model.py": sha(PROJECT_ROOT / "models/patchtst_mtl_model.py"),
                          "this_script": sha(Path(__file__))},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                        "numpy": np.__version__, "platform": platform.platform()},
        "wall_seconds": time.time() - t0,
    }
    (out_dir / "replay_report.json").write_text(json.dumps(summary, indent=2))
    if failed:
        log("REPLAY GATE FAILED for:", failed, "-- the item stops here; nothing is retrained and no checkpoint is dropped.")
        raise SystemExit(2)
    log("replay gate passed for", len(report), "checkpoint(s);", "smoke only, no predictions written" if args.smoke else "predictions written for six splits")
    log("written to", out_dir)


if __name__ == "__main__":
    main()
