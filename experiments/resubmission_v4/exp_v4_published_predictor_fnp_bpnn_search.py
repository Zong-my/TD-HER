#!/usr/bin/env python3
"""FNP-BPNN search and final phases — separate additive driver revision (2026-10-14, rev 2: own target loader, bound provenance).

The extraction driver (exp_v4_published_predictor_fnp_bpnn.py) stays frozen; this driver imports
its helpers read-only and adds the NEW validation-selection mask of
protocols/comparator_fnp_bpnn_addendum_v1.json: a validation row whose workbook sha256 occurs among
the eligible training rows is excluded from model selection (input identity only).

Phases
  search  verify the completed extraction (non-smoke), build and record the mask, run the benchmark
          grid on train/val (no test arrays)
  final   (after the search gate) fit the selected settings on the training rows; evaluate on the
          frozen common cohorts with the stored test FNP predictions as inputs; L2/L3 frozen and
          retrained (train + extracted calibration rows) variants
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
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_v4_published_predictor_fnp_bpnn as X  # frozen extraction driver, used read-only  # noqa: E402

ADDENDUM = REPO_ROOT / "configs/comparator_fnp_bpnn_addendum_v1.json"
DRIVER_PATH = Path(__file__).resolve()
SCENARIO_SPLITS = {"L1": {"cal": None, "test": "test"},
                   "L2": {"cal": "cross_cond_finetune", "test": "cross_cond_test"},
                   "L3": {"cal": "cross_cond_topo_finetune", "test": "cross_cond_topo_test"}}


def provenance(protocol: dict, phase: str, **extra) -> dict:
    p = X.runtime_provenance(protocol)
    p.update({"phase": phase, "search_driver_path": str(DRIVER_PATH.relative_to(X.PROJECT_ROOT)),
              "search_driver_sha256": X.sha256_file(DRIVER_PATH), "extraction_driver_sha256": p["driver_sha256"],
              "driver_sha256": X.sha256_file(DRIVER_PATH),
              "addendum_id": json.loads(ADDENDUM.read_text())["addendum_id"], "addendum_sha256": X.sha256_file(ADDENDUM)})
    p.update(extra)
    return p


def build_validation_mask(train_rows: pd.DataFrame, val_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Exclude validation rows whose workbook hash occurs among the eligible training rows (identity only)."""
    train_hashes = set(train_rows["workbook_sha256"])
    excluded = val_rows[val_rows["workbook_sha256"].isin(train_hashes)]
    kept = val_rows[~val_rows["workbook_sha256"].isin(train_hashes)]
    record = {
        "rule": "exclude validation rows whose workbook_sha256 occurs in the eligible training rows (input identity only)",
        "n_val_before": int(len(val_rows)), "n_val_excluded": int(len(excluded)), "n_val_used": int(len(kept)),
        "n_train": int(len(train_rows)),
        "excluded": [{"val_row_index": int(r.row_index), "val_file_name": r.file_name, "workbook_sha256": r.workbook_sha256,
                      "matching_train_row_indices": [int(x) for x in train_rows.loc[train_rows["workbook_sha256"] == r.workbook_sha256, "row_index"]],
                      "y1": float(r.y1), "y2": float(r.y2)} for r in excluded.itertuples()],
    }
    return kept, record


def phase_search(args, protocol: dict) -> None:
    out_dir = args.out_dir / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        verification = {s: X.verify_extraction(args.out_dir, s) for s in ("train", "val")}
        X.write_json(out_dir / "extraction_verification.json", verification)
    tr = X.load_training_rows(args.out_dir, "train")
    va_all = X.load_training_rows(args.out_dir, "val")
    va, mask = build_validation_mask(tr, va_all)
    X.write_json(out_dir / "validation_selection_mask.json", {**mask, "addendum_sha256": X.sha256_file(ADDENDUM)})
    if len(tr) < 4 or len(va) < 2:
        raise RuntimeError(f"too few eligible rows: train {len(tr)}, val {len(va)}")
    Xtr, Ytr = tr[["x1", "x2"]].to_numpy(float), tr[["y1", "y2"]].to_numpy(float)
    Xva, Yva = va[["x1", "x2"]].to_numpy(float), va[["y1", "y2"]].to_numpy(float)
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-12
    my, sy = Ytr.mean(0), Ytr.std(0) + 1e-12
    grid = protocol["model"]["candidate_grid"]
    max_iter = int(grid["max_iter_smoke"] if args.smoke else grid["max_iter_full"])
    rows = []
    ref = np.mean(np.abs(Xva - Yva), axis=0)
    rows.append({"candidate": "FNP_identity_reference", "hidden": "", "lr": "", "seed": "",
                 "val_mae_y1_hz": float(ref[0]), "val_mae_y2_s": float(ref[1]),
                 "val_objective": float(np.mean(ref / sy)), "fit_seconds": 0.0, "n_iter": ""})
    for hidden in grid["hidden_units"]:
        for lr in grid["learning_rate_init"]:
            t0 = time.time()
            model = X.fit_bpnn((Xtr - mx) / sx, (Ytr - my) / sy, int(hidden), float(lr), max_iter, args.seed)
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
    X.write_json(out_dir / "best_settings.json", {
        **best, "label": "benchmark settings selected on the validation split; the paper fixes none",
        "n_train_rows": int(len(tr)), "n_val_rows": int(len(va)), "n_val_rows_before_mask": int(len(va_all)),
        "n_val_excluded": int(mask["n_val_excluded"]), "max_iter": max_iter,
        "standardization": {"x_mean": mx.tolist(), "x_std": sx.tolist(), "y_mean": my.tolist(), "y_std": sy.tolist()},
        "mask_sha256": X.sha256_file(out_dir / "validation_selection_mask.json"),
    })
    X.write_json(out_dir / "provenance.json", provenance(protocol, "search", smoke=bool(args.smoke),
                                                        files_read=list(X._FILES_READ), max_iter=max_iter))
    X.assert_no_test_reads()
    logger.info(f"best: {best}; validation rows used {len(va)} of {len(va_all)}")


def load_targets(split: str) -> np.ndarray:
    """Read-only target loader (repA/ms10 y arrays, float64); test arrays are read only in the final phase."""
    return np.load(X.ACTIVE / "repA" / f"ms{X.MS}" / f"y_{split}.npy").astype(np.float64)


def load_cohort_rows(scenario: str, protocol: dict) -> np.ndarray:
    return X.load_common_cohort(scenario, protocol)


def load_rows(root: Path, split: str) -> pd.DataFrame:
    return X.load_training_rows(root, split)


def load_stored_test_inputs(scenario: str, rows: np.ndarray) -> np.ndarray:
    ap = pd.read_csv(X.SHARED / "analytical/physical_causal_sfr_fnp_v2/analytical_predictions.csv")
    ap = ap[(ap["scenario"] == scenario) & (ap["method"] == "FNP") & (ap["prediction_status"] == "OK")].set_index("row_index")
    missing = [int(r) for r in rows if r not in ap.index]
    if missing:
        raise RuntimeError(f"{scenario}: {len(missing)} common-cohort rows lack a stored FNP prediction")
    return ap.loc[rows, ["delta_f_hat_hz", "t_extremum_hat_rel_s"]].to_numpy(float)


def _check_matrix(name: str, arr: np.ndarray, n: int) -> None:
    if arr.shape != (n, 2) or not np.isfinite(arr).all():
        raise RuntimeError(f"{name}: expected finite shape ({n}, 2), got {arr.shape}")


def phase_final(args, protocol: dict) -> None:
    import joblib
    search_dir = args.out_dir / "search"
    out_dir = args.out_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path, mask_path = search_dir / "best_settings.json", search_dir / "validation_selection_mask.json"
    best = json.loads(best_path.read_text())
    hidden, lr, max_iter = int(best["hidden"]), float(best["lr"]), int(best["max_iter"])
    binding = {"best_settings_sha256": X.sha256_file(best_path), "validation_mask_sha256": X.sha256_file(mask_path),
               "selected": {"candidate": best["candidate"], "hidden": hidden, "lr": lr, "max_iter": max_iter}}
    tr = load_rows(args.out_dir, "train")
    Xtr, Ytr = tr[["x1", "x2"]].to_numpy(float), tr[["y1", "y2"]].to_numpy(float)
    _check_matrix("training inputs", Xtr, len(tr)); _check_matrix("training targets", Ytr, len(tr))
    expected = protocol["evaluation"]["common_cohort"]["expected_counts"]
    record = {"files": {rel: X.sha256_file(X.PROJECT_ROOT / rel) for rel in protocol["evaluation"]["common_cohort"]["source_files"]},
              "scenarios": {}}
    labels, inputs = {}, {}
    for s, v in SCENARIO_SPLITS.items():
        y = load_targets(v["test"])
        crows = np.asarray(load_cohort_rows(s, protocol))
        if (len(crows) != int(expected[s]) or len(np.unique(crows)) != len(crows)
                or crows.min() < 0 or crows.max() >= len(y)):
            raise RuntimeError(f"common cohort {s}: {len(crows)} rows (expected {expected[s]}); unique/in-range check failed")
        np.save(out_dir / f"common_cohort_rows_{s}.npy", crows)
        record["scenarios"][s] = {"n": int(len(crows)), "sha256_rows": hashlib.sha256(crows.tobytes()).hexdigest()}
        labels[s] = y[crows]; inputs[s] = load_stored_test_inputs(s, crows)
        _check_matrix(f"{s} stored test inputs", inputs[s], len(crows)); _check_matrix(f"{s} labels", labels[s], len(crows))
    X.write_json(out_dir / "common_cohort_record.json", record)
    cal_rows = {s: load_rows(args.out_dir, v["cal"]) for s, v in SCENARIO_SPLITS.items() if v["cal"]}
    rows, n_models = [], 0
    for seed in args.seeds:
        fits = {"frozen": (Xtr, Ytr, int(len(tr)), 0)}
        for s, c in cal_rows.items():
            Xc, Yc = c[["x1", "x2"]].to_numpy(float), c[["y1", "y2"]].to_numpy(float)
            _check_matrix(f"{s} calibration inputs", Xc, len(c))
            fits[f"retrained_{s}"] = (np.vstack([Xtr, Xc]), np.vstack([Ytr, Yc]), int(len(tr)), int(len(c)))
        models = {}
        for name, (Xf, Yf, n_tr, n_cal) in fits.items():
            mx, sx = Xf.mean(0), Xf.std(0) + 1e-12
            my, sy = Yf.mean(0), Yf.std(0) + 1e-12
            t0 = time.time()
            model = X.fit_bpnn((Xf - mx) / sx, (Yf - my) / sy, hidden, lr, max_iter, seed)
            mdir = out_dir / "models" / f"seed_{seed}" / name
            mdir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, mdir / "model.joblib")
            scaler = {"x_mean": mx.tolist(), "x_std": sx.tolist(), "y_mean": my.tolist(), "y_std": sy.tolist()}
            X.write_json(mdir / "scaler.json", scaler)
            X.write_json(mdir / "fit.json", {"fit": name, "seed": seed, "n_train_rows": n_tr, "n_cal_rows": n_cal,
                                              "n_rows": int(len(Xf)), "hidden": hidden, "lr": lr, "max_iter": max_iter,
                                              "n_iter": int(model.n_iter_), "final_loss": float(model.loss_),
                                              "fit_seconds": time.time() - t0, **binding})
            models[name] = (model, scaler, n_tr, n_cal); n_models += 1
        for s in SCENARIO_SPLITS:
            modes = {"frozen": ("frozen", models["frozen"])}
            if f"retrained_{s}" in models:
                modes["retrained"] = (f"retrained_{s}", models[f"retrained_{s}"])
            for mode, (fit_name, (model, scaler, n_tr, n_cal)) in modes.items():
                mx, sx = np.asarray(scaler["x_mean"]), np.asarray(scaler["x_std"])
                my, sy = np.asarray(scaler["y_mean"]), np.asarray(scaler["y_std"])
                pred = model.predict((inputs[s] - mx) / sx) * sy + my
                _check_matrix(f"{s} {mode} predictions", pred, len(labels[s]))
                err = pred - labels[s]
                run_dir = out_dir / f"seed_{seed}" / f"{s}_{mode}"
                run_dir.mkdir(parents=True, exist_ok=True)
                np.save(run_dir / "preds.npy", pred)
                metrics = {"seed": seed, "scenario": s, "mode": mode, "fit": fit_name, "n_common": int(len(labels[s])),
                           "y1_MAE_common": float(np.mean(np.abs(err[:, 0]))), "y2_MAE_common": float(np.mean(np.abs(err[:, 1]))),
                           "y1_RMSE_common": float(np.sqrt(np.mean(err[:, 0] ** 2))), "y2_RMSE_common": float(np.sqrt(np.mean(err[:, 1] ** 2))),
                           "n_train_rows": n_tr, "n_cal_rows": n_cal, "hidden": hidden, "lr": lr, "max_iter": max_iter}
                X.write_json(run_dir / "metrics.json", metrics)
                X.write_json(run_dir / "config.json", {**metrics, "model_dir": str(out_dir / "models" / f"seed_{seed}" / fit_name),
                                                      "scaler": scaler, "driver_sha256": X.sha256_file(DRIVER_PATH), **binding})
                rows.append(metrics)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "all_runs.csv", index=False)
    cols = ["y1_MAE_common", "y2_MAE_common", "y1_RMSE_common", "y2_RMSE_common"]
    summ = frame.groupby(["scenario", "mode"])[cols].agg(["mean", "std", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(out_dir / "summary.csv", index=False)
    X.write_json(out_dir / "provenance.json", provenance(protocol, "final", seeds=args.seeds, files_read=list(X._FILES_READ),
                                                        n_models_saved=n_models, n_evaluations=len(rows), **binding,
                                                        printed_references_frozen=protocol["evaluation"]["printed_references_frozen"]))
    logger.info(f"final: {n_models} models, {len(rows)} evaluations written to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=("search", "final"), required=True)
    ap.add_argument("--protocol", type=Path, default=X.DEFAULT_PROTOCOL)
    ap.add_argument("--out-dir", type=Path, default=X.DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    protocol = X.load_protocol(args.protocol)
    if args.seeds is None:
        args.seeds = list(protocol["model"]["seeds"]["final"])
    if args.smoke and not str(args.out_dir).endswith("_smoke"):
        args.out_dir = Path(str(args.out_dir) + "_smoke")
    for never in protocol["never_write"]:
        if str(args.out_dir.resolve()).startswith(str((X.PROJECT_ROOT / never).resolve())):
            raise RuntimeError(f"output directory {args.out_dir} lies inside a frozen path {never}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(args.out_dir / f"{args.phase}_v2.log", enqueue=True)
    logger.info(f"phase={args.phase} out_dir={args.out_dir} addendum={ADDENDUM}")
    {"search": phase_search, "final": phase_final}[args.phase](args, protocol)


if __name__ == "__main__":
    main()
