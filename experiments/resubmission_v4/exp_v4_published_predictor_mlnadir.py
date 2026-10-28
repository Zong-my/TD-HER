#!/usr/bin/env python3
"""Published learning-based nadir predictor (MLNadir decision-tree family) on the
TD-HER benchmark. Additive research phase 2026-10-14, task 1.

Reference: Jimenez-Navarro et al., Neurocomputing 626 (2025) 129583; official
code github.com/manjimnav/MLNadir. Inputs are the unwrapped voltage phase angles
at the ten generator buses over the stored 100 ms window (11 samples); the
regression target is the absolute centre-of-inertia frequency 60 + y1 (Hz).

Phases
  describe  angle-unit statistics, cohort sizes (train/val native cohort and the
            frozen common test cohorts from the stored status files); no training
  search    per prespecified variant (cohort x context) a 20-trial search over the
            official decision-tree space; selection by validation MAE on the fixed
            validation split; loads ONLY the train and val arrays
  final     fit the selected parameters on train (seeds), evaluate on the L1 test
            split and the frozen common cohort; L2/L3 frozen and retrained
            variants (reads test arrays; run only after the search gate)

Usage (repository root, fepfm interpreter):
  python experiments/resubmission_v4/exp_v4_published_predictor_mlnadir.py --phase describe
  python experiments/resubmission_v4/exp_v4_published_predictor_mlnadir.py --phase search --smoke
  python experiments/resubmission_v4/exp_v4_published_predictor_mlnadir.py --phase search
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
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = WORK_ROOT
DEFAULT_PROTOCOL = ARTIFACT_ROOT / "protocols/comparator_mlnadir.json"
DEFAULT_OUT = work_path('results/ieee39/resubmission_v4_published_predictor_mlnadir_v1')
REP_DIR = Path(str(data_path('ieee39_v8_80_10_10')))
HARNESS = work_path("analytical/shared")
DRIVER_PATH = Path(__file__).resolve()
SCENARIO_SPLITS = {
    "L1": {"cal": None, "test": "test"},
    "L2": {"cal": "cross_cond_finetune", "test": "cross_cond_test"},
    "L3": {"cal": "cross_cond_topo_finetune", "test": "cross_cond_topo_test"},
}

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
        import optuna
        optuna_version = optuna.__version__
    except Exception:
        optuna_version = None
    try:
        import skopt
        skopt_version = skopt.__version__
    except Exception:
        skopt_version = None
    return {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "optuna": optuna_version,
        "skopt": skopt_version,
        "git_head": git_head(),
        "driver_path": str(DRIVER_PATH.relative_to(PROJECT_ROOT)),
        "driver_sha256": sha256_file(DRIVER_PATH),
        "protocol_id": protocol.get("protocol_id"),
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, default=str)


def max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# --------------------------------------------------------------------------
# protocol, data and features
# --------------------------------------------------------------------------
def load_protocol(path: Path) -> dict:
    with open(path) as handle:
        protocol = json.load(handle)
    for key in ("inputs", "target", "variants", "model", "search", "final",
                "evaluation", "never_write"):
        if key not in protocol:
            raise KeyError(f"protocol lacks '{key}'")
    return protocol


def sheet_index() -> int:
    with open(REP_DIR / "repB/ms10/meta.json") as handle:
        sheets = json.load(handle)["sheets"]
    return sheets.index("ANGL")


def load_temporal(split: str) -> np.ndarray:
    return np.load(REP_DIR / "repB/ms10" / f"X_temporal_{split}.npy")


def load_targets(split: str) -> np.ndarray:
    # Stored labels are float32. Cast to float64 BEFORE any arithmetic with the
    # 60 Hz base so that (60 + y1) - 60 reproduces y1 to double precision
    # (Codex preliminary review, 2026-10-14). Stored files are unchanged.
    return np.load(REP_DIR / "repA/ms10" / f"y_{split}.npy").astype(np.float64)


def angle_stats(X: np.ndarray, idx: int) -> dict:
    deg = np.asarray(X[:, :, :, idx], dtype=np.float64)
    step = np.abs(np.diff(deg, axis=1))
    return {
        "n_events": int(len(deg)),
        "min_deg": float(deg.min()), "max_deg": float(deg.max()),
        "p01_deg": float(np.percentile(deg, 1)), "p50_deg": float(np.percentile(deg, 50)),
        "p99_deg": float(np.percentile(deg, 99)),
        "frac_abs_gt_180": float((np.abs(deg) > 180).mean()),
        "events_abs_gt_360": int((np.abs(deg).max(axis=(1, 2)) > 360).sum()),
        "max_abs_step_deg": float(step.max()),
        "steps_gt_180_deg": int((step > 180).sum()),
        "unwrap_is_noop": bool((step > 180).sum() == 0),
    }


def angle_features(X: np.ndarray, idx: int) -> np.ndarray:
    rad = np.deg2rad(np.asarray(X[:, :, :, idx], dtype=np.float64))
    rad = np.unwrap(rad, axis=1)          # per event, along time only
    return rad.reshape(len(rad), -1)


def cohort_mask(y: np.ndarray, cohort: str) -> np.ndarray:
    if cohort == "native_downward":
        return y[:, 0] < 0
    if cohort == "all_events":
        return np.ones(len(y), dtype=bool)
    raise ValueError(f"unknown cohort {cohort}")


def augment(X: np.ndarray, y_abs: np.ndarray, context: str, protocol: dict,
            seed: int) -> tuple[np.ndarray, np.ndarray]:
    if context == "context2":
        return X, y_abs
    if context == "context7":
        sigma = float(protocol.get("augmentation_sigma_rad", 0.01))
        rng = np.random.default_rng(seed)
        noisy = X + rng.normal(0.0, sigma, size=X.shape)
        return np.vstack([X, noisy]), np.concatenate([y_abs, y_abs])
    raise ValueError(f"unknown context {context}")


def variant_pairs(protocol: dict) -> list[tuple[str, str]]:
    pairs = []
    for name in protocol["variants"]["prespecified_pairs"]:
        cohort, context = name.split("__")
        pairs.append((cohort, context))
    return pairs


def make_tree(params: dict, seed: int):
    from sklearn.tree import DecisionTreeRegressor
    return DecisionTreeRegressor(
        splitter=params["splitter"], max_depth=int(params["max_depth"]),
        criterion=params["criterion"], random_state=seed,
    )


def fit_eval(Xtr, ytr_abs, Xva, yva_abs, params: dict, seed: int):
    start = time.time()
    model = make_tree(params, seed).fit(Xtr, ytr_abs)
    pred = model.predict(Xva)
    mae = float(np.mean(np.abs(pred - yva_abs)))
    return mae, time.time() - start, model


# --------------------------------------------------------------------------
# frozen common cohort (status files only; no test arrays)
# --------------------------------------------------------------------------
def load_common_cohort(scenario: str, protocol: dict) -> np.ndarray:
    rule = protocol["evaluation"]["common_cohort"]
    ap = pd.read_csv(HARNESS / "analytical/physical_causal_sfr_fnp_v2/analytical_predictions.csv")
    ap = ap[ap["scenario"] == scenario]
    tc = None
    keep = None
    for cond in rule["conditions"]:
        if cond["file"] == "analytical_predictions":
            ids = set(ap[(ap["method"] == cond["method"])
                         & (ap["prediction_status"] == cond["status"])]["event_id"])
        elif cond["file"] == "target_compatibility":
            if tc is None:
                tc = pd.read_csv(HARNESS / "eval_truth/truth_v1/target_compatibility.csv")
            col = tc[cond["column"]].astype(str).str.lower()
            ids = set(tc[col == str(cond["value"]).lower()]["event_id"])
        else:
            raise ValueError(f"unknown cohort source {cond['file']}")
        keep = ids if keep is None else keep & ids
    ei = pd.read_csv(HARNESS / "event_index.csv")
    ei = ei[ei["scenario"] == scenario]
    rows = np.sort(ei[ei["event_id"].isin(keep)]["row_index"].to_numpy())
    return rows


# --------------------------------------------------------------------------
# search backends
# --------------------------------------------------------------------------
def search_optuna(name: str, Xtr, ytr_abs, Xva, yva_abs, protocol: dict, args,
                  out_dir: Path) -> dict:
    import optuna
    space = protocol["model"]["search_space"]
    out_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=name, storage=f"sqlite:///{out_dir / 'study.db'}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
        load_if_exists=True,
    )

    def objective(trial):
        params = {
            "splitter": trial.suggest_categorical("splitter", list(space["splitter"])),
            "max_depth": trial.suggest_int("max_depth", int(space["max_depth"]["low"]),
                                           int(space["max_depth"]["high"]),
                                           step=int(space["max_depth"]["step"])),
            "criterion": trial.suggest_categorical("criterion", list(space["criterion"])),
        }
        mae, secs, _ = fit_eval(Xtr, ytr_abs, Xva, yva_abs, params, args.seed)
        trial.set_user_attr("fit_seconds", secs)
        logger.info(f"{name} trial {trial.number}: val MAE {mae:.6f} Hz ({secs:.1f}s) {params}")
        return mae

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, args.n_trials - done)
    if remaining:
        study.optimize(objective, n_trials=remaining)
    frame = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs",
                                          "datetime_start", "datetime_complete", "duration"))
    frame.to_csv(out_dir / "trials.csv", index=False)
    best = study.best_trial
    return {"backend": "optuna_tpe", "params": best.params, "val_mae_hz": float(best.value),
            "fit_seconds": best.user_attrs.get("fit_seconds"),
            "n_trials_complete": int(len([t for t in study.trials
                                          if t.state == optuna.trial.TrialState.COMPLETE]))}


def search_skopt(name: str, Xtr, ytr_abs, Xva, yva_abs, protocol: dict, args,
                 out_dir: Path) -> dict:
    from skopt import BayesSearchCV
    from skopt.space import Categorical
    space = protocol["model"]["search_space"]
    out_dir.mkdir(parents=True, exist_ok=True)
    X = np.vstack([Xtr, Xva])
    y = np.concatenate([ytr_abs, yva_abs])
    cv = [(np.arange(len(Xtr)), np.arange(len(Xtr), len(X)))]
    depths = list(range(int(space["max_depth"]["low"]), int(space["max_depth"]["high"]) + 1,
                        int(space["max_depth"]["step"])))
    opt = BayesSearchCV(
        make_tree({"splitter": "best", "max_depth": 1, "criterion": "squared_error"}, args.seed),
        {"splitter": Categorical(list(space["splitter"])),
         "max_depth": Categorical(depths),
         "criterion": Categorical(list(space["criterion"]))},
        n_iter=args.n_trials, cv=cv, scoring="neg_mean_absolute_error",
        random_state=args.sampler_seed, refit=False, n_jobs=1,
    )
    opt.fit(X, y)
    pd.DataFrame(opt.cv_results_).to_csv(out_dir / "trials.csv", index=False)
    best = dict(opt.best_params_)
    return {"backend": "skopt_bayessearchcv", "params": best,
            "val_mae_hz": float(-opt.best_score_), "fit_seconds": None,
            "n_trials_complete": int(len(opt.cv_results_["params"]))}


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------
def phase_describe(args, protocol: dict) -> None:
    idx = sheet_index()
    yva = load_targets("val")
    ytr = load_targets("train")
    Xva = load_temporal("val")
    stats = angle_stats(Xva, idx)
    kinds = pd.read_csv(REP_DIR / "csv/val_ms10.csv", usecols=["distu_kind"])["distu_kind"]
    native = yva[:, 0] < 0
    cohorts = {}
    for scenario, expected in protocol["evaluation"]["common_cohort"]["expected_counts"].items():
        rows = load_common_cohort(scenario, protocol)
        cohorts[scenario] = {"n": int(len(rows)), "expected": int(expected),
                             "ok": bool(len(rows) == int(expected))}
    assert_no_test_reads()
    out = {
        "protocol_id": protocol["protocol_id"],
        "angle_sheet_index": idx,
        "angle_stats_val": stats,
        "feature_vector_length": int(Xva.shape[1] * Xva.shape[2]),
        "target_abs_min_hz_train": float(60.0 + ytr[:, 0].min()),
        "native_downward_train": int((ytr[:, 0] < 0).sum()), "n_train": int(len(ytr)),
        "native_downward_val": int(native.sum()), "n_val": int(len(yva)),
        "native_downward_val_by_kind": {k: int(v) for k, v in
                                        kinds[native].value_counts().to_dict().items()},
        "common_cohorts": cohorts,
        "files_read": _FILES_READ,
        "provenance": runtime_provenance(protocol),
    }
    print(json.dumps(out, indent=2, default=str))


def phase_search(args, protocol: dict) -> None:
    out_dir = args.out_dir / "search"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = sheet_index()
    t0 = time.time()
    Xtr_raw, Xva_raw = load_temporal("train"), load_temporal("val")
    ytr, yva = load_targets("train"), load_targets("val")
    assert_no_test_reads()
    stats = {"train": angle_stats(Xtr_raw, idx), "val": angle_stats(Xva_raw, idx)}
    write_json(out_dir / "angle_stats.json", stats)
    Ftr, Fva = angle_features(Xtr_raw, idx), angle_features(Xva_raw, idx)
    del Xtr_raw, Xva_raw
    f_base = float(protocol["target"]["f_base_hz"])
    load_seconds = time.time() - t0
    provenance = runtime_provenance(protocol)
    provenance.update({
        "phase": "search", "smoke": bool(args.smoke), "backend": args.search_backend,
        "n_trials_per_variant": args.n_trials, "seed": args.seed,
        "sampler_seed": args.sampler_seed, "data_load_seconds": load_seconds,
        "files_read": list(_FILES_READ),
        "files_read_sha256": {Path(p).name: sha256_file(Path(p)) for p in dict.fromkeys(_FILES_READ)
                              if p.endswith(".npy")},
    })
    write_json(out_dir / "provenance.json", provenance)
    write_json(out_dir / "protocol_copy.json", protocol)

    summary = {}
    for cohort, context in variant_pairs(protocol):
        name = f"{cohort}__{context}"
        mtr, mva = cohort_mask(ytr, cohort), cohort_mask(yva, cohort)
        Xtr, ytr_abs = augment(Ftr[mtr], f_base + ytr[mtr, 0], context, protocol, args.seed)
        Xva, yva_abs = Fva[mva], f_base + yva[mva, 0]
        vdir = out_dir / name
        t1 = time.time()
        if args.search_backend == "skopt":
            best = search_skopt(name, Xtr, ytr_abs, Xva, yva_abs, protocol, args, vdir)
        else:
            best = search_optuna(name, Xtr, ytr_abs, Xva, yva_abs, protocol, args, vdir)
        best.update({"variant": name, "cohort": cohort, "context": context,
                     "n_train_rows": int(len(Xtr)), "n_train_events": int(mtr.sum()),
                     "n_val": int(mva.sum()), "search_seconds": time.time() - t1})
        write_json(vdir / "best_params.json", best)
        summary[name] = best
        logger.info(f"{name}: best val MAE {best['val_mae_hz']:.6f} Hz with {best['params']}")
    assert_no_test_reads()
    write_json(out_dir / "search_summary.json", {
        "smoke": bool(args.smoke), "variants": summary,
        "test_arrays_read": [p for p in _FILES_READ if "test" in Path(p).name],
        "max_rss_mb": max_rss_mb(), "total_seconds": time.time() - t0,
    })
    logger.info(json.dumps({k: {"val_mae_hz": v["val_mae_hz"], "params": v["params"]}
                            for k, v in summary.items()}, indent=2))


def phase_final(args, protocol: dict) -> None:
    """Selected parameters on train (seeds); L1 test, common cohorts, L2/L3 frozen+retrained."""
    search_dir = args.out_dir / "search"
    out_dir = args.out_dir / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = sheet_index()
    f_base = float(protocol["target"]["f_base_hz"])
    Ftr = angle_features(load_temporal("train"), idx)
    ytr = load_targets("train")
    tests = {s: (angle_features(load_temporal(v["test"]), idx), load_targets(v["test"]))
             for s, v in SCENARIO_SPLITS.items()}
    cals = {s: (angle_features(load_temporal(v["cal"]), idx), load_targets(v["cal"]))
            for s, v in SCENARIO_SPLITS.items() if v["cal"]}
    cohorts = {s: load_common_cohort(s, protocol) for s in SCENARIO_SPLITS}
    expected = protocol["evaluation"]["common_cohort"]["expected_counts"]
    cohort_record = {"files": {}, "scenarios": {}}
    for rel in protocol["evaluation"]["common_cohort"]["source_files"]:
        cohort_record["files"][rel] = sha256_file(PROJECT_ROOT / rel)
    for s, crows in cohorts.items():
        n_test = len(tests[s][1])
        if (len(crows) != int(expected[s]) or len(np.unique(crows)) != len(crows)
                or crows.min() < 0 or crows.max() >= n_test):
            raise RuntimeError(f"common cohort {s}: {len(crows)} rows (expected {expected[s]}); "
                               "unique/in-range check failed")
        np.save(out_dir / f"common_cohort_rows_{s}.npy", crows)
        cohort_record["scenarios"][s] = {
            "n": int(len(crows)), "sha256_rows": hashlib.sha256(crows.tobytes()).hexdigest(),
        }
    write_json(out_dir / "common_cohort_record.json", cohort_record)
    rows = []
    for cohort, context in variant_pairs(protocol):
        name = f"{cohort}__{context}"
        with open(search_dir / name / "best_params.json") as handle:
            params = json.load(handle)["params"]
        mtr = cohort_mask(ytr, cohort)
        Xtr, ytr_abs = augment(Ftr[mtr], f_base + ytr[mtr, 0], context, protocol, args.seed)
        multiplier = 2 if context == "context7" else 1
        for seed in args.seeds:
            model = make_tree(params, seed).fit(Xtr, ytr_abs)
            for scenario, (Xte, yte) in tests.items():
                evaluations = {"frozen": (model, {
                    "n_train_rows": int(len(Xtr)), "n_train_events": int(mtr.sum()),
                    "n_cal_events": 0, "augmentation_multiplier": multiplier})}
                if scenario in cals:
                    Xc, yc = cals[scenario]
                    mc = cohort_mask(yc, cohort)
                    Xr, yr = augment(np.vstack([Ftr[mtr], Xc[mc]]),
                                     np.concatenate([f_base + ytr[mtr, 0], f_base + yc[mc, 0]]),
                                     context, protocol, args.seed)
                    evaluations["retrained"] = (make_tree(params, seed).fit(Xr, yr), {
                        "n_train_rows": int(len(Xr)), "n_train_events": int(mtr.sum()),
                        "n_cal_events": int(mc.sum()), "augmentation_multiplier": multiplier})
                for mode, (mdl, train_meta) in evaluations.items():
                    pred_y1 = mdl.predict(Xte) - f_base
                    err = pred_y1 - yte[:, 0]
                    run_dir = out_dir / name / f"seed_{seed}" / f"{scenario}_{mode}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    np.save(run_dir / "preds_y1.npy", pred_y1)
                    joblib.dump(mdl, run_dir / "model.joblib")
                    write_json(run_dir / "config.json", {
                        "variant": name, "cohort": cohort, "context": context, "seed": seed,
                        "scenario": scenario, "mode": mode, "params": params,
                        **train_meta, "f_base_hz": f_base,
                        "driver_sha256": sha256_file(DRIVER_PATH),
                        "protocol_id": protocol["protocol_id"],
                    })
                    cohort_rows = cohorts[scenario]
                    down = yte[:, 0] < 0
                    metrics = {
                        "variant": name, "seed": seed, "scenario": scenario, "mode": mode,
                        "n_test": int(len(yte)),
                        "y1_MAE_full": float(np.mean(np.abs(err))),
                        "y1_RMSE_full": float(np.sqrt(np.mean(err ** 2))),
                        "n_common": int(len(cohort_rows)),
                        "y1_MAE_common": float(np.mean(np.abs(err[cohort_rows]))),
                        "n_downward": int(down.sum()),
                        "y1_MAE_downward": float(np.mean(np.abs(err[down]))),
                        "params": params,
                    }
                    write_json(run_dir / "metrics.json", metrics)
                    rows.append(metrics)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "all_runs.csv", index=False)
    cols = ["y1_MAE_full", "y1_RMSE_full", "y1_MAE_common", "y1_MAE_downward"]
    summary = frame.groupby(["variant", "scenario", "mode"])[cols].agg(["mean", "std", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(out_dir / "summary.csv", index=False)
    write_json(out_dir / "provenance.json", {**runtime_provenance(protocol), "phase": "final",
                                             "seeds": args.seeds, "files_read": list(_FILES_READ),
                                             "printed_references_frozen":
                                             protocol["evaluation"]["printed_references_frozen"]})
    logger.info(f"final results written to {out_dir}")


# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("describe", "search", "final"), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--search-backend", choices=("optuna", "skopt"), default="optuna")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123, help="tree random_state (published 123)")
    parser.add_argument("--sampler-seed", type=int, default=123)
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="final phase seeds")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    if args.n_trials is None:
        args.n_trials = int(protocol["search"]["trials_per_variant"])
    if args.seeds is None:
        args.seeds = list(protocol["final"]["seeds"])
    if args.smoke:
        args.n_trials = min(args.n_trials, 3)
        if not str(args.out_dir).endswith("_smoke"):
            args.out_dir = Path(str(args.out_dir) + "_smoke")
    for never in protocol["never_write"]:
        never_path = (PROJECT_ROOT / never).resolve()
        if str(args.out_dir.resolve()).startswith(str(never_path)):
            raise RuntimeError(f"output directory {args.out_dir} lies inside a frozen path {never}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger.add(args.out_dir / f"{args.phase}.log", enqueue=True)
    logger.info(f"phase={args.phase} out_dir={args.out_dir} protocol={args.protocol}")
    {"describe": phase_describe, "search": phase_search, "final": phase_final}[args.phase](args, protocol)


if __name__ == "__main__":
    main()
