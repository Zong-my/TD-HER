#!/usr/bin/env python3
"""Label-budget study of the TD-HER admission rule - runner (harness v3a).

Design: claude_label_budget_design_v1.md with amendments 1, 2, 3 and 3a in
the study design described in docs/EXPERIMENTS.md. The study protocol (question, rules,
budgets, seeds, cell statistic, classification, decision criterion; design v1 with amendments 1
and 2) was frozen on 2026-09-21 before any fit. Amendments 3 and 3a change only the acceptance
condition of the level-2 harness check and were written after the first stage stopped at that
check; at that point the full-budget harness diagnostics had been inspected and no reduced-budget
or pilot outcome had been observed.

Question: how many labelled events from a shifted system make including the
retrained adapter worthwhile, and how do the candidate admission rules behave
as that number changes? This is a new, retrospective, descriptive study. It is
not a check of any number in the manuscript.

This file creates every fitted output of the study. Planning of the executed
grid and aggregation live in summarize_label_budget.py, which only reads JSON.

Safety properties
- Production routing/admission/affine functions are imported unchanged from
  experiments/exp_tdh_router.py. train_lgb_adapter_oof, get_cached_prediction
  and main() are never called, because they read and write caches inside the
  frozen production results directory.
- Everything is written under one new results directory. A configuration
  directory is created once, by atomic rename of a temporary directory, and is
  never overwritten or deleted. Existing configurations are reused.
- Identity is enforced, not only recorded (v2): on every invocation the rule
  record, the SHA-256 of every input and code file (this file included) and the
  effective protocol (resolved feature indices, adapter parameters, router
  constants, library versions) are recomputed and compared with the files
  written when the directory was created. Any difference stops the run before
  a stage-pass file or a completed configuration is trusted. A changed harness
  therefore needs a fresh versioned results directory.
- The level-2 check is a direct bound on the adapter predictions (amendments 3
  and 3a): every retrained full-budget adapter array must reproduce the frozen
  one within 1e-8, or differ from it in at most 1 % of its rows with a mean
  absolute prediction difference of at most 0.1 % of the frozen adapter's MAE on
  that array; at most two of the eight arrays may use the second condition. The
  difference of two MAEs is reported but is not a criterion (it can hide changed
  predictions by cancellation). Downstream MAE or decisions cannot clear a
  mismatch. These are disclosed engineering tolerances, not statistical bounds.
  The training path is identical to harness v2.

Stages: level1 (no training) | level2 | pilot | full
"""

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
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORK_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DEFAULT = work_path('results/ieee39/resubmission_v4_label_budget_v3')
PROD = work_path('results/ieee39/exp_tdh_router')
CSV_DIR = work_path('TD-HER/data/ieee39/csv')
EXP1 = work_path('results/ieee39/exp1')

SCENARIOS = {
    "L2": {"prod": "L2_cross_cond_fewshot", "cal": "cross_cond_finetune",
           "test": "cross_cond_test", "s": 2, "budgets": [100, 250, 510]},
    "L3": {"prod": "L3_cross_topo_fewshot", "cal": "cross_cond_topo_finetune",
           "test": "cross_cond_topo_test", "s": 3, "budgets": [100, 250, 500, 1000, 2337]},
}
N_SEEDS = 5
FALLBACK_SEEDS = 3
SEED0 = 20260921
FULL_K = 9                      # seed index used for the (unsampled) full budget
BOOT_ROUNDS = 2000
RULES = ["A_margin", "B_margin_and_bootstrap", "C_always", "D_never"]
ADAPTER = "LightGBM-Adapter"
FIRST_STAGE = [("L2", 510, FULL_K), ("L3", 2337, FULL_K), ("L3", 500, 0)]
TIME_LIMIT_H = 8.0
PARALLEL_SLOTS = 3
LEVEL1_TOL = 1e-10
LEVEL2_TOL = 1e-8               # reproduction: max|diff| of an adapter array
LEVEL2_MAX_ROW_SHARE = 0.01     # bounded discrepancy: share of rows with |diff| > LEVEL2_TOL
LEVEL2_MAX_REL_SHIFT = 1e-3     # bounded discrepancy: mean|new - frozen| over the frozen adapter MAE on that array
LEVEL2_MAX_BOUNDED_ARRAYS = 2   # of the eight adapter arrays (2 scenarios x OOF/test x 2 targets)
IDENTITY_TOL = 1e-12

RULE_RECORD = {
    "study": "label-budget study of the TD-HER admission rule",
    "design": ["claude_label_budget_design_v1.md", "claude_label_budget_design_v1_amendment1.md",
               "claude_label_budget_design_v1_amendment2.md", "claude_label_budget_design_v1_amendment3.md",
               "claude_label_budget_design_v1_amendment3a.md"],
    "nature": "retrospective, descriptive; not a check of manuscript numbers; "
              "not independent validation of the rule that is eventually chosen",
    "budget_meaning": "the whole calibration stage (adapter OOF training and final refit, admission, "
                      "simplex weights, affine correction) sees only the n sampled shifted-domain events; "
                      "base experts frozen as in V3",
    "scenarios": {k: {"budgets": v["budgets"], "cal": v["cal"], "test": v["test"]} for k, v in SCENARIOS.items()},
    "subsampling": "stratified by distu_kind, nested across budgets, largest-remainder quotas, rows kept in "
                   "original order; seeds SEED0+k, k=0..4; full budget = all rows in original order, run once",
    "seed0": SEED0, "n_seeds": N_SEEDS,
    "adapter": "production protocol: fixed LightGBM parameters from results/ieee39/exp1/LightGBM, trees capped "
               "at 800, random_state 42, n_jobs 8, KFold(min(3,n), shuffle=True, random_state=42), final refit "
               "on source training set plus the n events",
    "rules": {"A_margin": "admission_affine_convex_blend_router (tau=0.20)",
              "B_margin_and_bootstrap": "certified_admission_affine_convex_blend_router (reviewed rule)",
              "C_always": "affine_convex_blend_router on base + adapter",
              "D_never": "affine_convex_blend_router on base only",
              "post": "enforce_y2_nonnegative on every rule",
              "excluded_on_purpose": "interval-only rule (never part of the reviewed method)"},
    "cell": "scenario x budget x seed x target; full budget counted once; 64 cells with five seeds, 40 with the "
            "three-seed fallback; the executed grid is fixed in run_plan.json before the full run and is the "
            "only grid used for dispatch, completeness and aggregation",
    "cell_statistic": "MAE(C) - MAE(D) on the fixed test split with one paired event-bootstrap 95% interval, "
                      "2000 resamples, seed = SEED0 + 1000*s + 100*b + 10*k + t (L2 s=2, L3 s=3; b budget index; "
                      "k seed index, 9 for the full budget; t 0 for y1, 1 for y2); shared by all rules",
    "identities_checked_per_cell": "routed A equals routed C when A admits and routed D when it rejects, likewise "
                                   "for B (max|diff| <= 1e-12); B admits implies A admits; A's decision equals the "
                                   "margin gate inside B. A failed identity stops the study.",
    "classification": "only cells where A admits and B rejects: interval > 0 = harmful admission avoided by the "
                      "second criterion; interval < 0 = recovery missed by it; interval containing 0 = neither",
    "decision": "parsimony default margin-only; the second criterion is retained only if avoided >= 1 and "
                "missed <= avoided, summed over all cells of the executed grid",
    "harness_check": {"level1": "no training. Frozen adapter arrays through the harness, full budget, original "
                                "order: max|diff| <= 1e-10 against production arrays for rules A, B, D (and C "
                                "before projection); the complete per-configuration evaluation is run on the same "
                                "frozen arrays, its identities must hold and its A/B decisions must equal the "
                                "decisions stored in production metrics_summary.json",
                      "level2": "retrained full-budget adapter arrays (2 scenarios x out-of-fold/test x 2 targets = 8) "
                                "vs the frozen adapter arrays. Each array must satisfy (a) reproduction, max|diff| <= "
                                "1e-8, or (b) bounded discrepancy, at most 1% of rows with |diff| > 1e-8 and "
                                "mean|new - frozen| <= 0.1% of the frozen adapter's MAE on that array (fails closed if "
                                "that MAE is zero); at most two of the eight arrays may use (b); A/B decisions must "
                                "equal the production ones. Otherwise the study stops with diagnostics. The difference "
                                "of the two MAEs, the routed output and the routed MAE are reported only. Disclosed "
                                "engineering tolerances (amendments 3 and 3a), recorded in the study protocol before the "
                                "reduced-budget study; not a "
                                "statistical bound. A mismatch concerns the new script and never changes V3 numbers."},
    "identity_enforcement": "rule record, SHA-256 of every input/code file and the effective protocol are "
                            "recomputed on every invocation and must equal the files written at creation",
    "time_rule": "8 h wall-clock for the configurations left after the first stage (29 with five seeds), projected "
                 "as (number left) x (mean wall time of the three first-stage configurations); if the sequential "
                 "projection exceeds 8 h, three parallel slots, projected as sequential/3; if that still exceeds "
                 "8 h, seeds k=0,1,2 with three slots. Uses timing only, never a result value. Written once to "
                 "run_plan.json.",
    "claim_boundary": "answers whether including the adapter was worthwhile in these scenarios at these budgets; "
                      "supports no generally applicable minimum number of labels; does not establish that "
                      "tau = 0.20 is optimal",
}


# ----------------------------------------------------------------------------- utilities
def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(o):
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def normalise(obj):
    return json.loads(json.dumps(obj, sort_keys=True, default=_json_default))


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def config_name(scen, n, k):
    return f"{scen}_n{n:04d}_k{k}"


def all_configs(n_seeds=N_SEEDS):
    out = []
    for scen, spec in SCENARIOS.items():
        out.append((scen, spec["budgets"][-1], FULL_K))
        for n in spec["budgets"][:-1]:
            for k in range(n_seeds):
                out.append((scen, n, k))
    return out


# ----------------------------------------------------------------------------- identity of inputs, code and protocol
def input_files(R, rep_dir, ms):
    rep_a = Path(rep_dir) / f"repA/ms{ms}"
    files = []
    for spec in SCENARIOS.values():
        for m in R.ROUTER_MODELS:
            for split in (spec["cal"], spec["test"]):
                files.append(PROD / "experts" / m / f"{split}_preds.npy")
        ad = PROD / "adapters" / spec["prod"] / ADAPTER
        files += [ad / "cal_oof_preds.npy", ad / "test_preds.npy"]
        sd = PROD / "scenarios" / spec["prod"]
        files += [sd / f"{v}_preds.npy" for v in (
            "admission_affine_convex_blend_nonnegative_y2",
            "certified_admission_affine_convex_blend_nonnegative_y2",
            "base_affine_convex_blend_nonnegative_y2", "affine_convex_blend")]
        files += [sd / "metrics_summary.json"]
        files += [CSV_DIR / f"{spec['cal']}_ms{ms}.csv"]
        for split in (spec["cal"], spec["test"]):
            files += [rep_a / f"X_{split}.npy", rep_a / f"y_{split}.npy"]
    files += [rep_a / "X_train.npy", rep_a / "y_train.npy"]
    # feature selection, feature order and preprocessing actually used by the adapter (v2)
    files += [rep_a / "feature_names.json", EXP1 / "mrmr_features.json"]
    files += [EXP1 / "LightGBM" / f"{t}_lgb_model.pkl" for t in ("y1", "y2")]
    files += [PROJECT_ROOT / "experiments/exp_tdh_router.py", PROJECT_ROOT / "experiments/exp4_generalization.py",
              Path(__file__).resolve()]
    # Not listed on purpose: the repB/repC arrays and static mRMR list that the production loaders also read.
    # They feed the neural experts, whose predictions enter this study only through the hashed cached arrays.
    return files


def manifest_key(p):
    return str(p.relative_to(PROJECT_ROOT)) if str(p).startswith(str(PROJECT_ROOT)) else str(p)


def compute_manifest(R, rep_dir, ms):
    files = input_files(R, rep_dir, ms)
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit("missing frozen inputs:\n" + "\n".join(missing))
    return {manifest_key(p): sha256(p) for p in files}


def effective_protocol(R, rep_dir, ms):
    import platform
    import lightgbm
    import scipy
    import sklearn
    names = load_json(Path(rep_dir) / f"repA/ms{ms}/feature_names.json")
    union = load_json(EXP1 / "mrmr_features.json")["union"]
    idx = [names.index(f) for f in union]
    return normalise({
        "feature_resolution": "indices of results/ieee39/exp1/mrmr_features.json['union'] in repA "
                              "feature_names.json, the resolution used by "
                              "experiments/exp4_generalization.py::load_train_scalers; checked against the "
                              "production loader's own feat_idx before the first fit of every invocation",
        "n_raw_features": len(names), "n_selected_features": len(idx), "feature_indices": idx,
        "feature_indices_sha256": hashlib.sha256(json.dumps(idx).encode()).hexdigest(),
        "adapter_params": R.load_exp1_lgb_specs(),
        "router_constants": {"ROUTER_MODELS": list(R.ROUTER_MODELS), "ADAPTER_N_FOLDS": R.ADAPTER_N_FOLDS,
                             "ADAPTER_MAX_TREES": R.ADAPTER_MAX_TREES, "ADAPTER_N_JOBS": R.ADAPTER_N_JOBS,
                             "ADAPTER_ADMISSION_REL_IMPROV": R.ADAPTER_ADMISSION_REL_IMPROV,
                             "BOOTSTRAP_ROUNDS": R.BOOTSTRAP_ROUNDS, "MS": ms},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__},
    })


def prepare_out(out, R, rep_dir, ms):
    """Create the results directory once; afterwards enforce identical rule record, inputs, code and protocol."""
    rr, mf, ep = out / "rule_record.json", out / "input_sha256_manifest.json", out / "effective_protocol.json"
    manifest = compute_manifest(R, rep_dir, ms)
    protocol = effective_protocol(R, rep_dir, ms)
    if not out.exists():
        out.mkdir(parents=True)
        dump_json(RULE_RECORD, rr)
        dump_json(manifest, mf)
        dump_json(protocol, ep)
        print(f"[init] created {out}: rule record, {len(manifest)} input identities, effective protocol "
              f"({protocol['n_selected_features']} of {protocol['n_raw_features']} features)")
    else:
        problems = []
        if load_json(rr) != normalise(RULE_RECORD):
            problems.append("rule_record.json differs from the record in this script")
        saved = load_json(mf)
        for key in sorted(set(saved) | set(manifest)):
            if saved.get(key) != manifest.get(key):
                problems.append(f"input identity changed: {key}: saved {str(saved.get(key))[:16]} "
                                f"now {str(manifest.get(key))[:16]}")
        saved_protocol = load_json(ep)
        for key in sorted(set(saved_protocol) | set(protocol)):
            if saved_protocol.get(key) != protocol.get(key):
                problems.append(f"effective protocol changed: {key}")
        if problems:
            raise SystemExit(f"refusing to reuse {out}; stage-pass files and completed configurations are not "
                             "trusted. Use a fresh versioned directory.\n  " + "\n  ".join(problems))
        print(f"[init] reusing {out}: rule record, {len(manifest)} input identities and effective protocol verified")
    return {"manifest_sha256": sha256(mf), "effective_protocol_sha256": sha256(ep), "protocol": protocol}


# ----------------------------------------------------------------------------- data
def load_frozen_expert_preds(R, split):
    preds = {}
    for m in R.ROUTER_MODELS:
        p = PROD / "experts" / m / f"{split}_preds.npy"
        if not p.exists():
            raise SystemExit(f"frozen expert prediction missing: {p}")
        preds[m] = np.load(p)
    return preds


def load_categories(spec, y_cal, ms):
    import pandas as pd
    df = pd.read_csv(CSV_DIR / f"{spec['cal']}_ms{ms}.csv", usecols=["distu_kind", "fpu_deltamax", "t_delta"])
    y_csv = df[["fpu_deltamax", "t_delta"]].to_numpy(dtype=np.float32)
    if y_csv.shape != y_cal.shape or not np.array_equal(y_csv, y_cal.astype(np.float32)):
        raise SystemExit(f"calibration CSV and array targets are not row-identical for {spec['cal']}")
    return df["distu_kind"].to_numpy()


def nested_stratified_indices(categories, budgets, seed):
    """Indices for every budget (nested, stratified, original row order)."""
    n_total = len(categories)
    cats = sorted(set(categories))
    rng = np.random.default_rng(seed)
    perm = {c: rng.permutation(np.flatnonzero(categories == c)) for c in cats}
    prev_quota, out = {c: 0 for c in cats}, {}
    for n in sorted(budgets):
        if n >= n_total:
            out[n] = np.arange(n_total)
            continue
        exact = {c: n * len(perm[c]) / n_total for c in cats}
        quota = {c: int(np.floor(exact[c])) for c in cats}
        rest = n - sum(quota.values())
        for c in sorted(cats, key=lambda c: (-(exact[c] - quota[c]), c))[:rest]:
            quota[c] += 1
        for c in cats:
            if quota[c] < prev_quota[c] or quota[c] > len(perm[c]):
                raise SystemExit(f"nested stratified quotas not monotone for category {c} at n={n}")
        prev_quota = quota
        out[n] = np.sort(np.concatenate([perm[c][:quota[c]] for c in cats]))
    return out


def production_decisions(spec):
    """A/B admission decisions stored by the production run (read for comparison only)."""
    s = load_json(PROD / "scenarios" / spec["prod"] / "metrics_summary.json")
    out = {}
    for target in ("y1", "y2"):
        a = s["admission_affine_convex_blend"]["details"][target]["admission"]["adapter_details"][ADAPTER]
        b = s["certified_admission_affine_convex_blend"]["details"][target]["admission"]["adapter_details"][ADAPTER]
        out[target] = {"A_admits": bool(a["admitted"]), "B_admits": bool(b["admitted"])}
    return out


# ----------------------------------------------------------------------------- adapter (production protocol, new output location)
def train_adapter(R, specs, X_base, y_base, X_cal, y_cal, X_test):
    import lightgbm as lgb
    from sklearn.model_selection import KFold
    n = len(y_cal)
    kf = KFold(n_splits=min(R.ADAPTER_N_FOLDS, n), shuffle=True, random_state=42)
    oof = np.zeros_like(y_cal, dtype=float)
    test = np.zeros((len(X_test), 2), dtype=float)
    fit_seconds = []
    for t_idx, target in enumerate(["y1", "y2"]):
        params = dict(specs[target])
        for tr, va in kf.split(X_cal):
            t0 = time.time()
            model = lgb.LGBMRegressor(**params)
            model.fit(np.concatenate([X_base, X_cal[tr]], axis=0),
                      np.concatenate([y_base[:, t_idx], y_cal[tr, t_idx]]))
            oof[va, t_idx] = model.predict(X_cal[va])
            fit_seconds.append({"target": target, "kind": "oof", "seconds": time.time() - t0,
                                "trees": int(params["n_estimators"])})
        t0 = time.time()
        final = lgb.LGBMRegressor(**params)
        final.fit(np.concatenate([X_base, X_cal], axis=0), np.concatenate([y_base[:, t_idx], y_cal[:, t_idx]]))
        test[:, t_idx] = final.predict(X_test)
        fit_seconds.append({"target": target, "kind": "final", "seconds": time.time() - t0,
                            "trees": int(params["n_estimators"])})
    return oof, test, fit_seconds


# ----------------------------------------------------------------------------- rules and statistics
def apply_rules(R, y_cal, cal_preds, test_preds):
    base_cal = {m: p for m, p in cal_preds.items() if m not in R.ADAPTER_NAMES}
    base_test = {m: p for m, p in test_preds.items() if m not in R.ADAPTER_NAMES}
    raw = {
        "A_margin": R.admission_affine_convex_blend_router(y_cal, cal_preds, test_preds),
        "B_margin_and_bootstrap": R.certified_admission_affine_convex_blend_router(y_cal, cal_preds, test_preds),
        "C_always": R.affine_convex_blend_router(y_cal, cal_preds, test_preds),
        "D_never": R.affine_convex_blend_router(y_cal, base_cal, base_test),
    }
    routed = {k: R.enforce_y2_nonnegative(v[0]) for k, v in raw.items()}
    details = {k: v[1] for k, v in raw.items()}
    unprojected_C = raw["C_always"][0]
    return routed, details, unprojected_C


def admitted(details, rule, target):
    adm = details[rule][target]["admission"]["adapter_details"][ADAPTER]
    return bool(adm["admitted"]), adm


def cell_interval(err_c, err_d, seed):
    d = err_c - err_d
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(BOOT_ROUNDS, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def evaluate_config(R, scen, n, k, y_cal, cal_preds, test_preds, y_test):
    spec = SCENARIOS[scen]
    routed, details, _ = apply_rules(R, y_cal, cal_preds, test_preds)
    b = spec["budgets"].index(n)
    res = {"scenario": scen, "n": int(n), "k": int(k), "targets": {}}
    for t, target in enumerate(["y1", "y2"]):
        mae = {r: float(np.mean(np.abs(routed[r][:, t] - y_test[:, t]))) for r in RULES}
        seed = SEED0 + 1000 * spec["s"] + 100 * b + 10 * k + t
        delta, lo, hi = cell_interval(np.abs(routed["C_always"][:, t] - y_test[:, t]),
                                      np.abs(routed["D_never"][:, t] - y_test[:, t]), seed)
        a_adm, a_info = admitted(details, "A_margin", target)
        b_adm, b_info = admitted(details, "B_margin_and_bootstrap", target)
        ident = {}
        for rule, adm in (("A_margin", a_adm), ("B_margin_and_bootstrap", b_adm)):
            ref = "C_always" if adm else "D_never"
            ident[rule] = {"expected_equal_to": ref,
                           "max_abs_diff": float(np.max(np.abs(routed[rule][:, t] - routed[ref][:, t])))}
        ident["B_implies_A"] = bool(a_adm or not b_adm)
        ident["A_equals_margin_gate_of_B"] = bool(a_adm == bool(b_info["practical_gate"]))
        ident["pass"] = bool(ident["A_margin"]["max_abs_diff"] <= IDENTITY_TOL
                             and ident["B_margin_and_bootstrap"]["max_abs_diff"] <= IDENTITY_TOL
                             and ident["B_implies_A"] and ident["A_equals_margin_gate_of_B"])
        res["targets"][target] = {
            "test_MAE": mae,
            "C_minus_D": {"delta": delta, "ci95": [lo, hi], "bootstrap_seed": int(seed)},
            "A_admits": a_adm, "B_admits": b_adm,
            "oof_adapter_MAE": float(a_info["cal_MAE"]), "required_MAE": float(a_info["required_MAE"]),
            "base_best_model": details["A_margin"][target]["admission"]["base_best_model"],
            "base_best_cal_MAE": float(details["A_margin"][target]["admission"]["base_best_cal_MAE"]),
            "B_certificate_ci95": [float(x) for x in b_info["certificate"]["ci95"]],
            "weights": {r: details[r][target]["weights"] for r in RULES},
            "identities": ident,
        }
    res["identity_pass"] = bool(all(v["identities"]["pass"] for v in res["targets"].values()))
    return res, routed


# ----------------------------------------------------------------------------- stages
def stage_level1(R, out, rep_dir, ms):
    report, ok = {}, True
    for scen, spec in SCENARIOS.items():
        y_cal = np.load(f"{rep_dir}/repA/ms{ms}/y_{spec['cal']}.npy")
        y_test = np.load(f"{rep_dir}/repA/ms{ms}/y_{spec['test']}.npy")
        cal_preds = load_frozen_expert_preds(R, spec["cal"])
        test_preds = load_frozen_expert_preds(R, spec["test"])
        ad = PROD / "adapters" / spec["prod"] / ADAPTER
        cal_preds[ADAPTER] = np.load(ad / "cal_oof_preds.npy")
        test_preds[ADAPTER] = np.load(ad / "test_preds.npy")
        routed, _, unproj_c = apply_rules(R, y_cal, cal_preds, test_preds)
        sd = PROD / "scenarios" / spec["prod"]
        pairs = {"A_margin": ("admission_affine_convex_blend_nonnegative_y2", routed["A_margin"]),
                 "B_margin_and_bootstrap": ("certified_admission_affine_convex_blend_nonnegative_y2",
                                            routed["B_margin_and_bootstrap"]),
                 "D_never": ("base_affine_convex_blend_nonnegative_y2", routed["D_never"]),
                 "C_always_unprojected": ("affine_convex_blend", unproj_c)}
        report[scen] = {"arrays": {}}
        for rule, (name, arr) in pairs.items():
            diff = float(np.max(np.abs(arr - np.load(sd / f"{name}_preds.npy"))))
            report[scen]["arrays"][rule] = {"production_array": name, "max_abs_diff": diff,
                                            "pass": diff <= LEVEL1_TOL}
            ok &= diff <= LEVEL1_TOL
            print(f"[level1] {scen} {rule:<24} vs {name:<58} max|diff| = {diff:.3e}")
        # dry run of the complete per-configuration evaluation on the frozen arrays (no fit): exercises the
        # parsing of admission details, the shared cell statistic and the identities before any training
        full = spec["budgets"][-1]
        res, _ = evaluate_config(R, scen, full, FULL_K, y_cal, cal_preds, test_preds, y_test)
        prod = production_decisions(spec)
        same = all(res["targets"][t]["A_admits"] == prod[t]["A_admits"]
                   and res["targets"][t]["B_admits"] == prod[t]["B_admits"] for t in ("y1", "y2"))
        report[scen]["evaluation_dry_run_on_frozen_arrays"] = res
        report[scen]["production_decisions"] = prod
        report[scen]["decisions_identical_to_production"] = bool(same)
        ok &= bool(res["identity_pass"]) and bool(same)
        print(f"[level1] {scen} evaluation dry run: identities={'ok' if res['identity_pass'] else 'FAILED'} "
              f"decisions_same_as_production={same} | " + " | ".join(
                  f"{t}: A={'adm' if v['A_admits'] else 'rej'} B={'adm' if v['B_admits'] else 'rej'}"
                  for t, v in res["targets"].items()))
    report["pass"] = bool(ok)
    path = out / "level1_check.json"
    if path.exists():
        print("[level1] level1_check.json already exists; not overwritten")
    else:
        dump_json(report, path)
    print("[level1] PASS" if ok else "[level1] FAIL - stop and debug the harness")
    return ok


def run_configs(R, out, rep_dir, ms, configs, ids):
    from experiments.exp4_generalization import load_level_data, load_train_scalers
    (out / "configs").mkdir(exist_ok=True)
    for d in sorted((out / "configs").glob("[!.]*")):
        rj = d / "result.json"
        if rj.exists() and not load_json(rj).get("identity_pass", False):
            raise SystemExit(f"configuration {d.name} has a failed identity check; the study is stopped")
    todo = [c for c in configs if not (out / "configs" / config_name(*c) / "result.json").exists()]
    print(f"[run] {len(configs)} requested, {len(configs) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        return
    scalers = R.attach_base_training_data(load_train_scalers())
    specs = R.load_exp1_lgb_specs()
    protocol = ids["protocol"]
    if [int(i) for i in scalers["feat_idx"]] != protocol["feature_indices"]:
        raise SystemExit("feature indices of the production loader differ from effective_protocol.json")
    if normalise(specs) != protocol["adapter_params"]:
        raise SystemExit("adapter parameters differ from effective_protocol.json")
    X_base, y_base = scalers["X_train_n"], scalers["y_train"]
    print(f"[run] source training matrix {X_base.shape}; adapter trees y1={specs['y1']['n_estimators']} "
          f"y2={specs['y2']['n_estimators']}; n_jobs={specs['y1']['n_jobs']}")
    script_hash = sha256(Path(__file__).resolve())
    thread_env = {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    cache = {}
    for scen, n, k in todo:
        spec = SCENARIOS[scen]
        if scen not in cache:
            cal, test = load_level_data(spec["cal"], scalers), load_level_data(spec["test"], scalers)
            cache[scen] = {"cal": cal, "test": test,
                           "cal_preds": load_frozen_expert_preds(R, spec["cal"]),
                           "test_preds": load_frozen_expert_preds(R, spec["test"]),
                           "cats": load_categories(spec, cal["y"], ms), "idx": {}}
        c = cache[scen]
        full = spec["budgets"][-1]
        if n == full:
            idx = np.arange(len(c["cal"]["y"]))
        else:
            if k not in c["idx"]:
                c["idx"][k] = nested_stratified_indices(c["cats"], spec["budgets"], SEED0 + k)
            idx = c["idx"][k][n]
        assert len(idx) == n, (scen, n, len(idx))
        name = config_name(scen, n, k)
        t0 = time.time()
        oof, test_pred, fits = train_adapter(R, specs, X_base, y_base, c["cal"]["X_n"][idx], c["cal"]["y"][idx],
                                             c["test"]["X_n"])
        cal_preds = {m: p[idx] for m, p in c["cal_preds"].items()}
        test_preds = dict(c["test_preds"])
        cal_preds[ADAPTER], test_preds[ADAPTER] = oof, test_pred
        res, routed = evaluate_config(R, scen, n, k, c["cal"]["y"][idx], cal_preds, test_preds, c["test"]["y"])
        res.update({"fits": fits, "wall_seconds": time.time() - t0, "peak_rss_gb": peak_rss_gb(),
                    "script_sha256": script_hash, "manifest_sha256": ids["manifest_sha256"],
                    "effective_protocol_sha256": ids["effective_protocol_sha256"], "thread_env": thread_env,
                    "categories_in_sample": {str(a): int(b) for a, b in
                                             zip(*np.unique(c["cats"][idx], return_counts=True))}})
        tmp = out / "configs" / f".tmp_{name}_{os.getpid()}"
        tmp.mkdir()
        np.save(tmp / "subsample_indices.npy", idx)
        np.save(tmp / "adapter_cal_oof.npy", oof)
        np.save(tmp / "adapter_test.npy", test_pred)
        for r in RULES:
            np.save(tmp / f"routed_{r}.npy", routed[r])
        dump_json(res, tmp / "result.json")
        os.rename(tmp, out / "configs" / name)          # atomic; fails if the target exists
        print(f"[run] {name}: {res['wall_seconds']:.1f} s, peak RSS {res['peak_rss_gb']:.1f} GB | " + " | ".join(
            f"{t}: A={'adm' if v['A_admits'] else 'rej'} B={'adm' if v['B_admits'] else 'rej'} "
            f"MAE A/C/D={v['test_MAE']['A_margin']:.5f}/{v['test_MAE']['C_always']:.5f}/{v['test_MAE']['D_never']:.5f}"
            for t, v in res["targets"].items()), flush=True)
        if not res["identity_pass"]:
            raise SystemExit(f"identity check failed in {name} (saved for inspection); the study is stopped")


def adapter_array_condition(new, ref, truth):
    """Level-2 condition for one adapter array (one scenario, split and target); amendments 3 and 3a."""
    d = np.abs(new - ref)
    mae_new = float(np.mean(np.abs(new - truth)))
    mae_ref = float(np.mean(np.abs(ref - truth)))
    share = float(np.mean(d > LEVEL2_TOL))
    shift = float(d.mean() / mae_ref) if mae_ref > 0 else float("inf")      # fails closed
    reproduced = bool(d.max() <= LEVEL2_TOL)
    bounded = bool(share <= LEVEL2_MAX_ROW_SHARE and shift <= LEVEL2_MAX_REL_SHIFT)
    return {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "rows_beyond_1e-8": int(np.sum(d > LEVEL2_TOL)), "share_of_rows_beyond_1e-8": share,
            "share_of_rows_differing_at_all": float(np.mean(d > 0)),
            "mean_abs_diff_over_frozen_adapter_MAE": shift,
            "adapter_MAE_harness": mae_new, "adapter_MAE_frozen": mae_ref,
            "reported_only_relative_MAE_difference": abs(mae_new - mae_ref) / mae_ref if mae_ref > 0 else float("inf"),
            "condition": "reproduced" if reproduced else ("bounded_discrepancy" if bounded else "FAILED")}


def stage_level2(R, out, rep_dir, ms):
    report, ok, n_bounded = {}, True, 0
    for scen, spec in SCENARIOS.items():
        cfg = out / "configs" / config_name(scen, spec["budgets"][-1], FULL_K)
        ad = PROD / "adapters" / spec["prod"] / ADAPTER
        truth = {"cal_oof": np.load(f"{rep_dir}/repA/ms{ms}/y_{spec['cal']}.npy"),
                 "test": np.load(f"{rep_dir}/repA/ms{ms}/y_{spec['test']}.npy")}
        new = {"cal_oof": np.load(cfg / "adapter_cal_oof.npy"), "test": np.load(cfg / "adapter_test.npy")}
        ref = {"cal_oof": np.load(ad / "cal_oof_preds.npy"), "test": np.load(ad / "test_preds.npy")}
        arrays = {}
        for name in ("cal_oof", "test"):
            for t, target in enumerate(("y1", "y2")):
                arrays[f"{name}_{target}"] = adapter_array_condition(new[name][:, t], ref[name][:, t],
                                                                     truth[name][:, t])
                n_bounded += int(arrays[f"{name}_{target}"]["condition"] == "bounded_discrepancy")
        res = load_json(cfg / "result.json")
        prod = production_decisions(spec)
        same = all(res["targets"][t]["A_admits"] == prod[t]["A_admits"]
                   and res["targets"][t]["B_admits"] == prod[t]["B_admits"] for t in ("y1", "y2"))
        # reported only; never a pass criterion, because a rejected adapter does not enter the routed output
        prod_routed = np.load(PROD / "scenarios" / spec["prod"] / "admission_affine_convex_blend_nonnegative_y2_preds.npy")
        routed_diff = float(np.max(np.abs(np.load(cfg / "routed_A_margin.npy") - prod_routed)))
        arrays_ok = all(v["condition"] != "FAILED" for v in arrays.values())
        report[scen] = {"adapter_arrays": arrays, "every_array_reproduced_or_bounded": bool(arrays_ok),
                        "decisions_identical_to_production": bool(same),
                        "reported_only_routed_A_max_abs_diff": routed_diff,
                        "pass": bool(arrays_ok and same)}
        ok &= report[scen]["pass"]
        print(f"[level2] {scen}: " + ", ".join(f"{k}={v['condition']}" for k, v in arrays.items())
              + f" | decisions_same={same} -> {'PASS' if report[scen]['pass'] else 'FAIL'}")
    report["arrays_using_bounded_discrepancy"] = n_bounded
    report["limit_arrays_using_bounded_discrepancy"] = LEVEL2_MAX_BOUNDED_ARRAYS
    ok &= n_bounded <= LEVEL2_MAX_BOUNDED_ARRAYS
    report["pass"] = bool(ok)
    report["note"] = ("Direct bound on the adapter predictions (amendments 3 and 3a): disclosed engineering "
                      "tolerances, not a statistical bound. The difference of the two MAEs and the routed output are "
                      "reported only. On failure the study stops here. A mismatch concerns the new harness only. The "
                      "author-verified V3 numbers are not affected, re-examined or replaced.")
    path = out / "level2_check.json"
    if path.exists():
        print("[level2] level2_check.json already exists; not overwritten")
    else:
        dump_json(report, path)
    print(f"[level2] arrays under bounded discrepancy: {n_bounded} (limit {LEVEL2_MAX_BOUNDED_ARRAYS})")
    print("[level2] PASS" if ok else "[level2] FAIL - study stopped; see level2_check.json for diagnostics")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["level1", "level2", "pilot", "full"])
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--slot", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)

    import experiments.exp_tdh_router as R            # pure functions and constants only
    from experiments.exp4_generalization import MS, REP_DIR

    ids = prepare_out(out, R, REP_DIR, MS)
    if a.stage == "level1":
        sys.exit(0 if stage_level1(R, out, REP_DIR, MS) else 1)
    if not (out / "level1_check.json").exists() or not load_json(out / "level1_check.json")["pass"]:
        raise SystemExit("level1 has not passed; refusing to train")
    if a.stage == "level2":
        run_configs(R, out, REP_DIR, MS, FIRST_STAGE[:2], ids)
        sys.exit(0 if stage_level2(R, out, REP_DIR, MS) else 1)
    if not (out / "level2_check.json").exists() or not load_json(out / "level2_check.json")["pass"]:
        raise SystemExit("level2 has not passed; refusing to continue")
    if a.stage == "pilot":
        run_configs(R, out, REP_DIR, MS, FIRST_STAGE[2:], ids)
        return
    plan_path = out / "run_plan.json"
    if not plan_path.exists():
        raise SystemExit("run_plan.json missing; run 'summarize_label_budget.py plan' after the first stage")
    plan = load_json(plan_path)
    if not 0 <= a.slot < plan["nslots"]:
        raise SystemExit(f"--slot must be in [0, {plan['nslots']}) for this plan")
    # the plan fixes which configurations each slot runs, so parallel slots never race for the same one
    run_configs(R, out, REP_DIR, MS, [tuple(c) for c in plan["slots"][a.slot]], ids)


if __name__ == "__main__":
    main()
