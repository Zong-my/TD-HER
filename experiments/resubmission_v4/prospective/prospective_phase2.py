#!/usr/bin/env python3
"""Prospective families F1/F2, phase 2: calibration stage and endpoints. Frozen before the batch exists.

Protocol: claude_pssE_F1F2_protocol_v5_20261005.md with addenda a, b and c (analysis plan).

Everything that is fitted on a new event is fitted here, by the functions of the label-budget study
(experiments/resubmission_v4/label_budget/run_label_budget.py, unchanged, hash checked) and of the
production calibration stage (experiments/exp_tdh_router.py, unchanged, hash checked). For each family:

  budgets     100, 250 (if smaller than the calibration set) and the full calibration set;
  draws       five nested, class-stratified draws at each reduced budget; the full set once;
  per draw    the whole calibration stage sees only the drawn events: adapter (out-of-fold and final refit on
              source data plus the drawn events), admission rule (tau = 0.20), simplex weights per target,
              bounded affine correction, projection of the arrival time;
  rules       A: production rule; C: adapter always included; D: adapter never included;
  cell        family x budget x draw x target: MAE(C) - MAE(D) on the family's test set with a paired
              event-bootstrap 95% interval (2,000 resamples), and the decision of rule A;
  full budget in addition the rows of the manuscript's comparison table: each expert, the best single
              expert chosen on calibration data (raw), the same choice from the admitted set with the affine
              correction and projection, the base route, the adapter alone, TD-HER; paired contrasts; errors
              by disturbance class.

Stages:  selftest  the same functions on the existing L2 and L3 arrays must reproduce stored configurations
                   of the label-budget study (no new data involved);
         plan      checks the frozen output of phase 1 and writes the list of configurations;
         run       fits the configurations of one slot;
         summarize reads the configurations and writes the tables.
Every output goes to a new directory or a new file; nothing is overwritten.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import argparse, csv, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(str(REPO_ROOT))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_ROOT)
sys.path.insert(0, str(ROOT))

LB_PATH = ROOT / "experiments/resubmission_v4/label_budget/run_label_budget.py"
LB_SHA_PREFIX, ROUTER_SHA_PREFIX = '9db4a269fc96d93a277a91c79cab0c965a2172b2195b3b58d4db9a8101fc9e6b', '293ed1318ec2b3d0a038125ac51b9f8b5c065ea77bd17f6bcec7bad51c6b2939'
LB_RESULTS = work_path('results/ieee39/resubmission_v4_label_budget_v3')
SEED0P = 20261005
FAMILY_S = {"F1": 4, "F2": 5}
REDUCED, N_SEEDS, FULL_K = [100, 250], 5, 9
RULES = ["A_margin", "C_always", "D_never"]
ADAPTER = "LightGBM-Adapter"
TAU = 0.20
TARGETS = ["y1", "y2"]
IDENTITY_TOL = 1e-12
MS = 10


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_modules():
    import experiments.exp_tdh_router as R
    import experiments.exp4_generalization as G4
    if not sha(ROOT / "experiments/exp_tdh_router.py").startswith(ROUTER_SHA_PREFIX):
        sys.exit("exp_tdh_router.py is not the reviewed version")
    if not sha(LB_PATH).startswith(LB_SHA_PREFIX):
        sys.exit("run_label_budget.py is not the audited version")
    if abs(R.ADAPTER_ADMISSION_REL_IMPROV - TAU) > 0:
        sys.exit("production tau is not 0.20")
    spec = importlib.util.spec_from_file_location("label_budget_harness", LB_PATH)
    LB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LB)
    return R, G4, LB


def dump(obj, path):
    with open(path, "x") as f:                       # never overwrites
        json.dump(obj, f, indent=1, sort_keys=True, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


def mae2(y, p):
    return [float(np.mean(np.abs(y[:, t] - p[:, t]))) for t in range(2)]


# ----------------------------------------------------------------------------- data
def load_family(fam, p1, tables, R):
    """Arrays of one family from the frozen output of phase 1."""
    d = {"name": fam}
    for sp in ("cal", "test"):
        name = f"{fam}_{sp}"
        y = np.load(tables / f"repA/ms{MS}/y_{name}.npy")[:, :2].astype(float)
        Xn = np.load(p1 / f"X_n_{name}.npy")
        preds = {m: np.load(p1 / "experts" / m / f"{name}_preds.npy").astype(float) for m in R.ROUTER_MODELS}
        import pandas as pd
        df = pd.read_csv(tables / "csv" / f"{name}_ms{MS}.csv", usecols=["distu_kind", "fpu_deltamax", "t_delta"])
        if not np.array_equal(df[["fpu_deltamax", "t_delta"]].to_numpy(dtype=np.float32), y.astype(np.float32)):
            sys.exit(f"{name}: table and array targets are not row-identical")
        for m, p in preds.items():
            if p.shape != y.shape or not np.isfinite(p).all():
                sys.exit(f"{name}: prediction array of {m} has a wrong shape or non-finite values")
        if len(Xn) != len(y):
            sys.exit(f"{name}: inputs and targets differ in length")
        d[sp] = {"y": y, "X_n": Xn, "preds": preds, "cats": df["distu_kind"].to_numpy()}
    return d


def load_existing(scen, R, G4, LB, scalers):
    """The same structure from the existing shifted sets (self-test only)."""
    spec = LB.SCENARIOS[scen]
    d = {"name": scen}
    for sp, split in (("cal", spec["cal"]), ("test", spec["test"])):
        lv = G4.load_level_data(split, scalers)
        preds = {m: np.load(R.EXPERT_DIR / m / f"{split}_preds.npy").astype(float) for m in R.ROUTER_MODELS}
        cats = LB.load_categories({"cal": split}, lv["y"], MS) if sp == "cal" else None
        d[sp] = {"y": lv["y"][:, :2].astype(float), "X_n": lv["X_n"], "preds": preds, "cats": cats, "y_raw": lv["y"]}
    return d


def budgets_for(n_cal):
    return [b for b in REDUCED if b < n_cal] + [n_cal]


def configs_for(name, n_cal):
    b = budgets_for(n_cal)
    return [(name, b[-1], FULL_K)] + [(name, n, k) for n in b[:-1] for k in range(N_SEEDS)]


def cname(name, n, k):
    return f"{name}_n{n:04d}_k{k}"


# ----------------------------------------------------------------------------- rules and statistics
def apply_rules(R, y_cal, cal_preds, test_preds):
    """Rules A, C and D as in the label-budget study (its apply_rules without the withdrawn two-criterion rule)."""
    base_cal = {m: p for m, p in cal_preds.items() if m not in R.ADAPTER_NAMES}
    base_test = {m: p for m, p in test_preds.items() if m not in R.ADAPTER_NAMES}
    raw = {"A_margin": R.admission_affine_convex_blend_router(y_cal, cal_preds, test_preds),
           "C_always": R.affine_convex_blend_router(y_cal, cal_preds, test_preds),
           "D_never": R.affine_convex_blend_router(y_cal, base_cal, base_test)}
    return {k: R.enforce_y2_nonnegative(v[0]) for k, v in raw.items()}, {k: v[1] for k, v in raw.items()}


def classify(lo, hi):
    return "helps" if hi < 0 else ("harms" if lo > 0 else "unresolved")


def evaluate(R, LB, name, n, k, b_index, seed_of, y_cal, cal_preds, test_preds, y_test):
    routed, details = apply_rules(R, y_cal, cal_preds, test_preds)
    res = {"family": name, "n": int(n), "k": int(k), "targets": {}}
    for t, target in enumerate(TARGETS):
        mae = {r: float(np.mean(np.abs(routed[r][:, t] - y_test[:, t]))) for r in RULES}
        seed = seed_of(b_index, k, t)
        delta, lo, hi = LB.cell_interval(np.abs(routed["C_always"][:, t] - y_test[:, t]),
                                         np.abs(routed["D_never"][:, t] - y_test[:, t]), seed)
        adm, info = LB.admitted(details, "A_margin", target)
        ref = "C_always" if adm else "D_never"
        ident = float(np.max(np.abs(routed["A_margin"][:, t] - routed[ref][:, t])))
        a = details["A_margin"][target]["admission"]
        inclusion = classify(lo, hi)
        res["targets"][target] = {
            "test_MAE": mae, "C_minus_D": {"delta": delta, "ci95": [lo, hi], "bootstrap_seed": int(seed)},
            "inclusion": inclusion, "A_admits": bool(adm),
            "outcome": ("admitted, " if adm else "declined, ") + {"helps": "inclusion helps", "harms": "inclusion harms",
                                                                  "unresolved": "unresolved"}[inclusion],
            "oof_adapter_MAE": float(info["cal_MAE"]), "required_MAE": float(info["required_MAE"]),
            "base_best_model": a["base_best_model"], "base_best_cal_MAE": float(a["base_best_cal_MAE"]),
            "oof_improvement_over_best_base": float(1.0 - info["cal_MAE"] / a["base_best_cal_MAE"]),
            "weights": {r: details[r][target]["weights"] for r in RULES},
            "affine": {r: {"slope": details[r][target]["slope"], "bias": details[r][target]["bias"]} for r in RULES},
            "identity": {"A_equals": ref, "max_abs_diff": ident, "pass": bool(ident <= IDENTITY_TOL)}}
    res["identity_pass"] = bool(all(v["identity"]["pass"] for v in res["targets"].values()))
    return res, routed, details


def full_budget_rows(R, y_cal, cal_preds, test_preds, y_test, routed, cats_test):
    """Rows of the manuscript's comparison table at the full budget, paired contrasts, errors by class."""
    base_cal = {m: p for m, p in cal_preds.items() if m not in R.ADAPTER_NAMES}
    base_test = {m: p for m, p in test_preds.items() if m not in R.ADAPTER_NAMES}
    rows, info = {}, {}
    for m in base_test:
        rows[f"expert:{m}"] = base_test[m]
    best_raw, sel = R.best_expert_router(y_cal, base_cal, base_test)
    rows["best_single_raw"] = best_raw
    info["best_single_raw"] = {t: sel[t]["expert"] for t in TARGETS}
    a1 = np.zeros_like(y_test)
    info["matched_single"] = {}
    for t, tn in enumerate(TARGETS):
        models, _ = R.admitted_models_for_target(y_cal, cal_preds, t, TAU)
        scores = {m: R._mae(y_cal[:, t], cal_preds[m][:, t]) for m in models}
        best = min(scores, key=scores.get)
        a, b, _ = R.fit_restricted_affine(y_cal[:, t], cal_preds[best][:, t])
        a1[:, t] = R.apply_affine(test_preds[best][:, t], a, b)
        info["matched_single"][tn] = {"expert": best, "slope": a, "bias": b, "admitted_set": models}
    rows["matched_single"] = R.enforce_y2_nonnegative(a1)
    rows["adapter_alone_raw"] = test_preds[ADAPTER]
    adj = np.zeros_like(y_test)
    for t in range(2):
        a, b, _ = R.fit_restricted_affine(y_cal[:, t], cal_preds[ADAPTER][:, t])
        adj[:, t] = R.apply_affine(test_preds[ADAPTER][:, t], a, b)
    rows["adapter_alone_matched"] = R.enforce_y2_nonnegative(adj)
    rows["base_route"] = routed["D_never"]
    rows["with_adapter_always"] = routed["C_always"]
    rows["tdher"] = routed["A_margin"]
    mae = {k: mae2(y_test, p) for k, p in rows.items()}

    def boot(cand, ref):
        b = R.paired_bootstrap_delta(y_test, rows[cand], rows[ref])
        out = {}
        for tn in TARGETS:
            lo, hi = b[tn]["ci95"]
            d, r = b[tn]["delta_MAE_candidate_minus_reference"], b[tn]["reference_MAE"]
            same = bool(np.array_equal(rows[cand][:, TARGETS.index(tn)], rows[ref][:, TARGETS.index(tn)]))
            out[tn] = {"delta": d, "relative_change_pct": (100.0 * d / r if r > 0 else None), "ci95": [lo, hi],
                       "identical_predictions": same, "resolved": bool((hi < 0 or lo > 0) and not same)}
        return out
    contrasts = {f"{a}_minus_{b}": boot(a, b) for a, b in [("tdher", "matched_single"), ("tdher", "best_single_raw"),
                                                             ("tdher", "base_route"), ("tdher", "adapter_alone_raw"),
                                                             ("matched_single", "best_single_raw")]}
    by_class = {}
    for c in sorted(set(cats_test)):
        m = cats_test == c
        by_class[c] = {"n": int(m.sum()), **{k: mae2(y_test[m], rows[k][m]) for k in ["tdher", "matched_single", "base_route", "best_single_raw"]}}
    return rows, mae, info, contrasts, by_class


# ----------------------------------------------------------------------------- one configuration
def run_config(R, LB, out, data, n, k, seed0, s, specs, X_base, y_base, budgets, full_extras=True):
    name = data["name"]
    cal, test = data["cal"], data["test"]
    full = budgets[-1]
    if n == full:
        idx = np.arange(len(cal["y"]))
    else:
        idx = LB.nested_stratified_indices(cal["cats"], budgets, seed0 + k)[n]
    assert len(idx) == n, (name, n, len(idx))
    t0 = time.time()
    oof, test_pred, fits = LB.train_adapter(R, specs, X_base, y_base, cal["X_n"][idx], cal["y_fit"][idx], test["X_n"])
    cal_preds = {m: p[idx] for m, p in cal["preds"].items()}
    test_preds = dict(test["preds"])
    cal_preds[ADAPTER], test_preds[ADAPTER] = oof, test_pred
    seed_of = lambda b, kk, t: seed0 + 1000 * s + 100 * b + 10 * kk + t
    res, routed, details = evaluate(R, LB, name, n, k, budgets.index(n), seed_of, cal["y"][idx], cal_preds, test_preds, test["y"])
    res.update({"fits": fits, "wall_seconds": time.time() - t0, "peak_rss_gb": LB.peak_rss_gb(),
                "categories_in_sample": {str(a): int(b) for a, b in zip(*np.unique(cal["cats"][idx], return_counts=True))},
                "n_cal_total": int(len(cal["y"])), "n_test": int(len(test["y"]))})
    tmp = out / "configs" / f".tmp_{cname(name, n, k)}_{os.getpid()}"
    tmp.mkdir(parents=True)
    np.save(tmp / "subsample_indices.npy", idx)
    np.save(tmp / "adapter_cal_oof.npy", oof)
    np.save(tmp / "adapter_test.npy", test_pred)
    for r in RULES:
        np.save(tmp / f"routed_{r}.npy", routed[r])
    if n == full and full_extras:
        rows, mae, info, contrasts, by_class = full_budget_rows(R, cal["y"][idx], cal_preds, test_preds, test["y"], routed, test["cats"])
        res["full_budget"] = {"test_MAE": mae, "selection": info, "contrasts": contrasts, "by_class": by_class}
        np.save(tmp / "row_matched_single.npy", rows["matched_single"])
        np.save(tmp / "row_best_single_raw.npy", rows["best_single_raw"])
    with open(tmp / "result.json", "w") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=float)
    os.rename(tmp, out / "configs" / cname(name, n, k))          # atomic; fails if the target exists
    print(f"[run] {cname(name, n, k)}: {res['wall_seconds']:.0f} s | " + " | ".join(
        f"{t}: {'adm' if v['A_admits'] else 'rej'}" for t, v in res["targets"].items()), flush=True)
    if not res["identity_pass"]:
        sys.exit(f"identity check failed in {cname(name, n, k)}")
    return res


def base_training(R, G4):
    scalers = R.attach_base_training_data(G4.load_train_scalers())
    return scalers, R.load_exp1_lgb_specs(), scalers["X_train_n"], scalers["y_train"]


# ----------------------------------------------------------------------------- stages
def stage_selftest(out, slot):
    """Existing L2/L3 data through this script must reproduce stored configurations of the label-budget study."""
    R, G4, LB = load_modules()
    jobs = [("L2", 510, FULL_K), ("L3", 2337, FULL_K), ("L2", 100, 0)]
    if slot is not None:
        jobs = jobs[slot::3]
    scalers, specs, X_base, y_base = base_training(R, G4)
    out.mkdir(parents=True, exist_ok=True)
    for scen, n, k in jobs:
        spec = LB.SCENARIOS[scen]
        data = load_existing(scen, R, G4, LB, scalers)
        data["cal"]["y_fit"] = data["cal"]["y_raw"]              # the harness fits on the stored targets as loaded
        data["test"]["cats"] = np.array(["all"] * len(data["test"]["y"]))
        budgets = spec["budgets"]
        res = run_config(R, LB, out, data, n, k, LB.SEED0, spec["s"], specs, X_base, y_base, budgets, full_extras=False)
        ref_dir = LB_RESULTS / "configs" / LB.config_name(scen, n, k)
        ref = json.load(open(ref_dir / "result.json"))
        new_dir = out / "configs" / cname(scen, n, k)
        rep = {"config": cname(scen, n, k), "reference": str(ref_dir)}
        rep["subsample_indices_equal"] = bool(np.array_equal(np.load(new_dir / "subsample_indices.npy"), np.load(ref_dir / "subsample_indices.npy")))
        idx = np.load(ref_dir / "subsample_indices.npy")
        truth = {"adapter_cal_oof": data["cal"]["y"][idx], "adapter_test": data["test"]["y"]}
        rep["adapter_arrays"] = {}
        for arr in ("adapter_cal_oof", "adapter_test"):
            for t, tn in enumerate(TARGETS):
                rep["adapter_arrays"][f"{arr}_{tn}"] = LB.adapter_array_condition(
                    np.load(new_dir / f"{arr}.npy")[:, t], np.load(ref_dir / f"{arr}.npy")[:, t], truth[arr][:, t])
        rep["targets"] = {}
        for tn in TARGETS:
            a, b = res["targets"][tn], ref["targets"][tn]
            rep["targets"][tn] = {
                "decision_equal": a["A_admits"] == b["A_admits"],
                "test_MAE_new": a["test_MAE"], "test_MAE_ref": {r: b["test_MAE"][r] for r in RULES},
                "max_rel_MAE_diff": max(abs(a["test_MAE"][r] - b["test_MAE"][r]) / b["test_MAE"][r] for r in RULES),
                "C_minus_D_new": a["C_minus_D"], "C_minus_D_ref": b["C_minus_D"],
                "seed_equal": a["C_minus_D"]["bootstrap_seed"] == b["C_minus_D"]["bootstrap_seed"]}
        for r in RULES:
            rep[f"routed_{r}_max_abs_diff"] = float(np.max(np.abs(np.load(new_dir / f"routed_{r}.npy") - np.load(ref_dir / f"routed_{r}.npy"))))
        rep["pass"] = bool(rep["subsample_indices_equal"]
                           and all(v["condition"] != "FAILED" for v in rep["adapter_arrays"].values())
                           and all(v["decision_equal"] and v["seed_equal"] and v["max_rel_MAE_diff"] <= 1e-3 for v in rep["targets"].values()))
        dump(rep, out / f"selftest_{cname(scen, n, k)}.json")
        print(f"[selftest] {cname(scen, n, k)}: pass={rep['pass']} | arrays "
              + ", ".join(f"{a}={v['condition']}" for a, v in rep["adapter_arrays"].items())
              + " | max rel MAE diff " + ", ".join(f"{tn}={v['max_rel_MAE_diff']:.2e}" for tn, v in rep["targets"].items()), flush=True)


def check_phase1(p1, tables):
    man = json.load(open(p1 / "phase1_manifest.json"))
    bad = [f for f, h in man["files"].items() if not Path(f).exists() or sha(f) != h]
    if bad:
        sys.exit(f"phase-1 output changed since its manifest was written: {bad[:5]}")
    rep = json.load(open(p1 / "phase1_report.json"))
    if rep["mode"] != "batch" or rep.get("smoke_v5"):
        sys.exit("phase 2 runs on the batch output of phase 1 only")
    return sha(p1 / "phase1_manifest.json"), rep


def stage_plan(p1, tables, out):
    R, G4, LB = load_modules()
    man_sha, rep = check_phase1(p1, tables)
    out.mkdir(parents=True, exist_ok=False)
    plan = {"phase1_manifest_sha256": man_sha, "seed0": SEED0P, "families": {}, "configs": [],
            "code_sha256": {"prospective_phase2.py": sha(os.path.abspath(__file__)), "run_label_budget.py": sha(LB_PATH),
                            "exp_tdh_router.py": sha(ROOT / "experiments/exp_tdh_router.py"),
                            "exp4_generalization.py": sha(ROOT / "experiments/exp4_generalization.py")}}
    for fam in ("F1", "F2"):
        d = load_family(fam, p1, tables, R)
        n_cal, n_test = len(d["cal"]["y"]), len(d["test"]["y"])
        cats = {sp: {str(a): int(b) for a, b in zip(*np.unique(d[sp]["cats"], return_counts=True))} for sp in ("cal", "test")}
        plan["families"][fam] = {"n_cal": n_cal, "n_test": n_test, "budgets": budgets_for(n_cal), "classes": cats, "s": FAMILY_S[fam]}
        plan["configs"] += [list(c) for c in configs_for(fam, n_cal)]
    dump(plan, out / "run_plan.json")
    print(json.dumps(plan["families"], indent=1)); print(len(plan["configs"]), "configurations")


def stage_run(p1, tables, out, slot, nslots):
    R, G4, LB = load_modules()
    plan = json.load(open(out / "run_plan.json"))
    man_sha, _ = check_phase1(p1, tables)
    if man_sha != plan["phase1_manifest_sha256"] or plan["code_sha256"]["prospective_phase2.py"] != sha(os.path.abspath(__file__)):
        sys.exit("phase-1 output or this script changed since the plan was written")
    todo = [tuple(c) for c in plan["configs"]][slot::nslots]
    todo = [c for c in todo if not (out / "configs" / cname(*c) / "result.json").exists()]
    if not todo:
        return
    scalers, specs, X_base, y_base = base_training(R, G4)
    cache = {}
    for fam, n, k in todo:
        if fam not in cache:
            d = load_family(fam, p1, tables, R)
            d["cal"]["y_fit"] = np.load(tables / f"repA/ms{MS}/y_{fam}_cal.npy")     # targets as stored, as in the harness
            cache[fam] = d
        run_config(R, LB, out, cache[fam], n, k, SEED0P, FAMILY_S[fam], specs, X_base, y_base, plan["families"][fam]["budgets"])


def stage_summarize(out):
    plan = json.load(open(out / "run_plan.json"))
    cells, full = [], {}
    for fam, n, k in [tuple(c) for c in plan["configs"]]:
        p = out / "configs" / cname(fam, n, k) / "result.json"
        if not p.exists():
            sys.exit(f"configuration missing: {cname(fam, n, k)}")
        r = json.load(open(p))
        if not r["identity_pass"]:
            sys.exit(f"identity check failed in {cname(fam, n, k)}")
        for tn in TARGETS:
            v = r["targets"][tn]
            cells.append({"family": fam, "n": n, "k": k, "target": tn, "n_cal_total": r["n_cal_total"], "n_test": r["n_test"],
                          "A_admits": v["A_admits"], "inclusion": v["inclusion"], "outcome": v["outcome"],
                          "oof_adapter_MAE": v["oof_adapter_MAE"], "required_MAE": v["required_MAE"],
                          "base_best_model": v["base_best_model"], "base_best_cal_MAE": v["base_best_cal_MAE"],
                          "oof_improvement_over_best_base": v["oof_improvement_over_best_base"],
                          "MAE_A": v["test_MAE"]["A_margin"], "MAE_C": v["test_MAE"]["C_always"], "MAE_D": v["test_MAE"]["D_never"],
                          "C_minus_D": v["C_minus_D"]["delta"], "ci_lo": v["C_minus_D"]["ci95"][0], "ci_hi": v["C_minus_D"]["ci95"][1]})
        if "full_budget" in r:
            full[fam] = r["full_budget"]
            full[fam]["n_cal"], full[fam]["n_test"] = r["n_cal_total"], r["n_test"]
    from collections import Counter
    summary = {"cells": len(cells), "outcomes_all": dict(Counter(c["outcome"] for c in cells)),
               "outcomes_by_family": {f: dict(Counter(c["outcome"] for c in cells if c["family"] == f)) for f in ("F1", "F2")},
               "harmful_admissions": [c for c in cells if c["outcome"] == "admitted, inclusion harms"],
               "full_budget": full, "plan": plan["families"]}
    dump(summary, out / "summary.json")
    with open(out / "summary_cells.csv", "x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
        w.writeheader(); w.writerows(cells)
    print(json.dumps({k: summary[k] for k in ("cells", "outcomes_all", "outcomes_by_family")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["selftest", "plan", "run", "summarize"])
    ap.add_argument("--phase1-results")
    ap.add_argument("--tables")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slot", type=int)
    ap.add_argument("--nslots", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)
    if a.stage == "selftest":
        return stage_selftest(out, a.slot)
    if a.stage == "summarize":
        return stage_summarize(out)
    p1, tables = Path(a.phase1_results), Path(a.tables)
    if a.stage == "plan":
        return stage_plan(p1, tables, out)
    stage_run(p1, tables, out, a.slot or 0, a.nslots)


if __name__ == "__main__":
    main()
