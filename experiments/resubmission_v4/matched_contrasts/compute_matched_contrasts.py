#!/usr/bin/env python3
"""Protocol A (v2, 2026-10-05): matched single-expert contrasts on frozen predictions.

No training and no model inference. Every row is computed from the frozen expert and
adapter predictions saved by ``experiments/exp_tdh_router.py`` with that module's own
functions, imported unchanged. Saved production rows (TD-HER, base route, shared route,
best single expert) are loaded, never refitted. Nothing is fitted or selected on test
labels. Outputs go to a new versioned directory.

New rows
  A1   target-wise selection, matched: per target, the member of the admitted set M_j
       (same admission call as TD-HER, tau = 0.20) with the lowest calibration MAE of its
       raw prediction (out-of-fold for the adapter); restricted affine fitted on its
       calibration prediction; y2 projection.  = the routing problem restricted to the
       vertices of the simplex, same post-processing.
  A1s  sensitivity: as A1, expert chosen by calibration MAE after the affine fit.
  A2   one expert for both targets, matched: pool = union of the two admitted sets (the
       convention of the shared route); expert minimising the mean over targets of
       calibration MAE / scale (scales from _target_balance_scales on that pool);
       per-target affine; projection.
  A3   uniform average, matched: mean of the raw predictions over M_j; affine; projection.

Also: test MAE by disturbance class (all classes, with counts) for the seven experts, the
adapter, the best single expert, A1 and TD-HER, and the paired TD-HER minus A1 contrast per
class. Descriptive; nothing is tuned or selected on these slices.
"""
from __future__ import annotations

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_ROOT)  # exp_tdh_router uses project-relative paths

import experiments.exp_tdh_router as R  # noqa: E402

ROUTER_SHA256_PREFIX = "800d957e"
TAU = 0.20
OUT_BASE = work_path('results/ieee39/resubmission_v4_matched_contrasts')
EQUAL_BANK = work_path('results/ieee39/resubmission_v3_equal_bank')
EVENT_INDEX = (WORK_ROOT / "analytical"
               / "execution/shared/event_index.csv")
TRUTH = (WORK_ROOT / "analytical"
         / "execution/shared/eval_truth/truth_v1/security_truth.csv")
SCEN_SHORT = {"L1_same_dist": "L1", "L2_cross_cond_fewshot": "L2", "L3_cross_topo_fewshot": "L3"}
TARGETS = ("y1", "y2")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mae2(y: np.ndarray, p: np.ndarray) -> list[float]:
    return [float(v) for v in np.mean(np.abs(p - y), axis=0)]


def fresh_out_dir() -> Path:
    v = 1
    while (OUT_BASE.parent / f"{OUT_BASE.name}_v{v}").exists():
        v += 1
    out = OUT_BASE.parent / f"{OUT_BASE.name}_v{v}"
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


def fail(log, msg: str):
    log("CHECK FAILED:", msg)
    raise SystemExit(f"CHECK FAILED: {msg}")


def boot(y, cand, ref) -> dict:
    """Paired event bootstrap of MAE(cand) - MAE(ref); production function, seed 42."""
    b = R.paired_bootstrap_delta(y, cand, ref)
    out = {}
    for t in TARGETS:
        lo, hi = b[t]["ci95"]
        out[t] = {
            "delta": b[t]["delta_MAE_candidate_minus_reference"],
            "candidate_MAE": b[t]["candidate_MAE"],
            "reference_MAE": b[t]["reference_MAE"],
            "relative_change_pct": (100.0 * b[t]["delta_MAE_candidate_minus_reference"] / b[t]["reference_MAE"]
                                    if b[t]["reference_MAE"] > 0 else None),
            "ci95": [lo, hi],
            "resolved": bool(hi < 0.0 or lo > 0.0),
        }
    return out


def load_classes(log) -> dict:
    """Disturbance class per test row, from the verified Phase-1 event index."""
    rows = list(csv.DictReader(open(EVENT_INDEX)))
    truth = {}
    first_endpoint = None
    with open(TRUTH) as f:
        for r in csv.DictReader(f):
            if first_endpoint is None:
                first_endpoint = r["endpoint_id"]
            if r["endpoint_id"] == first_endpoint:
                truth[r["event_id"]] = (float(r["y1_true_hz"]), float(r["y2_true_rel_s"]))
    out = {}
    for s in ("L1", "L2", "L3"):
        # the index holds test events only; the split name differs by scenario (test, cross_cond_test, ...)
        sel = [r for r in rows if r["scenario"] == s]
        splits = sorted({r["active_split"] for r in sel})
        if not sel or any("test" not in sp for sp in splits):
            fail(log, f"event index of {s}: unexpected splits {splits}")
        log(f"  event index {s}: {len(sel)} rows, split(s) {splits}")
        n = len(sel)
        idx = np.array([int(r["row_index"]) for r in sel])
        if sorted(idx.tolist()) != list(range(n)):
            fail(log, f"event index of {s} is not a permutation of 0..{n-1}")
        cls = np.empty(n, dtype=object)
        yt = np.full((n, 2), np.nan)
        for r in sel:
            i = int(r["row_index"])
            cls[i] = r["disturbance_category"]
            if r["event_id"] in truth:
                yt[i] = truth[r["event_id"]]
        out[s] = {"cls": cls, "truth": yt, "sha_index": sha(EVENT_INDEX)}
    return out


def main() -> None:
    out_dir = fresh_out_dir()
    log = Tee(out_dir / "run.log")
    t0 = time.time()
    log("Protocol A v2: matched single-expert contrasts")
    log("output directory:", out_dir)

    router_sha = sha(PROJECT_ROOT / "experiments/exp_tdh_router.py")
    if not router_sha.startswith(ROUTER_SHA256_PREFIX):
        fail(log, f"exp_tdh_router.py hash {router_sha[:12]} differs from the reviewed {ROUTER_SHA256_PREFIX}")
    if abs(R.ADAPTER_ADMISSION_REL_IMPROV - TAU) > 0:
        fail(log, "production tau is not 0.20")

    audit = json.loads((R.OUT_DIR / "audit_report.json").read_text())
    rep = Path(audit["rep_dir"]) / "repA" / "ms10"
    eq_manifest = json.loads((EQUAL_BANK / "input_sha256_manifest.json").read_text())
    eq_results = json.loads((EQUAL_BANK / "equal_bank_results.json").read_text())
    prod_summary = json.loads((R.OUT_DIR / "metrics_summary.json").read_text())
    classes = load_classes(log)

    manifest: dict[str, str] = {}
    n_manifest_checked = 0

    def track(path: Path) -> None:
        nonlocal n_manifest_checked
        key = str(path)
        h = sha(path)
        manifest[key] = h
        if key in eq_manifest:
            n_manifest_checked += 1
            if eq_manifest[key] != h:
                fail(log, f"hash of {key} differs from the equal-bank manifest")

    results: dict = {}
    per_class_rows: list[dict] = []
    rows_dir = out_dir / "rows"
    rows_dir.mkdir()

    for scen, (cal, test) in R.SCENARIOS.items():
        s = SCEN_SHORT[scen]
        log(f"\n=== {scen} (cal={cal}, test={test}) ===")
        y_cal_p, y_test_p = rep / f"y_{cal}.npy", rep / f"y_{test}.npy"
        track(y_cal_p), track(y_test_p)
        y_cal = np.load(y_cal_p)[:, :2].astype(float)
        y_test = np.load(y_test_p)[:, :2].astype(float)

        cal_preds, test_preds = {}, {}
        for m in R.ROUTER_MODELS:
            cp, tp = R.EXPERT_DIR / m / f"{cal}_preds.npy", R.EXPERT_DIR / m / f"{test}_preds.npy"
            track(cp), track(tp)
            cal_preds[m] = np.load(cp).astype(float)
            test_preds[m] = np.load(tp).astype(float)
        ad_dir = R.ADAPTER_DIR / scen / "LightGBM-Adapter"
        track(ad_dir / "cal_oof_preds.npy"), track(ad_dir / "test_preds.npy")
        ad_cal = np.load(ad_dir / "cal_oof_preds.npy").astype(float)
        ad_test = np.load(ad_dir / "test_preds.npy").astype(float)
        cal_all = dict(cal_preds); cal_all["LightGBM-Adapter"] = ad_cal
        test_all = dict(test_preds); test_all["LightGBM-Adapter"] = ad_test
        for d in (cal_all, test_all):
            for m, p in d.items():
                ref_shape = y_cal.shape if d is cal_all else y_test.shape
                if p.shape != ref_shape or not np.isfinite(p).all():
                    fail(log, f"{scen}: prediction array of {m} has a wrong shape or non-finite values")

        sc = R.OUT_DIR / "scenarios" / scen

        def saved(name: str) -> np.ndarray:
            p = sc / f"{name}_preds.npy"
            track(p)
            return np.load(p).astype(float)

        rows: dict[str, np.ndarray] = {}
        rows["best_single_raw"] = saved("base_best_expert")
        rows["simplex_base_full"] = saved("base_affine_convex_blend_nonnegative_y2")
        rows["shared_full"] = saved("admission_shared_affine_convex_blend_nonnegative_y2")
        rows["tdher"] = saved("admission_affine_convex_blend_nonnegative_y2")

        # --- check (ii): loaded rows reproduce the equal-bank table exactly
        eqr = eq_results[scen]["rows"]
        for key_new, key_eq in (("best_single_raw", "best_single"), ("simplex_base_full", "simplex_base_full"),
                                ("shared_full", "shared_full"), ("tdher", "tdher")):
            m_now = mae2(y_test, rows[key_new])
            m_ref = [eqr[key_eq]["y1_MAE"], eqr[key_eq]["y2_MAE"]]
            if max(abs(a - b) for a, b in zip(m_now, m_ref)) > 1e-12:
                fail(log, f"{scen}: loaded row {key_new} does not reproduce the equal-bank value")

        # --- admitted sets (same call as TD-HER)
        admitted, admission = {}, {}
        for t, tn in enumerate(TARGETS):
            models, adm = R.admitted_models_for_target(y_cal, cal_all, t, TAU)
            admitted[tn], admission[tn] = models, adm
            ad = adm["adapter_details"]["LightGBM-Adapter"]
            log(f"  admission {tn}: best base {adm['base_best_model']} cal MAE {adm['base_best_cal_MAE']:.6f}; "
                f"required {ad['required_MAE']:.6f}; adapter OOF {ad['cal_MAE']:.6f}; admitted={ad['admitted']}")
        union_models, union_adm = R.admitted_models_for_shared_route(y_cal, cal_all, TAU)
        sets_coincide = (admitted["y1"] == admitted["y2"] == union_models)

        # --- A1 and A1s
        a1 = np.zeros_like(y_test); a1s = np.zeros_like(y_test)
        a1_info, a1s_info = {}, {}
        for t, tn in enumerate(TARGETS):
            raw_scores = {m: R._mae(y_cal[:, t], cal_all[m][:, t]) for m in admitted[tn]}
            best = min(raw_scores, key=raw_scores.get)
            a, b, info = R.fit_restricted_affine(y_cal[:, t], cal_all[best][:, t])
            a1[:, t] = R.apply_affine(test_all[best][:, t], a, b)
            a1_info[tn] = {"expert": best, "raw_cal_MAE": raw_scores[best], "slope": a, "bias": b,
                           "affine_cal_MAE": info["cal_MAE"], "affine_success": info["success"],
                           "all_raw_cal_MAE": raw_scores}
            cands = {}
            for m in admitted[tn]:
                am, bm, im = R.fit_restricted_affine(y_cal[:, t], cal_all[m][:, t])
                cands[m] = {"slope": am, "bias": bm, "affine_cal_MAE": im["cal_MAE"], "success": im["success"]}
            best_s = min(cands, key=lambda m: cands[m]["affine_cal_MAE"])
            a1s[:, t] = R.apply_affine(test_all[best_s][:, t], cands[best_s]["slope"], cands[best_s]["bias"])
            a1s_info[tn] = {"expert": best_s, **cands[best_s],
                            "all_affine_cal_MAE": {m: c["affine_cal_MAE"] for m, c in cands.items()}}
        rows["A1"] = R.enforce_y2_nonnegative(a1)
        rows["A1s"] = R.enforce_y2_nonnegative(a1s)

        # --- A2: one expert for both targets, pool = union of admitted sets
        scales = R._target_balance_scales(y_cal, cal_all, union_models)
        obj = {m: float(np.mean([R._mae(y_cal[:, t], cal_all[m][:, t]) / scales[t] for t in range(2)]))
               for m in union_models}
        best2 = min(obj, key=obj.get)
        a2 = np.zeros_like(y_test); a2_info = {"expert": best2, "balanced_objective": obj[best2],
                                               "all_balanced_objective": obj,
                                               "scales": {"y1": float(scales[0]), "y2": float(scales[1])},
                                               "pool": union_models}
        for t, tn in enumerate(TARGETS):
            a, b, info = R.fit_restricted_affine(y_cal[:, t], cal_all[best2][:, t])
            a2[:, t] = R.apply_affine(test_all[best2][:, t], a, b)
            a2_info[tn] = {"slope": a, "bias": b, "affine_cal_MAE": info["cal_MAE"]}
        rows["A2"] = R.enforce_y2_nonnegative(a2)

        # --- A3: uniform average over M_j, matched post-processing
        a3 = np.zeros_like(y_test); a3_info = {}
        for t, tn in enumerate(TARGETS):
            cmean = np.mean(np.stack([cal_all[m][:, t] for m in admitted[tn]], axis=1), axis=1)
            tmean = np.mean(np.stack([test_all[m][:, t] for m in admitted[tn]], axis=1), axis=1)
            a, b, info = R.fit_restricted_affine(y_cal[:, t], cmean)
            a3[:, t] = R.apply_affine(tmean, a, b)
            a3_info[tn] = {"members": admitted[tn], "slope": a, "bias": b, "affine_cal_MAE": info["cal_MAE"]}
        rows["A3"] = R.enforce_y2_nonnegative(a3)

        # --- adapter only, matched (as in the equal-bank table) for check (iii)
        adj = np.zeros_like(ad_test)
        for t in range(2):
            a, b, _ = R.fit_restricted_affine(y_cal[:, t], ad_cal[:, t])
            adj[:, t] = R.apply_affine(ad_test[:, t], a, b)
        rows["adapter_matched"] = R.enforce_y2_nonnegative(adj)
        m_now = mae2(y_test, rows["adapter_matched"])
        m_ref = [eqr["adapter_matched"]["y1_MAE"], eqr["adapter_matched"]["y2_MAE"]]
        rel = max(abs(a - b) / b for a, b in zip(m_now, m_ref))
        log(f"  adapter-only matched: {m_now[0]:.6f}/{m_now[1]:.6f}; stored {m_ref[0]:.6f}/{m_ref[1]:.6f}; rel diff {rel:.2e}")
        if rel > 1e-9:
            fail(log, f"{scen}: adapter-only matched row differs from the stored value by {rel:.2e} relative")

        # --- check (iii)/(iv): expected selections
        if s == "L3":
            if not all(a1_info[tn]["expert"] == "LightGBM-Adapter" for tn in TARGETS):
                fail(log, "L3: A1 does not select the adapter for both targets")
            if np.max(np.abs(rows["A1"] - rows["adapter_matched"])) > 0:
                fail(log, "L3: A1 differs from the adapter-only matched row")
        else:
            prod = prod_summary[scen]["base_best_expert"]["details"]
            for tn in TARGETS:
                if a1_info[tn]["expert"] != prod[tn]["expert"]:
                    fail(log, f"{scen}: A1 expert for {tn} differs from the production best single expert")
        for k, p in rows.items():
            if p.shape != y_test.shape or not np.isfinite(p).all():
                fail(log, f"{scen}: row {k} has a wrong shape or non-finite values")
            np.save(rows_dir / f"{k}_{s}.npy", p)

        # --- MAE and contrasts
        mae = {k: mae2(y_test, p) for k, p in rows.items()}
        mae["uniform_raw_base"] = mae2(y_test, np.mean(np.stack([test_preds[m] for m in R.ROUTER_MODELS]), axis=0))
        if max(abs(a - b) for a, b in zip(mae["uniform_raw_base"],
                                          [eqr["uniform_avg"]["y1_MAE"], eqr["uniform_avg"]["y2_MAE"]])) > 1e-12:
            fail(log, f"{scen}: uniform average does not reproduce the equal-bank value")
        contrasts = {
            "tdher_minus_A1": boot(y_test, rows["tdher"], rows["A1"]),
            "tdher_minus_A2": boot(y_test, rows["tdher"], rows["A2"]),
            "A1_minus_A2": boot(y_test, rows["A1"], rows["A2"]),
            "tdher_minus_A3": boot(y_test, rows["tdher"], rows["A3"]),
            "tdher_minus_A1s": boot(y_test, rows["tdher"], rows["A1s"]),
            "tdher_minus_best_single_raw": boot(y_test, rows["tdher"], rows["best_single_raw"]),
            "A1_minus_best_single_raw": boot(y_test, rows["A1"], rows["best_single_raw"]),
        }
        # the existing interval for TD-HER vs the raw best single expert must be reproduced exactly
        ref_b = eqr["best_single"]["bootstrap_full_minus_row"]
        for tn in TARGETS:
            if max(abs(a - b) for a, b in zip(contrasts["tdher_minus_best_single_raw"][tn]["ci95"], ref_b[tn]["ci95"])) > 1e-12:
                fail(log, f"{scen}: bootstrap interval vs best single expert is not reproduced for {tn}")

        for k in ("best_single_raw", "A1", "A1s", "A2", "A3", "simplex_base_full", "shared_full", "adapter_matched", "tdher"):
            log(f"  {k:20s} y1={mae[k][0]:.6f}  y2={mae[k][1]:.6f}")
        log(f"  A1 experts: y1={a1_info['y1']['expert']}, y2={a1_info['y2']['expert']}; "
            f"A1s: y1={a1s_info['y1']['expert']}, y2={a1s_info['y2']['expert']}; A2: {best2}; sets coincide: {sets_coincide}")
        for name in ("tdher_minus_A1", "tdher_minus_A2", "A1_minus_A2", "tdher_minus_A3", "tdher_minus_A1s"):
            c = contrasts[name]
            log("  %-18s " % name + "; ".join(
                f"{tn}: {c[tn]['delta']:+.6f} ({c[tn]['relative_change_pct']:+.2f}%) CI [{c[tn]['ci95'][0]:+.6f}, {c[tn]['ci95'][1]:+.6f}]"
                f"{'' if c[tn]['resolved'] else ' unresolved'}" for tn in TARGETS))

        # --- by disturbance class (all classes)
        cinfo = classes[s]
        if len(cinfo["cls"]) != len(y_test):
            fail(log, f"{scen}: event index has {len(cinfo['cls'])} rows, test split has {len(y_test)}")
        dmax = float(np.nanmax(np.abs(cinfo["truth"] - y_test)))
        n_missing = int(np.isnan(cinfo["truth"]).any(axis=1).sum())
        log(f"  class index alignment: max |truth - y_test| = {dmax:.3e}; rows without truth: {n_missing}")
        if dmax > 1e-6 or n_missing > 0:
            fail(log, f"{scen}: event index is not aligned with the test labels")
        by_class = {}
        for c in sorted(set(cinfo["cls"].tolist())):
            msk = cinfo["cls"] == c
            n_c = int(msk.sum())
            entry = {"N": n_c, "MAE": {}}
            for m in R.ROUTER_MODELS:
                entry["MAE"][m] = mae2(y_test[msk], test_preds[m][msk])
            entry["MAE"]["LightGBM-Adapter (final, raw)"] = mae2(y_test[msk], ad_test[msk])
            for k in ("best_single_raw", "A1", "simplex_base_full", "tdher"):
                entry["MAE"][k] = mae2(y_test[msk], rows[k][msk])
            entry["mean_abs_y1"] = float(np.mean(np.abs(y_test[msk, 0])))
            entry["median_y2"] = float(np.median(y_test[msk, 1]))
            entry["tdher_minus_A1"] = boot(y_test[msk], rows["tdher"][msk], rows["A1"][msk]) if n_c >= 2 else None
            by_class[c] = entry
            for k, v in entry["MAE"].items():
                per_class_rows.append({"scenario": s, "class": c, "N": n_c, "row": k, "y1_MAE": v[0], "y2_MAE": v[1]})
            log(f"  class {c:14s} N={n_c:5d}  A1 {entry['MAE']['A1'][0]:.6f}/{entry['MAE']['A1'][1]:.6f}  "
                f"TD-HER {entry['MAE']['tdher'][0]:.6f}/{entry['MAE']['tdher'][1]:.6f}")

        results[scen] = {
            "n_cal": int(len(y_cal)), "n_test": int(len(y_test)),
            "admitted_sets": admitted, "admission": admission, "union_pool": union_models,
            "admitted_sets_coincide_with_union": bool(sets_coincide),
            "A1": a1_info, "A1s": a1s_info, "A2": a2_info, "A3": a3_info,
            "MAE": {k: {"y1": v[0], "y2": v[1]} for k, v in mae.items()},
            "contrasts": contrasts, "by_class": by_class,
        }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "input_sha256_manifest.json").write_text(json.dumps(manifest, indent=2))
    with open(out_dir / "by_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scenario", "class", "N", "row", "y1_MAE", "y2_MAE"])
        w.writeheader(); w.writerows(per_class_rows)
    import scipy, sklearn, lightgbm  # noqa: E401
    (out_dir / "protocol.json").write_text(json.dumps({
        "protocol": "A v2 (claude_protocols_stage2_v2_20261005.md)",
        "script_sha256": sha(Path(__file__)),
        "exp_tdh_router_sha256": router_sha,
        "tau": TAU,
        "bootstrap": {"function": "exp_tdh_router.paired_bootstrap_delta", "rounds": R.BOOTSTRAP_ROUNDS, "seed": 42},
        "rows": {
            "A1": "vertex of the simplex with the lowest raw calibration MAE in M_j; restricted affine; y2 projection",
            "A1s": "as A1, vertex chosen by calibration MAE after the affine fit",
            "A2": "one expert for both targets from the union of admitted sets, balanced objective of the shared route; per-target affine; projection",
            "A3": "uniform mean over M_j; affine; projection",
        },
        "loaded_not_refitted": ["base_best_expert", "base_affine_convex_blend_nonnegative_y2",
                                "admission_shared_affine_convex_blend_nonnegative_y2",
                                "admission_affine_convex_blend_nonnegative_y2"],
        "checks_passed": ["router hash", "input hashes vs equal-bank manifest (%d files)" % n_manifest_checked,
                          "loaded rows reproduce equal_bank_results.json to 1e-12",
                          "adapter-only matched within 1e-9 relative", "A1 = adapter-only at L3",
                          "A1 experts = production best single at L1/L2",
                          "bootstrap interval vs best single reproduced", "class index aligned with test labels"],
        "class_source": {"event_index": str(EVENT_INDEX), "sha256": sha(EVENT_INDEX), "truth": str(TRUTH)},
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                        "platform": platform.platform(), "cpu": platform.processor()},
        "wall_seconds": time.time() - t0,
    }, indent=2))
    log(f"\nall checks passed; {n_manifest_checked} input hashes matched the equal-bank manifest")
    log("written to", out_dir)


if __name__ == "__main__":
    main()
