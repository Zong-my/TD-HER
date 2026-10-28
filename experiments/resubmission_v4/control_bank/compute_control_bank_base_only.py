#!/usr/bin/env python3
"""Protocol B, post hoc supplement (2026-10-05): the same banks WITHOUT the adapter.

Added after step 2 showed that the admission rule admits the LightGBM adapter into the
control-family bank for the L1 arrival time (admission R/A), so that the bank's arrival-time
error there is the adapter's. To attribute correctly, this supplement reports, for every bank
and scenario, the base-only route: simplex weights over the bank's base members, restricted
affine, y2 projection (the path of the production row base_affine_convex_blend_nonnegative_y2).
Descriptive; no pre-declared reading is changed. New files only, in the directory of step 2.
"""
from __future__ import annotations

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT)); WORK_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_ROOT)
import experiments.exp_tdh_router as R  # noqa: E402

TARGETS = ("y1", "y2"); SEEDS = (42, 52, 62)
C15 = [f"{s}_s{k}" for s in ("vanilla", "uncertainty", "gradnorm", "pcgrad", "mmoe") for k in SEEDS]
C18 = C15 + [f"wide_s{k}" for k in SEEDS]
SCEN_SHORT = {"L1_same_dist": "L1", "L2_cross_cond_fewshot": "L2", "L3_cross_topo_fewshot": "L3"}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mae2(y, p): return [float(v) for v in np.mean(np.abs(p - y), axis=0)]


def boot(y, cand, ref):
    b = R.paired_bootstrap_delta(y, cand, ref); out = {}
    for t in TARGETS:
        lo, hi = b[t]["ci95"]
        out[t] = {"delta": b[t]["delta_MAE_candidate_minus_reference"], "candidate_MAE": b[t]["candidate_MAE"],
                  "reference_MAE": b[t]["reference_MAE"],
                  "relative_change_pct": 100.0 * b[t]["delta_MAE_candidate_minus_reference"] / b[t]["reference_MAE"],
                  "ci95": [lo, hi], "resolved": bool(hi < 0.0 or lo > 0.0)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bank-dir", required=True); a = ap.parse_args()
    bank_dir = PROJECT_ROOT / a.bank_dir
    out_json = bank_dir / "supplement_base_only.json"
    if out_json.exists():
        raise SystemExit("supplement already exists")
    logf = open(bank_dir / "supplement_base_only.log", "w")
    def log(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    if not sha(PROJECT_ROOT / "experiments/exp_tdh_router.py").startswith("800d957e"):
        raise SystemExit("router changed")
    audit = json.loads((R.OUT_DIR / "audit_report.json").read_text())
    rep = Path(audit["rep_dir"]) / "repA" / "ms10"
    res = {}
    for scen, (cal, test) in R.SCENARIOS.items():
        y_cal = np.load(rep / f"y_{cal}.npy")[:, :2].astype(float)
        y_test = np.load(rep / f"y_{test}.npy")[:, :2].astype(float)
        ex_cal = {m: np.load(R.EXPERT_DIR / m / f"{cal}_preds.npy").astype(float) for m in R.ROUTER_MODELS}
        ex_test = {m: np.load(R.EXPERT_DIR / m / f"{test}_preds.npy").astype(float) for m in R.ROUTER_MODELS}
        c_cal = {n: np.load(bank_dir / "preds" / n / f"{cal}_preds.npy").astype(float) for n in C18}
        c_test = {n: np.load(bank_dir / "preds" / n / f"{test}_preds.npy").astype(float) for n in C18}
        sc = R.OUT_DIR / "scenarios" / scen
        tdher = np.load(sc / "admission_affine_convex_blend_nonnegative_y2_preds.npy").astype(float)
        base_saved = np.load(sc / "base_affine_convex_blend_nonnegative_y2_preds.npy").astype(float)
        banks = {"H7": (ex_cal, ex_test),
                 "B15": ({n: c_cal[n] for n in C15}, {n: c_test[n] for n in C15}),
                 "B18": (c_cal, c_test),
                 "Bplus": ({**ex_cal, **{n: c_cal[n] for n in C15}}, {**ex_test, **{n: c_test[n] for n in C15}})}
        res[scen] = {"TDHER_saved_MAE": dict(zip(TARGETS, mae2(y_test, tdher))),
                     "TDHER_base_route_saved_MAE": dict(zip(TARGETS, mae2(y_test, base_saved))), "banks": {}}
        log(f"=== {scen} ===  TD-HER {res[scen]['TDHER_saved_MAE']['y1']:.6f}/{res[scen]['TDHER_saved_MAE']['y2']:.6f}; "
            f"TD-HER base route {res[scen]['TDHER_base_route_saved_MAE']['y1']:.6f}/{res[scen]['TDHER_base_route_saved_MAE']['y2']:.6f}")
        for b, (bc, bt) in banks.items():
            routed, det = R.affine_convex_blend_router(y_cal, bc, bt)
            pred = R.enforce_y2_nonnegative(routed)
            e = {"MAE": dict(zip(TARGETS, mae2(y_test, pred))),
                 "weights": {tn: {m: w for m, w in det[tn]["weights"].items() if w > 1e-4} for tn in TARGETS},
                 "tdher_base_route_minus_bank_base_route": boot(y_test, base_saved, pred),
                 "tdher_saved_minus_bank_base_route": boot(y_test, tdher, pred)}
            if b == "H7":
                e["refit_vs_saved_base_route_max_abs_diff"] = float(np.max(np.abs(pred - base_saved)))
            if b == "Bplus":
                e["weight_by_family"] = {tn: {"experts": float(sum(v for m, v in det[tn]["weights"].items() if m in R.ROUTER_MODELS)),
                                              "control_models": float(sum(v for m, v in det[tn]["weights"].items() if m in C15))} for tn in TARGETS}
            np.save(bank_dir / "rows" / f"{b}_baseonly_slsqp_{SCEN_SHORT[scen]}.npy", pred)
            res[scen]["banks"][b] = e
            c = e["tdher_base_route_minus_bank_base_route"]
            log(f"  {b:6s} base-only route {e['MAE']['y1']:.6f}/{e['MAE']['y2']:.6f}   TD-HER base route minus this: " + "; ".join(
                f"{tn}: {c[tn]['delta']:+.6f} ({c[tn]['relative_change_pct']:+.1f}%) CI [{c[tn]['ci95'][0]:+.6f}, {c[tn]['ci95'][1]:+.6f}]{'' if c[tn]['resolved'] else ' unresolved'}" for tn in TARGETS))
    out_json.write_text(json.dumps({"note": "post hoc supplement, see module docstring", "script_sha256": sha(Path(__file__)), "results": res}, indent=2))
    log("written", out_json)


if __name__ == "__main__":
    main()
