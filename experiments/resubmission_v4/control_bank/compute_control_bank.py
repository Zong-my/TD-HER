#!/usr/bin/env python3
"""Protocol B (v2), step 2: banks drawn from the control family, through the TD-HER calibration stage.

No training. Inputs: the frozen predictions of the 18 control checkpoints written by step 1
(after its replay gate), the frozen predictions of the seven experts and the cached LightGBM
adapter arrays of the main run. Every bank goes through the production functions, imported
unchanged: admission (tau = 0.20, recomputed per bank because it compares the adapter with
that bank's best base member), simplex weights per target, restricted affine, y2 projection.

Banks
  B15    the 15 control models (four training methods on PatchTST, and MMoE-head; three seeds)
  B18    B15 plus the three wide fixed-loss models
  Bplus  the seven experts plus the 15 control models (add-in)
  H7     the seven experts (reference; TD-HER's own bank, refitted here only to report the
         solver sensitivity -- the published TD-HER values stay those of the saved arrays)

Solver: SLSQP from uniform weights (production) is the primary, same-workflow result for
every bank. The exact LP minimum of the calibration MAE on the simplex is reported as a
sensitivity result for every bank, whatever it shows.
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
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_ROOT)

import experiments.exp_tdh_router as R  # noqa: E402

ROUTER_SHA256_PREFIX = "800d957e"
TAU = 0.20
TARGETS = ("y1", "y2")
SEEDS = (42, 52, 62)
C15 = [f"{s}_s{k}" for s in ("vanilla", "uncertainty", "gradnorm", "pcgrad", "mmoe") for k in SEEDS]
C18 = C15 + [f"wide_s{k}" for k in SEEDS]
ADAPTER = "LightGBM-Adapter"
SCEN_SHORT = {"L1_same_dist": "L1", "L2_cross_cond_fewshot": "L2", "L3_cross_topo_fewshot": "L3"}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()


def mae2(y, p):
    return [float(v) for v in np.mean(np.abs(p - y), axis=0)]


def lad_simplex_lp(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact minimum of mean |y - P w| over the simplex (formulation of compute_constraint_ablation.py, sparse)."""
    n, k = P.shape
    c = np.concatenate([np.zeros(k), np.ones(n) / n])
    eye = sparse.identity(n, format="csr")
    Ps = sparse.csr_matrix(P)
    A_ub = sparse.vstack([sparse.hstack([Ps, -eye]), sparse.hstack([-Ps, -eye])], format="csr")
    b_ub = np.concatenate([y, -y])
    A_eq = sparse.csr_matrix(np.concatenate([np.ones(k), np.zeros(n)])[None, :])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.array([1.0]),
                  bounds=[(0, None)] * (k + n), method="highs", options={"presolve": True})
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    w = np.clip(res.x[:k], 0.0, None)
    w = w / w.sum()
    return w, float(np.mean(np.abs(y - P @ w)))


def boot(y, cand, ref) -> dict:
    b = R.paired_bootstrap_delta(y, cand, ref)
    out = {}
    for t in TARGETS:
        lo, hi = b[t]["ci95"]
        out[t] = {"delta": b[t]["delta_MAE_candidate_minus_reference"], "candidate_MAE": b[t]["candidate_MAE"],
                  "reference_MAE": b[t]["reference_MAE"],
                  "relative_change_pct": 100.0 * b[t]["delta_MAE_candidate_minus_reference"] / b[t]["reference_MAE"],
                  "ci95": [lo, hi], "resolved": bool(hi < 0.0 or lo > 0.0)}
    return out


def run_bank(y_cal, cal_all, test_all, y_test) -> dict:
    """Production pipeline (SLSQP) and LP sensitivity for one bank; returns predictions and details."""
    routed, details = R.admission_affine_convex_blend_router(y_cal, cal_all, test_all, TAU)
    pred_slsqp = R.enforce_y2_nonnegative(routed)
    pred_lp = np.zeros_like(y_test)
    lp_details = {}
    for t, tn in enumerate(TARGETS):
        models, admission = R.admitted_models_for_target(y_cal, cal_all, t, TAU)
        if models != list(details[tn]["weights"].keys()):
            raise SystemExit("admitted set differs between the two code paths")
        P_cal = np.stack([cal_all[m][:, t] for m in models], axis=1)
        P_test = np.stack([test_all[m][:, t] for m in models], axis=1)
        w, obj = lad_simplex_lp(P_cal, y_cal[:, t])
        a, b, info = R.fit_restricted_affine(y_cal[:, t], P_cal @ w)
        pred_lp[:, t] = R.apply_affine(P_test @ w, a, b)
        slsqp_obj = details[tn]["blend_cal_MAE"]
        lp_details[tn] = {"weights": {m: float(x) for m, x in zip(models, w)}, "cal_MAE_route": obj,
                          "slsqp_cal_MAE_route": slsqp_obj,
                          "slsqp_excess_over_lp_pct": 100.0 * (slsqp_obj - obj) / obj,
                          "slope": a, "bias": b, "affine_cal_MAE": info["cal_MAE"]}
    pred_lp = R.enforce_y2_nonnegative(pred_lp)
    return {"pred_slsqp": pred_slsqp, "pred_lp": pred_lp, "details_slsqp": details, "details_lp": lp_details}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True, help="output directory of step 1 (replay passed)")
    args = ap.parse_args()
    bank_dir = (PROJECT_ROOT / args.bank_dir).resolve() if not os.path.isabs(args.bank_dir) else Path(args.bank_dir)
    log = Tee(bank_dir / "control_bank.log")
    t0 = time.time()
    log("Protocol B v2, step 2: control-family banks")
    replay = json.loads((bank_dir / "replay_report.json").read_text())
    if replay.get("smoke") or not replay.get("all_passed") or len(replay["checkpoints"]) != 18:
        raise SystemExit("step 1 is not a complete, passed replay of 18 checkpoints")
    if (bank_dir / "results.json").exists():
        raise SystemExit("results.json already exists in this directory; rerun step 1 into a new directory")
    router_sha = sha(PROJECT_ROOT / "experiments/exp_tdh_router.py")
    if not router_sha.startswith(ROUTER_SHA256_PREFIX):
        raise SystemExit("exp_tdh_router.py differs from the reviewed version")
    params = {n: replay["checkpoints"][n]["parameters"] for n in C18}
    bank_params = {"B15": sum(params[n] for n in C15), "B18": sum(params[n] for n in C18)}
    log("parameters: B15", f"{bank_params['B15']:,}", "B18", f"{bank_params['B18']:,}")

    audit = json.loads((R.OUT_DIR / "audit_report.json").read_text())
    rep = Path(audit["rep_dir"]) / "repA" / "ms10"
    manifest, results = {}, {}
    rows_dir = bank_dir / "rows"
    rows_dir.mkdir()

    for scen, (cal, test) in R.SCENARIOS.items():
        s = SCEN_SHORT[scen]
        log(f"\n=== {scen} ===")
        y_cal = np.load(rep / f"y_{cal}.npy")[:, :2].astype(float)
        y_test = np.load(rep / f"y_{test}.npy")[:, :2].astype(float)
        manifest[str(rep / f"y_{cal}.npy")] = sha(rep / f"y_{cal}.npy")
        manifest[str(rep / f"y_{test}.npy")] = sha(rep / f"y_{test}.npy")
        ex_cal, ex_test = {}, {}
        for m in R.ROUTER_MODELS:
            cp, tp = R.EXPERT_DIR / m / f"{cal}_preds.npy", R.EXPERT_DIR / m / f"{test}_preds.npy"
            ex_cal[m], ex_test[m] = np.load(cp).astype(float), np.load(tp).astype(float)
            manifest[str(cp)], manifest[str(tp)] = sha(cp), sha(tp)
        ad_dir = R.ADAPTER_DIR / scen / ADAPTER
        ad_cal = np.load(ad_dir / "cal_oof_preds.npy").astype(float)
        ad_test = np.load(ad_dir / "test_preds.npy").astype(float)
        manifest[str(ad_dir / "cal_oof_preds.npy")] = sha(ad_dir / "cal_oof_preds.npy")
        manifest[str(ad_dir / "test_preds.npy")] = sha(ad_dir / "test_preds.npy")
        c_cal, c_test = {}, {}
        for n in C18:
            cp, tp = bank_dir / "preds" / n / f"{cal}_preds.npy", bank_dir / "preds" / n / f"{test}_preds.npy"
            c_cal[n], c_test[n] = np.load(cp).astype(float), np.load(tp).astype(float)
            manifest[str(cp.relative_to(PROJECT_ROOT))], manifest[str(tp.relative_to(PROJECT_ROOT))] = sha(cp), sha(tp)
            if c_cal[n].shape != y_cal.shape or c_test[n].shape != y_test.shape:
                raise SystemExit(f"{n}: prediction shape mismatch")
        tdher_p = R.OUT_DIR / "scenarios" / scen / "admission_affine_convex_blend_nonnegative_y2_preds.npy"
        tdher = np.load(tdher_p).astype(float)
        manifest[str(tdher_p)] = sha(tdher_p)

        banks = {
            "H7": (dict(ex_cal), dict(ex_test)),
            "B15": ({n: c_cal[n] for n in C15}, {n: c_test[n] for n in C15}),
            "B18": ({n: c_cal[n] for n in C18}, {n: c_test[n] for n in C18}),
            "Bplus": ({**ex_cal, **{n: c_cal[n] for n in C15}}, {**ex_test, **{n: c_test[n] for n in C15}}),
        }
        scen_res = {"n_cal": int(len(y_cal)), "n_test": int(len(y_test)),
                    "TDHER_saved_MAE": dict(zip(TARGETS, mae2(y_test, tdher))), "banks": {}}
        for bname, (bc, bt) in banks.items():
            cal_all = dict(bc); cal_all[ADAPTER] = ad_cal
            test_all = dict(bt); test_all[ADAPTER] = ad_test
            out = run_bank(y_cal, cal_all, test_all, y_test)
            entry = {"members": list(bc.keys()), "n_base_members": len(bc)}
            entry["admission"] = {tn: {"best_base": out["details_slsqp"][tn]["admission"]["base_best_model"],
                                       "best_base_cal_MAE": out["details_slsqp"][tn]["admission"]["base_best_cal_MAE"],
                                       "adapter_OOF_MAE": out["details_slsqp"][tn]["admission"]["adapter_details"][ADAPTER]["cal_MAE"],
                                       "required_MAE": out["details_slsqp"][tn]["admission"]["adapter_details"][ADAPTER]["required_MAE"],
                                       "admitted": bool(out["details_slsqp"][tn]["admission"]["adapter_details"][ADAPTER]["admitted"])}
                                  for tn in TARGETS}
            # context rows for the bank: its calibration-selected best member (raw) and uniform mean of its base members
            best_members = {}
            best_pred = np.zeros_like(y_test)
            for t, tn in enumerate(TARGETS):
                sc = {m: R._mae(y_cal[:, t], bc[m][:, t]) for m in bc}
                bm = min(sc, key=sc.get)
                best_members[tn] = bm
                best_pred[:, t] = bt[bm][:, t]
            uni = np.mean(np.stack([bt[m] for m in bt]), axis=0)
            entry["best_base_member_raw"] = {"members": best_members, "MAE": dict(zip(TARGETS, mae2(y_test, best_pred)))}
            entry["uniform_raw_base"] = {"MAE": dict(zip(TARGETS, mae2(y_test, uni)))}
            for solver in ("slsqp", "lp"):
                pred = out[f"pred_{solver}"]
                np.save(rows_dir / f"{bname}_{solver}_{s}.npy", pred)
                det = out["details_slsqp"] if solver == "slsqp" else out["details_lp"]
                e = {"MAE": dict(zip(TARGETS, mae2(y_test, pred))),
                     "weights": {tn: {m: w for m, w in det[tn]["weights"].items() if w > 1e-4} for tn in TARGETS},
                     "cal_MAE_route": {tn: (det[tn]["blend_cal_MAE"] if solver == "slsqp" else det[tn]["cal_MAE_route"]) for tn in TARGETS},
                     "affine": {tn: {"slope": det[tn]["slope"], "bias": det[tn]["bias"]} for tn in TARGETS},
                     "tdher_saved_minus_bank": boot(y_test, tdher, pred)}
                if bname == "Bplus":
                    fam = {}
                    for tn in TARGETS:
                        w = det[tn]["weights"]
                        fam[tn] = {"experts": float(sum(v for m, v in w.items() if m in R.ROUTER_MODELS)),
                                   "control_models": float(sum(v for m, v in w.items() if m in C15)),
                                   "adapter": float(w.get(ADAPTER, 0.0))}
                    e["weight_by_family"] = fam
                    e["bank_minus_tdher_saved"] = boot(y_test, pred, tdher)
                entry[solver] = e
            entry["slsqp_excess_over_lp_pct"] = {tn: out["details_lp"][tn]["slsqp_excess_over_lp_pct"] for tn in TARGETS}
            if bname == "H7":
                entry["slsqp_refit_vs_saved_tdher_max_abs_diff"] = float(np.max(np.abs(out["pred_slsqp"] - tdher)))
            scen_res["banks"][bname] = entry
            adm = "/".join("A" if entry["admission"][tn]["admitted"] else "R" for tn in TARGETS)
            log(f"  {bname:6s} admission {adm}  SLSQP {entry['slsqp']['MAE']['y1']:.6f}/{entry['slsqp']['MAE']['y2']:.6f}  "
                f"LP {entry['lp']['MAE']['y1']:.6f}/{entry['lp']['MAE']['y2']:.6f}  "
                f"SLSQP excess over LP (cal) {entry['slsqp_excess_over_lp_pct']['y1']:+.3f}%/{entry['slsqp_excess_over_lp_pct']['y2']:+.3f}%  "
                f"best member raw {entry['best_base_member_raw']['MAE']['y1']:.6f}/{entry['best_base_member_raw']['MAE']['y2']:.6f}")
            c = entry["slsqp"]["tdher_saved_minus_bank"]
            log("         TD-HER minus bank (SLSQP): " + "; ".join(
                f"{tn}: {c[tn]['delta']:+.6f} ({c[tn]['relative_change_pct']:+.1f}%) CI [{c[tn]['ci95'][0]:+.6f}, {c[tn]['ci95'][1]:+.6f}]"
                f"{'' if c[tn]['resolved'] else ' unresolved'}" for tn in TARGETS))
            if bname == "Bplus":
                log("         weight by family (SLSQP): " + "; ".join(
                    f"{tn}: experts {entry['slsqp']['weight_by_family'][tn]['experts']:.3f}, controls {entry['slsqp']['weight_by_family'][tn]['control_models']:.3f}, "
                    f"adapter {entry['slsqp']['weight_by_family'][tn]['adapter']:.3f}" for tn in TARGETS))
        log(f"  TD-HER saved: {scen_res['TDHER_saved_MAE']['y1']:.6f}/{scen_res['TDHER_saved_MAE']['y2']:.6f}; "
            f"H7 SLSQP refit vs saved max|d| {scen_res['banks']['H7']['slsqp_refit_vs_saved_tdher_max_abs_diff']:.2e}")
        results[scen] = scen_res

    import scipy
    (bank_dir / "results.json").write_text(json.dumps(results, indent=2))
    (bank_dir / "input_sha256_manifest.json").write_text(json.dumps(manifest, indent=2))
    (bank_dir / "protocol.json").write_text(json.dumps({
        "protocol": "B v2 step 2 (claude_protocols_stage2_v2_20261005.md)",
        "script_sha256": sha(Path(__file__)), "exp_tdh_router_sha256": router_sha, "tau": TAU,
        "banks": {"B15": C15, "B18": C18, "Bplus": "seven experts + B15", "H7": "seven experts (reference)"},
        "bank_parameters": {**bank_params, "per_checkpoint": params,
                            "heterogeneous_bank_disclosed": "Obtain neural parameter and tree counts from your own fitted base bank."},
        "solver": {"primary": "SLSQP from uniform weights (exp_tdh_router.optimize_convex_weights)",
                   "sensitivity": "exact LP minimum of the calibration MAE on the simplex (HiGHS), reported for every bank"},
        "contrast": "paired event bootstrap, exp_tdh_router.paired_bootstrap_delta, 2000 rounds, seed 42; TD-HER = saved production array",
        "members_are": "L1 checkpoints, no fine-tuning on shifted data; predictions from step 1 after its replay gate",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform()},
        "wall_seconds": time.time() - t0,
    }, indent=2))
    log("\nwritten to", bank_dir)


if __name__ == "__main__":
    main()
