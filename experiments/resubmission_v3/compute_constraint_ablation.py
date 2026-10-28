#!/usr/bin/env python3
"""Constraint-only ablation of the target-wise routing rule (re-plot, no training).

For each scenario (L1-L3) and target, fit linear combinations of the SAME seven
frozen base-expert predictions on the SAME calibration split with the SAME
mean-absolute-error objective, varying ONLY the weight constraint:

  * unconstrained : w in R^K            (LAD regression, no intercept)
  * nonnegative   : w >= 0              (no sum constraint)
  * simplex       : w >= 0, sum(w) = 1  (the TD-HER routing constraint)

Every variant then receives the same bounded affine correction (fitted on the
calibration split) and the same y2 >= 0 projection used by TD-HER, and is
evaluated once on the held-out test split.  Paired bootstrap intervals compare
each variant with the simplex variant.  All three LAD problems are solved as
linear programs (HiGHS) so that the solver is identical across variants; the
SLSQP simplex solution used by the paper pipeline is reported alongside as a
cross-check.

Inputs are the cached expert predictions in results/ieee39/exp_tdh_router/experts
and the frozen labels of the representation directory recorded in
results/ieee39/exp_tdh_router/audit_report.json.  Nothing is trained.
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
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
import experiments.exp_tdh_router as R  # noqa: E402

OUT_DIR = work_path('results/ieee39/resubmission_v3_constraint_ablation')
EXPERT_DIR = R.OUT_DIR / "experts"
SCENARIOS = {
    "L1_same_dist": ("val", "test"),
    "L2_cross_cond_fewshot": ("cross_cond_finetune", "cross_cond_test"),
    "L3_cross_topo_fewshot": ("cross_cond_topo_finetune", "cross_cond_topo_test"),
}
VARIANTS = ("unconstrained", "nonnegative", "simplex")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lad_fit(P: np.ndarray, y: np.ndarray, constraint: str) -> np.ndarray:
    """Least-absolute-deviation weights via LP: min sum(t) s.t. -t <= y - P w <= t."""
    n, k = P.shape
    # variables: w (k), t (n)
    c = np.concatenate([np.zeros(k), np.ones(n) / n])
    A_ub = np.block([[P, -np.eye(n)], [-P, -np.eye(n)]])
    b_ub = np.concatenate([y, -y])
    if constraint == "unconstrained":
        bounds = [(None, None)] * k + [(0, None)] * n
        A_eq = b_eq = None
    elif constraint == "nonnegative":
        bounds = [(0, None)] * k + [(0, None)] * n
        A_eq = b_eq = None
    elif constraint == "simplex":
        bounds = [(0, None)] * k + [(0, None)] * n
        A_eq = np.concatenate([np.ones(k), np.zeros(n)])[None, :]
        b_eq = np.array([1.0])
    else:
        raise ValueError(constraint)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                  method="highs", options={"presolve": True})
    if not res.success:
        raise RuntimeError(f"LP failed ({constraint}): {res.message}")
    return res.x[:k]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = json.loads((R.OUT_DIR / "audit_report.json").read_text())
    rep = Path(audit["rep_dir"]) / "repA" / "ms10"
    models = list(R.ROUTER_MODELS)
    manifest, rows, detail = [], [], {}

    for scen, (cal_split, test_split) in SCENARIOS.items():
        y_cal = np.load(rep / f"y_{cal_split}.npy")[:, :2].astype(float)
        y_te = np.load(rep / f"y_{test_split}.npy")[:, :2].astype(float)
        manifest += [{"file": str(rep / f"y_{s}.npy"), "sha256": sha(rep / f"y_{s}.npy")} for s in (cal_split, test_split)]
        cal_preds, te_preds = {}, {}
        for m in models:
            pc, pt = EXPERT_DIR / m / f"{cal_split}_preds.npy", EXPERT_DIR / m / f"{test_split}_preds.npy"
            cal_preds[m], te_preds[m] = np.load(pc).astype(float), np.load(pt).astype(float)
            manifest += [{"file": str(pc), "sha256": sha(pc)}, {"file": str(pt), "sha256": sha(pt)}]
            assert cal_preds[m].shape == y_cal.shape and te_preds[m].shape == y_te.shape

        outputs = {}
        detail[scen] = {}
        for variant in VARIANTS:
            pred_te = np.zeros_like(y_te)
            detail[scen][variant] = {}
            for j, tgt in enumerate(("y1", "y2")):
                P_cal = np.stack([cal_preds[m][:, j] for m in models], axis=1)
                P_te = np.stack([te_preds[m][:, j] for m in models], axis=1)
                w = lad_fit(P_cal, y_cal[:, j], variant)
                route_cal, route_te = P_cal @ w, P_te @ w
                a, b, aff = R.fit_restricted_affine(y_cal[:, j], route_cal)
                pred_te[:, j] = R.apply_affine(route_te, a, b)
                detail[scen][variant][tgt] = {
                    "weights": {m: float(x) for m, x in zip(models, w)},
                    "sum_w": float(np.sum(w)), "min_w": float(np.min(w)),
                    "cal_MAE_route": float(np.mean(np.abs(y_cal[:, j] - route_cal))),
                    "affine": {"a": a, "b": b},
                }
            pred_te = R.enforce_y2_nonnegative(pred_te)
            outputs[variant] = pred_te

        # cross-check: the paper pipeline's SLSQP simplex route (base bank, affine, projection)
        ref = np.load(R.OUT_DIR / "scenarios" / scen / "base_affine_convex_blend_nonnegative_y2_preds.npy").astype(float)
        outputs["simplex_slsqp_pipeline"] = ref
        manifest.append({"file": str(R.OUT_DIR / "scenarios" / scen / "base_affine_convex_blend_nonnegative_y2_preds.npy"),
                         "sha256": sha(R.OUT_DIR / "scenarios" / scen / "base_affine_convex_blend_nonnegative_y2_preds.npy")})

        for name, pred in outputs.items():
            mae = np.mean(np.abs(pred - y_te), axis=0)
            row = {"scenario": scen, "variant": name, "y1_MAE": float(mae[0]), "y2_MAE": float(mae[1])}
            if name != "simplex":
                bs = R.paired_bootstrap_delta(y_te, pred, outputs["simplex"])
                for t in ("y1", "y2"):
                    row[f"{t}_delta_vs_simplex"] = bs[t]["delta_MAE_candidate_minus_reference"]
                    row[f"{t}_ci_lo"], row[f"{t}_ci_hi"] = bs[t]["ci95"]
            rows.append(row)
            print(f"{scen:24s} {name:24s} y1={mae[0]:.6f} y2={mae[1]:.6f}"
                  + ("" if name == "simplex" else f"  d_y1=[{row['y1_ci_lo']:+.5f},{row['y1_ci_hi']:+.5f}] d_y2=[{row['y2_ci_lo']:+.5f},{row['y2_ci_hi']:+.5f}]"))

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("scenario", "variant"), k))
    with open(OUT_DIR / "constraint_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    (OUT_DIR / "details.json").write_text(json.dumps(detail, indent=2))
    (OUT_DIR / "input_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT_DIR / "protocol.json").write_text(json.dumps({
        "objective": "calibration MAE (least absolute deviations), no intercept, solved as LP (HiGHS)",
        "inputs": "seven same-target base-expert predictions (adapter excluded)",
        "postprocessing": "bounded affine correction fitted on calibration route + y2>=0 projection (same functions as TD-HER)",
        "variants": list(VARIANTS), "bootstrap_rounds": R.BOOTSTRAP_ROUNDS,
        "selection": "none; every variant evaluated once on the held-out test split",
        "experts": models,
    }, indent=2))
    print("written to", OUT_DIR)


if __name__ == "__main__":
    main()
