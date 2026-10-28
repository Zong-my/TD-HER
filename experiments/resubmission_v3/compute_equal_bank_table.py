#!/usr/bin/env python3
"""Equal-expert-bank comparison for the resubmission (re-plot only, no training).

Every row reuses frozen expert / adapter predictions saved by
``experiments/exp_tdh_router.py``. New rows (uniform average, base-only ridge
stacking, adapter-only with matched affine+projection) are computed from those
arrays with the *same* helper functions as the paper pipeline. Nothing is fit
on test labels. Outputs go to ``results/ieee39/resubmission_v3_equal_bank``.
"""
from __future__ import annotations

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(WORK_ROOT)  # exp_tdh_router uses project-relative paths

import experiments.exp_tdh_router as R  # noqa: E402

OUT_DIR = work_path('results/ieee39/resubmission_v3_equal_bank')
OUT_DIR.mkdir(parents=True, exist_ok=True)
TEX_OUT = work_path("tables")
TEX_OUT.mkdir(parents=True, exist_ok=True)

audit = json.loads((R.OUT_DIR / "audit_report.json").read_text())
REP = Path(audit["rep_dir"]) / "repA" / "ms10"

ROWS = [
    ("best_single", "Best single expert (calibration-selected)"),
    ("uniform_avg", "Uniform average of 7 experts"),
    ("ridge_base", "Ridge stacking, unconstrained (7 experts)"),
    ("ridge_saved", "Ridge stacking, unconstrained (7 experts + adapter)"),
    ("simplex_base", "Simplex route, 7 experts, no correction"),
    ("simplex_base_full", "Simplex route, 7 experts, affine + projection"),
    ("shared_full", "Shared route, full pipeline"),
    ("adapter_raw", "Adapter only, raw"),
    ("adapter_matched", "Adapter only, affine + projection"),
    ("tdher", "Full TD-HER"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mae(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(p - y), axis=0)


def main() -> None:
    results: dict[str, dict] = {}
    manifest: dict[str, str] = {}
    for scen, (cal, test) in R.SCENARIOS.items():
        y_cal_p, y_test_p = REP / f"y_{cal}.npy", REP / f"y_{test}.npy"
        y_cal = np.load(y_cal_p)[:, :2].astype(float)
        y_test = np.load(y_test_p)[:, :2].astype(float)
        manifest[str(y_cal_p)] = sha(y_cal_p)
        manifest[str(y_test_p)] = sha(y_test_p)

        cal_preds, test_preds = {}, {}
        for m in R.ROUTER_MODELS:
            cp, tp = R.EXPERT_DIR / m / f"{cal}_preds.npy", R.EXPERT_DIR / m / f"{test}_preds.npy"
            cal_preds[m] = np.load(cp).astype(float)
            test_preds[m] = np.load(tp).astype(float)
            manifest[str(cp)] = sha(cp)
            manifest[str(tp)] = sha(tp)
        ad_dir = R.ADAPTER_DIR / scen / "LightGBM-Adapter"
        ad_cal = np.load(ad_dir / "cal_oof_preds.npy").astype(float)
        ad_test = np.load(ad_dir / "test_preds.npy").astype(float)
        manifest[str(ad_dir / "cal_oof_preds.npy")] = sha(ad_dir / "cal_oof_preds.npy")
        manifest[str(ad_dir / "test_preds.npy")] = sha(ad_dir / "test_preds.npy")

        sc = R.OUT_DIR / "scenarios" / scen

        def saved(name: str) -> np.ndarray:
            p = sc / f"{name}_preds.npy"
            manifest[str(p)] = sha(p)
            return np.load(p).astype(float)

        preds: dict[str, np.ndarray] = {}
        preds["best_single"] = saved("base_best_expert")
        preds["uniform_avg"] = np.mean(np.stack([test_preds[m] for m in R.ROUTER_MODELS]), axis=0)
        preds["ridge_base"], ridge_details = R.ridge_stack_router(y_cal, cal_preds, test_preds)
        preds["ridge_saved"] = saved("ridge_stack")
        preds["simplex_base"] = saved("base_convex_blend")
        preds["simplex_base_full"] = saved("base_affine_convex_blend_nonnegative_y2")
        preds["shared_full"] = saved("admission_shared_affine_convex_blend_nonnegative_y2")
        preds["adapter_raw"] = ad_test
        adj = np.zeros_like(ad_test)
        affine_info = {}
        for t, tn in enumerate(("y1", "y2")):
            a, b, info = R.fit_restricted_affine(y_cal[:, t], ad_cal[:, t])
            adj[:, t] = R.apply_affine(ad_test[:, t], a, b)
            affine_info[tn] = {"slope": a, "bias": b, "cal_MAE": info["cal_MAE"]}
        preds["adapter_matched"] = R.enforce_y2_nonnegative(adj)
        preds["tdher"] = saved("admission_affine_convex_blend_nonnegative_y2")

        full = preds["tdher"]
        scen_res = {"n_cal": int(len(y_cal)), "n_test": int(len(y_test)),
                    "adapter_affine": affine_info, "rows": {}}
        for key, label in ROWS:
            p = preds[key]
            assert p.shape == y_test.shape and np.isfinite(p).all(), key
            m = mae(y_test, p)
            entry = {"label": label, "y1_MAE": float(m[0]), "y2_MAE": float(m[1])}
            if key != "tdher":
                boot = R.paired_bootstrap_delta(y_test, full, p)
                entry["bootstrap_full_minus_row"] = boot
            scen_res["rows"][key] = entry
        results[scen] = scen_res
        print(f"\n=== {scen}  (n_cal={len(y_cal)}, n_test={len(y_test)}) ===")
        for key, label in ROWS:
            e = scen_res["rows"][key]
            print(f"  {label:52s} y1={e['y1_MAE']:.6f}  y2={e['y2_MAE']:.6f}")

    (OUT_DIR / "equal_bank_results.json").write_text(json.dumps(results, indent=2))
    (OUT_DIR / "input_sha256_manifest.json").write_text(json.dumps(manifest, indent=2))

    # LaTeX table: rows x (L1 y1 y2 | L2 y1 y2 | L3 y1 y2), bold = column minimum.
    scens = list(R.SCENARIOS)
    cols = [(s, t) for s in scens for t in ("y1_MAE", "y2_MAE")]
    mins = {c: min(results[c[0]]["rows"][k][c[1]] for k, _ in ROWS) for c in cols}
    lines = []
    for key, label in ROWS:
        cells = []
        for c in cols:
            v = results[c[0]]["rows"][key][c[1]]
            s = f"{v:.4f}"
            cells.append(rf"\textbf{{{s}}}" if abs(v - mins[c]) < 5e-7 else s)
        row_label = rf"\textbf{{{label}}}" if key == "tdher" else label
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")
    tex = "\n".join(lines) + "\n"
    (TEX_OUT / "equal_bank_rows.tex").write_text(tex)
    print(f"\nWrote {OUT_DIR / 'equal_bank_results.json'} and {TEX_OUT / 'equal_bank_rows.tex'}")


if __name__ == "__main__":
    main()
