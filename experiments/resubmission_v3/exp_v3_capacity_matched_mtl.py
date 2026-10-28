#!/usr/bin/env python3
"""Capacity-matched shared-backbone MTL control for the resubmission.

Re-uses ``experiments/exp_r2_mtl_baselines.py`` unchanged (same data, splits,
target scaling, optimiser, scheduler, early stopping, L2/L3 fine-tuning
protocol and no test-set selection). The only change is the PatchTST width so
that the single shared-backbone model has approximately the same number of
trainable parameters as the six neural experts of the TD-HER bank
(5,614,923 trainable parameters over the seven saved neural instances, incl. both KAN targets, buffers excluded; LightGBM is tree-based and reported separately).

Usage:
  python exp_v3_capacity_matched_mtl.py --size-only            # print candidate sizes
  python exp_v3_capacity_matched_mtl.py --d-model 224 --d-ff 448   # run
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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import experiments.exp_r2_mtl_baselines as M  # noqa: E402

BANK_NEURAL_PARAMS = 5_614_923  # seven saved neural expert instances (incl. both KAN targets), buffers excluded
OUT_DIR = work_path('results/ieee39/resubmission_v3_mtl_capacity_matched')

_orig_load_fixed_config = M.load_fixed_config
_OVERRIDE: dict = {}


def load_wide_config() -> dict:
    cfg = _orig_load_fixed_config()
    cfg["model"].update(_OVERRIDE)
    cfg["source"] = (
        "results/ieee39/exp1/PatchTST/best_params.json with width override "
        + str(_OVERRIDE)
    )
    cfg["capacity_match"] = (
        "shared-backbone PatchTST widened to approximately the six-neural-expert "
        f"bank total of {BANK_NEURAL_PARAMS:,} parameters; optimiser, scheduler, "
        "loss, splits, target scaling, early stopping and fine-tuning protocol "
        "are identical to revision_r2_mtl_exact"
    )
    return cfg


def count(cfg: dict, data: dict) -> int:
    model = M.make_model(data, cfg)
    n = int(sum(p.numel() for p in model.parameters()))
    del model
    return n


def size_table(data: dict) -> None:
    base = _orig_load_fixed_config()
    print(f"baseline PatchTST control: {count(base, data):,}")
    print(f"target (neural bank total): {BANK_NEURAL_PARAMS:,}\n")
    rows = []
    for d_model in (128, 160, 192, 208, 224, 240, 256):
        for ff_mult in (2, 4):
            for n_layers in (4, 6):
                for tab_hidden in (256, 512):
                    ov = {"d_model": d_model, "d_ff": ff_mult * d_model,
                          "n_layers": n_layers, "tab_hidden": tab_hidden,
                          "n_heads": 4 if d_model % 4 == 0 else 1}
                    cfg = _orig_load_fixed_config()
                    cfg["model"].update(ov)
                    n = count(cfg, data)
                    rows.append((abs(n - BANK_NEURAL_PARAMS), n, ov))
    rows.sort(key=lambda r: r[0])
    for diff, n, ov in rows[:12]:
        print(f"{n:>10,}  diff={n - BANK_NEURAL_PARAMS:+,}  {ov}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-only", action="store_true")
    ap.add_argument("--d-model", type=int)
    ap.add_argument("--d-ff", type=int)
    ap.add_argument("--n-layers", type=int)
    ap.add_argument("--tab-hidden", type=int)
    ap.add_argument("--n-heads", type=int)
    ap.add_argument("--strategies", nargs="+", default=["vanilla"])
    ap.add_argument("--seeds", nargs="+", default=["42", "52", "62"])
    ap.add_argument("--max-epochs", default="200")
    ap.add_argument("--patience", default="20")
    ap.add_argument("--fine-tune-epochs", default="50")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    data = M.load_fixed_data()
    if args.size_only:
        size_table(data)
        return
    for key, val in (("d_model", args.d_model), ("d_ff", args.d_ff),
                     ("n_layers", args.n_layers), ("tab_hidden", args.tab_hidden),
                     ("n_heads", args.n_heads)):
        if val is not None:
            _OVERRIDE[key] = val
    M.load_fixed_config = load_wide_config
    n = count(load_wide_config(), data)
    print(f"widened PatchTST parameters: {n:,} (target {BANK_NEURAL_PARAMS:,})")
    del data
    sys.argv = [sys.argv[0], "--strategies", *args.strategies, "--seeds", *args.seeds,
                "--max-epochs", args.max_epochs, "--patience", args.patience,
                "--fine-tune-epochs", args.fine_tune_epochs,
                "--out-dir", str(OUT_DIR)] + (["--smoke"] if args.smoke else [])
    M.main()


if __name__ == "__main__":
    main()
