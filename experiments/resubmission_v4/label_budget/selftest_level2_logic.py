#!/usr/bin/env python3
"""No-training test of the level-2 acceptance logic of harness v3a (amendments 3 and 3a).

Part 1 is purely synthetic and needs no project data: it includes Codex's sign-reflection
counterexample (LB3-M1), which the withdrawn MAE-difference criterion accepted.
Part 2 feeds stage_level2 with copies of adapter arrays that already exist: the two full-budget
configurations saved by the stopped v2 run, unmodified and with synthetic perturbations.
Nothing is fitted. Everything is written into fresh temporary directories; the v2 directory and
the production arrays are only read. A pass here is a test of the logic, NOT the level-2 result
of the study, which can only come from the rerun in the v3 results directory.
Run with --synthetic-only to skip part 2 (no prediction array is opened).
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_label_budget as RB  # noqa: E402

checks = []


def check(name, cond):
    checks.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name)


def old_mae_difference_expression(new, ref, truth):
    """The criterion of amendment 3 as first written (withdrawn): row share and difference of two MAEs."""
    d = np.abs(new - ref)
    mae_new, mae_ref = np.mean(np.abs(new - truth)), np.mean(np.abs(ref - truth))
    return bool(np.mean(d > RB.LEVEL2_TOL) <= 0.01 and abs(mae_new - mae_ref) / mae_ref <= 1e-3)


# ---------------------------------------------------------------- part 1: synthetic
print("--- part 1: synthetic cases")
truth = np.full(1000, 10.0)
ref = np.full(1000, 9.0)
new = ref.copy()
new[:10] = 11.0                                   # 1 % of rows reflected to the other side of the target
c = RB.adapter_array_condition(new, ref, truth)
check("sign reflection (Codex LB3-M1): the withdrawn MAE-difference criterion accepts it",
      old_mae_difference_expression(new, ref, truth))
check("sign reflection: the direct prediction-distance bound rejects it (shift = 2 % of the MAE)",
      c["condition"] == "FAILED" and abs(c["mean_abs_diff_over_frozen_adapter_MAE"] - 0.02) < 1e-12
      and c["reported_only_relative_MAE_difference"] == 0.0)
new = ref.copy()
new[:5] += 0.19                                   # 0.5 % of rows, shift 0.095 % of the MAE
check("small shift in few rows -> bounded discrepancy",
      RB.adapter_array_condition(new, ref, truth)["condition"] == "bounded_discrepancy")
new[:5] += 0.02                                   # shift 0.105 %
check("just above 0.1 % of the MAE -> FAILED", RB.adapter_array_condition(new, ref, truth)["condition"] == "FAILED")
check("identical arrays -> reproduced", RB.adapter_array_condition(ref.copy(), ref, truth)["condition"] == "reproduced")
perfect = truth.copy()
moved = perfect.copy()
moved[0] += 1e-3
check("zero frozen MAE, identical arrays -> reproduced",
      RB.adapter_array_condition(perfect.copy(), perfect, truth)["condition"] == "reproduced")
check("zero frozen MAE, one row moved -> fails closed",
      RB.adapter_array_condition(moved, perfect, truth)["condition"] == "FAILED")

if "--synthetic-only" in sys.argv:
    print(f"\n{sum(checks)}/{len(checks)} checks passed (synthetic part only)")
    sys.exit(0 if all(checks) else 1)

# ---------------------------------------------------------------- part 2: saved arrays of the stopped v2 run
from experiments.exp4_generalization import MS, REP_DIR  # noqa: E402

V2 = RB.PROJECT_ROOT / "results/ieee39/resubmission_v4_label_budget_v2/configs"
FULL = [RB.config_name(s, RB.SCENARIOS[s]["budgets"][-1], RB.FULL_K) for s in RB.SCENARIOS]
L2, L3 = FULL


def case(label, mutate=None):
    d = Path(tempfile.mkdtemp(prefix="lb_level2_logic_"))
    (d / "configs").mkdir()
    for name in FULL:
        shutil.copytree(V2 / name, d / "configs" / name)
    if mutate:
        mutate(d / "configs")
    print(f"--- {label}")
    ok = RB.stage_level2(None, d, REP_DIR, MS)
    return ok, RB.load_json(d / "level2_check.json")


def perturb(name, array, target, share, size, seed=0):
    def f(cfgs):
        p = cfgs / name / array
        a = np.load(p)
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(a), size=max(1, int(round(share * len(a)))), replace=False)
        a[rows, target] += size
        np.save(p, a)
    return f


def reflect_worst_rows(cfgs):
    """Reflect the five worst L2 test y1 predictions (0.98 % of rows) about their targets: every absolute
    error, hence the MAE, is unchanged, while the predictions move by twice their error."""
    p = cfgs / L2 / "adapter_test.npy"
    a = np.load(p)
    y = np.load(f"{REP_DIR}/repA/ms{MS}/y_{RB.SCENARIOS['L2']['test']}.npy")
    rows = np.argsort(-np.abs(a[:, 0] - y[:, 0]))[:5]
    a[rows, 0] = 2 * y[rows, 0] - a[rows, 0]
    np.save(p, a)


ok, rep = case("v2 arrays as saved")
obs = rep["L3"]["adapter_arrays"]["cal_oof_y2"]
check("saved v2 arrays: L2 all reproduced",
      all(v["condition"] == "reproduced" for v in rep["L2"]["adapter_arrays"].values()))
check("saved v2 arrays: L3 out-of-fold y2 is the only bounded discrepancy (3 rows beyond 1e-8, shift about 0.0074 %)",
      obs["condition"] == "bounded_discrepancy" and obs["rows_beyond_1e-8"] == 3
      and 7.0e-5 < obs["mean_abs_diff_over_frozen_adapter_MAE"] < 7.7e-5
      and rep["arrays_using_bounded_discrepancy"] == 1)
check("saved v2 arrays: logic returns pass", ok and rep["pass"])

ok, rep = case("five worst L2 test y1 predictions reflected about their targets (MAE unchanged)", reflect_worst_rows)
r = rep["L2"]["adapter_arrays"]["test_y1"]
check("reflection on real arrays: MAE difference ~0 yet FAILED on prediction distance, stage fails",
      (not ok) and r["condition"] == "FAILED" and r["reported_only_relative_MAE_difference"] < 1e-9
      and r["share_of_rows_beyond_1e-8"] <= 0.01)

ok, rep = case("every row of L2 test y1 moved by 1e-6 (what a protocol error looks like)",
               perturb(L2, "adapter_test.npy", 0, 1.0, 1e-6))
check("all rows moved slightly -> FAILED on row share, stage fails",
      (not ok) and rep["L2"]["adapter_arrays"]["test_y1"]["condition"] == "FAILED")

ok, rep = case("5 % of L3 test y2 rows moved by 1e-3", perturb(L3, "adapter_test.npy", 1, 0.05, 1e-3))
check("5 % of rows -> FAILED on row share", (not ok) and rep["L3"]["adapter_arrays"]["test_y2"]["condition"] == "FAILED")

ok, rep = case("0.5 % of L2 out-of-fold y2 rows moved by 1.0 s", perturb(L2, "adapter_cal_oof.npy", 1, 0.005, 1.0))
check("few rows, large shift -> FAILED on prediction distance",
      (not ok) and rep["L2"]["adapter_arrays"]["cal_oof_y2"]["condition"] == "FAILED")


def three(cfgs):
    perturb(L2, "adapter_test.npy", 0, 0.002, 1e-5, 1)(cfgs)
    perturb(L2, "adapter_test.npy", 1, 0.002, 1e-5, 2)(cfgs)


ok, rep = case("two further arrays with tiny bounded discrepancies (three in all)", three)
check("three arrays under bounded discrepancy -> stage fails on the cap of two",
      (not ok) and rep["arrays_using_bounded_discrepancy"] == 3
      and all(v["condition"] != "FAILED" for s in ("L2", "L3") for v in rep[s]["adapter_arrays"].values()))

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
