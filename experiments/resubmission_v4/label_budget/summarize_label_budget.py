#!/usr/bin/env python3
"""Label-budget study - planning of the executed grid and aggregation (no training, JSON only).

Commands
  plan       after the first stage (two full-budget configurations and the pilot): applies the
             pre-declared 8-hour rule to the measured wall times and writes run_plan.json once.
             The plan fixes the executed grid (five seeds = 32 configurations / 64 cells, or the
             three-seed fallback = 20 / 40) and the slot assignment. Runner, completeness check
             and aggregation all use this one grid. The rule uses timing only, never a result.
  summarize  collects the cells of the executed grid, checks that every configuration was produced
             by one runner version under one input manifest and one effective protocol, counts
             avoided harmful admissions and missed recoveries, and applies the pre-declared
             decision only when the executed grid is complete and consistent.

This file never fits, never reads prediction arrays and never writes inside a configuration
directory. It is not part of the enforced input manifest, so it can be corrected without
invalidating fitted configurations; its own SHA-256 is recorded in everything it writes.
Grid constants are imported from run_label_budget.py, so there is a single definition.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_label_budget as RB  # noqa: E402  (constants and small helpers only; no production import)

SELF_SHA = RB.sha256(Path(__file__).resolve())


def cmd_plan(out):
    l2 = out / "level2_check.json"
    if not l2.exists() or not RB.load_json(l2)["pass"]:
        raise SystemExit("level2 has not passed; no plan is written")
    times = {}
    for c in RB.FIRST_STAGE:
        p = out / "configs" / RB.config_name(*c) / "result.json"
        if not p.exists():
            raise SystemExit(f"first-stage configuration missing: {RB.config_name(*c)}")
        times[RB.config_name(*c)] = float(RB.load_json(p)["wall_seconds"])
    mean_s = sum(times.values()) / len(times)
    left_five = len(RB.all_configs(RB.N_SEEDS)) - len(RB.FIRST_STAGE)
    seq_h = left_five * mean_s / 3600.0
    par_h = seq_h / RB.PARALLEL_SLOTS
    if seq_h <= RB.TIME_LIMIT_H:
        n_seeds, nslots, branch = RB.N_SEEDS, 1, "sequential projection within the limit"
    elif par_h <= RB.TIME_LIMIT_H:
        n_seeds, nslots, branch = RB.N_SEEDS, RB.PARALLEL_SLOTS, "sequential projection exceeds the limit; three slots"
    else:
        n_seeds, nslots, branch = (RB.FALLBACK_SEEDS, RB.PARALLEL_SLOTS,
                                   "three-slot projection exceeds the limit; seeds k=0,1,2 with three slots")
    grid = RB.all_configs(n_seeds)
    left = sorted((c for c in grid if c not in RB.FIRST_STAGE), key=lambda c: (-c[1], c[0], c[2]))
    slots = [left[i::nslots] for i in range(nslots)]
    plan = {
        "rule": RB.RULE_RECORD["time_rule"], "limit_hours": RB.TIME_LIMIT_H, "branch": branch,
        "first_stage_wall_seconds": times, "mean_wall_seconds": mean_s,
        "projected_hours_five_seeds_sequential": seq_h, "projected_hours_five_seeds_three_slots": par_h,
        "n_seeds": n_seeds, "nslots": nslots,
        "configs": [list(c) for c in grid], "n_configs": len(grid), "cells_expected": 2 * len(grid),
        "slots": [[list(c) for c in s] for s in slots], "n_left_after_first_stage": len(left),
        "projected_hours_of_this_plan": len(left) * mean_s / 3600.0 / nslots,
        "note": "Fixed before the full run and never changed afterwards. Starting the full run needs the "
                "author's go-ahead. Projections are idealised (no contention allowance); a longer actual run "
                "does not change the grid.",
        "summarizer_sha256": SELF_SHA,
    }
    path = out / "run_plan.json"
    if path.exists():
        print("[plan] run_plan.json already exists; not overwritten")
        plan = RB.load_json(path)
    else:
        RB.dump_json(plan, path)
    print(f"[plan] mean {plan['mean_wall_seconds']:.1f} s per configuration; five seeds: "
          f"{plan['projected_hours_five_seeds_sequential']:.2f} h sequential, "
          f"{plan['projected_hours_five_seeds_three_slots']:.2f} h with three slots -> {plan['branch']}; "
          f"{plan['n_configs']} configurations / {plan['cells_expected']} cells, {plan['nslots']} slot(s), "
          f"{plan['n_left_after_first_stage']} left, about {plan['projected_hours_of_this_plan']:.2f} h")


def cmd_summarize(out):
    plan = RB.load_json(out / "run_plan.json") if (out / "run_plan.json").exists() else None
    grid = [tuple(c) for c in plan["configs"]] if plan else RB.all_configs(RB.N_SEEDS)
    rows, avoided, missed, identity_failures = [], 0, 0, []
    seen = {"script_sha256": set(), "manifest_sha256": set(), "effective_protocol_sha256": set()}
    for scen, n, k in grid:
        p = out / "configs" / RB.config_name(scen, n, k) / "result.json"
        if not p.exists():
            continue
        res = RB.load_json(p)
        for key in seen:
            seen[key].add(res.get(key))
        if not res.get("identity_pass", False):
            identity_failures.append(RB.config_name(scen, n, k))
        for target, v in res["targets"].items():
            lo, hi = v["C_minus_D"]["ci95"]
            sign = "above_zero" if lo > 0 else ("below_zero" if hi < 0 else "contains_zero")
            disagree = bool(v["A_admits"] and not v["B_admits"])
            avoided += int(disagree and sign == "above_zero")
            missed += int(disagree and sign == "below_zero")
            rows.append({"scenario": scen, "n": n, "k": k, "target": target, "A_admits": v["A_admits"],
                         "B_admits": v["B_admits"], "C_minus_D": v["C_minus_D"]["delta"], "ci_lo": lo, "ci_hi": hi,
                         "interval": sign, "A_admits_B_rejects": disagree,
                         **{f"MAE_{r}": v["test_MAE"][r] for r in RB.RULES},
                         "oof_adapter_MAE": v["oof_adapter_MAE"], "required_MAE": v["required_MAE"],
                         "wall_seconds": res["wall_seconds"]})
    directory_ids = {"manifest_sha256": RB.sha256(out / "input_sha256_manifest.json"),
                     "effective_protocol_sha256": RB.sha256(out / "effective_protocol.json")}
    problems = [f"{key}: {sorted(map(str, vals))}" for key, vals in seen.items() if len(vals) > 1]
    problems += [f"{key} of a configuration differs from the directory's file"
                 for key, val in directory_ids.items() if seen[key] - {val}]
    problems += [f"identity check failed in {name}" for name in identity_failures]
    expected = 2 * len(grid) if plan else None
    complete = plan is not None and len(rows) == expected
    final = complete and not problems
    if final:
        tag = "final"
    elif plan is None:
        tag = f"preplan_{len(rows)}_cells"
    else:
        tag = f"{'inconsistent' if complete else 'partial'}_{len(rows)}_of_{expected}"
    if rows:
        path = out / f"cells_{tag}.csv"
        if path.exists():
            print(f"[summarize] {path.name} exists; not overwritten")
        else:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    decision = {"cells_present": len(rows), "cells_expected": expected, "executed_grid_fixed": plan is not None,
                "n_seeds": plan["n_seeds"] if plan else None, "consistency_problems": problems,
                "avoided_harmful_admissions": avoided, "missed_recoveries": missed,
                "second_criterion_retained": bool(avoided >= 1 and missed <= avoided) if final else None,
                "rule": RB.RULE_RECORD["decision"], "runner_sha256_of_configurations": sorted(map(str, seen["script_sha256"])),
                "summarizer_sha256": SELF_SHA}
    dpath = out / f"decision_{tag}.json"
    if dpath.exists():
        print(f"[summarize] {dpath.name} exists; not overwritten")
    else:
        RB.dump_json(decision, dpath)
    print(f"[summarize] {tag}: {len(rows)} cells of {expected}; avoided={avoided} missed={missed} "
          f"problems={len(problems)} -> second criterion retained: {decision['second_criterion_retained']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plan", "summarize"])
    ap.add_argument("--out", default=str(RB.OUT_DEFAULT))
    a = ap.parse_args()
    (cmd_plan if a.command == "plan" else cmd_summarize)(Path(a.out))


if __name__ == "__main__":
    main()
