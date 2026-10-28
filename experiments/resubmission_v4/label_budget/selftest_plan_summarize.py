#!/usr/bin/env python3
"""No-training self-test of summarize_label_budget.py (plan + summarize) on synthetic result files.

Writes only inside fresh temporary directories. Touches no study output, no prediction array, no model.
Usage: python selftest_plan_summarize.py            (from anywhere; run with the project interpreter)
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_label_budget as RB  # noqa: E402

SUMM = str(HERE / "summarize_label_budget.py")
RUNNER_HASH = "r" * 64


def fake_result(scen, n, k, wall, ids, cells=None, script=RUNNER_HASH, identity=True):
    cells = cells or {}
    targets = {}
    for t in ("y1", "y2"):
        a, b, lo, hi = cells.get(t, (False, False, -0.1, 0.1))
        targets[t] = {"C_minus_D": {"delta": (lo + hi) / 2, "ci95": [lo, hi]}, "A_admits": a, "B_admits": b,
                      "test_MAE": {r: 1.0 for r in RB.RULES}, "oof_adapter_MAE": 1.0, "required_MAE": 1.0}
    return {"scenario": scen, "n": n, "k": k, "targets": targets, "wall_seconds": wall, "identity_pass": identity,
            "script_sha256": script, **ids}


def make_dir(wall):
    d = Path(tempfile.mkdtemp(prefix="lb_selftest_"))
    (d / "configs").mkdir()
    RB.dump_json({"pass": True}, d / "level2_check.json")
    RB.dump_json({"dummy": "manifest"}, d / "input_sha256_manifest.json")
    RB.dump_json({"dummy": "protocol"}, d / "effective_protocol.json")
    ids = {"manifest_sha256": RB.sha256(d / "input_sha256_manifest.json"),
           "effective_protocol_sha256": RB.sha256(d / "effective_protocol.json")}
    for c in RB.FIRST_STAGE:
        put(d, c, fake_result(*c, wall, ids))
    return d, ids


def put(d, c, res):
    p = d / "configs" / RB.config_name(*c)
    p.mkdir()
    RB.dump_json(res, p / "result.json")


def run(cmd, d):
    r = subprocess.run([sys.executable, SUMM, cmd, "--out", str(d)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def decision(d, tag):
    return RB.load_json(d / f"decision_{tag}.json")


checks = []


def check(name, cond):
    checks.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# 0. without a plan nothing can be final
d, ids = make_dir(600)
run("summarize", d)
check("pre-plan summary is not final and carries no decision",
      decision(d, "preplan_6_cells")["second_criterion_retained"] is None)

# 1-3. the three branches of the time rule
for wall, exp in ((600, (5, 1, 32, 64)), (1500, (5, 3, 32, 64)), (4000, (3, 3, 20, 40))):
    d, ids = make_dir(wall)
    print("   ", run("plan", d))
    plan = RB.load_json(d / "run_plan.json")
    got = (plan["n_seeds"], plan["nslots"], plan["n_configs"], plan["cells_expected"])
    check(f"wall {wall} s -> seeds/slots/configs/cells {exp}", got == exp)
    slotted = [tuple(c) for s in plan["slots"] for c in s]
    left = [c for c in RB.all_configs(plan["n_seeds"]) if c not in RB.FIRST_STAGE]
    check(f"wall {wall} s -> slots partition the {len(left)} remaining configurations exactly once",
          sorted(slotted) == sorted(left) and len(set(slotted)) == len(slotted)
          and max(map(len, plan["slots"])) - min(map(len, plan["slots"])) <= 1)
    before = (d / "run_plan.json").read_bytes()
    run("plan", d)
    check(f"wall {wall} s -> a second plan call does not rewrite run_plan.json",
          (d / "run_plan.json").read_bytes() == before)
    run("summarize", d)
    check(f"wall {wall} s -> partial summary (6 of {exp[3]}) has no decision",
          decision(d, f"partial_6_of_{exp[3]}")["second_criterion_retained"] is None)
    special = {left[0]: {"y1": (True, False, 0.2, 0.4)},            # disagreement, harmful -> avoided
               left[1]: {"y2": (True, False, -0.4, -0.2)},          # disagreement, helpful -> missed
               left[2]: {"y1": (True, False, -0.1, 0.1)},           # disagreement, unresolved -> neither
               left[3]: {"y1": (True, True, 0.2, 0.4)}}             # no disagreement -> never counted
    for c in left:
        put(d, c, fake_result(*c, wall, ids, special.get(c)))
    run("summarize", d)
    dec = decision(d, "final")
    check(f"wall {wall} s -> final summary over {exp[3]} cells: avoided 1, missed 1, criterion retained",
          dec["cells_present"] == exp[3] and dec["avoided_harmful_admissions"] == 1
          and dec["missed_recoveries"] == 1 and dec["second_criterion_retained"] is True)
    rows = (d / "cells_final.csv").read_text().strip().splitlines()
    check(f"wall {wall} s -> cells_final.csv has {exp[3]} rows", len(rows) - 1 == exp[3])

# 4. avoided 0 -> margin-only
d, ids = make_dir(600)
run("plan", d)
for c in [c for c in RB.all_configs(5) if c not in RB.FIRST_STAGE]:
    put(d, c, fake_result(*c, 600, ids))
run("summarize", d)
check("no disagreement cell -> second criterion not retained (margin-only)",
      decision(d, "final")["second_criterion_retained"] is False)

# 5. mixed runner versions or a failed identity are never final
for label, kw in (("two runner versions", {"script": "x" * 64}), ("failed identity check", {"identity": False})):
    d, ids = make_dir(600)
    run("plan", d)
    left = [c for c in RB.all_configs(5) if c not in RB.FIRST_STAGE]
    for i, c in enumerate(left):
        put(d, c, fake_result(*c, 600, ids, **(kw if i == 0 else {})))
    run("summarize", d)
    dec = decision(d, "inconsistent_64_of_64")
    check(f"{label} -> tagged inconsistent, no decision",
          dec["second_criterion_retained"] is None and len(dec["consistency_problems"]) >= 1)

print(f"\n{sum(ok for _, ok in checks)}/{len(checks)} checks passed")
print("summarizer sha256:", hashlib.sha256(Path(SUMM).read_bytes()).hexdigest())
sys.exit(0 if all(ok for _, ok in checks) else 1)
