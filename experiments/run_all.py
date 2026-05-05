#!/usr/bin/env python3
"""Master orchestrator: run all phases sequentially.

Usage:
    # Run every phase end-to-end
    python -m experiments.run_all --phase all

    # Run a single phase
    python -m experiments.run_all --phase exp_tdh_router
    python -m experiments.run_all --phase data
    python -m experiments.run_all --phase audits

The phase ordering matches the manuscript's reproducibility chain. Each phase
shells out to the corresponding script via ``python -m <module>`` so that the
project root stays on the import path.
"""

import argparse
import os
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PYTHON = sys.executable

# ── Default data paths (override via experiment_config.yaml or CLI flags) ──
IEEE39_RAW = "data/ieee39_v8"
IEEE39_REP = "data/ieee39_v8_80_10_10"
IEEE300_RAW = "data/ieee300_v2"
IEEE300_REP = "data/ieee300_v2_posttrigger"


def _run(cmd: str) -> None:
    logger.info(f"$ {cmd}")
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Phase command failed (rc={rc}): {cmd}")


# ─────────────────────────────────────────────────────────────────
# Phase 1 — Data pipeline (xlsx → CSV → adjacency → representations)
# ─────────────────────────────────────────────────────────────────
def phase_data() -> None:
    logger.info("PHASE 1 — Data pipeline")
    # IEEE 39: extract → adjacency → representations
    _run(
        f"{PYTHON} {PROJECT_ROOT}/data_proc/extract_features.py "
        f"--system ieee39 --data-dir {IEEE39_RAW} "
        f"--output-dir {IEEE39_RAW}/csv --ms-list 1 5 10 15 25 --threads 23"
    )
    _run(f"{PYTHON} {PROJECT_ROOT}/data_proc/build_adjacency.py")
    _run(
        f"{PYTHON} {PROJECT_ROOT}/data_proc/build_representations.py "
        f"--csv-dir {IEEE39_RAW}/csv --output-dir {IEEE39_REP} --ms-list 1 5 10 15 25"
    )
    # IEEE 300: same chain
    _run(
        f"{PYTHON} {PROJECT_ROOT}/data_proc/extract_features.py "
        f"--system ieee300 --data-dir {IEEE300_RAW} "
        f"--output-dir {IEEE300_REP}/csv --ms-list 10 --threads 23"
    )
    _run(
        f"{PYTHON} {PROJECT_ROOT}/data_proc/build_representations.py "
        f"--csv-dir {IEEE300_REP}/csv --output-dir {IEEE300_REP} --ms-list 10"
    )


# ─────────────────────────────────────────────────────────────────
# Phase 2 — Expert bank training (Sec. IV.B, Table II)
# ─────────────────────────────────────────────────────────────────
def phase_exp1() -> None:
    logger.info("PHASE 2 — Expert bank (exp1_main_comparison)")
    _run(f"{PYTHON} -m experiments.exp1_main_comparison")


# ─────────────────────────────────────────────────────────────────
# Phase 3 — Generalization protocols & TD-HER routing (Sec. IV.C–F)
# ─────────────────────────────────────────────────────────────────
def phase_generalization() -> None:
    logger.info("PHASE 3a — Cross-protocol baseline metrics (exp4_generalization)")
    _run(f"{PYTHON} -m experiments.exp4_generalization")


def phase_tdh_router() -> None:
    logger.info("PHASE 3b — TD-HER routing on IEEE 39-bus (exp_tdh_router)")
    _run(f"{PYTHON} -m experiments.exp_tdh_router")


# ─────────────────────────────────────────────────────────────────
# Phase 4 — Sensitivity analyses (Figs. 6–10)
# ─────────────────────────────────────────────────────────────────
def phase_sensitivity() -> None:
    logger.info("PHASE 4 — Threshold / expert / multi-window sensitivity")
    _run(f"{PYTHON} -m experiments.exp_tdh_expert_sensitivity")
    _run(f"{PYTHON} -m experiments.exp_tdh_multi_window")


# ─────────────────────────────────────────────────────────────────
# Phase 5 — Robustness redesign (Sec. IV.G)
# ─────────────────────────────────────────────────────────────────
def phase_robustness() -> None:
    logger.info("PHASE 5 — Robustness redesign chain")
    _run(f"{PYTHON} -m experiments.exp_tdh_robustness")
    _run(f"{PYTHON} -m experiments.exp_tdh_stable_lgbm_robustness")
    _run(f"{PYTHON} -m experiments.exp_tdh_robust_router")
    _run(f"{PYTHON} -m experiments.exp_tdh_quality_router")
    _run(f"{PYTHON} -m experiments.exp_tdh_quality_router_sweep")


# ─────────────────────────────────────────────────────────────────
# Phase 6 — Pipeline timing (Fig. 11)
# ─────────────────────────────────────────────────────────────────
def phase_timing() -> None:
    logger.info("PHASE 6 — Pipeline timing")
    _run(f"{PYTHON} -m experiments.exp_tdh_pipeline_timing")


# ─────────────────────────────────────────────────────────────────
# Phase 7 — IEEE 300-bus scalability (Sec. IV.H)
# ─────────────────────────────────────────────────────────────────
def phase_ieee300() -> None:
    logger.info("PHASE 7 — IEEE 300-bus scalability")
    _run(f"{PYTHON} -m experiments.exp7_scalability")
    _run(f"{PYTHON} -m experiments.exp_tdh_router_ieee300 --device cuda")


# ─────────────────────────────────────────────────────────────────
# Phase 8 — Interpretability (supplementary figS4–figS9)
# ─────────────────────────────────────────────────────────────────
def phase_interpretability() -> None:
    logger.info("PHASE 8 — Interpretability (mRMR / ALE / KAN activation)")
    _run(f"{PYTHON} -m experiments.exp5_interpretability")


# ─────────────────────────────────────────────────────────────────
# Phase 9 — Reproducibility audits (supplementary §VIII)
# ─────────────────────────────────────────────────────────────────
def phase_audits() -> None:
    logger.info("PHASE 9 — Reproducibility audits")
    _run(f"{PYTHON} -m experiments.audit_dataset_representations")
    _run(f"{PYTHON} -m experiments.audit_tdh_router --fail-on-error")
    _run(f"{PYTHON} -m experiments.audit_ieee300_exp7")
    _run(f"{PYTHON} -m experiments.audit_ieee300_tdh_router --fail-on-error")


# ─────────────────────────────────────────────────────────────────
# Phase 10 — Tables and figures
# ─────────────────────────────────────────────────────────────────
def phase_artifacts() -> None:
    logger.info("PHASE 10 — Tables and figures")
    _run(f"{PYTHON} -m experiments.export_tii_tables")
    _run(
        f"{PYTHON} -m experiments.render_tii_artifacts "
        f"--table-dir results/paper_tables "
        f"--router-dir results/ieee39/exp_tdh_router "
        f"--ieee300-dir results/ieee300/exp7_rebuild "
        f"--out-dir results/paper_artifacts"
    )
    _run(f"{PYTHON} -m experiments.render_all_figures_v2")


PHASES = {
    "data": phase_data,
    "exp1": phase_exp1,
    "generalization": phase_generalization,
    "exp_tdh_router": phase_tdh_router,
    "sensitivity": phase_sensitivity,
    "robustness": phase_robustness,
    "timing": phase_timing,
    "ieee300": phase_ieee300,
    "interpretability": phase_interpretability,
    "audits": phase_audits,
    "artifacts": phase_artifacts,
    "all": None,
}

ORDER = [
    "data", "exp1", "generalization", "exp_tdh_router",
    "sensitivity", "robustness", "timing", "ieee300",
    "interpretability", "audits", "artifacts",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=list(PHASES.keys()), default="all")
    args = parser.parse_args()

    if args.phase == "all":
        for p in ORDER:
            logger.info("\n" + "#" * 60 + f"\n# PHASE: {p}\n" + "#" * 60)
            PHASES[p]()
    else:
        PHASES[args.phase]()

    logger.info("\n" + "=" * 60 + "\nDONE\n" + "=" * 60)


if __name__ == "__main__":
    main()
