# -*- coding: utf-8 -*-
"""
IEEE 300-Bus Dynamic Simulation Data Generator (v3).

Generates post-disturbance frequency response data for three disturbance types:
  - circuit_short: three-phase bus fault → line trip → reclose
  - load_change:   sudden active power load variation
  - cut_machine:   simultaneous generator tripping

Requires PSS/E 33.04 with Python 2.7 API (pssepath, psspy, dyntools).

Usage:
    Run this script in a PSS/E-enabled Python 2.7 environment on Windows.
    Configure paths and parameters in the __main__ block at the bottom.
"""

import copy
import os
import re
import sys
import time
import random

import cStringIO
import numpy as np
import pandas as pd
from tqdm import tqdm

import pssepath
pssepath.add_pssepath(33)
import psspy, dyntools

# PSS/E default placeholders
_i = psspy.getdefaultint()
_f = psspy.getdefaultreal()
_s = psspy.getdefaultchar()

# Convergence thresholds
TAIL_LINES = 30       # Number of log tail lines to check for errors
NNC_THRESHOLD = 2     # Max allowed "Network not converged" occurrences


# ══════════════════════════════════════════════════════════════════
# Utility functions
# ══════════════════════════════════════════════════════════════════

def tail(filepath, n, block=-1024):
    """Read the last n lines of a file efficiently."""
    with open(filepath, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        while True:
            if filesize >= abs(block):
                f.seek(block, 2)
                lines = f.readlines()
                if len(lines) > n:
                    return lines[-n:]
                block *= 2
            else:
                block = -filesize


class TimeoutException(Exception):
    pass


def check_simulation_errors(log_path):
    """Check PSS/E log for simulation errors.

    Returns:
        (has_error: bool, message: str)
    """
    console = ''.join(tail(log_path, TAIL_LINES))

    nnc_count = console.count("Network not converged")
    if nnc_count > NNC_THRESHOLD:
        return True, "Network not converged (%d times)" % nnc_count
    if "NaN" in console:
        return True, "NaN detected in simulation"
    if "INITIAL CONDITIONS CHECK O.K." in console:
        return False, "OK"
    return False, "No fatal errors detected"


def fetch_results(d, e, z):
    """Parse PSS/E dynamic simulation output into DataFrames per channel type.

    Args:
        d: Header dict from dyntools
        e: Channel labels dict
        z: Time series data dict

    Returns:
        dict of {channel_type: pd.DataFrame}
    """
    channel_types = set()
    for ch in range(1, len(e)):
        parts = re.split(r' |\[|\]', e[ch])
        channel_types.add(parts[0])

    result = {ct: pd.DataFrame() for ct in channel_types}

    for ch in range(1, len(e)):
        parts = re.split(r' |\[|\]', e[ch])
        ch_type, ch_id = parts[0], parts[1]

        if len(result[ch_type]) == 0:
            result[ch_type] = pd.DataFrame(
                z[ch], columns=[ch_id], index=z['time']
            )
        else:
            result[ch_type].insert(
                result[ch_type].shape[1], ch_id, z[ch], allow_duplicates=True
            )

    return result


def save_results_to_xlsx(basepath, out_file, d, e, z):
    """Convert simulation output to xlsx with one sheet per channel type."""
    try:
        data = dyntools.CHNF(os.path.join(basepath, out_file))
        d, e, z = data.get_data()
        sheets = fetch_results(d, e, z)
        xlsx_path = os.path.join(basepath, out_file[:-4] + '.xlsx')
        with pd.ExcelWriter(xlsx_path) as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name)
        print("Saved: %s" % xlsx_path)
        return True
    except Exception as exc:
        print("Error saving results: %s" % exc)
        return False


# ══════════════════════════════════════════════════════════════════
# System info & steady-state setup (shared by all disturbance types)
# ══════════════════════════════════════════════════════════════════

def get_system_info(sav=None):
    """Read current system state: loads, generators, branches.

    Args:
        sav: Path to .sav file. If provided, loads the case first.

    Returns:
        (loadbuses, demand_mw, generators, genp, loads, branch_flows)
    """
    if sav is not None:
        psspy.case(sav)

    _, demand = psspy.systot('LOAD')
    _, supply = psspy.systot('GEN')
    demand_mw = demand.real

    _, (genbuses,) = psspy.amachint(-1, 1, 'NUMBER')
    _, (genid,) = psspy.amachchar(-1, 1, 'ID')
    _, (genp,) = psspy.amachreal(-1, 1, 'PGEN')
    _, (genpmax,) = psspy.amachreal(-1, 1, 'PMAX')
    generators = zip(genbuses, genid, genp, genpmax)

    _, (loadbuses,) = psspy.aloadint(-1, 1, 'NUMBER')
    _, (loadid,) = psspy.aloadchar(-1, 1, 'ID')
    _, (loadt,) = psspy.aloadcplx(-1, 1, 'TOTALACT')
    loads = zip(loadbuses, loadid, loadt)

    _, (tobus,) = psspy.aflowint(-1, 1, 1, 1, 'TONUMBER')
    _, (frmbus,) = psspy.aflowint(-1, 1, 1, 1, 'FROMNUMBER')
    _, cktid = psspy.aflowchar(-1, 1, 1, 1, 'ID')
    branches = zip(frmbus, tobus, cktid[0])

    branch_flows = []
    for br in branches:
        _, pflow = psspy.brnflo(br[0], br[1], br[2])
        branch_flows.append((br[0], br[1], br[2], pflow))

    return loadbuses, demand_mw, generators, genp, loads, branch_flows


def setup_operating_conditions(sav, dyr, sp, h_originals, gen_bus_nums,
                               random_cut_bus=False, cut_branch=None):
    """Configure steady-state: load level, reserve, ZIP model, inertia, channels.

    This is the shared setup for all three disturbance types.

    Args:
        sav: Path to .sav case file
        dyr: Path to .dyr dynamics file
        sp: Operating condition dict with keys: le, lz, rr, hi
        h_originals: DataFrame with columns [bus_num, H]
        gen_bus_nums: List of generator bus numbers for channel monitoring
        random_cut_bus: Whether to trip a random branch
        cut_branch: (fbus, tbus, cktid) tuple for topology change

    Returns:
        (generators, loads, branch_flows) after power flow re-solve,
        or None if power flow diverges.
    """
    le, lz, rr, hi = sp['le'], sp['lz'], sp['rr'], sp['hi']

    # Load case
    psspy.case(sav)

    # Read current system state
    _, demand = psspy.systot('LOAD')
    demand_mw = demand.real
    _, (genbuses,) = psspy.amachint(-1, 1, 'NUMBER')
    _, (genid,) = psspy.amachchar(-1, 1, 'ID')
    _, (genp,) = psspy.amachreal(-1, 1, 'PGEN')
    _, (genpmax,) = psspy.amachreal(-1, 1, 'PMAX')
    generators = list(zip(genbuses, genid, genp, genpmax))

    # Scale load level
    psspy.scal_2(0, 1, 1, [0, 0, 0, 0, 0], [0.0]*7)
    psspy.scal_2(0, 1, 2, [_i, 2, 0, 1, 0], [le - 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Scale generator capacity (spinning reserve)
    psspy.bsys(0, 0, [0.6, 345.], 0, [], len(generators),
               [g[0] for g in generators], 0, [], 0, [])
    psspy.scal_2(0, 0, 1, [0, 0, 0, 0, 0], [0.0]*7)
    psspy.scal_2(0, 1, 2, [_i, 2, 0, 1, 0], [0.0, rr - 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Optional topology change (trip a branch before power flow)
    if random_cut_bus and cut_branch:
        psspy.dist_branch_trip(cut_branch[0], cut_branch[1], cut_branch[2])

    # Solve power flow
    try:
        err = psspy.fnsl([0, 0, 0, 1, 1, 0, 99, 0])
        if err != 0:
            print("Power flow diverged (err=%d)" % err)
            return None
    except Exception as exc:
        print("Power flow exception: %s" % exc)
        return None

    # Re-read system state after power flow
    _, _, generators_new, genp_new, loads_new, branch_flows = get_system_info()

    # Convert loads to ZIP model
    psspy.cong(0)
    psspy.conl(0, 1, 1, [0, 0], [lz[1], lz[0], lz[1], lz[0]])
    psspy.conl(0, 1, 2, [0, 0], [lz[1], lz[0], lz[1], lz[0]])
    psspy.conl(0, 1, 3, [0, 0], [lz[1], lz[0], lz[1], lz[0]])

    # Initialize dynamic simulation
    psspy.ordr(1)
    psspy.fact()
    psspy.tysl(0)
    psspy.dyre_new([1, 1, 1, 1], dyr, "", "", "")

    # Set inertia constants
    for b, gid, p, pm in generators:
        gen_model = psspy.mdlnam(b, gid, 'GEN')[-1].strip()
        h_orig = h_originals[h_originals["bus_num"] == b]['H'].values[0]
        h_new = h_orig * hi

        if gen_model == 'GENROU':
            err = psspy.change_plmod_con(b, r"""1""", r"""GENROU""", 5, h_new)
            assert err == 0, "GENROU inertia change failed at bus %d" % b
        elif gen_model == 'GENCLS':
            err = psspy.change_plmod_con(b, r"""1""", r"""GENCLS""", 1, h_new)
            assert err == 0, "GENCLS inertia change failed at bus %d" % b

    # Set reference machine angle and monitoring channels
    psspy.set_relang(1, gen_bus_nums[0], r"""1""")

    n, m = 1, 0
    psspy.bsys(n, m, [0.6, 345.], 0, [], len(gen_bus_nums), gen_bus_nums, 0, [], 0, [])

    # Select output channels (generator-only monitoring)
    CHANNELS = [
        (1,  "ANGLE"),    # Machine relative rotor angle (degrees)
        (2,  "PELEC"),    # Machine electrical power (pu on SBASE)
        (6,  "PMECH"),    # Turbine mechanical power (pu on MBASE)
        (7,  "SPEED"),    # Machine speed deviation from nominal (pu)
        (8,  "XADIFD"),   # Machine field current (pu)
        (12, "BSFREQ"),   # Bus frequency deviations (pu)
        (13, "VOLT"),     # Bus voltages (complex pu)
        (25, "PLOAD"),    # Active power load
        (27, "GREF"),     # Turbine governor reference
        (28, "LCREF"),    # Turbine load control reference
    ]
    for ch_id, ch_name in CHANNELS:
        psspy.chsb(n, m, [-1, -1, -1, 1, ch_id, 0])

    return generators_new, loads_new, branch_flows


# ══════════════════════════════════════════════════════════════════
# Disturbance simulation functions
# ══════════════════════════════════════════════════════════════════

def run_circuit_short(basepath, out_file, tst, fbus, tbus, bid, runtime,
                      log_path):
    """Execute circuit short disturbance simulation.

    Sequence: start → run to tst → bus fault → clear after 80ms →
              trip line → reclose after 1.2s → run to end.
    """
    psspy.strt(0, os.path.join(basepath, out_file))
    psspy.run(0, tst, 0, 1, 1)                   # Run to fault trigger time
    psspy.dist_bus_fault(fbus)                     # Apply three-phase bus fault
    psspy.run(0, tst + 0.08, 0, 1, 1)             # Fault duration: 80ms
    psspy.dist_branch_trip(fbus, tbus, bid)        # Trip faulted line
    psspy.dist_clear_fault(1)                      # Clear fault
    psspy.run(0, tst + 0.08 + 1.2, 0, 1, 1)       # Wait 1.2s for reclose
    psspy.dist_branch_close(fbus, tbus, bid)       # Reclose line
    psspy.run(0, runtime, 0, 1, 1)                 # Run to end
    psspy.delete_all_plot_channels()

    time.sleep(1)
    has_err, msg = check_simulation_errors(log_path)
    if has_err:
        print("  [SKIP] %s" % msg)
    return not has_err


def run_load_change(basepath, out_file, tst, lodbus, lodid, ld, lz,
                    loads, runtime, log_path):
    """Execute load change disturbance simulation.

    Applies sudden active power change at the specified load bus.
    """
    psspy.strt(0, os.path.join(basepath, out_file))
    psspy.run(0, tst, 0, 1, 1)

    # Find the current load value at the target bus
    lodcplx = None
    for lbus, lid, lcpx in loads:
        if lodbus == lbus and lodid == lid:
            lodcplx = lcpx
            break

    if lodcplx is None:
        print("  [SKIP] Load bus %d not found" % lodbus)
        return False

    # Apply load change with ZIP proportions
    load_new = (1.0 + float(ld) / 100.0) * lodcplx.real
    z_ratio = float(lz[0]) / 100.0
    i_ratio = float(lz[1]) / 100.0
    p_ratio = float(lz[2]) / 100.0
    psspy.load_data_4(lodbus, lodid, [_i]*6,
                      [load_new * p_ratio, _f, load_new * i_ratio, _f,
                       load_new * z_ratio, _f])

    psspy.run(0, runtime, 0, 1, 1)
    psspy.delete_all_plot_channels()

    time.sleep(0.1)
    has_err, msg = check_simulation_errors(log_path)
    if has_err:
        print("  [SKIP] %s" % msg)
    return not has_err


def run_cut_machine(basepath, out_file, tst, tet, gbus_list, runtime,
                    log_path):
    """Execute generator tripping disturbance simulation.

    Trips multiple generators simultaneously.
    """
    psspy.strt(0, os.path.join(basepath, out_file))
    psspy.run(0, tst, 0, 1, 1)

    for gbus in gbus_list:
        psspy.dist_machine_trip(gbus)

    psspy.run(0, tet, 0, 1, 1)
    psspy.dist_clear_fault(1)
    psspy.run(0, runtime, 0, 1, 1)
    psspy.delete_all_plot_channels()

    has_err, msg = check_simulation_errors(log_path)
    if has_err:
        print("  [SKIP] %s" % msg)
    return not has_err


# ══════════════════════════════════════════════════════════════════
# Recorder: tracks completed simulations to enable resume
# ══════════════════════════════════════════════════════════════════

class SimRecorder:
    """Simple file-based recorder to skip already-completed simulations."""

    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            open(path, 'w').close()
        with open(path, 'r') as f:
            self.completed = set(line.strip() for line in f if line.strip())

    def is_done(self, key):
        return key in self.completed

    def mark_done(self, key):
        self.completed.add(key)
        with open(self.path, 'a') as f:
            f.write(key + '\n')


# ══════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    random_cut_bus = False

    # ── PSS/E initialization ──
    err = psspy.psseinit(200000)
    assert err == 0, "PSS/E initialization failed"

    log_path = r'E:\simulation\code\psse3304_tutorials\logs\log300_v3.txt'
    psspy.progress_output(2, log_path)
    psspy.alert_output(2, log_path)
    psspy.report_output(2, log_path)
    psspy.prompt_output(2, log_path)

    # ── File paths ──
    base_path = r'E:\simulation\code\psse3304_tutorials\ieee300\Outputs\IEEE300\V1'
    sav_file = r'E:\simulation\code\psse3304_tutorials\Demo_Models\IEEE300\IEEE300.sav'
    dyr_file = r'E:\simulation\code\psse3304_tutorials\Demo_Models\IEEE300\IEEE300_dyn_v2.dyr'
    h_originals = pd.read_csv(
        r'E:\simulation\code\psse3304_tutorials\ieee300\ieee300_gen_Hs.csv'
    )
    gen_bus_nums = h_originals['bus_num'].values.flatten().tolist()

    # ── Simulation timing ──
    trigger_start = 1.0     # Disturbance trigger time (s)
    trigger_end = 1.17      # Fault clearing time (s)
    runtime = 20            # Total simulation duration (s)
    channel_option = 'GEN'  # Monitor generator buses only

    # ── Operating condition parameter space ──
    load_levels = np.arange(90.0, 130.0, 20.0)        # Load level (%)
    zip_models = [[0, 100, 0], [20.0, 50.0, 30.0]]    # ZIP proportions [Z, I, P]
    reserve_ratios = np.arange(90.0, 130.0, 20.0)      # Generator capacity (%)
    inertia_scales = np.arange(0.2, 2.2, 0.8)          # Inertia scaling factor

    # ── Disturbance parameters ──
    n_gen_trip = 7           # Number of generators to trip per cut_machine case
    n_load_samples = 10      # Number of random load buses for load_change
    n_branch_samples = 10    # Number of random branches for circuit_short
    load_deltas = [-5000.0, 5000.0, -15000.0, 15000.0]  # Load change (MW)

    # ── Main simulation loop ──
    for le in tqdm(load_levels, desc="Load level"):
        for lz in zip_models:
            for rr in reserve_ratios:
                for hi in inertia_scales:
                    t0 = time.time()
                    sp = {'le': le, 'lz': lz, 'rr': rr, 'hi': hi}

                    # Read baseline system info
                    loadbuses, demand_mw, generators, genp, loads, branch_flows = \
                        get_system_info(sav_file)
                    generators = list(generators)
                    loads = list(loads)

                    # Random branch for topology change
                    ftb = random.choice([(bf[0], bf[1], bf[2]) for bf in branch_flows])

                    sav_tag = 'IEEE300_le%.1f_zip%s-%s-%s_rr%.1f_hi%.1f' % (
                        le, lz[0], lz[1], lz[2], rr, hi
                    )

                    # ────────────────────────────────────
                    # Cut machine
                    # ────────────────────────────────────
                    cm_path = os.path.join(base_path, 'cut_machine')
                    if not os.path.exists(cm_path):
                        os.makedirs(cm_path)
                    recorder = SimRecorder(os.path.join(cm_path, 'recorder.txt'))

                    for _ in range(len(generators)):
                        gbus = [g[0] for g in random.sample(generators, n_gen_trip)]
                        gbus_str = '-'.join(str(g) for g in gbus)
                        case = 'psse3304_%s-cut_machine-gbus_%s-%s_%ds' % (
                            sav_tag, gbus_str, channel_option, runtime
                        )

                        if recorder.is_done(case):
                            continue
                        recorder.mark_done(case)

                        out_file = case + '.out'
                        print(case)

                        try:
                            result = setup_operating_conditions(
                                sav_file, dyr_file, sp, h_originals,
                                gen_bus_nums, random_cut_bus, ftb
                            )
                            if result is None:
                                continue
                            _, _, _ = result

                            success = run_cut_machine(
                                cm_path, out_file, trigger_start, trigger_end,
                                gbus, runtime, log_path
                            )
                            if success:
                                save_results_to_xlsx(
                                    cm_path, out_file, None, None, None
                                )
                        except Exception as exc:
                            print("  Exception: %s" % exc)

                        # Clean up .out file
                        out_abs = os.path.join(cm_path, out_file)
                        if os.path.exists(out_abs):
                            try: os.remove(out_abs)
                            except: pass

                    # ────────────────────────────────────
                    # Load change
                    # ────────────────────────────────────
                    lc_path = os.path.join(base_path, 'load_change')
                    if not os.path.exists(lc_path):
                        os.makedirs(lc_path)
                    recorder = SimRecorder(os.path.join(lc_path, 'recorder.txt'))

                    for ld in load_deltas:
                        selected = random.sample(loads, min(n_load_samples, len(loads)))
                        for lodbus, lodid, _ in selected:
                            case = 'psse3304_%s_ld_%.1f-load_change-lodbus_%s-%s_%ds' % (
                                sav_tag, ld, lodbus, channel_option, runtime
                            )

                            if recorder.is_done(case):
                                continue
                            recorder.mark_done(case)

                            out_file = case + '.out'
                            print(case)

                            try:
                                result = setup_operating_conditions(
                                    sav_file, dyr_file, sp, h_originals,
                                    gen_bus_nums, random_cut_bus, ftb
                                )
                                if result is None:
                                    continue
                                _, loads_new, _ = result

                                success = run_load_change(
                                    lc_path, out_file, trigger_start,
                                    lodbus, lodid, ld, lz, loads_new,
                                    runtime, log_path
                                )
                                if success:
                                    save_results_to_xlsx(
                                        lc_path, out_file, None, None, None
                                    )
                            except Exception as exc:
                                print("  Exception: %s" % exc)

                            out_abs = os.path.join(lc_path, out_file)
                            if os.path.exists(out_abs):
                                try: os.remove(out_abs)
                                except: pass

                    # ────────────────────────────────────
                    # Circuit short
                    # ────────────────────────────────────
                    cs_path = os.path.join(base_path, 'circuit_short')
                    if not os.path.exists(cs_path):
                        os.makedirs(cs_path)
                    recorder = SimRecorder(os.path.join(cs_path, 'recorder.txt'))

                    selected_bfs = random.sample(
                        branch_flows, min(n_branch_samples, len(branch_flows))
                    )
                    for frmbus, tobus, cktid, _ in selected_bfs:
                        case = 'psse3304_%s-circuit_short-frmbus%s_tobus%s-%s_%ds' % (
                            sav_tag, frmbus, tobus, channel_option, runtime
                        )

                        if recorder.is_done(case):
                            continue
                        recorder.mark_done(case)

                        out_file = case + '.out'
                        print(case)

                        try:
                            result = setup_operating_conditions(
                                sav_file, dyr_file, sp, h_originals,
                                gen_bus_nums, False, None
                            )
                            if result is None:
                                continue

                            success = run_circuit_short(
                                cs_path, out_file, trigger_start,
                                frmbus, tobus, cktid, runtime, log_path
                            )
                            if success:
                                save_results_to_xlsx(
                                    cs_path, out_file, None, None, None
                                )
                        except Exception as exc:
                            print("  Exception: %s" % exc)

                        out_abs = os.path.join(cs_path, out_file)
                        if os.path.exists(out_abs):
                            try: os.remove(out_abs)
                            except: pass

                    elapsed = time.time() - t0
                    print("Condition le=%.0f lz=%s rr=%.0f hi=%.1f done in %.1fs" % (
                        le, lz, rr, hi, elapsed
                    ))
