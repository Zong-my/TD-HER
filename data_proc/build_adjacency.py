#!/usr/bin/env python3
"""
Build graph adjacency matrices from PSS/E RAW files for ST-GCN.

Constructs generator-level subgraphs by:
1. Parsing RAW file → full bus/branch/transformer topology
2. Building full admittance matrix Y
3. Kron reduction to generator buses only
4. Converting electrical distance to adjacency via Gaussian kernel

Usage:
    python data_proc/build_adjacency.py
"""

import re
import numpy as np
from pathlib import Path
from loguru import logger


# ── RAW file parsing ─────────────────────────────────────────────

def parse_raw_file(raw_path: str) -> dict:
    """Parse a PSS/E RAW file and extract buses, branches, and transformers."""
    lines = Path(raw_path).read_text().splitlines(keepends=True)

    buses = {}       # {bus_num: bus_type}
    branches = []    # [(from, to, r, x)]
    transformers = [] # [(from, to, x)]

    section = 'header'
    skip_header = 3  # first 3 lines are header

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        # Detect section boundaries
        if '/' in s:
            s_upper = s.upper()
            if 'END OF BUS DATA' in s_upper:
                section = 'load'; i += 1; continue
            elif 'END OF LOAD DATA' in s_upper:
                section = 'shunt'; i += 1; continue
            elif 'END OF FIXED SHUNT' in s_upper or 'END OF SWITCHED SHUNT' in s_upper:
                section = 'gen'; i += 1; continue
            elif 'END OF GENERATOR DATA' in s_upper:
                section = 'branch'; i += 1; continue
            elif 'END OF BRANCH DATA' in s_upper:
                section = 'xfmr'; i += 1; continue
            elif 'END OF TRANSFORMER DATA' in s_upper:
                section = 'done'; i += 1; continue

        parts = [p.strip() for p in s.split(',')]

        if skip_header > 0 and section == 'header':
            skip_header -= 1
            if skip_header > 0:
                i += 1; continue
            # After skipping 3 lines, we're in bus data
            section = 'bus'

        try:
            if section == 'bus' and len(parts) >= 4:
                bus_num = int(parts[0])
                bus_type = int(parts[3])
                buses[bus_num] = bus_type

            elif section == 'branch' and len(parts) >= 5:
                fb = int(parts[0])
                tb = int(parts[1])
                r = float(parts[3])
                x = float(parts[4])
                branches.append((fb, tb, r, x))

            elif section == 'xfmr' and len(parts) >= 4:
                fb = int(parts[0])
                tb = int(parts[1])
                third = int(parts[2])
                if third == 0:  # 2-winding transformer
                    # Read next line for impedance
                    i += 1
                    xfmr_line = lines[i].strip().split(',')
                    r_xfmr = float(xfmr_line[0])
                    x_xfmr = float(xfmr_line[1])
                    transformers.append((fb, tb, r_xfmr, x_xfmr))
                    i += 2  # skip remaining transformer lines
                    continue
        except (ValueError, IndexError):
            pass

        i += 1

    gen_buses = sorted([b for b, t in buses.items() if t in [2, 3]])

    logger.info(f"Parsed {raw_path}: {len(buses)} buses, {len(branches)} branches, "
                f"{len(transformers)} transformers, {len(gen_buses)} generators")

    return {
        'buses': buses,
        'branches': branches,
        'transformers': transformers,
        'gen_buses': gen_buses,
    }


# ── Admittance matrix & Kron reduction ───────────────────────────

def build_admittance_matrix(parsed: dict) -> tuple:
    """Build full bus admittance matrix Y from branches and transformers."""
    all_buses = sorted(parsed['buses'].keys())
    n = len(all_buses)
    bus_to_idx = {b: i for i, b in enumerate(all_buses)}

    Y = np.zeros((n, n), dtype=complex)

    # Add branches
    for fb, tb, r, x in parsed['branches']:
        if fb not in bus_to_idx or tb not in bus_to_idx:
            continue
        i, j = bus_to_idx[fb], bus_to_idx[tb]
        z = complex(r, x)
        if abs(z) < 1e-12:
            continue
        y = 1.0 / z
        Y[i, i] += y
        Y[j, j] += y
        Y[i, j] -= y
        Y[j, i] -= y

    # Add transformers
    for fb, tb, r, x in parsed['transformers']:
        if fb not in bus_to_idx or tb not in bus_to_idx:
            continue
        i, j = bus_to_idx[fb], bus_to_idx[tb]
        z = complex(r, x)
        if abs(z) < 1e-12:
            continue
        y = 1.0 / z
        Y[i, i] += y
        Y[j, j] += y
        Y[i, j] -= y
        Y[j, i] -= y

    return Y, all_buses, bus_to_idx


def kron_reduce(Y: np.ndarray, all_buses: list, bus_to_idx: dict,
                retain_buses: list) -> np.ndarray:
    """Kron-reduce admittance matrix to retain only specified buses."""
    retain_idx = sorted([bus_to_idx[b] for b in retain_buses if b in bus_to_idx])
    eliminate_idx = sorted(set(range(len(all_buses))) - set(retain_idx))

    if len(eliminate_idx) == 0:
        return Y[np.ix_(retain_idx, retain_idx)]

    # Partition Y into [Yrr, Yre; Yer, Yee]
    Yrr = Y[np.ix_(retain_idx, retain_idx)]
    Yre = Y[np.ix_(retain_idx, eliminate_idx)]
    Yer = Y[np.ix_(eliminate_idx, retain_idx)]
    Yee = Y[np.ix_(eliminate_idx, eliminate_idx)]

    # Kron reduction: Y_reduced = Yrr - Yre @ inv(Yee) @ Yer
    try:
        Yee_inv = np.linalg.inv(Yee)
        Y_reduced = Yrr - Yre @ Yee_inv @ Yer
    except np.linalg.LinAlgError:
        logger.warning("Yee is singular, using pseudoinverse")
        Yee_inv = np.linalg.pinv(Yee)
        Y_reduced = Yrr - Yre @ Yee_inv @ Yer

    return Y_reduced


def admittance_to_distance(Y_reduced: np.ndarray) -> np.ndarray:
    """Convert reduced admittance matrix to electrical distance matrix."""
    n = Y_reduced.shape[0]
    Z = np.zeros((n, n))

    # Impedance magnitude between each pair
    for i in range(n):
        for j in range(i + 1, n):
            y_ij = Y_reduced[i, j]
            if abs(y_ij) > 1e-12:
                z_ij = abs(1.0 / y_ij)
            else:
                # No direct connection, use diagonal elements
                z_ij = abs(1.0 / Y_reduced[i, i]) + abs(1.0 / Y_reduced[j, j])
            Z[i, j] = z_ij
            Z[j, i] = z_ij

    return Z


def distance_to_adjacency(dist_matrix: np.ndarray, sigma: float = None) -> np.ndarray:
    """Convert distance matrix to adjacency using Gaussian kernel.

    A[i,j] = exp(-dist^2 / (2 * sigma^2))
    sigma defaults to median of non-zero distances.
    """
    n = dist_matrix.shape[0]

    # Compute sigma from median distance
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]
    nonzero = upper_tri[upper_tri > 1e-12]
    if sigma is None:
        sigma = float(np.median(nonzero)) if len(nonzero) > 0 else 1.0

    A = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

    # Zero out diagonal (will add self-loops separately)
    np.fill_diagonal(A, 0.0)

    # Add self-loops
    A = A + np.eye(n)

    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-12))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A, A_norm, sigma


# ── Main entry points ────────────────────────────────────────────

def build_gen_adjacency(raw_path: str, gen_buses: list = None,
                        output_dir: str = None) -> dict:
    """Build generator-level adjacency matrix from RAW file.

    Args:
        raw_path: Path to PSS/E .RAW file
        gen_buses: List of generator bus numbers. If None, auto-detect from RAW.
        output_dir: If provided, save adjacency matrices as .npy files.

    Returns:
        dict with keys: adjacency, adjacency_norm, distance, gen_buses, sigma
    """
    parsed = parse_raw_file(raw_path)

    if gen_buses is None:
        gen_buses = parsed['gen_buses']

    logger.info(f"Building adjacency for {len(gen_buses)} generators: {gen_buses}")

    # Build full admittance matrix
    Y, all_buses, bus_to_idx = build_admittance_matrix(parsed)
    logger.info(f"Full admittance matrix: {Y.shape}")

    # Kron reduce to generator buses
    Y_reduced = kron_reduce(Y, all_buses, bus_to_idx, gen_buses)
    logger.info(f"Reduced admittance matrix: {Y_reduced.shape}")

    # Convert to distance
    dist = admittance_to_distance(Y_reduced)

    # Convert to adjacency
    A, A_norm, sigma = distance_to_adjacency(dist)
    logger.info(f"Adjacency: {A.shape}, sigma={sigma:.6f}")

    # Save
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'adjacency.npy', A)
        np.save(out / 'adjacency_norm.npy', A_norm)
        np.save(out / 'distance_matrix.npy', dist)
        np.save(out / 'gen_buses.npy', np.array(gen_buses))
        logger.info(f"Saved adjacency matrices to {out}")

    return {
        'adjacency': A,
        'adjacency_norm': A_norm,
        'distance': dist,
        'gen_buses': gen_buses,
        'sigma': sigma,
    }


def build_fullbus_adjacency(raw_path: str, output_dir: str = None) -> dict:
    """Build full-bus adjacency matrix (for Experiment 3 spatial ablation)."""
    parsed = parse_raw_file(raw_path)
    all_buses = sorted(parsed['buses'].keys())

    logger.info(f"Building full-bus adjacency for {len(all_buses)} buses")

    Y, all_buses_sorted, bus_to_idx = build_admittance_matrix(parsed)

    # Direct adjacency from branches + transformers (binary + weighted)
    n = len(all_buses_sorted)
    A_binary = np.zeros((n, n))
    A_weighted = np.zeros((n, n))

    for fb, tb, r, x in parsed['branches'] + parsed['transformers']:
        if fb in bus_to_idx and tb in bus_to_idx:
            i, j = bus_to_idx[fb], bus_to_idx[tb]
            A_binary[i, j] = 1.0
            A_binary[j, i] = 1.0
            impedance = np.sqrt(r**2 + x**2)
            if impedance > 1e-12:
                A_weighted[i, j] = 1.0 / impedance
                A_weighted[j, i] = 1.0 / impedance

    # Add self-loops
    A_binary += np.eye(n)

    # Normalize weighted
    A_w = A_weighted + np.eye(n)
    D = np.diag(A_w.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-12))
    A_w_norm = D_inv_sqrt @ A_w @ D_inv_sqrt

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / 'adjacency_fullbus.npy', A_binary)
        np.save(out / 'adjacency_fullbus_weighted_norm.npy', A_w_norm)
        np.save(out / 'bus_order.npy', np.array(all_buses_sorted))
        logger.info(f"Saved full-bus adjacency to {out}")

    return {
        'adjacency': A_binary,
        'adjacency_weighted_norm': A_w_norm,
        'bus_order': all_buses_sorted,
    }


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    # IEEE 39 Gen-only (10 generators)
    ieee39_raw = "data/topology/IEEE39.RAW"
    ieee39_gens = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    ieee39_out = "data/ieee39_v8/adjacency"

    logger.info("=== IEEE 39 Gen-only adjacency ===")
    result39 = build_gen_adjacency(ieee39_raw, ieee39_gens, ieee39_out)
    print(f"IEEE 39 adjacency (10x10), sigma={result39['sigma']:.6f}")
    print(f"Adjacency matrix:\n{np.round(result39['adjacency'], 3)}")

    # IEEE 39 Full-bus (for Experiment 3)
    ieee39_fullbus_out = "data/ieee39_v8/adjacency_fullbus"
    logger.info("\n=== IEEE 39 Full-bus adjacency ===")
    result39_full = build_fullbus_adjacency(ieee39_raw, ieee39_fullbus_out)
    print(f"IEEE 39 full-bus adjacency ({result39_full['adjacency'].shape})")

    # IEEE 300 Gen-only (69 generators)
    ieee300_raw = "data/topology/IEEE300Bus_modified_noHVDC_v2.raw"
    ieee300_gens = list(range(10000, 10069))
    ieee300_out = "data/ieee300_v2/adjacency"

    logger.info("\n=== IEEE 300 Gen-only adjacency ===")
    result300 = build_gen_adjacency(ieee300_raw, ieee300_gens, ieee300_out)
    print(f"IEEE 300 adjacency (69x69), sigma={result300['sigma']:.6f}")
