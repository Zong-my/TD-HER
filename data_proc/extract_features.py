#!/usr/bin/env python3
"""
Extract features from raw PSS/E xlsx simulation files.

Optimized version: uses fastexcel + polars (no pandas conversion),
numpy-vectorized feature assembly. ~1s/file on SSD.

Usage:
    python data_proc/extract_features.py --system ieee39 --data-dir /path/to/v8
    python data_proc/extract_features.py --system ieee300 --data-dir /path/to/v2
"""

import re
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastexcel import read_excel as fast_read


# ── Static feature parsing from filename ─────────────────────────

def parse_static_features(cat: str, filename: str) -> dict:
    """Extract operating condition parameters from simulation filename."""
    fe = {
        'load_level': 0.0, 'load_zip_z': 0.0, 'load_zip_i': 0.0,
        'load_zip_p': 0.0, 'reserve_ratio': 0.0, 'h_inertia': 0.0,
        'load_delta': 0.0,
    }

    if cat == 'cut_machine':
        parts = filename.split('_')
        for p in parts:
            if p.startswith('le'):
                try: fe['load_level'] = float(p[2:])
                except ValueError: pass
            elif p.startswith('zip'):
                try:
                    zips = p[3:].split('-')
                    fe['load_zip_z'] = float(zips[0])
                    fe['load_zip_i'] = float(zips[1])
                    fe['load_zip_p'] = float(zips[2])
                except (ValueError, IndexError): pass
            elif p.startswith('rr'):
                try: fe['reserve_ratio'] = float(p[2:])
                except ValueError: pass
            elif p.startswith('hi'):
                try: fe['h_inertia'] = float(p[2:].split('-')[0])
                except ValueError: pass
    else:
        # circuit_short / load_change
        floats = re.findall(r'\d+\.\d+', filename)
        if len(floats) >= 6:
            fe['load_level'] = float(floats[0])
            fe['load_zip_z'] = float(floats[1])
            fe['load_zip_i'] = float(floats[2])
            fe['load_zip_p'] = float(floats[3])
            fe['reserve_ratio'] = float(floats[4])
            fe['h_inertia'] = float(floats[5])
        if cat == 'load_change' and len(floats) >= 7:
            fe['load_delta'] = float(floats[6])

    return fe


# ── Optimized single file extraction ─────────────────────────────

def extract_single_file(filepath: str, cat: str, gen_bus_strs: set,
                        freq_bus_strs: list, freq_weights: np.ndarray,
                        base_freq: float, trigger_time: float,
                        max_ms: int = 25, full_bus: bool = False) -> dict | None:
    """Extract features from a single xlsx file using fastexcel + numpy.

    Args:
        full_bus: If True, extract ALL bus columns (not just gen_bus_strs).
                  Used for Exp3 spatial ablation.

    Returns dict with keys: static, temporal, targets, meta.
    Returns None on failure.
    """
    try:
        parser = fast_read(filepath)
        time_index = None
        sheet_data = {}  # {sheet: {bus_str: np.ndarray}}

        skip_sheets = {'PLOD'}  # PLOD: 80%+ zero-padded, excluded from all representations
        for sheet_name in parser.sheet_names:
            if sheet_name in skip_sheets:
                continue
            ws = parser.load_sheet_by_name(sheet_name)
            pl_df = ws.to_polars()
            cols = pl_df.columns

            # Time index from unnamed column (handles both fastexcel naming conventions)
            unnamed = [c for c in cols if c.startswith('__UNNAMED__') or c.startswith('Unnamed')]
            if time_index is None and unnamed:
                time_index = pl_df[unnamed[0]].to_numpy().astype(np.float64)

            # Keep bus columns: all numeric-named cols (full_bus) or gen_bus_strs only
            bus_arrays = {}
            for c in cols:
                is_bus = (c in gen_bus_strs) if not full_bus else c.isdigit()
                if is_bus:
                    arr = pl_df[c].to_numpy()
                    # Handle potential null values
                    if arr.dtype == object:
                        arr = np.array([float(x) if x is not None else 0.0 for x in arr])
                    bus_arrays[c] = arr.astype(np.float64)
            sheet_data[sheet_name] = bus_arrays

        if time_index is None or len(time_index) == 0:
            return None

        # ── Weighted system frequency ──
        freq_data = sheet_data.get('FREQ', {})
        w_sum = 0.0
        freq_system = np.zeros(len(time_index))
        for i, bus_str in enumerate(freq_bus_strs):
            if bus_str in freq_data:
                freq_system += freq_data[bus_str] * freq_weights[i]
                w_sum += freq_weights[i]
        if w_sum > 1e-12:
            freq_system /= w_sum

        # ── Locate trigger rows ──
        abs_diff = np.abs(time_index - trigger_time)
        closest = int(np.argmin(abs_diff))
        if time_index[closest] >= trigger_time:
            closest -= 1
        t0_row = max(0, closest)

        trans_row = t0_row + 1
        while trans_row < len(time_index) and time_index[trans_row] <= trigger_time:
            trans_row += 1
        tf0_row = trans_row - 1

        # ── Frequency extreme ──
        # The arrival-time target is physically defined after the disturbance.
        # Searching the full trace can select a small pre-trigger numerical
        # deviation and produce negative t_delta, which invalidates IEEE300.
        search_start = min(trans_row, len(time_index) - 1)
        if search_start < 0 or search_start >= len(time_index):
            return None
        abs_freq = np.abs(freq_system[search_start:])
        max_row = search_start + int(np.argmax(abs_freq))
        fpu_deltamax = float(freq_system[max_row]) * base_freq
        t_delta = float(time_index[max_row]) - trigger_time

        # ── Extract temporal features ──
        rows = [t0_row]  # timestep 0 = pre-disturbance
        for ms in range(1, max_ms + 1):
            r = tf0_row + ms
            if r < len(time_index):
                rows.append(r)

        temporal = {}
        for sheet_name, bus_arrays in sheet_data.items():
            for bus_str, arr in bus_arrays.items():
                for ts_idx, row_idx in enumerate(rows):
                    temporal[f'{sheet_name}_{bus_str}_{ts_idx}'] = float(arr[row_idx])

        # ── Static features ──
        filename = os.path.basename(filepath)
        static = parse_static_features(cat, filename)

        return {
            'static': static,
            'temporal': temporal,
            'targets': {'fpu_deltamax': fpu_deltamax, 't_delta': t_delta},
            'meta': {'distu_kind': cat, 'file_name': filename},
        }

    except Exception as e:
        logger.debug(f"Failed: {filepath}: {e}")
        return None


# ── Batch processing ─────────────────────────────────────────────

def process_file_batch(files: list, cat: str, base_dir: str,
                       gen_bus_strs: set, freq_bus_strs: list,
                       freq_weights: np.ndarray, base_freq: float,
                       trigger_time: float, max_ms: int,
                       thread_id: int, full_bus: bool = False) -> list:
    """Process a batch of files in a single thread."""
    results = []
    from tqdm import tqdm
    for f in tqdm(files, desc=f"T{thread_id}", position=thread_id, leave=False):
        filepath = os.path.join(base_dir, f)
        result = extract_single_file(
            filepath, cat, gen_bus_strs, freq_bus_strs, freq_weights,
            base_freq, trigger_time, max_ms, full_bus=full_bus
        )
        if result is not None:
            results.append(result)
    return results


def extract_split(split_dir: str, cats: list, gen_bus_strs: set,
                  freq_bus_strs: list, freq_weights: np.ndarray,
                  base_freq: float, trigger_time: float,
                  max_ms: int = 25, num_threads: int = 23,
                  full_bus: bool = False) -> list:
    """Extract features from all files in a split directory."""
    all_results = []

    for cat in cats:
        cat_dir = os.path.join(split_dir, cat)
        if not os.path.isdir(cat_dir):
            logger.warning(f"Not found: {cat_dir}")
            continue

        xlsx_files = sorted([f for f in os.listdir(cat_dir) if f.endswith('.xlsx')])
        logger.info(f"  {cat}: {len(xlsx_files)} files")

        if not xlsx_files:
            continue

        # Split across threads
        chunk_size = max(1, len(xlsx_files) // num_threads)
        chunks = [xlsx_files[i:i + chunk_size] for i in range(0, len(xlsx_files), chunk_size)]

        with ThreadPoolExecutor(max_workers=min(num_threads, len(chunks))) as executor:
            futures = []
            for idx, chunk in enumerate(chunks):
                futures.append(executor.submit(
                    process_file_batch, chunk, cat, cat_dir,
                    gen_bus_strs, freq_bus_strs, freq_weights,
                    base_freq, trigger_time, max_ms, idx,
                    full_bus=full_bus
                ))
            for future in as_completed(futures):
                all_results.extend(future.result())

    return all_results


def results_to_dataframe(results: list) -> pd.DataFrame:
    """Convert list of result dicts to a flat DataFrame."""
    rows = []
    for r in results:
        row = {}
        row.update(r['meta'])
        row.update(r['static'])
        row.update(r['temporal'])
        row.update(r['targets'])
        rows.append(row)
    return pd.DataFrame(rows)


def save_per_ms(df: pd.DataFrame, output_dir: str, split_name: str, ms_list: list):
    """Save separate CSVs for each ms value."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    meta_cols = ['distu_kind', 'file_name']
    static_cols = ['load_level', 'load_zip_z', 'load_zip_i', 'load_zip_p',
                   'reserve_ratio', 'h_inertia', 'load_delta']
    target_cols = ['fpu_deltamax', 't_delta']
    temporal_cols = [c for c in df.columns if c not in meta_cols + static_cols + target_cols]

    for ms in ms_list:
        ms_cols = [col for col in temporal_cols
                   if int(col.rsplit('_', 1)[-1]) <= ms]
        selected = meta_cols + static_cols + sorted(ms_cols) + target_cols
        df_ms = df[selected]
        out_path = os.path.join(output_dir, f"{split_name}_ms{ms}.csv")
        df_ms.to_csv(out_path, index=False)
        logger.info(f"  Saved {out_path}: {len(df_ms)} rows × {len(selected)} cols")


# ── System configurations ────────────────────────────────────────

IEEE39_CONFIG = {
    'gen_buses': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    'freq_bus_strs': ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39'],
    # Bus 33 weight=0 (no generator in standard IEEE 39)
    'freq_weights': np.array([6.05, 3.41, 6.05, 0.0, 3.41, 5.016, 3.141, 3.141, 5.32, 500]),
    'base_freq': 60.0,
    'trigger_time': 1.0,
    'cats': ['circuit_short', 'cut_machine', 'load_change'],
}

def _load_ieee300_weights():
    """Load IEEE300 generator inertia weights from the bundled reference table
    at ``simulation/ieee300_gen_Hs.csv`` (resolved relative to the project root).
    """
    from pathlib import Path
    csv_path = str(Path(__file__).resolve().parent.parent
                   / 'simulation' / 'ieee300_gen_Hs.csv')
    try:
        import csv
        weights = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                weights.append(float(row['H']))
        return np.array(weights)
    except Exception:
        return np.ones(69)

IEEE300_CONFIG = {
    'gen_buses': list(range(10000, 10069)),
    'freq_bus_strs': [str(b) for b in range(10000, 10069)],
    'freq_weights': _load_ieee300_weights(),
    'base_freq': 60.0,
    'trigger_time': 1.0,
    'cats': ['circuit_short', 'cut_machine', 'load_change'],
}


# ── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract features from PSS/E xlsx files")
    parser.add_argument('--system', choices=['ieee39', 'ieee300'], required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--splits', nargs='+', default=None)
    parser.add_argument('--ms-list', nargs='+', type=int, default=[1, 5, 10, 15, 25])
    parser.add_argument('--max-ms', type=int, default=25)
    parser.add_argument('--threads', type=int, default=23)
    parser.add_argument('--full-bus', action='store_true',
                        help='Extract ALL bus columns, not just generators (for Exp3 spatial ablation)')
    args = parser.parse_args()

    config = IEEE39_CONFIG if args.system == 'ieee39' else IEEE300_CONFIG
    data_dir = args.data_dir
    output_dir = args.output_dir or os.path.join(data_dir, 'csv')

    gen_bus_strs = set(str(b) for b in config['gen_buses'])

    # Auto-detect splits
    if args.splits:
        splits = args.splits
    else:
        splits = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
            and any(os.path.isdir(os.path.join(data_dir, d, c)) for c in config['cats'])
        ])

    logger.info(f"System: {args.system} | Data: {data_dir} | Splits: {splits}")
    logger.info(f"MS: {args.ms_list} | Threads: {args.threads}")

    for split in splits:
        split_dir = os.path.join(data_dir, split)
        logger.info(f"\n{'='*60}\nProcessing: {split}")

        results = extract_split(
            split_dir, config['cats'], gen_bus_strs,
            config['freq_bus_strs'], config['freq_weights'],
            config['base_freq'], config['trigger_time'],
            args.max_ms, args.threads,
            full_bus=args.full_bus,
        )

        if not results:
            logger.warning(f"No results for {split}")
            continue

        logger.info(f"Extracted {len(results)} samples")

        df = results_to_dataframe(results)

        # Save full CSV
        full_path = os.path.join(output_dir, f"{split}_full.csv")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(full_path, index=False)
        logger.info(f"Full: {full_path} ({df.shape})")

        # Save per-ms CSVs
        save_per_ms(df, output_dir, split, args.ms_list)

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
