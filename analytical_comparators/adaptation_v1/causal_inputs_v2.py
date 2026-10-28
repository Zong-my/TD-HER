"""Frozen observation reader plus separate timestamp-only evaluation grid.

The original v1 reader is retained and reused unchanged. Time-only metadata
may identify the fixed stored horizon; no future frequency/power is returned.
"""

# Resolve source imports independently of the caller's working directory.
import sys as _tdher_sys
from pathlib import Path as _TDHERPath
_tdher_repo = next(x for x in _TDHERPath(__file__).resolve().parents if (x / "tdher_paths.py").is_file())
if str(_tdher_repo) not in _tdher_sys.path:
    _tdher_sys.path.insert(0, str(_tdher_repo))
from tdher_paths import REPO_ROOT, WORK_ROOT, DATA_ROOT, RESULTS_ROOT, data_path, work_path
from pathlib import Path
import numpy as np
import fastexcel
from causal_inputs import read_frozen_window as _read


def read_frozen_window(workbook, trigger_s=1., n_post=10, bus_order=tuple(range(30,40))):
    result=_read(workbook,trigger_s,n_post,bus_order)
    result['power_channel_system_base_mva']=100.
    result['power_base_status']='PSSE33_API_PELEC_PU_ON_SBASE_AND_READONLY_SAV_SBASE_100_VERIFIED'
    result['input_parameters_status']='CALLER_BINDS_DECLARED_P0_INERTIA_AND_EQUIPMENT'
    return result


def read_post_event_times_metadata(workbook, trigger_s=1.):
    book=fastexcel.read_excel(Path(workbook))
    df=book.load_sheet('FREQ',use_columns=[0]).to_polars()
    times=np.asarray(df.to_series(0),dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times)<0):
        raise ValueError('Invalid stored time metadata')
    selected=times[times>trigger_s]-trigger_s
    if not len(selected):raise ValueError('No stored post-trigger horizon')
    return selected
