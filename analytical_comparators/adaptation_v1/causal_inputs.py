"""Read exactly the frozen ML observation rows; no future measurement values.

Only the time column is inspected to locate rows. Measurement sheets are read
with a hard n_rows bound ending at the last permitted row. Labels, event names,
filenames, PLOD and evaluator caches are not used to construct model inputs.
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

Buses=tuple(range(30,40))

def read_frozen_window(workbook, trigger_s=1., n_post=10, bus_order=Buses):
    book=fastexcel.read_excel(Path(workbook))
    # Timing metadata only. 128 rows suffices for the frozen 10 ms dataset;
    # fail explicitly for a different sampling rate rather than widening input.
    td=book.load_sheet('FREQ',n_rows=128,use_columns=[0]).to_polars()
    t=np.asarray(td.to_series(0),dtype=float)
    post=np.flatnonzero(t>trigger_s)
    if not len(post):raise ValueError('Trigger not located within frozen timing probe')
    trans=int(post[0]);tf0=trans-1
    closest=int(np.argmin(abs(t-trigger_s)))
    t0=closest-1 if t[closest]>=trigger_s else closest
    idx=np.r_[t0,tf0+np.arange(1,n_post+1)]
    if t0<0 or idx[-1]>=len(t):raise ValueError('Incomplete observation prefix')
    selected=t[idx]
    if np.any(np.diff(selected)<0) or selected[-1]>trigger_s+.100+1e-10:
        raise ValueError('Window violates recorded order or 100 ms ceiling')
    arrs={}
    for sheet in ['FREQ','POWR']:
        df=book.load_sheet(sheet,n_rows=int(idx[-1])+1).to_polars()
        if any(str(b) not in df.columns for b in bus_order):raise ValueError('Missing generator measurement channel')
        sheet_t=np.asarray(df.to_series(0),dtype=float)
        if not np.array_equal(sheet_t[idx],selected):raise ValueError('Measurement sheet timestamp mismatch')
        arrs[sheet]=df.select([str(b) for b in bus_order]).to_numpy()[idx]
        if not np.all(np.isfinite(arrs[sheet])):raise ValueError('Nonfinite causal measurement')
    return {'time_relative_s':selected-trigger_s,'frequency_deviation_hz':arrs['FREQ']*60.,
            'power_channel_native_pu':arrs['POWR'],'bus_order':bus_order,'row_indices':idx,
            'measurement_read_end_row':int(idx[-1]),'observation_end_s':float(selected[-1]-trigger_s),
            'power_base_status':'REQUIRES_AUDITED_CHANNEL_BASE','input_parameters_status':'P0_STATE_BINDING_REQUIRED'}
