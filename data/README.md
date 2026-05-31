# TD-HER Dataset

Pre-processed representations for reproducing all results in the paper
*"Target-Dependent Heterogeneous Expert Routing for Power System Frequency
Extremum and Arrival-Time Prediction"*.

## Quick Start

All experiment scripts default to `data/` as the dataset root.
No additional configuration is needed if the dataset is placed here.

## Dataset Structure

```
data/
├── ieee39/                     # IEEE 39-bus New England system (10 generators)
│   ├── repA/                   # Representation A (tabular features)
│   │   ├── ms1/                #   ~10 ms early window (1 post-trigger step)
│   │   ├── ms5/                #   ~50 ms early window
│   │   ├── ms10/               #   ~100 ms early window (main setting)
│   │   ├── ms15/               #   ~150 ms early window
│   │   └── ms25/               #   ~250 ms early window
│   ├── repB/                   # Representation B (spatiotemporal tensor + static)
│   │   └── ms{1,5,10,15,25}/
│   ├── repC/                   # Representation C (graph: tensor + adjacency)
│   │   └── ms{1,5,10,15,25}/
│   ├── csv/                    # Raw ms10 CSV files (for robustness stress test)
│   └── adjacency/              # 10×10 Kron-reduced electrical-distance adjacency
│
└── ieee300/                    # IEEE 300-bus system (69 generators)
    ├── repA/ms10/              # Tabular features (same-distribution only)
    ├── repB/ms10/              # Spatiotemporal tensor + static
    ├── repC/ms10/              # Graph representation
    └── adjacency/              # 69×69 electrical-distance adjacency
```

## Split Convention

Each `ms{N}/` directory contains NumPy arrays following the naming convention:

| File pattern | Description |
|---|---|
| `X_train.npy`, `y_train.npy` | Training set (L1 same-distribution) |
| `X_val.npy`, `y_val.npy` | Validation/calibration set (L1) |
| `X_test.npy`, `y_test.npy` | Held-out test set (L1) |
| `X_cross_cond_finetune.npy` | L2 cross-condition calibration set |
| `X_cross_cond_test.npy` | L2 cross-condition test set |
| `X_cross_cond_topo_finetune.npy` | L3 cross-topology calibration set |
| `X_cross_cond_topo_test.npy` | L3 cross-topology test set |
| `feature_names.json` | Ordered feature names for RepA |
| `meta.json` | Tensor dimensions and channel info for RepB |

L2/L3 splits are only available at `ms10` (the main early-window setting).

## Prediction Targets

- **y1** (`y[:,0]`): Signed COI frequency extremum (Hz)
- **y2** (`y[:,1]`): Nonnegative time from disturbance trigger to frequency extremum (s)

## Sample Counts

| System | Split | Samples |
|---|---|---|
| IEEE 39-bus | Train | 101,390 |
| IEEE 39-bus | Validation/Calibration | 12,673 |
| IEEE 39-bus | Test | 12,676 |
| IEEE 39-bus | L2 cross-condition calibration | 510 |
| IEEE 39-bus | L2 cross-condition test | 511 |
| IEEE 39-bus | L3 cross-topology calibration | 2,337 |
| IEEE 39-bus | L3 cross-topology test | 2,337 |
| IEEE 300-bus | Train | 7,391 |
| IEEE 300-bus | Validation | 923 |
| IEEE 300-bus | Test | 926 |

## Channels

| Channel | Source | Description |
|---|---|---|
| `FREQ` | PMU-observable | Bus frequency |
| `VOLT` | PMU-observable | Bus voltage magnitude |
| `ANGL` | PMU-observable | Bus voltage angle |
| `POWR` | PMU-computable | Generator-terminal active power |
| `SPD` | Generator-side state | Generator speed deviation (PMU-proxy compatible) |

## Paper Section Mapping

| Data | Paper section | Experiment |
|---|---|---|
| ieee39/rep{A,B,C}/ms10 | §IV.B–E | Main expert comparison, TD-HER routing, ablation, weights |
| ieee39/rep{A,B,C}/ms{1,5,15,25} | §IV.G, Fig. 10 | Multi-window latency-accuracy study |
| ieee39/csv/*_ms10.csv | Supplementary §III | Robustness stress test (sensor noise, gaps) |
| ieee39/adjacency/ | §IV (ST-GCN input) | Graph expert adjacency matrix |
| ieee300/rep{A,B,C}/ms10 | §IV.F | 300-bus scalability experiment |
| ieee300/adjacency/ | §IV.F (ST-GCN input) | 300-bus graph adjacency matrix |

## Data Generation

The raw time-domain simulation data were generated in PSS/E under randomized
disturbances and operating conditions (see Section IV-A of the paper).
The representations were constructed by the pipeline in `data_proc/`:

1. `data_proc/extract_features.py` — xlsx/CSV extraction from PSS/E output
2. `data_proc/build_representations.py` — RepA/RepB/RepC construction
3. `data_proc/build_adjacency.py` — Kron-reduced adjacency matrices

## IEEE Test System RAW Topology Files

The graph-expert adjacency builder (`data_proc/build_adjacency.py`) consumes
PSS/E `.RAW` topology files. These are standard public IEEE test cases and are
not redistributed here.

| File | Public source |
|---|---|
| `IEEE39.RAW` | IEEE PES Test Feeder Working Group / ICSEG |
| `IEEE300Bus_modified_noHVDC_v2.raw` | RLGC repository (`testData/IEEE300/`) |

Place them in `data/topology/` if you need to regenerate adjacency matrices
from scratch (not required if using the pre-computed `.npy` files above).

## Size

| Component | Size |
|---|---|
| ieee39 (all windows) | 15.5 GB |
| ieee300 (ms10 only) | 2.5 GB |
| **Total** | **18.0 GB** |

205 files total (184 `.npy`, 12 `.json`, 7 `.csv`, 2 other).

## License

Released under the same license as the TD-HER repository.
