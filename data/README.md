# Data Directory — Placeholder

Raw simulation data is **not bundled** with this repository (multi-GB scale).
This directory is reserved for the dataset layout expected by the data pipeline.

## Expected Layout

```
data/
├── ieee39_v8/                         # IEEE 39-bus raw simulation set (xlsx)
│   ├── train/                         # 121,668 samples (.xlsx per scenario)
│   ├── val/                           # 12,673 samples
│   ├── test/                          # 12,676 samples
│   ├── cross_cond_test/               # 511   samples (L2 protocol)
│   ├── cross_cond_topo_test/          # 2,337 samples (L3 protocol)
│   ├── adjacency/
│   │   └── adjacency.npy              # 10×10 generator-bus electrical-distance kernel
│   └── adjacency_fullbus/
│       └── adjacency_fullbus.npy      # full-bus variant for ablation
├── ieee39_v8_80_10_10/                # IEEE 39-bus split-and-built representations
│   ├── csv/{train,val,test,...}_ms{1,5,10,15,25}.csv
│   └── {train,val,test,...}/
│       ├── repA.npy   (tabular features)
│       ├── repB.npy   (T × N × C tensor)
│       ├── repC.npy   (graph-form per-node tensor)
│       ├── y1.npy     (frequency extremum, signed)
│       └── y2.npy     (time to extremum, seconds)
├── ieee300_v2/                        # IEEE 300-bus raw simulation set
│   ├── train/                         # 7,391 samples
│   ├── val/                           #   923 samples
│   ├── test/                          #   926 samples
│   └── adjacency/
│       └── adjacency.npy              # 69×69 generator-bus kernel
├── ieee300_v2_posttrigger/            # IEEE 300-bus rebuilt post-trigger representations
│   └── {train,val,test}/{repA,repB,repC,y1,y2}.npy
└── topology/                          # PSS/E reference RAW files (not bundled)
    ├── IEEE39.RAW
    └── IEEE300Bus_modified_noHVDC_v2.raw
```

## Channels

| Channel | Source              | Description                              |
|---------|---------------------|------------------------------------------|
| `FREQ`  | PMU-observable      | Bus frequency                            |
| `VOLT`  | PMU-observable      | Bus voltage magnitude                    |
| `ANGL`  | PMU-observable      | Bus voltage angle                        |
| `POWR`  | PMU-computable      | Generator-terminal active power          |
| `SPD`   | Generator-side state| Generator speed deviation (PMU-proxy OK) |

## Targets

| Target | Notation       | Definition                                                    |
|--------|----------------|---------------------------------------------------------------|
| `y1`   | `Δf_ext` (Hz)  | Signed post-trigger COI frequency extremum                    |
| `y2`   | `t_Δf` (s)     | Non-negative time from trigger to that extremum               |

## Reproducing the Dataset

The IEEE 39-bus and IEEE 300-bus dynamic simulations are generated in PSS/E
(version 33 or later). Each sample is a single time-domain disturbance run
covering one of:

- Three-phase short circuits at randomly chosen buses
- Generator tripping
- Step load changes

Operating-condition randomization parameters (load level, ZIP composition,
reserve ratio, inertia distribution) follow the description in the manuscript
Section IV.A. The IEEE 300-bus generator inertia table is included at
`simulation/ieee300_gen_Hs.csv` for reproducibility, and a reference
generation script is provided at `simulation/generate_freq_data300_v3.py`.

## IEEE Test System RAW Topology Files

The graph-expert adjacency builder (`data_proc/build_adjacency.py`) and a few
figure-rendering helpers consume the PSS/E `.RAW` topology specifications of
the IEEE 39-bus and IEEE 300-bus reference systems. These are **standard
public test cases** distributed by the IEEE Power & Energy Society and not
redistributed in this repository.

| File expected by the code | Public source                                                     |
|---------------------------|-------------------------------------------------------------------|
| `IEEE39.RAW`              | IEEE PES Test Feeder Working Group / Illinois Center for a Smarter Electric Grid (ICSEG) |
| `IEEE300Bus_modified_noHVDC_v2.raw` | RLGC repository on GitHub (`testData/IEEE300/`); a modified IEEE 300-bus case without HVDC links |

**Recommended placement** (consistent with `configs/experiment_config.yaml`):

```
data/
├── topology/
│   ├── IEEE39.RAW
│   └── IEEE300Bus_modified_noHVDC_v2.raw
└── ...
```

All scripts in this repository read these files from
`data/topology/IEEE39.RAW` and `data/topology/IEEE300Bus_modified_noHVDC_v2.raw`
(relative to the project root, declared in `configs/experiment_config.yaml`).
Drop the public-domain RAW files into `data/topology/` and the pipeline
will resolve them automatically.

### Where to obtain

- **IEEE 39-bus**: search for "New England 39-bus IEEE PES test case PSS/E RAW"
  on the IEEE PES Test Feeder Working Group site or on the ICSEG benchmark
  archive. The standard 60 Hz, 10-generator system is sufficient.
- **IEEE 300-bus**: the modified-noHVDC variant used here is published in the
  RLGC test-case archive (Liu et al., open-source GitHub repository under the
  `testData/IEEE300/` subdirectory). MATPOWER's `case300.m` can also be
  converted to PSS/E RAW format if the HVDC line is removed manually.

## Obtaining the Data

The full simulation set used in the paper is available on request. Contact
the corresponding author with institutional affiliation. A minimal sample
subset for demo purposes will be released once the manuscript is accepted.
