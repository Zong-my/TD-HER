# Paper-Aligned PSS/E Case Records

This directory provides the four versioned PSS/E text records corresponding to
the IEEE 39-bus and IEEE 300-bus system configurations documented in the
revised TD-HER manuscript.

## Files

| File | Role |
|---|---|
| `IEEE39_paper_aligned_mixed_v2.raw` | IEEE 39-bus network and steady-state equipment record |
| `IEEE39_paper_aligned_mixed_v2.dyr` | IEEE 39-bus dynamic-model record |
| `IEEE300_paper_aligned_v2.raw` | IEEE 300-bus network and steady-state equipment record |
| `IEEE300_paper_aligned_v2.dyr` | IEEE 300-bus dynamic-model record |

## Benchmark Names and Explicit PSS/E Buses

The system names follow the standard transmission-network benchmarks, while
the PSS/E records retain additional equipment-attachment buses needed by the
simulation model. In the 39-bus case, buses 504, 507, 508, and 518 are
auxiliary load-model buses connected to benchmark buses 4, 7, 8, and 18,
respectively, through dedicated low-impedance coupling transformers. They are
equipment-attachment points that leave the standard 39-bus topology unchanged.
The RAW record therefore contains 43 explicit bus records, whereas the manuscript
schematic shows the standard benchmark buses 1--39.

In the 300-bus case, PSS/E explicitly represents 69 low-voltage
generating-unit terminal buses 10000--10068 behind their step-up transformers.
These are machine-side connection and frequency-monitoring points, not
additional transmission buses in the standard 300-bus benchmark. The RAW record
therefore contains 369 explicit bus records. The manuscript figure is a
representative section of the standard transmission topology and omits these
terminal buses.

The 39-bus pair contains the mixed renewable configuration aligned with the
revised system description. It retains the historical 632-MW WT4G1/WT4E1
Type-4 wind plant at bus 33, adds full PVGU1/PVEU1/PANELU1/IRRADU1 chains at
buses 5 and 14, and adds WT4G2/WT4E2 Type-4 wind aggregates at buses 9 and 26.
Each added renewable point is rated 166.667 MVA, injects 150 MW at unity power
factor, and serves 75 MW of co-located demand. The PV active-current limit is
1.2 p.u., and the irradiance input is fixed at 866.667 W/m2 during the matched
15-s sensitivity test. No added converter uses synthetic or virtual inertia.
The IEEE 300-bus pair preserves the 69-synchronous-generator scalability
configuration and contains no wind or PV converter units.

## Validation

The records were exported and validated with PSS/E 33.4 under its supported
Python 2.7 runtime. The validation sequence comprised:

1. RAW case loading and solved power flow;
2. DYR loading, ordering, factorization, and dynamic initialization, including
   the PV and Type-4 wind model chains in the 39-bus record;
3. independent reloading of the exported DYR record;
4. execution of the specified disturbance; and
5. completion of the 15-s dynamic simulation.

File identities are fixed by `SHA256SUMS`.

## Reproducibility Scope

The records support direct inspection and PSS/E reconstruction of the network,
equipment locations, steady-state operating point, and dynamic-model
configuration reported in the revised manuscript. They complement two other
public reproducibility layers:

- the 18-GB processed TD-HER dataset on Hugging Face, which is the source for
  reproducing the model-training and held-out evaluation results; and
- the experiment and data-processing code in this repository.

The three renewable-composition trajectories reported in the revision are an
additional controlled sensitivity study derived from the aligned 39-bus setup;
they do not replace the historical benchmark used for TD-HER training and
evaluation. The four files in this directory provide the mixed reference
configuration. The matched PV-only and wind-only records, trajectories, and
validation logs are retained as separately versioned analysis evidence.

## Provenance

The files are versioned, paper-aligned derivatives of standard IEEE test-system
cases. They are released for research reproducibility with explicit file
hashes. Original test-system provenance and any terms attached to the source
cases remain applicable.
