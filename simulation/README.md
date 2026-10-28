# Simulation sources and inputs

## Environments

Run these scripts with the Windows Python/PSS/E installation for which the vendor API
is available. They are separate from the Linux learning environment. Set TDHER_SIM_OUTPUT
to a new external directory before running either legacy generator. Never target this
source checkout. Configure vendor Python paths for your licensed installation.

## 39-bus benchmark

ieee39/generate_freq_datav7_3.py contains operating-condition and disturbance generation.
Its default SAV/DYR inputs are the benchmark_ieee39 pair in data/psse_cases.
The supplied ieee39/tdher_f1f2_runner_v7.py drives the two additional event families.
Inspect its --help for plan and simulation modes; every run needs a fresh external
output directory. Its input hashes bind the supplied case and generator source.

The mixed-resource RAW/DYR pair is a separate controlled-study input. It does not replace
the original benchmark pair in the legacy generator.

## 300-bus benchmark

generate_freq_data300_v3.py uses the supplied generator-inertia input table
ieee300_gen_Hs.csv and the paired 300-bus DYR. Set TDHER_IEEE300_SAV to your compatible
saved network prepared using licensed PSS/E. Set TDHER_SIM_OUTPUT outside the checkout.

## Learning handoff

Generated workbooks and logs stay in the external simulation directory. Feature extraction,
representation construction and adjacency construction are in data_proc/. Preserve
training-only preprocessing and the declared split roles. The prospective phase1/phase2
drivers verify event identities, case bindings and upstream manifests before evaluation.
No simulation has been run by the release validation; no generated trajectory, fitted
network state or evaluation result is supplied.
