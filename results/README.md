# Results Directory — Placeholder

Experiment outputs are **not bundled** with this repository
(prediction arrays, model checkpoints, and figures are generated locally
when experiments are run). This directory documents the layout produced
by the experiment scripts in [`experiments/`](../experiments/).

## Expected Layout

```
results/
├── ieee39/
│   ├── exp1/                          # Main expert benchmark (Sec. IV.B)
│   │   ├── LightGBM/{predictions.npy, metrics.json, ...}
│   │   ├── KAN/...
│   │   ├── ConvLSTM/...
│   │   ├── PatchTST/...
│   │   ├── Mamba/...
│   │   ├── ST-GCN/...
│   │   └── FT-Transformer/...
│   ├── exp4/                          # L1 / L2 / L3 generalization (Sec. IV.C)
│   ├── exp_tdh_router/                # Main TD-HER routing decisions
│   │   ├── L1/, L2/, L3/
│   │   └── audit_report.json
│   ├── exp_tdh_multi_window/          # 10/50/100/150/250 ms windows (Fig. 10)
│   ├── tdher_pipeline_timing/         # GPU/CPU latency breakdown (Fig. 11)
│   │   ├── pipeline_timing_gpu.json
│   │   └── pipeline_timing_cpu.json
│   ├── tdher_robustness/              # Quality-aware robustness (Sec. IV.G)
│   └── tdher_expert_sensitivity/      # Leave-one-out / top-m subset (Fig. 7)
├── ieee300/                           # 300-bus scalability (Sec. IV.H)
│   ├── exp7_rebuild/                  # Per-expert baselines
│   └── tdher_router/                  # 300-bus TD-HER routing
├── ieee300_posttrigger/               # Rebuilt 300-bus dataset audit
├── paper_tables/                      # Final CSV tables consumed by the manuscript
│   ├── tdher_main_results.csv
│   ├── tdher_ablation.csv
│   ├── tdher_threshold_sensitivity.csv
│   ├── tdher_final_routing_weights.csv
│   ├── tdher_route_conflict_diagnostics.csv
│   ├── tdher_certified_admission.csv
│   ├── tdher_l3_bootstrap.csv
│   ├── tdher_multiwindow_l1.csv
│   ├── tdher_expert_leave_one_out.csv
│   ├── tdher_expert_subset_size.csv
│   ├── tdher_expert_diversity.csv
│   ├── ieee300_scalability.csv
│   ├── ieee300_tdher_routing.csv
│   └── ieee300_tdher_routing_weights.csv
└── paper_artifacts/
    ├── figures/                       # Vector PDF + 600 DPI PNG + SVG
    └── latex/                         # Auto-generated LaTeX table fragments
```

## Reproducing the Results

Run the experiment scripts in the order indicated in the top-level
[`README.md`](../README.md). All randomized splits and Optuna searches use
seed 42; deterministic mode is enabled where supported. Full re-execution
on a single RTX-class GPU plus 40-core CPU takes approximately 55–65 hours.

## Audit Reports

Every experiment writes an `audit_report.json` documenting:

- Dataset split sizes
- Prediction-label alignment checks
- Metric recomputation against exported CSV tables
- Physical projection compliance (e.g., `t_Δf ≥ 0`)

These reports are summarized in the manuscript supplementary material.
