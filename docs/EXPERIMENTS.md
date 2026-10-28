# Experiment workflows

All commands below run through tdher.py from the source root and store generated artifacts
under TDHER_WORKDIR. Inspect each driver's --help before execution. This document describes
dependencies and run order, not a list of supplied results.

## Base benchmark and TD-HER

1. Obtain the processed benchmark separately, or generate workbooks with licensed PSS/E and
   use data_proc/extract_features.py, build_representations.py and build_adjacency.py.
   Keep the prescribed train/validation/calibration/test roles and train-fitted transforms.
2. Train base families with experiments/exp1_main_comparison.py or train_single_model.py.
   The exact identifiers are LightGBM, KAN, ConvLSTM, PatchTST, Mamba, ST-GCN and FT-Transformer.
   Optuna selection uses validation data. Per-model runs support --n-trials; the convenience
   default is 20. Select each budget explicitly and retain its configuration externally.
3. Run experiments/exp_tdh_router.py after the base fits and selected-feature records exist.
   It creates calibration/test prediction caches from your own models, fits adapters,
   applies per-target admission and compares routing rules. The current TD-HER method uses
   the margin-only admission rule, bounded affine correction and arrival-time projection.
4. Use exp_tdh_router_ieee300.py for the second system, after preparing its own data and fits.

## Additional analyses

| Sources | Upstream inputs and role |
|---|---|
| exp2_window_sensitivity.py, exp_tdh_multi_window.py | Window-specific representations and base models |
| exp4_generalization.py, exp4b_finetune.py | Source experts and shifted calibration/test partitions |
| exp5_interpretability.py, exp6_feature_selection.py, exp_p*.py | Feature, branch and gradient diagnostics |
| exp_r2_mtl_baselines.py, exp_r2_mmoe_baseline.py | Fixed representation and shared-backbone controls |
| resubmission_v3/exp_v3_capacity_matched_mtl.py | Widened PatchTST capacity controls |
| exp_r2_component_ablation.py, exp_r2_adapter_sensitivity.py | Frozen predictions and controlled component/adapter substitutions |
| exp_tdh_expert_sensitivity.py, resubmission_v3/compute_equal_bank_table.py | Common-bank comparisons |
| resubmission_v4/label_budget/ | Nested calibration subsets; margin-only, extra-bootstrap, always/never admission comparisons |
| resubmission_v4/prospective/ | New-event identity/preprocessing in phase1, routing/budget evaluation in phase2 |
| resubmission_v4/exp_v4_published_predictor*.py | MLNadir and FNP-BPNN benchmark implementations |
| resubmission_v4/exp_v4_wide_control_tuning.py | Independent 20-trial widened-model search |
| resubmission_v4/second_backbone/ | ConvLSTM controls under multiple multi-task strategies |
| resubmission_v4/control_bank/ | Banks formed from the control fits, followed by common routing/postprocessing |
| analytical_comparators/ | Physical SFR/FNP implementations and causal input construction |

For additional comparator and control studies, use [protocol binding](PROTOCOLS.md).
No fitted parameter files, study databases, saved predictions or populated evaluation manifests
are included. Historical identifiers in script/file names describe the workflow revision;
they do not imply that the old dual-condition gate is the current TD-HER admission rule.

## Label budgets and new events

The label-budget runner imports the same production admission/routing functions. Its
level1/level2 stages compare with the base outputs from your own preceding run before pilot/full
subsampling. summarize_label_budget.py and plot_label_budget.py consume your completed external
study outputs. Reduced calibration subsets apply to adapter OOF fitting, admission, route
fitting and affine correction together.

For new events, use simulation/ieee39/tdher_f1f2_runner_v7.py on Windows and transfer the
resulting workbooks, event ledger and source hashes into an external workspace. prospective_phase1.py
verifies the handoff and constructs model inputs; prospective_phase2.py requires that completed
phase1 manifest. Neither driver generates PSS/E trajectories on the learning server.

## Resource controls

Training cost depends on hardware, architecture, event count and search budget. Use --help
to set device, worker count, trial count, epochs and smoke/subsample options where supported.
A smoke run is not an experiment result. Use fresh versioned output directories when the input
or source identity changes; the consistency checks intentionally reject stale manifests.
