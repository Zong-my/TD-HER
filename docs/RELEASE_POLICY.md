# Source-only release policy

The public tree contains implementations, design settings, documentation and simulation input cases.
Keep every generated artifact in an external workspace. This includes predictions, features,
processed/raw trajectories, labels, fitted scalers, selected feature indices, fitted parameters,
weights, checkpoints, tables, figures, trial databases, timings and logs.
The separately published Hugging Face benchmark is not embedded in this Git repository.

Allowed case inputs are the listed RAW/DYR records, the original IEEE39 SAV input and the
generator-inertia CSV. A PSS/E network SAV input is not a trained machine-learning model.
Do not add other SAV files, including solved or modified states produced by a new run.

The supported launcher changes to TDHER_WORKDIR, disables bytecode and rejects Python writes
under the source root. This is a workflow safeguard, not a security sandbox for arbitrary
native extensions or subprocesses. Verify outputs before publishing; .gitignore alone does
not prevent already tracked or force-added files from being uploaded.

Run tools/check_release.py and the synthetic tests before staging. Review git diff --cached
and the exact staged file list. No experiment output is required by these release checks.
Scientific calculations and configured study settings are retained; changing portable paths
does not itself establish equality with historical experiment results.
