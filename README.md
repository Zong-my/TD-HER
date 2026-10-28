# TD-HER

**Early Prediction of Power System Frequency Extremum and Its Arrival Time via
Target-Dependent Heterogeneous Expert Routing**

TD-HER combines a frozen heterogeneous expert bank, target-specific convex
routing, and selective adaptation. It predicts the signed frequency extremum
and its arrival time from an early post-disturbance measurement window.

This is a **source-only release**. It contains Python implementations,
simulation input cases, experiment configuration and documentation.
It contains **no trained weights, prediction arrays, experiment results,
result tables, simulation trajectories, cached features or training logs**.
The existing processed benchmark is distributed separately on
[Hugging Face](https://huggingface.co/datasets/babycow/TD-HER-Dataset).
Do not download data or save generated artifacts inside this checkout.

## Method

1. Train the base experts on the source training split and select their settings
   using the designated validation split. Freeze them before calibration.
2. For each evaluation scenario, obtain base-expert predictions on the
   held-out calibration events. A LightGBM adapter is trained with source
   samples and the available new calibration labels. Its calibration
   predictions are out of fold; its final fit uses all available calibration
   events.
3. Decide admission separately for each target. The current rule admits an
   adapter when its out-of-fold calibration MAE is at most
   `(1 - tau) * best_base_calibration_MAE`, with `tau = 0.20`.
4. Fit nonnegative, unit-sum weights separately for the two targets on the
   admitted prediction columns. Apply the bounded affine correction and
   project the arrival-time prediction onto the nonnegative domain.

The additional paired-bootstrap admission condition is **not part of the
current TD-HER rule**. Functions named `certified_*` remain only as comparison
implementations for the historical two-condition variant. Bootstrap analysis
of prediction differences is distinct from the deployed admission decision.

The evaluated base bank contains LightGBM, KAN, ConvLSTM, PatchTST, Mamba,
ST-GCN and FT-Transformer. `models/` contains their architecture source code,
not trained model files. Additional model modules support explicitly named
comparison experiments; they do not enlarge the seven-expert bank.

## Layout

| Path | Contents |
|---|---|
| `models/` | Expert architectures, static-feature branches and multi-task control heads |
| `data_proc/` | Feature extraction, tabular/tensor/graph representations and adjacency construction |
| `experiments/exp1_main_comparison.py` | Base-expert training and validation-driven selection |
| `experiments/train_single_model.py` | Per-model training and hyperparameter search |
| `experiments/exp_tdh_router.py` | Target-wise admission, convex routing, affine correction and comparison rules |
| `experiments/exp_tdh_router_ieee300.py` | Routing for the 300-bus benchmark |
| `experiments/exp_r2_*.py` | Multi-task controls, attribution and diagnostic analyses |
| `experiments/resubmission_v3/` | Common-bank and capacity-control implementations |
| `experiments/resubmission_v4/` | Label budgets, new-event evaluation, published-predictor and second-backbone controls |
| `analytical_comparators/` | System-frequency-response and analytical nadir-prediction implementations |
| `simulation/` | PSS/E data-generation scripts and simulation input parameters |
| `data/psse_cases/` | Network and dynamic input records; see its provenance note |
| `configs/` | Design settings and configuration interfaces |
| `tests/` | Synthetic, data-free checks of routing and release boundaries |
| `tdher.py`, `tdher_paths.py` | External-workspace launcher and path configuration |

See [experiment workflows](docs/EXPERIMENTS.md),
[data interfaces](data/README.md),
[simulation instructions](simulation/README.md), and
[release policy](docs/RELEASE_POLICY.md).

## Installation

Use a separate Python environment. The machine-learning code uses Python 3.12.
PSS/E generation uses its own Windows/Python environment and is independent
of the machine-learning installation.

```bash
git clone https://github.com/Zong-my/TD-HER.git
cd TD-HER
python3.12 -m venv ../TD-HER-venv
source ../TD-HER-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your hardware before the optional
GPU extensions. Mamba requires its compatible `mamba-ssm` implementation;
ST-GCN requires `torch-geometric`; KAN uses `efficient-kan`.
Install the applicable packages from requirements-optional.txt.
See [dependencies](docs/DEPENDENCIES.md) for the tested environment and
optional packages. PSS/E, its licence and vendor binaries are not included.

## External workspace

```bash
export TDHER_WORKDIR="$HOME/TD-HER-work"
export TDHER_DATA_ROOT="$TDHER_WORKDIR/data"
python tdher.py paths
```

The default workspace is the sibling directory `../TD-HER-work`.
Both variables must point outside the source checkout. The supported
`tdher.py run` launcher also rejects writes into the checkout.

The expected processed-data layout is:

```text
TD-HER-work/
  data/
    ieee39/{repA,repB,repC,csv,adjacency}/
    ieee300/{repA,repB,repC,adjacency}/
    raw/                     # optional raw simulation workbooks
  results/                   # created by your own runs
  analytical/                # your own analytical preparation
  protocols/                 # your own study manifests, when needed
```

The repository names `ieee39_v8_80_10_10` and
`ieee300_v2_posttrigger` in legacy interfaces resolve to
`ieee39` and `ieee300` through `tdher_paths.py`.

A dataset download, if desired, must target the external data directory:

```python
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="babycow/TD-HER-Dataset",
    repo_type="dataset",
    local_dir=os.environ["TDHER_DATA_ROOT"],
)
```

Check the resulting directory structure against the layout above. The
separate dataset release is an input resource, not an output bundle from
the additional studies.

## Running the core pipeline

Inspect the CLI before selecting the compute budget:

```bash
python tdher.py run experiments/train_single_model.py -- --help
python tdher.py run experiments/exp1_main_comparison.py -- --help
python tdher.py list
```

Train the seven base families through the comparison driver, or train one
family at a time. For example:

```bash
python tdher.py run experiments/train_single_model.py -- --model PatchTST --n-trials 20
python tdher.py run experiments/exp1_main_comparison.py --   --models LightGBM KAN ConvLSTM PatchTST Mamba ST-GCN FT-Transformer
```

The exact accepted model identifiers are listed by `--help`.
After all required expert fits and feature-selection records exist in your
external workspace:

```bash
python tdher.py run experiments/exp_tdh_router.py
```

The current TD-HER output is the margin-rule, affine-corrected,
nonnegative-arrival-time variant. Historical comparison variants are named
separately in the generated output. No test labels are used to choose
admission or route weights.

The extension drivers require their declared upstream inputs: for example,
base-expert fits for control comparisons, and independently generated
analytical/cohort inputs for native-nadir comparisons. They are not supplied
as precomputed result files. [EXPERIMENTS.md](docs/EXPERIMENTS.md) explains the
dependency order and configuration preparation.

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python tools/check_release.py
```

These checks use synthetic fixtures or source inspection. They do not
retrain experts, re-evaluate the benchmark or assert published MAE values.
Full simulation requires a licensed PSS/E installation; full learning
experiments require the external input data and suitable compute resources.

## Citation and licence

Please cite the TD-HER article by its final bibliographic record when
available. This source release does not assign a DOI or publication status
to the manuscript.

The code is distributed under the [MIT licence](LICENSE). Third-party
libraries retain their own licences. Simulation cases are research
derivatives of IEEE test systems; original case provenance and applicable
source terms remain in force. The PSS/E software itself is not redistributed.
