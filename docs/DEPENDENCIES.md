# Dependencies and execution environments

The learning pipeline uses Python 3.12 on Linux. The inspected environment has PyTorch
2.9.1, torch-geometric 2.7.0, LightGBM 4.6.0, efficient-kan 0.1.0, mamba-ssm 2.3.1,
Optuna 4.8.0, NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.2 and scikit-learn 1.8.0.
These are environment versions, not accuracy or runtime results. requirements.txt gives
installation dependencies rather than an exact cross-platform lock file.

Install a PyTorch build compatible with your GPU and CUDA runtime first. Install
requirements-optional.txt for all seven families and optional comparisons. mamba-ssm
may require a matching compiler/CUDA toolchain. Do not replace a system GPU driver merely
to satisfy an example command. CPU-only users can run LightGBM and synthetic routing tests;
some model implementations require their optional package even when another model is chosen.

PSS/E generation is separate. Use the Python interpreter bundled with the installed PSS/E
version on Windows. The legacy generators target the PSS/E 33 API; vendor modules psspy,
dyntools and redirect require a licensed installation. Its executable, licence and runtime
are not shipped. Simulation outputs belong in TDHER_SIM_OUTPUT, outside this checkout.

## Manuscript workstation

The manuscript reports an Intel Xeon w5-3535X CPU at 3.20 GHz, 512 GB RAM,
and an NVIDIA RTX PRO 6000 GPU. These are the documented experiment resources,
not minimum hardware requirements. PSS/E 33.4 generation uses the separate
licensed Windows environment described in [simulation instructions](../simulation/README.md).
This release contains no measured timing or performance output files.
