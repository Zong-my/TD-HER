# Data inputs

Obtain the existing processed benchmark separately from
https://huggingface.co/datasets/babycow/TD-HER-Dataset .
Download to TDHER_DATA_ROOT, outside the source tree.

Expected folders are ieee39 and ieee300. Representation A uses repA/ms10;
Representation B uses repB/ms10; Representation C uses repC/ms10; adjacency
stores electrical-distance matrices. The csv folder supplies event identities and metadata.
The ms10 folder names are sampling-step identifiers; they do not denote a 10 ms observation
window. The main prediction window is 100 ms.

Array interfaces are X_<split>.npy and y_<split>.npy for tabular/graph data and
X_temporal_<split>.npy, X_static_<split>.npy, y_<split>.npy for tensor data.
Preserve the feature-name records, row order, target order [signed extremum in Hz,
arrival time in s], train-fitted transforms and split membership supplied with the data.
Consult data_proc/datasets.py and each driver's --help for the exact interface.

Historical split names include train, val, test, cross_cond_finetune, cross_cond_test,
cross_cond_topo_finetune and cross_cond_topo_test. Keep calibration and test roles distinct.
Do not fit preprocessing or select settings on test events.

Optional raw simulation workbooks belong under raw/ieee39 or raw/ieee300 in TDHER_DATA_ROOT.
They are needed for preprocessing and native-nadir input construction, but are not supplied
by this source release. No result file or trained model belongs in this directory.
See psse_cases/README.md for the simulation inputs that are included.
