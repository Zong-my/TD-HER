# Simulation input cases

These files are network/dynamic inputs, not simulated trajectories or learning results.

| Input | Role |
|---|---|
| benchmark_ieee39/IEEE39.sav | Original saved IEEE39 benchmark network used by the supplied legacy generator and F1/F2 runner |
| benchmark_ieee39/IEEE39bus_rew_10_v33.dyr | Dynamic records paired with that saved case |
| IEEE39_paper_aligned_mixed_v2.raw / .dyr | Paper-aligned mixed-resource 39-bus derivative for controlled case studies |
| IEEE300_paper_aligned_v2.raw / .dyr | Paper-aligned 300-bus network and dynamic input records |

The original IEEE39 benchmark pair and the mixed-resource derivative are distinct cases.
Do not substitute one silently for the other. The mixed-resource RAW includes auxiliary
buses; standard system names describe the underlying IEEE system rather than a claim that
every auxiliary or machine-terminal node has been removed.

The paired RAW/DYR files must be loaded together using a compatible licensed PSS/E runtime.
The 300-bus generator accepts TDHER_IEEE300_SAV, a saved network you prepare from the
appropriate RAW input using PSS/E; this release does not ship a newly solved 300-bus SAV.
Preserve case identity and document any conversion in your external workspace.

SHA256SUMS identifies the supplied input files. Hashes are source integrity metadata,
not experiment results. IEEE test-system provenance and applicable original source terms
remain in force; no PSS/E vendor software, licence or third-party executable is included.
