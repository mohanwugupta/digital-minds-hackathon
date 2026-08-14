# Ridge-linear future-return and persistence probes

## Bottom line

Future return is linearly decodable at layer **1** (validation/test R² **0.153 / 0.240**). The nonlinear benchmark test R² is 0.251 full / 0.179 sparse.
The direct persistence probe peaks at layer **31** (validation/test R² **0.999 / 0.998**).

## Exact matching

Matching retained **1180 states in 147 strata from 55 episodes**.
Ridge return → actual future return: beta **0.486**, p=2e-06.
Ridge return → persistence: beta **0.007**, p=0.951.
Direct persistence probe → persistence: beta **0.994**, p=0.

## Direction overlap

Maximum absolute return/persistence direction cosine was **0.154** at layer **14**. Maximum top-1% dimension Jaccard was **0.156** at layer **1**.

The direct probe is a localization/control analysis, not evidence for value. Continuation advantage remains the substantive missing target.

Layer and ridge alpha use validation episodes; all reported test metrics and exact-match diagnostics use untouched test episodes.
