# Ridge-linear future-return and persistence probes

## Bottom line

Future return is linearly decodable at layer **1** (validation/test R² **0.153 / 0.240**). The nonlinear benchmark test R² is 0.251 full / 0.179 sparse.
The direct persistence probe peaks at layer **31** (validation/test R² **0.999 / 0.998**).

## Direct replication against the nonlinear probes

All three decoders were evaluated on the same **1800 states from 78 untouched test episodes**. Prediction R² was **0.240** for ridge, **0.179** for the sparse nonlinear probe, and **0.251** for the full nonlinear probe.
Ridge and full-nonlinear predictions correlated **r=0.951**. The episode-bootstrap 95% interval for ridge minus full-nonlinear R² was **[-0.033, 0.013]**, which includes zero.

This is a strong replication of the central result: a flexible nonlinear decoder is not required to recover the held-out future-return signal.

## Selection robustness

For future return, validation and test layer ranks correlated **Spearman r=0.505**; the validation-selected layer ranked **2** on test (best test layer 0).
For persistence, validation and test layer ranks correlated **Spearman r=0.994**; the selected final layer also ranked **1** on test.

## Exact matching

Matching retained **1180 states in 147 strata from 55 episodes**.
Ridge return → actual future return: beta **0.486**, p=2e-06.
Ridge return → persistence: beta **0.007**, p=0.951.
Direct persistence probe → persistence: beta **0.994**, p=0.

## Important boundary condition

- Both-negative: 159 states / 20 episodes; return correlations ridge **-0.699**, sparse **-0.710**, full **-0.709**.
- One-positive: 873 states / 36 episodes; return correlations ridge **0.454**, sparse **0.421**, full **0.440**.
- Both-positive: 768 states / 22 episodes; return correlations ridge **0.562**, sparse **0.510**, full **0.565**.

All three probes fail in the both-negative regime and succeed in the one- or both-positive regimes. The replication therefore supports a linearly decodable future-return/trajectory signal, but not a uniformly calibrated signed value representation across every reward environment.

## Probe alone, history alone, and joint persistence models

The ridge return prediction alone explains **0.068** of persistence-logit variance. Recent history alone explains **0.753**; the joint model explains **0.755** (increment **0.0017**).
In the joint model, the standardized ridge-return coefficient is **0.049** (p=0.352); within exact matched states it is **0.007** (p=0.951).

Thus the linearly decoded return signal replicates, but it does not explain STOP beyond recent outcome, loss streak, and round. The near-perfect final-layer persistence probe confirms that the decision itself is linearly encoded and localizes where it becomes explicit; it is not evidence for value.

## Direction overlap

Maximum absolute return/persistence direction cosine was **0.154** at layer **14**. Maximum top-1% dimension Jaccard was **0.156** at layer **1** (chance-tail p=2.69e-09; 32-layer Bonferroni p=8.62e-08).

The sparse top-dimension overlap is above chance, but the corresponding signed whole-vector directions are nearly orthogonal. This suggests some shared dimensions without a common global linear axis.

## Adjudication

The ridge result rules out the narrow explanation that future return was recoverable only by a flexible nonlinear decoder. It does not establish that generic future return controls stopping: the exact-matched and joint persistence tests are null. The next discriminating experiment remains the counterfactual continuation-advantage probe.

Layer and ridge alpha use validation episodes; all reported test metrics and exact-match diagnostics use untouched test episodes.
