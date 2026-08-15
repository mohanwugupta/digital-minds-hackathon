# Continuation-advantage probe

## Bottom line

Continuation advantage is decodable but does not track persistence; the simple heuristic account is strengthened.

Selected layer: **2**. Validation/test advantage R²: **0.197 / 0.314**.
Exact matching retained **124 states in 31 strata from 50 episodes**.

## Critical matched-state coefficients

- Probe → continuation advantage: beta **0.575**, p=3.43e-11.
- Actual rollout advantage → persistence: beta **0.280**, p=0.00949.
- Advantage probe → persistence: beta **0.129**, p=0.297.
- Unique persistence R² beyond recent history: **0.0084**.

## Advantage/decision direction overlap

Maximum absolute cosine was **0.130** at layer **31**; maximum top-1% Jaccard was **0.083** at layer **2**.

## Target precision

Median per-state rollout SE: **9.629**; target variance: **990.046**; mean estimation-error variance: **115.977**.

continuation_advantage=max(Q_A,Q_B)-Q_STOP; Q_STOP=0
The max of finite-rollout Q estimates is upward biased; target CSVs retain per-arm standard errors and raw returns for sensitivity checks.
