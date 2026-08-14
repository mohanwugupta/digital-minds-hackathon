# Exploratory supervised future-return probe

Selected layer: **2**; sparse dimensions: **26**

## Future-return prediction

- Sparse validation/test R²: **0.152 / 0.179**
- Full validation/test R²: **0.150 / 0.251**
- Recent-history test R²: **0.020**
- Constant test R²: **-0.065**

## Persistence mechanism

Sparse adjusted beta: **0.016**; delta R²: **0.000**; p: **0.74**.
Exact matching retained **1180 states / 55 episodes**.

## Interpretation constraints

Monte Carlo return under the observed policy and outcome schedule, not a counterfactual continuation advantage.
The analysis is exploratory because probe redesign followed inspection of the TD test result.
A continuation-advantage target remains a separate, later analysis.
