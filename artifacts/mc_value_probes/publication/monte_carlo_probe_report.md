# Monte Carlo value probe: mechanism adjudication

## Bottom line

The supervised probe establishes a decodable, history-integrating future-return signal, but that signal does **not** explain the model's STOP policy after recent state is controlled. The behavioral evidence therefore favors a recent-state stopping heuristic. A distinct continuation-advantage representation remains plausible but untested.

Selected layer: **2**; sparse dimensions: **26**.

## Probe validity: does it predict future return?

- Sparse validation/test R²: **0.152 / 0.179**
- Full validation/test R²: **0.150 / 0.251**
- Recent-history test R²: **0.020**
- Constant test R²: **-0.065**
- Selected full/sparse checkpoints occurred at epochs **9 / 4**, rather than every layer collapsing to epoch one as in the TD run.

In descriptive regressions refit within the test set (a conservative advantage for the behavioral baseline), recent history explained **9.5%** of reconstructed future return. Sparse/full probes alone explained **24.7% / 27.3%**; joint history-plus-probe models explained **27.8% / 29.7%**.

## Exact matched-state test

Matching round, previous outcome, and loss streak retained **1180 states in 147 strata from 55 episodes**.
Within those strata, sparse probe value predicted future return (beta=**0.456**, SE=0.103, p=1.04e-05); the full probe did too (beta=**0.479**, SE=0.102, p=2.82e-06).
The mechanism CSV omits the final reward for 11 episodes censored at the 100-decision cap. Exhaustively assigning every combination of -2 and +3 left the matched beta positive: sparse **[0.431, 0.477]** and full **[0.455, 0.499]**.

## Does that representation explain persistence?

Recent-state controls explained **75.3%** of persistence-logit variance. Sparse/full probe-only R² values were **0.054 / 0.029**, and their unique increments beyond recent history were only **0.00015 / 0.00000**.
Within exact strata, sparse probe value did not predict persistence (beta=-0.070, p=0.48); neither did the full probe (beta=-0.063, p=0.564).
Adding cumulative score as a linear covariate produced a positive sparse coefficient (beta=0.213, p=1.14e-07), but this sign-reversing suppression result did not survive the more direct exact-matching test and should not be treated as robust mechanism evidence.

## Adjudicating the three possibilities

1. **Simple recent-loss/time stopping heuristic — supported for the behavioral policy.** Recent state predicts persistence extremely well, and the Monte Carlo probe adds essentially nothing after adjustment or exact matching.
2. **Integrated value exists but TD training failed — supported at the representational level.** Stable supervised probes predict held-out future return and retain that prediction among states with identical recent experience. This rescues the existence of a decodable integrated return signal, but not the claim that it drives STOP.
3. **Continuation advantage rather than generic value drives stopping — plausible, not established.** The dissociation between reliable return decoding and absent persistence prediction is exactly what motivates an advantage target. It is also compatible with the simpler account that the model encodes value epiphenomenally and stops via a heuristic.

## Sprint decision

Do not promote this Monte Carlo direction as a validated causal persistence direction. If time permits, the next discriminating analysis is a continuation-advantage probe with forced A/B counterfactual returns. Steering the current direction can still be reported as exploratory, but a null effect would be predicted by the observational mechanism results.

## Interpretation constraints

Monte Carlo return under the observed policy and outcome schedule, not a counterfactual continuation advantage.
The analysis is exploratory because probe redesign followed inspection of the TD test result.
The fitted probe is a two-layer ReLU network, so these results establish out-of-episode decodability rather than proving that the model exposes a single native linear value variable. A ridge-linear replication would strengthen the representational claim.
The reconstructed-return matched analysis uses exact outcomes for 67 STOP-terminated episodes and an explicitly enumerated terminal-reward sensitivity analysis for the 11 capped episodes.
