# Computational ingredients in the frozen L21/L22 persistence subspace

This is an exploratory representational analysis. The rank-4 `displacement-L21-k4` basis was frozen from the prior contrast search; no computational label was used to refit or rotate it.

## Computational-variable representation

- **history_finite_prediction**: best at l22, held-out R²=0.371.
- **history_summary**: best at l22, held-out R²=0.295.
- **progress_solvability**: best at displacement, held-out R²=0.215.
- **time_effort**: best at l22, held-out R²=0.150.
- **cost_pressure**: best at l21, held-out R²=0.026.

## Concentration and dimensionality

Targets retaining at least half of full-hidden R² in rank 4: none.
Persistence rank-4 decoding exceeded the matched-random 95th percentile for: none. Empirical p-values are in `random_subspace_controls.csv`.
The fitted four-target mapping ranks were l21=4, displacement=4, l22=4; canonical correlations and principal angles support subspace-level, not raw-PC, interpretation.

## Cross-task generalization and layer transition

Mean leave-one-task-out R² by target: progress_solvability=-0.019, history_summary=-0.381, cost_pressure=-0.771, history_finite_prediction=-0.806, time_effort=-3.199.
From L21 to L22, held-out R² increased for progress_solvability (+0.120), history_finite_prediction (+0.105), history_summary (+0.083), time_effort (+0.037); it decreased for cost_pressure (-0.079). The displacement itself decoded progress most strongly (R²=0.215).

## Persistence prediction beyond behavioral ingredients

For the displacement stage, adding rank-4 neural state to history/time/cost/progress changed held-out R² by +0.188; adding behavior to the neural-only model changed R² by +0.132.
These are incremental predictive comparisons, not causal mediation estimates.

## Specificity

Best nuisance-control R² values were arbitrary_choice/l21=0.938, generic_value/l21=0.751, terminality/displacement=0.890. These strong choice, terminality, and generic-value results do not support specificity to persistence computations.
Task identity peaked at accuracy=0.939 (chance=0.333); task-general claims must be read alongside leave-one-task-out transfer.

## Decision

Within-task decoding does not transfer reliably, favoring task-specific computational implementations (Outcome C). Neural state also adds substantial residual persistence information beyond the explicit behavioral set (Outcome D).

No causal mediation claim, subspace refit, activation recollection, or individual-PC naming is performed here.
