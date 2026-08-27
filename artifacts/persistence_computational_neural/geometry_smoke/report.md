# Computational ingredients in the frozen L21/L22 persistence subspace

This is an exploratory representational analysis. The rank-4 `displacement-L21-k4` basis was frozen from the prior contrast search; no computational label was used to refit or rotate it.

## Computational-variable representation

- **history_summary**: best at l22, held-out R²=0.177.
- **progress_solvability**: best at displacement, held-out R²=0.041.
- **history_finite_prediction**: best at l21, held-out R²=0.035.
- **time_effort**: best at l22, held-out R²=-0.067.
- **cost_pressure**: best at l22, held-out R²=-0.275.

## Concentration and dimensionality

Targets retaining at least half of full-hidden R² in rank 4: none.
Matched-random rank-4 percentiles and empirical exceedance counts are in `random_subspace_controls.csv`.
The fitted four-target mapping ranks were l21=4, displacement=4, l22=4; canonical correlations and principal angles support subspace-level, not raw-PC, interpretation.

## Cross-task generalization and layer transition

Mean leave-one-task-out R² by target: history_summary=-0.114, progress_solvability=-0.524, history_finite_prediction=-1.282, cost_pressure=-2.491, time_effort=-7.158.
The stage comparison distinguishes information already present at L21, introduced in the L21→L22 displacement, and expressed at L22; see `computational_decoding_by_layer_stage.png`.

## Persistence prediction beyond behavioral ingredients

For the displacement stage, adding rank-4 neural state to history/time/cost/progress changed held-out R² by +0.325; adding behavior to the neural-only model changed R² by -0.320.
These are incremental predictive comparisons, not causal mediation estimates.

## Specificity

The strongest nuisance continuous decoding result was arbitrary_choice/l21 (R²=0.938).
Task identity peaked at accuracy=0.982 (chance=0.333); task-general claims must be read alongside leave-one-task-out transfer.

## Decision

Within-task decoding does not transfer reliably, favoring task-specific computational implementations (Outcome C). Neural state also adds substantial residual persistence information beyond the explicit behavioral set (Outcome D).

No causal mediation claim, subspace refit, activation recollection, or individual-PC naming is performed here.
