# Computational models of cross-task persistence

This is an exploratory behavioral model comparison. Predictive superiority does not establish neural implementation.

## Corrected model rankings

The best cross-task model was **gru** (shared_architecture_task_observation; macro R²=0.897). Only candidates evaluated on Bandit, Foraging, and Solvability were eligible.
The best cross-task interpretable model was **finite_history** (macro R²=0.704, MSE=1.108).

Task-specific winners:

- **Bandit**: gru (R²=0.848, MSE=0.353)
- **Foraging**: gru (R²=0.945, MSE=0.085)
- **Solvability**: gru (R²=0.898, MSE=0.692)

## Flexible behavioral ceiling

The synthetic linear-recovery gate passed=True. The best flexible predictor was **gru** (R²=0.897, MSE=0.376, r=0.948, sampled-choice log loss=0.512). GRU is therefore not assumed to be the ceiling.

## Feature-family ablations

The largest macro leave-one-family-out contributions were **history** (delta R²=0.150), **time_effort** (delta R²=0.063), **cost** (delta R²=0.047).
Group-only performance is reported in `feature_group_only.csv`, separating standalone predictiveness from conditional necessity.
Using taskwise 95% pair/episode-clustered intervals, positive contributions were supported in all three tasks for: history.
Task-limited contributions were: cost (foraging, solvability); progress_solvability (solvability); time_effort (bandit). No taskwise positive effect was supported for: continuation_value, derived_termination, outside_option.

## Observable versus oracle state

Replacing observable value variables with oracle-state counterparts changed macro R² by -0.024. This is not a meaningful positive gain at the prespecified 0.02 descriptive threshold.

## Decision and mechanistic hypothesis

The behavioral follow-up gates passed=True. The strongest current computational ingredients are **history + time_effort + cost**, which should be used as targets for any later L21/L22 analysis rather than assuming a named recurrent model is the mechanism.

No L21/L22 activation analysis, steering, or neural-mechanistic claim is performed by this pipeline.
