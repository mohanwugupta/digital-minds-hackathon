# Computational models of cross-task persistence

This is an exploratory behavioral model comparison. Predictive superiority does not establish neural implementation.

## Behavioral predictability

The observable GRU explained 0.737 held-out persistence-logit R².
Sampled-choice performance is reported beside the oracle-policy ceiling in `model_metrics.csv`.

## Best interpretable model

The strongest observable interpretable candidate by held-out MSE was **mvt_like_foraging_threshold** (R²=0.717, MSE=0.434).
Its raw fraction of observable GRU R² was 0.972.

## Recurrence and value/history comparisons

Compare MLP versus GRU, finite history versus latent commitment, and RW/Bayesian versus history/sticky-termination rows in the accompanying tables. Confidence intervals are episode/pair clustered.

## Identifiability

The recovery matrix records which architectures were distinguishable in synthetic data. Near-equivalent families must not be distinguished more strongly in the real-data interpretation than this gate permits.

## Decision

Behavioral persistence is predictable, but the current results do not yet uniquely justify a computational-mechanistic interpretation.

No activation analysis, steering, or mechanistic claim is performed by this pipeline.
