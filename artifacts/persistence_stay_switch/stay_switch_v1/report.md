# Persistence as a shared history-dependent stay/switch computation

## Direct answers

1. **Is recent history shared?** The validation-selected finite kernel (5.0) changes held-out log loss relative to the intercept-only hazard by -0.104 (negative is better). Shared-history ceiling fraction=0.894.
2. **Cost of forcing evidence mappings to be shared:** fully shared adds 0.111 log loss versus the task-specific model; the rank-one shared stay/switch rule adds -0.002.
3. **Shared history plus task-specific evidence:** its held-out log-loss gap from the task-specific ceiling is 0.012, retaining 0.894 of ceiling improvement.
4. **Does recurrence matter?** Full GRU minus non-recurrent MLP R²=0.028; full GRU minus best finite window (limited_history_5) R²=0.001.
5. **Sufficient recurrent dimensions:** the smallest tested bottleneck within 95% of the best R² is 64 (best R²=0.929).
6. **Neural convergence with depth:** mean independent-readout cosine changes from 0.055 in early layers to 0.108 in late layers.
7. **Functional intervention convergence:** mean cross-profile correlation=0.031; every primary effect was measured with that task's own independently fitted readout.
8. **Specificity versus controls:** mean persistence-minus-control R² is arbitrary_choice=0.159, generic_value=0.144, terminality=0.079. One-shot controls were not assigned history or recurrence.

## Decision

**Outcome D — fully task-specific algorithms.** The shared models and convergence tests do not approach the task-specific ceiling.

## Causal gate

These analyses reuse existing trajectories and activations and do not establish causal mediation. Head localization, DAS, broad patching, or new steering should wait for reproducible behavioral sharing, a common functional stage, convergent manipulation profiles, and differentiation from generic-decision controls.

## Reuse and leakage safeguards

No Qwen trajectories were generated. Risk sets stop at the first termination event; episode and counterbalanced-pair splits remain intact; normalization and readout fitting use training data, ridge selection uses validation data, and test targets are evaluation-only. Activation memmaps are local caches excluded from Git.
