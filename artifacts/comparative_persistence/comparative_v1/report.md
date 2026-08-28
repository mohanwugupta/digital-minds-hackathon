# Comparative computational models of LLM persistence

Primary target: absorbing discrete-time termination hazard with task-macro evaluation.

## Direct answers

1. **Best interpretable model:** `finite_history` under `hierarchical` sharing (macro log loss 0.5149).
2. **Latent state versus finite history:** best latent=0.5162; best finite-history=0.5149; GRU=0.4923, nonrecurrent MLP=0.4865. A selected rho near zero is reported as collapse, not motivation.
3. **Zero-shot task transfer:** `perseveration/hierarchical` improves macro log loss by 0.0394 over the source-trained null (95% task bootstrap [-0.0471, 0.1628]).
4. **Architecture transfer:** best mean G=0.760 for `flexible_linear/fully_shared`; values are relative to separately fitted target-task ceilings.
5. **Few-shot adaptation:** the best task-macro point uses 22 target pairs for `finite_history/hierarchical` with log loss 0.5249.
6. **Most general ingredient:** `history` has the largest ablation cost (0.0411).
7. **Persistence-specific history:** PSH=0.0363 (95% task bootstrap [-0.0118, 0.0844]); mean kernel cosine is -0.648 for persistence–control versus 0.611 among persistence tasks; positive PSH means more history gain than independent repeated choice.
8. **Human/animal signatures reproduced directionally:** sampling-cost sensitivity, error-penalty sensitivity.
9. **Current hypothesis:** H2 — shared stay/switch computation.
10. **PRD 3 candidate:** finite recent-history integration.

## Frozen task inventory

Included: bandit, foraging, solvability, information_sampling, partial_reinforcement, independent_effort_control.

Excluded before fitting:

- `voluntary_waiting` — failed PRD-1 basic gate.
- `progressive_ratio` — failed PRD-1 basic gate.
- `sunk_cost` — failed PRD-1 basic gate.

## Guardrails

All normalization uses frozen environment specifications. Missing constructs retain explicit availability masks. LOTO/LOFO target records never enter fitting, normalization, hyperparameter selection, or model selection. Sunk cost, PREE, waiting-context, effort-breakpoint, and controllability effects are scientific outputs rather than task-validity criteria.

Task-level generalization uncertainty is necessarily substantial with this small number of task identities.

Synthetic H1–H4 recovery passed the 80% identifiability gate.
