# Bandit-to-foraging representational transfer

## Level 1 — behavioral generalization

Foraging produced **2195 states** across **768 episodes**; the semantic STAY choice rate was **0.650** and persistence-logit SD was **1.247**.

## Level 2 — representational generalization

Classification: **no convincing transfer**.
Decision-matrix status: **outcome d generic binary or output geometry**.

Strict zero-shot: R² **-20.108**, correlation **0.131**.
Bandit within-task reference: R² **0.998** at layer 31.
Foraging-specific ceiling: R² **0.999** at layer 28.
Transfer ratio: undefined because one R² is non-positive.
Non-persistence control correlation: **-0.279**.
Strict zero-shot episode-bootstrap R² interval: **-22.711 to -17.995**.

## Preregistered checks

- PASS — expected direction
- FAIL — label reversal consistency
- FAIL — exceeds random 95th percentile
- FAIL — negative control absent or weaker
- FAIL — at least half ceiling

The strict result applies the original bandit standardization and frozen weights with no fitted foraging parameter. The affine fit is diagnostic only.
