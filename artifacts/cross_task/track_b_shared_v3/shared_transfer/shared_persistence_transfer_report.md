# Shared persistence representational test

The primary direction was discovered using task-balanced Bandit + Foraging training targets. Solvability supplied no direction, layer, regularization, centering, or scaling parameter for the strict test.

Primary classification: **no convincing shared transfer**.
Held-out Solvability correlation: **0.898** (counterbalanced-pair-clustered 95% CI **0.878 to 0.917**).
Scale-free variance association (correlation²): **0.806**.
Fraction of Solvability-specific ceiling correlation²: **0.806**.
Selected discovery-only layer / ridge penalty: **31 / 0.001**.

## Discovery source-task gate

- Bandit: validation r **0.999**, R² **0.998**, matched-random 95th percentile **0.676**.
- Foraging: validation r **1.000**, R² **1.000**, matched-random 95th percentile **0.864**.

## Primary checks

- PASS — expected direction
- FAIL — label reversal consistency
- PASS — exceeds random 95th percentile
- FAIL — negative control absent or weaker
- FAIL — terminality control absent or weaker
- PASS — at least half ceiling

## Exact matched-history label replay

- try_again_m: r **0.912** (95% CI **0.892, 0.931**).
- try_again_n: r **0.910** (95% CI **0.887, 0.931**).
- Mean paired projection gap: **0.301 pooled SD**.
- Mean paired semantic-target gap: **0.275 pooled SD**.

## Specificity controls

- Arbitrary binary-choice r: **0.749** (95% CI **0.664, 0.817**).
- Rule-determined terminality r: **0.633** (95% CI **0.532, 0.722**).

## Leave-one-task-out robustness

- Held out solvability: r **0.898** (95% CI **0.878, 0.917**), layer **31**, no convincing shared transfer (confirmatory primary).
- Held out foraging: r **0.290** (95% CI **0.209, 0.363**), layer **31**, no convincing shared transfer (secondary leave one task out robustness).
- Held out bandit: r **0.750** (95% CI **0.699, 0.804**), layer **28**, no convincing shared transfer (secondary leave one task out robustness).

The validation-affine results are scale diagnostics only. Clearance uses the strict correlation, exact matched-history label replays, matched random directions, the arbitrary binary-choice control, and the rule-determined terminality control. The old Bandit-only transfer is not part of this gate.
