# Behavioral computational-model comparison

Data: 200 episodes and 3706 decision states.

Primary outcome: sampled STOP versus CONTINUE. Evaluation uses five held-out episode folds and equal total weight per episode.

| Model | Log loss (lower better) | AUC | Brier |
|---|---:|---:|---:|
| bayesian_hybrid | 0.4527 | 0.802 | 0.1429 |
| heuristic | 0.4553 | 0.805 | 0.1445 |
| rw_hybrid | 0.4612 | 0.787 | 0.1457 |
| bayesian | 0.4625 | 0.746 | 0.1453 |
| rw | 0.4711 | 0.737 | 0.1492 |
| time | 0.4990 | 0.753 | 0.1600 |

Selected RW learning rates by fold: [0.05, 0.25, 0.5499999999999999, 0.49999999999999994, 0.95].

Episode-bootstrap comparisons use the mean loss within each episode. Positive differences favor the candidate over the heuristic.

| Candidate | Improvement over heuristic | 95% CI | Two-sided p |
|---|---:|---:|---:|
| bayesian_hybrid | +0.0026 | [-0.0032, +0.0083] | 0.3788 |
| rw_hybrid | -0.0059 | [-0.0108, -0.0011] | 0.0142 |
| bayesian | -0.0071 | [-0.0188, +0.0047] | 0.2444 |
| rw | -0.0157 | [-0.0278, -0.0031] | 0.0160 |
| time | -0.0436 | [-0.0606, -0.0252] | 0.0000 |
## Interpretation

RW alone changes observation-weighted held-out log loss by -0.0157 relative to the recent-history heuristic; RW plus the heuristic changes it by -0.0059. Positive values favor the value model.

This comparison tests predictive sufficiency, not whether the network literally implements Rescorla--Wagner learning. A learning model should only be treated as behaviorally explanatory when it improves held-out prediction beyond the simpler loss-streak/time model.
