# Track B behavioral and counterbalancing gate

Overall gate passed: **False**.

This confirmatory gate uses only train and validation episodes; held-out test episodes were not used for behavioral tuning.

## Integrity and label counterbalancing

- Foraging: 768 episodes / 384 counterbalanced pairs; passed **True**.
- Binary control: 768 episodes / 384 counterbalanced pairs; passed **True**.
- Solvability: 768 episodes / 384 counterbalanced pairs; passed **True**.
- Rule terminality control: 768 episodes / 384 counterbalanced pairs; passed **True**.
- Every pair must use the same ecology/stimulus and exact inverse task-specific label mappings.

## Development-set behavior

- Episodes/states: **650 / 1831**.
- Semantic STAY rate: **0.645**.
- Mean decisions per episode: **2.817**.
- Episodes ending by LEAVE: **1.000**.
- Persistence-logit SD: **1.242**.
- STAY-probability P90−P10: **0.594**.
- Initial mapping gap: **0.038**.
- Higher-minus-lower outside-option logit effect: **-1.199**.
- Higher-minus-lower stay-cost logit effect: **-0.699**.

## Solvability development-set behavior

- Episodes/states: **650 / 1756**.
- Semantic TRY-AGAIN rate: **0.645**.
- Mean decisions per episode: **2.702**.
- Persistence-logit SD: **3.152**.

## Preregistered checks

- PASS — enough development episodes
- PASS — enough development states
- PASS — episodes contain repeated decisions
- PASS — persistence logit varies
- PASS — stay probability spans decisions
- PASS — semantic choices nondegenerate
- FAIL — episode termination nondegenerate
- PASS — label mapping initial gap bounded
- PASS — outside option reduces persistence
- PASS — stay cost reduces persistence
- PASS — solvability: enough development episodes
- PASS — solvability: enough development states
- PASS — solvability: episodes contain repeated decisions
- PASS — solvability: persistence logit varies
- PASS — solvability: persistence probability spans decisions
- PASS — solvability: semantic choices nondegenerate
- PASS — solvability: episode termination nondegenerate
- FAIL — solvability: label mapping initial gap bounded
- PASS — solvability: solvability evidence increases persistence
- PASS — solvability: attempt cost reduces persistence
- PASS — solvability: give up value reduces persistence
