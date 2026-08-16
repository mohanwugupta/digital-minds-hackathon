# STOP-payoff × CONTINUE-bonus value dissociation

Audit: **1187 complete states / 48 episodes**, 0 duplicate cells and 0 history failures.

## Behavioral causal effects

STOP payoff: standardized beta **-0.707**, raw slope **-0.1039 logit/point**, p=0.
CONTINUE bonus: standardized beta **0.533**, raw slope **0.1072 logit/point**, p=0.
Relative incentive alone explains **0.784** of within-state persistence variation; STOP-only and CONTINUE-only R² are **0.500** and **0.284**.

## Representational response

- Generic-return direction: STOP beta **0.453**; CONTINUE beta **0.133**; relative-only R² **0.083**.
- Provisional advantage direction: STOP beta **-0.236**; CONTINUE beta **0.266**; relative-only R² **0.121**.
- Direct persistence direction: STOP beta **-0.684**; CONTINUE beta **0.534**; relative-only R² **0.751**.

## Manipulation specificity

Because the continuation bonus applies equally to A and B, their relative logit should be stable. Arm-gap STOP/CONTINUE betas are **-0.080 / 0.376**.

continue_bonus - stop_payoff is exactly collinear with the two factor terms. Models use either STOP+CONTINUE or relative+common parameterizations, never all three in one design matrix.
