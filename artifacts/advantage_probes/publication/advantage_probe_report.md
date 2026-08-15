# Continuation-advantage probe

## Bottom line

The data support **integrated return information plus a strong recent-state persistence heuristic**, but they do **not** isolate a distinct continuation-advantage decision variable.

Target audit passed: **384 rows, 384 unique states, 0 duplicates**, with split counts {'train': 128, 'validation': 128, 'test': 128} and [10] rollouts per action.

Selected layer: **2**. Validation/test advantage R²: **0.197 / 0.314**.
The episode-bootstrap 95% interval for test R² is **[0.068, 0.472]**. Validation/test layer ranks correlate **0.773**; the selected layer ranks **4** on test.
Exact matching retained **124 states in 31 strata from 50 episodes**.

## What the three candidate accounts predict

1. **Recent-state STOP heuristic.** Round, previous outcome, and loss streak should dominate persistence; a value probe should add little after those variables are fixed.
2. **Integrated generic value.** Earlier hidden states should predict future return, but that signal need not be the quantity used to choose CONTINUE versus STOP.
3. **Continuation-advantage decision variable.** A probe trained on `max(Q_A,Q_B)-Q_STOP` should predict both its rollout target and persistence within matched states, with meaningful alignment to the direct persistence direction.

## Result 1: continuation return is linearly decodable

The ridge probe predicts held-out rollout advantage at layer 2 (test R² **0.314**, r **0.604**). Within exact recent-state matches, probe → advantage is beta **0.575** (p=3.43e-11).
This is evidence that the hidden state integrates information beyond the immediately preceding reward.

## Result 2: rollout advantage relates to behavior, but the probe does not

Within exact matches, actual rollout advantage predicts persistence (beta **0.280**, p=0.00949), whereas decoded advantage does not (beta **0.129**, p=0.297).
Recent history explains **0.687** of persistence variance. Adding the advantage probe raises this to **0.696** (increment **0.0084**; adjusted probe beta **0.113**, p=0.155).
Adding actual advantage instead raises R² to **0.714** (adjusted beta **0.176**, p=0.00238).
When actual and decoded advantage enter together, actual advantage remains associated with persistence (beta **0.173**, p=0.00609) while the probe coefficient is **0.007** (p=0.932).
This pattern is compatible with a noisy or incomplete advantage decoder, but it is not evidence that the fitted probe direction is the model's operative persistence variable.

## Result 3: the target is not sharply distinct from generic value

Forced-arm Q values are highly correlated (r **0.926**), and continuation advantage correlates **0.990** with their mean. The between-state standard deviation of mean Q is **30.1** reward units, versus only **4.5** for the max-over-arms premium.
The advantage and generic-return probe outputs correlate **0.913**. Within exact matches, the earlier generic-return probe predicts the advantage target (beta **0.666**, p=1.84e-15).
With both probes entered together for advantage, generic return remains strong (beta **0.746**, p=6.16e-06) while the nominal advantage probe is beta **-0.090** (p=0.574).
The generic probe was trained with many more labeled states, so this comparison is not a fair decoder competition. It does show that the present rollout target and probe do not identify a cleanly distinct computational axis.

## Result 4: the model does not implement a zero-value stopping threshold

All **24** held-out states with negative rollout advantage still had P(continue) above .5; their mean P(continue) was **0.944**, versus **0.959** for nonnegative states.
In both-negative environments the rollout target averages **-8.5**, but the probe predicts **15.0** and the model's mean P(continue) is **0.941**.
Thus persistence changes in the sensible direction with environment quality, but STOP is not governed by a rational `A_continue > 0` threshold. The strong continuation prior and recent loss/time cues dominate.

## Matched-state coefficients

- Advantage probe → continuation advantage: beta **0.575**, p=3.43e-11.
- Actual rollout advantage → persistence: beta **0.280**, p=0.00949.
- Advantage probe → persistence: beta **0.129**, p=0.297.
- Generic-return probe → persistence in this 128-state subset: beta **0.261**, p=0.0407. This subset result conflicts with the null effect in the prior 1,800-state analysis and should not supersede it.

## Advantage/decision direction overlap

Maximum absolute cosine was **0.130** at layer **31**; maximum top-1% Jaccard was **0.083** at layer **2**.

The weak geometric overlap further argues against treating the fitted advantage direction as the late persistence axis.

## Target precision and limitations

Median per-state rollout SE is **9.629**; target variance is **990.046** and mean estimation-error variance is **115.977**.
The test split contains only **12** both-negative states, compared with **63** one-positive and **53** both-positive states. The validation and test target means also differ (17.2 versus 34.4), reflecting reward-condition imbalance.
The advantage probe has only 128 labeled training states, whereas the generic-return probe used the much larger activation bank. Null differences between their coefficients are therefore not equivalence tests.
The rollout target follows the model's own downstream policy after a forced first action. It measures policy-contingent continuation return, not an environment-optimal Q function, and can inherit downstream stopping heuristics.

## Adjudication

- **Best-supported:** recent outcome, loss streak, and time implement the dominant STOP heuristic on top of a strong default-to-continue bias.
- **Also supported:** early hidden states contain integrated information about future/continuation return.
- **Not established:** a distinct continuation-advantage representation is read out to cause persistence. The present target is almost collinear with generic continuation value, the fitted probe adds little beyond recent history, and it fails the negative-value regime.

The decisive next step is causal steering with the frozen advantage direction and matched generic-return/random directions. A selective monotonic change in persistence would upgrade the advantage account; no effect would favor the heuristic interpretation.

continuation_advantage=max(Q_A,Q_B)-Q_STOP; Q_STOP=0
The max of finite-rollout Q estimates is upward biased; target CSVs retain per-arm standard errors and raw returns for sensitivity checks.
