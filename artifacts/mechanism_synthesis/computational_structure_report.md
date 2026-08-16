# Computational-structure synthesis

## Bottom line

The model contains decodable generic return information, but the identified generic-return and provisional-advantage directions do not causally move persistence. External behavior is organized primarily by the relative attractiveness of CONTINUE versus STOP, while the late persistence direction closely tracks and causally controls the final decision. A fixed recent-loss/time heuristic is therefore incomplete, although it remains a strong account of stopping in the original unmanipulated task.

## Integrity and behavioral intervention

The factorial audit retained **1187 states from 48 episodes** with no missing, duplicated, or history-mismatched cells.
STOP payoff beta: **-0.707**; CONTINUE bonus beta: **0.533**. Relative incentive explains **0.784** of within-state persistence variation.
The most pro-CONTINUE versus most pro-STOP manipulation changes persistence by **4.329 logits** (episode-bootstrap 95% CI 4.244 to 4.411). **99.9%** of histories cross the CONTINUE/STOP boundary somewhere in the factorial.

## Representational dissociation

- **Generic-return:** within-state correlation with persistence **r=-0.282**; relative-incentive R² **0.083**; STOP/CONTINUE betas **0.453 / 0.133**.
- **Provisional advantage:** within-state correlation with persistence **r=0.313**; relative-incentive R² **0.121**; STOP/CONTINUE betas **-0.236 / 0.266**.
- **Direct persistence:** within-state correlation with persistence **r=0.991**; relative-incentive R² **0.751**; STOP/CONTINUE betas **-0.684 / 0.534**.

The generic-return projection moves in the wrong direction when STOP becomes more valuable, so it is not an invariant action-relative decision signal under this manipulation. The advantage projection has the theoretically correct signs but explains only a modest share of the manipulated decision coordinate. The final persistence representation almost exactly follows behavior.

## Causal steering

The layer-31 persistence positive control passes: positive-minus-negative steering changes persistence by **1.994 logits** with 95% CI 1.978 to 2.011.
- **Generic return:** positive-minus-negative effect **0.0028 logits** (95% CI -0.0010 to 0.0069; bootstrap p=0.151); matched-random absolute empirical p **0.571**.
- **Continuation advantage:** positive-minus-negative effect **-0.0012 logits** (95% CI -0.0063 to 0.0041; bootstrap p=0.665); matched-random absolute empirical p **0.810**.
Both early value-direction effects are indistinguishable from zero and ordinary matched random directions, despite validated one-SD movement of their frozen probe outputs.

## Interpretation constraints

The zero/zero incentive prompt lowers persistence by **-1.444 logits** relative to the same underlying state with its unmodified decision prompt (95% CI -1.622 to -1.267). This does not invalidate within-factorial contrasts, but zero/zero is not a neutral copy of the original task.
A categorical relative-incentive model explains **0.883**, while unrestricted condition cells explain **0.949**. Relative value is dominant but not sufficient: numerical framing and/or nonlinear absolute-payoff effects remain.
The common CONTINUE manipulation also changes the A-minus-B logit gap, so it is not perfectly arm-specific. Claims should concern CONTINUE versus STOP behavior, not a fully isolated scalar utility computation.

## Adjudication

1. **Integrated value exists:** supported as decodable early-layer information by the earlier held-out return probes, but the identified linear axes are causally inert for persistence at the calibrated intervention scale.
2. **Generic future return drives stopping:** strongly disfavored; it neither predicted matched-state stopping, transformed appropriately under STOP-value manipulation, nor causally moved persistence.
3. **Continuation-versus-stop value matters:** strongly supported behaviorally. Explicitly changing either side of the comparison reverses the decision in nearly every held-out history. This alone cannot distinguish utility computation from a prompt-local numeric/instruction heuristic.
4. **A clean native continuation-advantage axis has been identified:** not supported for the fitted layer-2 direction. It responds correctly but weakly to external incentives and has no detectable causal effect; the late persistence axis is the operative variable identified here.
5. **Stopping is only a recent-loss/time heuristic:** rejected as a complete account. It describes the original-task baseline well, but cannot explain the large within-history response to externally manipulated payoffs. A broader policy-heuristic account combining recent history with prompt-local incentive cues remains viable.
