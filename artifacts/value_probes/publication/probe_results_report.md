# Value-probe result: held-out diagnostic

## Bottom line

The run successfully found a differentiable sparse direction and calibrated a tiny intervention, but it did **not** validate that direction as an integrated task-value representation. Confirmatory steering should be paused or explicitly relabeled exploratory until the probe target/training is repaired.

## Probe fitting

- Selected layer: **17 / 31**; selected dimensions: **26 / 2,560**.
- Sparse validation/test TD MSE: **6.155 / 6.373**.
- Full validation/test TD MSE at layer 17: **9.485 / 11.594**.
- Layer 17 beat the runner-up validation layer by only **0.008 MSE**; the entire 32-layer sparse range was **0.093**.
- Validation/test layer rankings were nevertheless similar (Spearman r=0.92), and layer 17 also had the lowest sparse test MSE.
- **32 / 32** layers had their best validation checkpoint at epoch 1, after which TD loss deteriorated before patience stopped training at epoch 11.
- No constant or recent-history TD baseline was stored, so the absolute MSE cannot establish predictive value by itself.
- Masking reduced the probe-output SD from **4.866** to **0.070**. The lower sparse TD error therefore largely accompanies shrinkage toward zero, rather than preservation of the full probe's value scale.

## Held-out mechanism test

The prespecified recent-history controls already explained **75.3%** of persistence-logit variance. Adding the sparse probe explained only **0.27 percentage points** (beta=-0.099, SE=0.040, p=0.014), and the coefficient was opposite the predicted direction.
Within states that had just received -2, the sparse coefficient was **-0.179** (delta R²=0.008, p=0.00444), again negative.
The full probe added essentially no adjusted persistence information (beta=-0.034, delta R²=0.0003, p=0.514).
The closest positive integrated-history result was cumulative score predicting sparse probe output within last-loss states (beta=0.153, delta R²=0.014), but its p=0.0703 did not cross the prespecified .05 threshold.

## Why the unadjusted pattern looks convincing

Raw probe–persistence correlations were positive (sparse r=0.49; full r=0.66), but both probes strongly tracked round (sparse r=0.72; full r=0.79). Once round and recent reward history were controlled, the apparent positive relationship vanished or reversed.

## Probe-only, history-only, and joint models

The sparse probe alone explained **24.0%** of persistence variance, while the full probe alone explained **44.0%**. Behavioral history alone explained **75.3%**. The joint models explained **75.6%** (sparse) and **75.3%** (full).
For the sparse probe, **23.7 percentage points** were shared with history and only **0.27 points** were unique. For the full probe, **44.0 points** were shared and **0.031 points** were unique.
The probe-only models are therefore meaningful descriptions of total prediction, while the joint models show that nearly all of that prediction duplicates information already available in recent history and round.

## Exact matched-history test

Matching exactly on round, previous outcome, and loss streak retained **1180 states in 147 strata from 55 episodes**. Older history was measured as cumulative score before the immediately preceding outcome.
Older history strongly predicted the full probe (beta=0.440, p=1.31e-05) but not the sparse probe or persistence. Within exact strata, neither the sparse probe (beta=0.002, p=0.976) nor the full probe (beta=-0.022, p=0.814) predicted persistence.
This supports a recent-state stopping heuristic over the claim that the current probe-defined value representation drives persistence. It also shows that the unpruned hidden-state-derived probe retains accumulated history; what remains unresolved is whether that history code represents expected future return.

## Calibration is not construct validation

Magnitude **0.01** ordered probe outputs on **100%** of validation states. Its reported relative RMS was **0.000198**. This verifies that stepping along the probe gradient changes the probe in the intended mathematical direction; it does not show that the probe represents value or that the perturbation will materially change action logits.

## Recommended sprint decision

1. Do not present the current layer-17 direction as a validated integrated-value representation.
2. Before causal confirmation, compare against a zero/constant TD baseline and a recent-outcome-plus-round baseline.
3. Replace unstable online bootstrapping with a frozen-target TD procedure or a supervised Monte Carlo future-return probe using the already stored future returns, then rerun the same untouched-test mechanism diagnostic.
4. Run causal steering only if the repaired probe adds positive held-out signal beyond recent history. Otherwise, any steering run should be labeled an exploratory perturbation of a probe-defined direction, not value steering.
