# PRD 2: Comparative Computational Models of LLM Persistence

**Project:** Computational Architecture of Persistence in LLMs
**Stage:** Cross-task behavioral modeling after persistence-battery construction
**Primary goal:** Determine which computational models developed for human/animal persistence best explain LLM persistence, what components generalize across tasks, and whether persistence is distinguishable from generic repeated decision-making.

---

## 1. Scientific objective

PRD 1 established a broader battery of structurally different persistence tasks grounded in human/animal paradigms.

PRD 2 asks:

> **What computational architecture best explains LLM persistence across these tasks?**

The central comparison is between four possibilities:

$$
\boxed{\text{unitary motivational state}}
$$

$$
\boxed{\text{shared stay/switch algorithm}}
$$

$$
\boxed{\text{shared motivational ingredients + task-specific computations}}
$$

$$
\boxed{\text{generic sequential decision-making}}
$$

The central methodological requirement is **generalization to entirely held-out tasks**, not simply prediction of held-out episodes within familiar tasks.

---

# 2. Current theoretical hypotheses

## H1 — Unitary motivational state

A slowly evolving latent variable represents something like motivation, commitment, patience, or persistence:

$$
M_t=\rho M_{t-1}+\beta^\top X_t+\epsilon_t
$$

with:

$$
P(\text{continue}_t)
=
\sigma(\alpha+\gamma M_t).
$$

This predicts:

* substantial recurrence;
* low-dimensional latent state sufficiency;
* similar latent dynamics across tasks;
* generalization of the latent-state model across contexts.

Current evidence makes this relatively unlikely, but it remains an explicit competitor.

---

## H2 — Shared stay/switch computation

Tasks construct different evidence, but a common termination rule integrates it:

$$
E_t^{(\tau)}
=
f_\tau(X_t,H_t)
$$

$$
P(\text{continue}_t)
=
g(E_t^{(\tau)}).
$$

The representations and observation mappings can differ while the downstream computational rule is shared.

---

## H3 — Shared ingredients, task-specific computation

The same broad classes of information matter:

$$
\text{history}
,\;
\text{progress}
,\;
\text{cost}
,\;
\text{alternatives}
,\;
\text{success evidence}
,\;
\text{elapsed effort},
$$

but each task combines them differently:

$$
P_\tau(\text{continue}_t)
=
g_\tau(X_t,H_t).
$$

This is the current leading hypothesis.

---

## H4 — Generic sequential choice

Persistence is not computationally special.

The same history-sensitive machinery predicts repeated independent decisions:

$$
P(a_t)
=
g(X_t,H_t)
$$

whether or not the choice concerns maintaining an ongoing goal.

The sequential non-persistence control from PRD 1 is the critical comparator.

---

# 3. Inputs

Use the frozen outputs of PRD 1 plus the original tasks.

Expected persistence tasks include:

* Bandit
* Foraging
* Solvability
* Voluntary Waiting
* Progressive Ratio
* Sunk-Cost Persistence
* Information Sampling
* Partial-Reinforcement Extinction

plus:

* Independent Effort Choice sequential control

and optionally:

* Controllability Transfer

Use only tasks that passed PRD 1's basic manipulation and nondegeneracy gates.

Do **not** repair or exclude a task because it fails to show a human-like motivational phenomenon.

---

# 4. Primary target: discrete-time termination hazard

For all ongoing-goal tasks define:

$$
Y_{it}
=
\begin{cases}
1,& \text{terminate at state }t\\
0,& \text{maintain current pursuit}
\end{cases}
$$

and:

$$
h_{it}
=
P(Y_{it}=1\mid Y_{i,<t}=0).
$$

Use the hazard formulation as primary because termination is absorbing.

The likelihood is:

$$
\mathcal L
=
\prod_i\prod_{t\in R_i}
h_{it}^{Y_{it}}
(1-h_{it})^{1-Y_{it}}.
$$

Never include post-termination states.

---

# 5. Standardized computational variables

Construct a frozen common semantic schema before model fitting.

## History

* previous semantic action
* previous outcome
* continue streak
* failure streak
* success streak
* finite recent outcomes
* finite recent actions

## Time / investment

* elapsed steps
* cumulative effort
* already invested cost
* distance from episode start

## Prospective evidence

* estimated probability of success
* progress evidence
* expected remaining effort
* remaining time
* expected continuation payoff

## Disengagement evidence

* outside option
* quit payoff
* alternative value
* evidence current goal is futile

## Cost

* immediate continuation cost
* marginal effort requirement
* sampling cost
* opportunity cost

Missing constructs remain explicitly missing.

**Do not invent a numerical “progress” variable for a task where progress has no meaningful definition.**

---

# 6. Scale normalization

Cross-task modeling will fail trivially if one task expresses cost as `3`, another as `30`, and another as `0.03`.

Define task variables in dimensionless terms wherever possible.

Examples:

$$
C^{norm}
=
\frac{\text{current continuation cost}}
{\text{task payoff scale}}
$$

$$
O^{norm}
=
\frac{\text{outside option}}
{\text{maximum attainable reward scale}}
$$

$$
T^{norm}
=
\frac{\text{elapsed effort}}
{\text{episode horizon}}.
$$

Normalization must be based on **environment specifications**, not held-out behavioral data.

No target-task outcome statistics may determine scaling in strict LOTO.

---

# 7. Model zoo

The primary model zoo should contain both biologically motivated interpretable models and flexible ceilings.

---

## M0 — Intercept / task base rate

$$
\operatorname{logit}(h_t)=\alpha_\tau.
$$

Required baseline.

---

## M1 — Time-only hazard

$$
\operatorname{logit}(h_t)
=
\alpha_\tau+\beta_TT_t.
$$

Tests simple increasing/decreasing quit hazard.

---

## M2 — Immediate-state model

$$
\operatorname{logit}(h_t)
=
\alpha_\tau+\beta^\top X_t.
$$

No behavioral history.

Tests dynamic re-evaluation based entirely on current conditions.

---

## M3 — Finite history

$$
\operatorname{logit}(h_t)
=
\alpha_\tau+
\sum_{k=1}^{K}
[
\beta^a_k a_{t-k}
+
\beta^r_k r_{t-k}
].
$$

Test:

```yaml
history_lags: [1, 2, 3, 5, 8]
```

Hyperparameter selected on validation only.

---

## M4 — Exponential reward-history integration

$$
R_t
=
\lambda_RR_{t-1}
+
r_t.
$$

Then:

$$
\operatorname{logit}(h_t)
=
\alpha_\tau+\beta_RR_t+\beta^\top X_t.
$$

This is analogous to reward-history integration in animal stay/leave models.

---

## M5 — Choice perseveration / hysteresis

$$
K_t
=
\lambda_KK_{t-1}
+
I(\text{continued}_{t-1}).
$$

Then:

$$
\operatorname{logit}(h_t)
=
\alpha_\tau
+
\kappa K_t
+
\beta^\top X_t.
$$

Tests whether maintaining a policy itself increases the tendency to maintain it again.

---

## M6 — Dual history

Include separate action and outcome memory:

$$
K_t
=
\lambda_KK_{t-1}+a_{t-1}
$$

$$
R_t
=
\lambda_RR_{t-1}+r_{t-1}.
$$

Then:

$$
\operatorname{logit}(h_t)
=
\alpha_\tau+
\kappa K_t+
\eta R_t+
\beta^\top X_t.
$$

This is an important candidate given the current results.

---

# 8. Dynamic re-evaluation model

Model the value of continuing relative to disengaging:

$$
A_t
=
V_{\text{continue},t}
-
V_{\text{terminate},t}.
$$

Then:

$$
P(\text{continue}_t)
=
\sigma(
\alpha_\tau+\beta_AA_t
).
$$

Whenever possible construct \(A_t\) from information genuinely available to the model.

Do not use realized future outcomes.

Variants:

### Observable re-evaluation

Uses only explicitly available state variables.

### Oracle re-evaluation

May use known environment parameters such as true success probability.

Oracle models must be clearly separated from behaviorally plausible models.

---

# 9. Option-termination model

Inspired by hierarchical RL:

$$
A^{option}_t
=
Q(\text{maintain current policy})
-
V(\text{alternative / terminate}).
$$

Termination probability:

$$
\beta_t
=
\sigma(-\eta A^{option}_t+b).
$$

Conceptually this asks whether persistence is best understood as termination of a temporally extended policy.

This should be one of the central interpretable models.

---

# 10. Competitive time–reward integration

Adapt animal patch-leaving / persistence models in which elapsed effort and recent rewards exert opposing pressure.

For example:

$$
DV_t
=
\beta_TT_t
-
\beta_RR_t
+
\beta_CC_t
+
\beta_OO_t.
$$

Then:

$$
P(\text{terminate}_t)
=
\sigma(DV_t).
$$

This is particularly relevant to:

* Foraging
* Waiting
* Progressive Ratio
* Sunk Cost
* PREE

but can be evaluated more broadly when variables have valid meanings.

---

# 11. Latent patience / commitment model

Explicitly retain a model analogous to slowly varying motivational state:

$$
M_t
=
\rho M_{t-1}
+
\beta_X^\top X_t
+
\epsilon_t
$$

$$
P(\text{terminate}_t)
=
\sigma(
\alpha-\gamma M_t
).
$$

Primary parameter:

$$
\rho.
$$

If the fit repeatedly drives:

$$
\rho\rightarrow0,
$$

the model has effectively collapsed into instantaneous re-evaluation.

Report that rather than continuing to call it a latent motivational state.

---

# 12. Sunk-investment extension

For relevant tasks add:

$$
S_t=\text{irrecoverable prior investment}.
$$

Compare:

$$
P(\text{continue})
=
f(\text{prospective variables})
$$

against:

$$
P(\text{continue})
=
f(\text{prospective variables},S_t).
$$

This tests whether previous investment affects current persistence beyond prospective value.

Estimate across all tasks where sunk investment can be independently varied.

---

# 13. Flexible linear model

Include all valid observable variables plus prespecified interactions:

$$
H + T + C + P + O + V
$$

plus selected terms such as:

$$
H\times P
$$

$$
H\times C
$$

$$
T\times P.
$$

Use regularization.

This serves as the flexible interpretable ceiling.

---

# 14. MLP ceiling

Nonrecurrent MLP using:

* current state
* explicit finite history window

This determines how much nonlinear behavior can be captured without latent recurrence.

---

# 15. GRU ceiling

A recurrent flexible model using the same observable information available to the interpretable models.

Do not allow oracle-only features unless running a separately labeled oracle ceiling.

Report:

* hidden size
* parameter count
* validation procedure
* held-out metrics.

---

# 16. Model comparison levels

Every relevant model should be evaluated under multiple sharing assumptions.

## Level A — Task-specific

Each task has independent parameters:

$$
\theta_\tau.
$$

This estimates the best version of that architecture when no sharing is required.

---

## Level B — Fully shared

$$
\theta_B=\theta_F=\dots=\theta.
$$

Only task intercepts may differ if prespecified.

This is the strongest parameter-sharing claim.

---

## Level C — Hierarchically shared

$$
\theta_\tau
=
\theta_{global}
+
\delta_\tau.
$$

Regularize:

$$
\delta_\tau\rightarrow0.
$$

This distinguishes:

> same architecture with parameter variation

from:

> unrelated algorithms.

For an unseen task, use:

$$
\theta_{new}=\theta_{global}.
$$

No held-out-task coefficients are estimated for zero-shot evaluation.

---

# 17. Central held-out-task experiment

This is the **primary analysis of PRD 2**.

For every persistence task \(\tau\):

1. hold out the entire task;
2. fit models to all remaining persistence tasks;
3. select hyperparameters using only the training tasks;
4. freeze model;
5. evaluate directly on the held-out task.

No target-task behavioral calibration.

No target-task normalization based on outcomes.

No target-task regression fitting.

Produce:

$$
\text{LOTO log loss}_{m,\tau}
$$

for every model \(m\).

---

# 18. Zero-shot computational generality

Define improvement over the held-out task's null/base-rate benchmark:

$$
\Delta L_{m,\tau}
=
L_{\text{null},\tau}
-
L_{m,\tau}.
$$

Macro-average equally over tasks:

$$
\overline{\Delta L}_m
=
\frac{1}{N}
\sum_{\tau=1}^N
\Delta L_{m,\tau}.
$$

Primary question:

> Which computational architecture predicts persistence in a genuinely unseen task?

This is more important than average within-task fit.

---

# 19. Few-shot adaptation curve

Secondary analysis.

After zero-shot evaluation, allow models:

$$
1,\;4,\;8,\;16,\;32
$$

target-task episodes for parameter adaptation.

Evaluate on remaining untouched episodes.

Plot:

$$
\text{performance}
$$

against:

$$
\text{target-task examples}.
$$

Interpretation:

A general architecture may fail parameter-free zero-shot but adapt rapidly.

This allows us to distinguish:

### Shared parameters

Immediate zero-shot transfer.

### Shared architecture

Fast adaptation with very little target-task data.

### Task-specific algorithm

Requires substantial target-task data.

This distinction is theoretically important.

---

# 20. Architecture-transfer metric

For each held-out task compute:

$$
G_{m,\tau}
=
\frac{
L_{\text{null},\tau}
-
L_{\text{LOTO},m,\tau}
}{
L_{\text{null},\tau}
-
L_{\text{task-specific},m,\tau}
}.
$$

Interpretation:

$$
G=1
$$

means the cross-task model captures all improvement achievable by fitting that architecture directly to the target.

$$
G=0
$$

means no transfer beyond null.

Negative values indicate harmful transfer.

Macro-average across tasks.

---

# 21. Leave-one-family-out analysis

Some tasks may be close relatives.

Define task families before examining outcomes.

Potential families:

```text
reward pursuit
effort expenditure
waiting
evidence gathering
learning/history
problem solving
```

Then hold out entire **families**.

This prevents the result:

> “Waiting transfers to another waiting variant”

from being interpreted as general motivational architecture.

LOFO is a stronger stretch test than LOTO.

---

# 22. Human/animal signature analyses

Separate model comparison from qualitative signatures.

Estimate, where applicable:

### Sunk cost

$$
\beta_{\text{sunk investment}}.
$$

### Partial reinforcement extinction

$$
\beta_{\text{partial acquisition}}.
$$

### Temporal-context adaptation

Difference in waiting hazard across learned timing distributions.

### Effort breakpoint

Sensitivity to progressive work requirement.

### Information sampling

Sensitivity to:

* uncertainty
* sample cost
* error penalty.

### Controllability

Effect of prior controllability on later persistence.

These are **outputs**, not validity gates.

---

# 23. Sequential-control analysis

Use the Independent Effort Choice task to test H3 versus H4.

Because it consists of repeated decisions but not maintenance of one ongoing goal, it is the key control.

Fit exactly the same history representations:

* finite action history
* finite outcome history
* perseveration
* exponentially integrated history
* MLP
* GRU.

Then quantify:

$$
\Delta L_{\text{history}}
$$

for persistence tasks versus the sequential control.

---

# 24. Persistence-specific history index

Define:

$$
PSH
=
\Delta L_{\text{history,persistence}}
-
\Delta L_{\text{history,control}}.
$$

Bootstrap by episodes/tasks.

If:

$$
PSH\approx0,
$$

history sensitivity may be generic sequential choice.

If:

$$
PSH>0,
$$

maintaining an ongoing goal exhibits additional history dependence.

Also compare the **shape** of the history kernel, not only performance.

---

# 25. History-kernel similarity

For each task estimate:

$$
K_\tau
=
[
\beta_1,\beta_2,\dots,\beta_K
].
$$

Compare:

* magnitude
* decay
* sign
* action versus outcome weighting.

Use:

* correlations;
* cosine similarity;
* hierarchical variance estimates.

This asks whether the shared component is genuinely the same history computation.

---

# 26. Strong test of human-like architecture

Do not define “human-like” as simply:

> higher reward → more persistence.

Instead evaluate whether LLMs exhibit a collection of distinctive signatures such as:

* finite recency weighting;
* perseveration;
* reward-history integration;
* context-sensitive waiting;
* sunk-cost sensitivity;
* partial-reinforcement effects;
* controllability effects.

The eventual paper should distinguish:

$$
\text{generic rational effects}
$$

from:

$$
\text{distinctive motivational signatures}.
$$

---

# 27. Main metrics

Primary:

### Held-out hazard log loss

$$
-\frac{1}{N}\sum
[
y\log p+(1-y)\log(1-p)
].
$$

Report task-macro and episode-weighted values.

Secondary:

* Brier score
* calibration error
* calibration slope/intercept
* ROC-AUC where informative
* deviance explained
* correlation with semantic persistence logit if logits are available

Avoid making \(R^2\) the sole criterion for probabilistic decisions.

---

# 28. Equal task weighting

The battery will contain different numbers of states per task.

Primary cross-task metric must be:

$$
\frac{1}{N_{\text{tasks}}}
\sum_\tau Metric_\tau.
$$

Do not allow long episodes such as Waiting or Progressive Ratio to dominate the result.

State-weighted metrics can be secondary.

---

# 29. Statistical uncertainty

Use task/episode-appropriate bootstrap.

For within-task quantities:

```yaml
bootstrap_unit: episode
samples: 2000
```

For cross-task conclusions:

prefer hierarchical/bootstrap procedures preserving complete tasks.

Report uncertainty over both:

* episodes;
* task identities where statistically meaningful.

With only ~8 tasks, be transparent that task-level generalization uncertainty is substantial.

---

# 30. Model recovery before real-data interpretation

This is mandatory.

Generate synthetic datasets from:

### Synthetic H1

Latent commitment.

### Synthetic H2

Shared stay/switch rule with task-specific observation mapping.

### Synthetic H3

Shared history but task-specific evaluation.

### Synthetic H4

Generic sequential choice.

Run the entire model-selection pipeline.

Verify that the intended model family is recoverable.

If candidate models cannot be distinguished synthetically at realistic sample sizes, do not interpret their empirical ranking strongly.

---

# 31. Model-confusion matrix

Produce:

| Generating model | Selected M1 | M2 | ... |
| ---------------- | ----------: | -: | --: |

across repeated synthetic datasets.

This should explicitly reveal where models are behaviorally indistinguishable.

Particularly important comparisons:

* latent commitment vs exponential history;
* finite history vs GRU;
* dynamic value vs option termination;
* choice perseveration vs generic reward history.

---

# 32. Feature ablations

For the best flexible model, perform leave-one-family-out ablations:

$$
\Delta L_H
$$

history

$$
\Delta L_T
$$

time/effort

$$
\Delta L_C
$$

cost

$$
\Delta L_P
$$

progress

$$
\Delta L_O
$$

outside option

$$
\Delta L_V
$$

prospective value.

Also run feature-family-only models.

This tells us which motivational ingredients actually generalize.

---

# 33. Cross-task feature support

For every feature family classify:

### Broadly shared

Useful in most applicable tasks with consistent sign.

### Family-specific

Useful only in a coherent subset.

### Task-specific

No cross-task consistency.

### Unsupported

Little reliable predictive value.

Do not treat missing variables as evidence of no effect.

---

# 34. GRU diagnostic

Given earlier evidence that five-step history nearly matched full recurrence, explicitly repeat the question across the expanded battery:

$$
\text{GRU}
$$

versus:

$$
\text{MLP + finite history}.
$$

If the expanded battery still shows:

$$
Performance_{\text{GRU}}
\approx
Performance_{\text{finite history}},
$$

this is strong evidence against a requirement for a persistent latent motivational state.

---

# 35. GRU bottleneck

Repeat:

```yaml
hidden_sizes: [1, 2, 4, 8, 16, 32, 64]
```

but interpret carefully.

The question is not:

> “Does Qwen have a 64-dimensional motivational state?”

It is:

> “How compact can a generic recurrent model be while reproducing these behavioral policies?”

This is a computational-complexity diagnostic only.

---

# 36. Predefined outcome taxonomy

## Outcome A — Common motivational computation

Evidence:

* strong LOTO/LOFO transfer;
* shared/hierarchical models approach task-specific ceiling;
* similar history kernels;
* biological models outperform generic alternatives;
* persistence differs from sequential control.

Interpretation:

> **Diverse LLM persistence behaviors share a common history-dependent policy-maintenance architecture.**

---

## Outcome B — Shared ingredients, different algorithms

Evidence:

* history/cost/progress families recur;
* task-specific models clearly beat shared mappings;
* LOTO parameter transfer limited;
* architecture adapts rapidly few-shot;
* kernel structure varies substantially.

Interpretation:

> **LLM persistence shares motivational ingredients without sharing a single computational rule.**

This is close to the current working hypothesis.

---

## Outcome C — Generic sequential choice

Evidence:

* sequential control shows equivalent history dependence;
* generic sequential models transfer as well as persistence-specific models;
* no distinctive persistence signatures remain after matching sequential structure.

Interpretation:

> **Persistence is a functional application of generic history-sensitive sequential decision machinery.**

---

## Outcome D — Task-specific behavior

Evidence:

* cross-task transfer poor;
* common feature families inconsistent;
* few-shot adaptation slow;
* little recoverable architecture-level regularity.

Interpretation:

> **Persistence is primarily a functional behavioral category instantiated through task-specific computations.**

---

# 37. Causal/mechanistic gate

Do **not** start PRD 3 mechanistic localization simply because one model wins average behavioral fit.

Proceed to targeted mechanistic analysis only if PRD 2 identifies a robust cross-task computational regularity.

Good targets would include:

* common finite history integration;
* policy hysteresis;
* reward-history summary;
* continuation-versus-termination advantage;
* interaction of history with current progress/cost.

The mechanistic hypothesis should be defined **before looking at activations**.

---

# 38. TDD requirements

All implementation follows:

```text
RED → GREEN → REFACTOR
```

Required tests include:

### Hazard risk set

No post-termination states.

### Future leakage

No realized future rewards or episode duration in observable models.

### Split integrity

No semantic episode or label-counterbalanced pair crosses splits.

### LOTO leakage

Held-out-task records may never enter:

* training
* hyperparameter selection
* feature normalization
* model selection.

### Normalization

Environment-spec normalization is reproducible.

### Missing variables

Missing semantic constructs cannot silently become zeros.

### Task weighting

Synthetic unequal task sizes must not alter macro rankings.

### Few-shot integrity

Adaptation episodes and evaluation episodes must be disjoint.

---

# 39. Synthetic TDD

Write failing tests for:

### Shared-rule recovery

Common rule + task offsets should be recognized as shared.

### Task-specific recovery

Distinct coefficients should favor task-specific architecture.

### History recovery

Known five-lag kernel should recover lag structure.

### Latent-state recovery

High-\(\rho\) latent generator should favor latent-state model.

### Zero-\(\rho\) collapse

Latent model with \(\rho=0\) should be identified as effectively immediate-state.

### Generic-control recovery

Identical sequential dynamics in persistence/control data should yield:

$$
PSH\approx0.
$$

---

# 40. Proposed code structure

```text
analysis/
    comparative_persistence/
        build_modeling_dataset.py
        semantic_features.py
        normalization.py

        hazard_models/
            baselines.py
            finite_history.py
            exponential_history.py
            perseveration.py
            dual_history.py
            dynamic_reevaluation.py
            option_termination.py
            competitive_accumulator.py
            latent_commitment.py

        flexible/
            linear.py
            mlp.py
            gru.py

        sharing/
            task_specific.py
            fully_shared.py
            hierarchical.py

        evaluation/
            within_task.py
            leave_one_task_out.py
            leave_one_family_out.py
            few_shot_adaptation.py
            feature_ablations.py

        controls/
            sequential_choice.py
            history_specificity.py

        synthetic/
            generators.py
            recovery.py
            confusion_matrix.py

        reporting/
            build_report.py
```

---

# 41. Configuration

Create:

```text
config/comparative_persistence.yaml
```

Suggested:

```yaml
protocol_version: comparative_persistence_v1

targets:
  primary: termination_hazard

sharing:
  - task_specific
  - fully_shared
  - hierarchical

history_lags: [1, 2, 3, 5, 8]

history_decay:
  - 0.0
  - 0.25
  - 0.5
  - 0.7
  - 0.85
  - 0.95

gru_hidden_sizes:
  - 1
  - 2
  - 4
  - 8
  - 16
  - 32
  - 64

evaluation:
  within_task: true
  leave_one_task_out: true
  leave_one_family_out: true
  few_shot_counts: [0, 1, 4, 8, 16, 32]

metrics:
  primary: log_loss
  secondary:
    - brier
    - calibration
    - auc
    - deviance_explained

bootstrap:
  samples: 2000

synthetic_recovery:
  repetitions: 100
```

---

# 42. Required outputs

Save to:

```text
artifacts/comparative_persistence/<run_id>/
```

Required:

```text
model_comparison/
    within_task.csv
    macro_average.csv
    model_rankings.csv

generalization/
    loto.csv
    loto_summary.csv
    lofo.csv
    few_shot_curves.csv
    architecture_transfer.csv

history/
    finite_kernels.csv
    exponential_kernels.csv
    task_kernel_similarity.csv
    persistence_vs_control.csv

features/
    family_ablation.csv
    family_only.csv
    cross_task_feature_support.csv

human_animal_signatures/
    signature_effects.csv

synthetic/
    recovery.csv
    confusion_matrix.csv

figures/
    model_zoo.png
    loto_performance.png
    architecture_transfer.png
    few_shot_adaptation.png
    history_kernels.png
    feature_ablation.png
    persistence_vs_control.png
    model_recovery.png

report.md
run_metadata.json
```

---

# 43. Primary figures for the paper

I would design the analysis around producing these figures.

### Figure 1 — Persistence battery

Tasks × manipulated motivational variables.

### Figure 2 — Computational model comparison

Human/animal-inspired models versus flexible models.

### Figure 3 — Held-out-task generalization

For each model:

$$
G_{m,\tau}.
$$

This is probably the central figure.

### Figure 4 — Shared computational ingredients

History, effort, progress, cost, outside-option ablations.

### Figure 5 — Persistence versus sequential choice

History dependence and kernel shape.

### Figure 6 — Human/animal motivational signatures

Which signatures appear across LLM tasks.

---

# 44. Automated report must answer

The final report should begin with direct answers to:

1. **Which interpretable computational model best predicts LLM persistence?**
2. **Does a latent commitment/patience state improve prediction over finite recent history?**
3. **How much predictive structure transfers zero-shot to entirely unseen tasks?**
4. **Does the same architecture transfer even when exact parameters do not?**
5. **How quickly can models adapt to a novel persistence task?**
6. **Which motivational ingredients generalize most consistently?**
7. **Are recent-history effects stronger or qualitatively different for maintaining an ongoing goal than for independent repeated decisions?**
8. **Which human/animal motivational signatures are reproduced?**
9. **Which of H1–H4 is currently best supported?**
10. **What specific computational variable, if any, has earned mechanistic investigation in PRD 3?**

---

# 45. Scientific endpoint

PRD 2 should leave us able to make one of two especially important claims.

The first would be:

> **Despite substantial variation in surface task and internal representation, LLM persistence exhibits a common computational architecture resembling biological models of dynamic policy maintenance.**

The second would be:

> **Persistence does not correspond to a unified motivational computation: common behavioral ingredients such as recent history recur across tasks, but their integration remains task-specific or indistinguishable from generic sequential decision-making.**

Either outcome is useful.

Most importantly, **PRD 2 determines what PRD 3 should mechanistically interrogate**. We should not decide in advance that PRD 3 is about a “persistence circuit.” It should target whatever computational regularity actually survives this broader behavioral battery.
