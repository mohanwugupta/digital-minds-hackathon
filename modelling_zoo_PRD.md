Below is a PRD written to hand directly to a coding agent. I’ve made the implementation requirements fairly strict because there are several easy ways to accidentally leak information, compare models unfairly, or turn the GRU into an uninterpretable “winner.”

# PRD: Computational Models of Cross-Task Persistence

**Project:** Digital Minds Sprint — Persistence Mechanisms
**Repository:** `mohanwugupta/digital-minds-hackathon`
**Status:** Implementation-ready
**Analysis role:** Exploratory model discovery, preceding mechanistic interpretation
**Primary model:** `Qwen/Qwen3.5-4B`

---

## 1. Objective

We want to determine what computational process best explains an LLM's changing willingness to continue pursuing a task rather than disengage.

Existing work identified:

1. strong within-task persistence signals;
2. behavioral sensitivity to incentives;
3. a cross-task direction that predicted persistence decisions but failed specificity controls because it also generalized to arbitrary binary choices and rule-determined terminality;
4. a candidate persistence-specific representational subspace around approximately layers 21–22 from subsequent exploratory work.

The next step is **not another probe trained directly on CONTINUE-vs-STOP logits**.

The goal of this PRD is to compare explicit computational accounts of persistence and generate interpretable latent variables that can subsequently be tested against the candidate neural persistence representation.

The core scientific question is:

> **What computational state or algorithm best explains the model's evolving tendency to persist across Bandit, Foraging, and Solvability tasks?**

Secondary question:

> **Is there a common computational architecture across tasks even when task-specific inputs and output mappings differ?**

---

# 2. High-level pipeline

Implement:

```text
behavioral records
      ↓
candidate computational models
      ↓
held-out behavioral prediction
      ↓
GRU / nonlinear ceilings
      ↓
model comparison
      ↓
winning interpretable latent variables
      ↓
future L21/L22 representation analysis
```

This PRD covers everything through **winning interpretable latent variables**.

Do **not** implement the L21/L22 neural analysis in this PRD unless explicitly requested later.

---

# 3. Existing code that must be reused

Do not create a parallel modeling framework when equivalent repository utilities already exist.

### Existing Bandit model comparison

Reuse/refactor:

```text
computational_modeling/analysis/compare_behavioral_models.py
```

It already implements:

* time-only model;
* recent-history heuristic;
* Rescorla–Wagner;
* RW + history;
* Bayesian learning;
* Bayesian + history;
* held-out whole-episode evaluation;
* episode-balanced weighting;
* log loss / AUC / Brier;
* episode bootstrap comparisons.

The existing implementation explicitly evaluates STOP vs CONTINUE on held-out episodes.

### Existing latent-state framework

Reuse/refactor:

```text
analysis/persistence_latent_state.py
analysis/run_persistence_latent_state.py
config/persistence_latent_state.yaml
```

This currently implements comparisons among:

* immediate decision;
* choice-history inertia;
* generic latent value;
* recurrent latent commitment.

It also contains synthetic architecture-recovery machinery.

### Existing cross-task data utilities

Reuse:

```text
experiments/cross_task_utils.py
```

Particularly:

* activation-shard loading;
* grouped pair-safe train/validation/test splits;
* record extraction.

The current loader already verifies episode integrity and state/activation alignment.

### Existing cross-task records

Use the existing activation banks from:

```text
artifacts/cross_task/track_b_shared_v3/
```

Paths are already specified in:

```text
config/persistence_discovery.yaml
```

including Bandit, Foraging, Solvability, arbitrary-choice, terminality, and generic-value controls.

The records already contain relevant behavioral fields including persistence logits, choice probabilities, semantic actions, reward/progress histories, and task parameters.

---

# 4. Scientific distinction that must be preserved

We need to distinguish two prediction targets.

## Target A: latent behavioral policy

Primary computational target:

$$
D_t =
\log
\frac{P(\text{continue}_t)}
{P(\text{disengage}_t)}
$$

Use the existing `persistence_logit` where available.

Examples:

### Bandit

$$
D_t =
\log
\frac{P(A)+P(B)}
{P(C)}
$$

### Foraging

$$
D_t =
\log
\frac{P(\mathrm{STAY})}
{P(\mathrm{LEAVE})}
$$

### Solvability

$$
D_t =
\log
\frac{P(\mathrm{TRY\ AGAIN})}
{P(\mathrm{GIVE\ UP})}
$$

This is the **primary outcome** because it removes random action-sampling noise.

Metrics:

* held-out \(R^2\);
* MSE;
* Pearson \(r\);
* taskwise metrics;
* macro-average across tasks.

---

## Target B: sampled semantic behavior

Secondary target:

```text
CONTINUE = 1
DISENGAGE = 0
```

Use the actual sampled semantic action.

Metrics:

* held-out log loss — primary;
* Brier score;
* AUC.

This maintains comparability with the original Bandit behavioral analysis.

---

# 5. Three performance ceilings

Every behavioral model comparison must report three reference levels.

## 5.1 Oracle policy ceiling

For sampled choices, the true LLM choice probability provides the irreducible stochastic ceiling.

For every state, use the model's actual:

```text
p_continue
p_stop
```

or the task-semantic equivalent.

Calculate:

* oracle sampled-choice log loss;
* oracle Brier.

This answers:

> How well could a predictor perform if it knew the LLM's actual policy exactly?

No fitted model can meaningfully be expected to beat this systematically on sampled choices.

---

## 5.2 Flexible GRU ceiling

Train a small GRU on task-observable history.

It predicts either:

$$
D_t
$$

or:

$$
P(\text{continue}_t).
$$

It may learn nonlinear/history-dependent structure but should remain intentionally low capacity.

This approximates:

> How much of the LLM's policy is recoverable from the recorded task state and history without assuming a particular cognitive model?

---

## 5.3 Non-recurrent nonlinear ceiling

Train a small MLP on the current observable state plus explicitly encoded recent-history features.

This distinguishes:

```text
nonlinearity
```

from:

```text
recurrence / latent memory
```

If:

```text
GRU >> MLP
```

that is evidence that sequence/history integration matters.

If:

```text
GRU ≈ MLP
```

the apparent persistence dynamics may be adequately captured by the current observable state.

---

# 6. Information-matching requirement

This is critical.

Each model must be tagged as one of:

```text
observable
oracle-state
```

## Observable models

May only use information available to the LLM in the prompt/history at that point.

Examples:

* past actions;
* past outcomes;
* cumulative score if displayed;
* current round;
* explicitly displayed costs;
* explicitly displayed outside options;
* explicitly displayed progress;
* previous success/failure.

## Oracle-state models

May additionally use latent environment parameters that the experimenter knows but the model does not directly observe.

Examples include:

* true Bandit \(p_A,p_B\);
* private Foraging patch probability;
* hidden environment probabilities.

The current code uses some privileged state variables—for example true Bandit probabilities and `patch_probability_private`—so these must not silently enter the observable model comparison.

Report observable and oracle-state models separately.

The **primary model ranking must use observable models**.

---

# 7. Model zoo

Implement the following candidate families.

Do not require every task to use every task-specific model.

---

## 7.1 Simple baselines

### M0: Task/intercept baseline

$$
D_t = \alpha_{\mathrm{task}}
$$

Tests whether anything dynamic needs explaining.

---

### M1: Time-only

Features such as:

$$
\log(1+t)
$$

and/or normalized progress through task.

Tests generic persistence decay.

Existing Bandit implementation should be reused.

---

# 8. History / heuristic models

## M2: Previous-outcome model

Features:

* previous reward/progress;
* time.

---

## M3: streak model

Features:

* loss/failure streak;
* success/progress streak;
* time.

This generalizes the existing Bandit heuristic.

---

## M4: choice-inertia model

Include:

* previous semantic continue/disengage;
* second previous semantic choice;
* time;
* optionally immediate outcome.

This model already conceptually exists in `persistence_latent_state.py`.

---

## M5: finite-history model

Explicitly encode the previous \(K\) semantic actions and outcomes.

Default:

```yaml
history_lags: [1, 2, 3, 5]
```

Select \(K\) using training/validation only.

Purpose:

Test whether apparent latent commitment can be reproduced by a short-memory heuristic.

---

# 9. Reinforcement/value-learning models

## M6: Rescorla–Wagner

Reuse existing implementation where meaningful.

For Bandit:

$$
Q_a(t+1)
=
Q_a(t)
+
\alpha[r_t-Q_a(t)].
$$

Use:

* best arm value;
* A/B value gap;
* time.

Learning rate selected using training/validation episodes only.

Do not tune alpha using test episodes.

---

## M7: Bayesian learner

Reuse current Beta-Bernoulli Bandit implementation.

Where a comparable Bayesian estimator makes sense for another task, implement it explicitly rather than pretending the Bandit update is universal.

---

## M8: learned value + history hybrid

For each learning model:

```text
value variables
+
recent outcomes
+
streak
+
choice inertia
+
time
```

Existing Bandit results already suggest this distinction matters, because value models alone did not outperform recent-history heuristics reliably.

---

# 10. Termination / meta-control models

These are scientifically central.

## M9: Immediate termination advantage

Construct:

$$
A_t =
Q_{\text{continue},t}
-
Q_{\text{disengage},t}.
$$

Then:

$$
D_t =
\beta_0+\beta_A A_t.
$$

Task-specific definitions of \(Q\) are permitted, but the architecture must remain the same.

Examples:

### Foraging

Compare estimated value of another search against outside option.

### Solvability

Compare expected value of another attempt against give-up value.

### Bandit

Compare estimated continuation value against STOP payoff.

---

## M10: Sticky termination model

Add persistence/inertia:

$$
D_t =
\beta_0
+
\beta_A A_t
+
\beta_K K_t
$$

where \(K_t\) is a choice/commitment kernel such as:

$$
K_t=\lambda K_{t-1}+I(\text{continued}_{t-1}).
$$

Estimate \(\lambda\) on train/validation only.

This is one of the main candidate explanations.

---

## M11: decomposed meta-control model

Instead of collapsing everything into one advantage, estimate separate contributions from:

$$
D_t =
\alpha
+\beta_V V_t
+\beta_C C_t
+\beta_P P_t
+\beta_O O_t
+\beta_H H_t.
$$

Conceptual components:

* continuation value;
* effort/cost pressure;
* progress evidence;
* outside option;
* history/inertia.

This model is important because a one-dimensional advantage may hide multiple control signals.

---

# 11. Foraging-specific model

## M12: MVT-like leave model

Implement an approximate marginal-value formulation.

At minimum compare:

$$
V_{\text{local patch}}
$$

against:

$$
V_{\text{outside}}
$$

plus search cost.

Do not claim this is literal Marginal Value Theorem unless the implemented assumptions warrant that interpretation.

Call it:

```text
MVT-like foraging threshold
```

in outputs unless it is formally derived.

---

# 12. Evidence-accumulation model

## M13: Leaky disengagement accumulator

Define:

$$
a_t =
\rho a_{t-1}
+
w^\top x_t.
$$

Then:

$$
D_t =
\alpha - \gamma a_t.
$$

where evidence \(x_t\) may contain:

* losses;
* cost;
* failure/progress evidence;
* outside-option evidence.

This differs conceptually from the existing latent-commitment model because the latent state is explicitly accumulated **evidence to disengage**.

Test both orientations but standardize output so:

```text
higher latent persistence state = more persistence
```

in saved artifacts.

---

# 13. Latent commitment model

## M14: Recurrent commitment state

Reuse/refactor the current model:

$$
w_t
=
\rho w_{t-1}
+
\beta^\top X_t
$$

with:

$$
D_t =
\alpha_{\text{task}}
+
\lambda_{\text{task}}w_t
+
\epsilon_t.
$$

The existing implementation currently uses:

```text
relative_value
cost_pressure
progress_evidence
```

and task-specific coefficients.

Extend it carefully to permit:

* observable-only inputs;
* oracle-state inputs;
* common versus task-specific dynamics.

---

# 14. Generic latent-value model

## M15: Generic recurrent value

A restricted version of M14:

$$
w_t =
\rho w_{t-1}
+
\beta V_t.
$$

Purpose:

Determine whether any benefit of the commitment state is simply persistent integration of generic value.

The current synthetic model-confusion framework already distinguishes this from latent commitment. Preserve that test.

---

# 15. Flexible models

## M16: regularized linear flexible model

Include a broad but predefined feature set.

This is useful as an interpretable upper benchmark before nonlinear models.

No automated feature search on the test set.

---

## M17: MLP

Suggested architecture:

```text
input
→ Linear(64)
→ GELU
→ Dropout
→ Linear(32)
→ GELU
→ output
```

Tune only a very small hyperparameter set.

Do not build a large deep network.

---

## M18: GRU

Primary flexible sequence ceiling.

Suggested:

```text
input projection
→ GRU(hidden_size=32 or 64, 1 layer)
→ linear output
```

Hyperparameter candidates:

```yaml
hidden_size: [32, 64]
dropout: [0.0, 0.1]
learning_rate: [1e-3, 3e-4]
```

Select on validation only.

No test-set tuning.

---

# 16. Cross-task architecture comparison

The most important model comparison is not simply:

```text
which model has lowest pooled error?
```

For appropriate model families, compare three levels of sharing.

## Fully task-specific

$$
\theta_B,\theta_F,\theta_S
$$

all parameters independent.

---

## Shared architecture, task-specific observation parameters

Example:

$$
w_t
=
\rho w_{t-1}
+
\beta_{\mathrm{task}}^\top X_t
$$

but common:

$$
\rho.
$$

Or:

$$
A_t=f_{\mathrm{task}}(X_t)
$$

followed by shared:

$$
D_t=\alpha_{\mathrm{task}}+\lambda A_t.
$$

This is likely the scientifically most plausible definition of a task-general computational mechanism.

---

## Fully shared

All compatible coefficients shared across tasks after standardized semantic input construction.

This is a deliberately strong test and should not be required for claiming a common architecture.

---

# 17. Splitting and leakage rules

These are hard requirements.

Use whole episodes as the unit of splitting.

Where counterbalanced episodes share a `pair_id`, both members of the pair must remain in the same split.

Reuse existing persisted splits whenever possible.

Never regenerate existing train/test splits merely because a different split is more convenient.

No:

* state-level random split;
* same episode across train/test;
* same counterbalanced pair across train/test;
* normalization using test data;
* hyperparameter selection on test;
* selecting model class based on test;
* selecting GRU architecture based on test.

---

# 18. Weighting

Long episodes must not count as more independent observations simply because the model persisted longer.

For aggregate fitting/evaluation:

$$
w_{it}\propto \frac{1}{N_i}
$$

where \(N_i\) is the number of states in episode \(i\).

Preserve the current episode-balanced logic.

For cross-task fits:

1. equalize total weight per episode;
2. equalize aggregate task contribution.

Thus a task with more recorded states or episodes should not automatically dominate the objective.

---

# 19. Primary evaluation

## Persistence-logit prediction

For each model/task report:

```text
held-out R²
held-out MSE
held-out r
```

Also report macro task averages.

---

## Sampled-choice prediction

Report:

```text
log loss
Brier score
AUC
```

Primary:

```text
log loss
```

---

# 20. Explainable-performance normalization

For persistence-logit prediction calculate:

$$
\mathrm{FractionGRU}
=
\frac{
R^2_{\mathrm{candidate}}
}{
R^2_{\mathrm{GRU}}
}.
$$

Cap descriptive plots at 1 if useful, but preserve raw values in tables.

Also report:

$$
\Delta R^2_{\mathrm{GRU-candidate}}.
$$

For sampled action log loss define:

$$
F_m
=
\frac{
L_{\mathrm{baseline}}-L_m
}{
L_{\mathrm{baseline}}-L_{\mathrm{oracle}}
}.
$$

This gives a rough fraction of reducible sampled-choice log loss captured by model \(m\).

Also calculate the GRU version.

Do not use these normalized metrics without reporting the raw metrics.

---

# 21. Statistical uncertainty

Cluster at episode level.

Where paired counterbalanced states exist, use pair-clustered resampling when appropriate.

Minimum:

```yaml
bootstrap_samples: 2000
```

Final report preferred:

```yaml
bootstrap_samples: 10000
```

Calculate 95% bootstrap confidence intervals for:

* each model metric;
* differences from key baselines;
* difference between best interpretable model and GRU;
* taskwise differences.

Avoid treating states as IID.

---

# 22. Model selection philosophy

Do not simply choose the model with the numerically smallest MSE.

Report:

1. predictive performance;
2. uncertainty;
3. parameter count;
4. architecture;
5. task generality;
6. observable vs oracle status.

If two models are effectively tied, prefer the simpler model.

Use a predefined tolerance such as:

```text
within 1% of best prediction error
```

before complexity tie-breaking.

This mirrors the existing latent-state model selection approach.

---

# 23. Synthetic model-recovery tests

Before interpreting real-data model selection, extend the existing synthetic recovery suite.

Generate synthetic datasets from at least:

```text
immediate decision
choice inertia
finite-history heuristic
RW/value learning
generic latent value
latent commitment
sticky termination
disengagement accumulator
```

Then run the model comparison blindly.

Required question:

> When architecture X generated the data, does model comparison recover X or at least the correct model family?

Produce a model-confusion matrix:

| Generating model | Recovered model |
| ---------------- | --------------- |
| immediate        | ...             |
| inertia          | ...             |
| value            | ...             |
| commitment       | ...             |

If two candidate architectures are systematically indistinguishable under the experimental design, **report this explicitly**.

Do not pretend the real-data comparison can distinguish models that synthetic recovery shows are observationally equivalent.

---

# 24. TDD requirement

Use strict RED → GREEN → REFACTOR.

Do not implement a feature first and add a passing test afterward.

For every core component:

```text
1. Write failing test.
2. Run test and confirm expected failure.
3. Implement minimum code.
4. Confirm test passes.
5. Run full existing suite.
6. Refactor only after green.
```

---

# 25. Mandatory tests

At minimum add tests for:

## Data integrity

* task records contain required fields;
* semantic continuation labels are correctly harmonized;
* no episode crosses split boundaries;
* no pair crosses split boundaries;
* state order within episode is correct.

## Information leakage

A test must deliberately insert an oracle-only field and verify:

```text
observable model → cannot access it
oracle model → can access it
```

## Standardization

Verify all normalization moments come from training data only.

## GRU sequence construction

Verify:

* reset hidden state between episodes;
* no state from episode \(i+1\) enters episode \(i\);
* sequence target at \(t\) cannot use future state \(t+1\).

## Episode weighting

Preserve/add test showing each episode receives equal total weight.

Existing test can be reused.

## Counterbalanced pair integrity

Verify both mappings remain in same split.

## Oracle ceiling

Synthetic sampled actions generated from known probabilities should yield oracle log loss matching analytical expectation within tolerance.

## Latent-state update

Hand-constructed sequence should produce exact expected state recursion.

## RW update

Preserve existing chosen-arm-only update test.

## Model recovery

One test per simulated generating architecture.

## Test blindness

Add a test or runtime assertion that the final test split is never passed into hyperparameter-selection functions.

---

# 26. Proposed code structure

Prefer:

```text
computational_modeling/
    README.md

    data/
        build_cross_task_behavioral_dataset.py
        feature_schema.py

    models/
        base.py
        baselines.py
        history.py
        rw.py
        bayesian.py
        termination.py
        accumulator.py
        latent_commitment.py
        mlp.py
        gru.py

    analysis/
        run_model_zoo.py
        evaluate_models.py
        model_recovery.py
        summarize_model_zoo.py

    tests/
        test_dataset_integrity.py
        test_information_sets.py
        test_episode_splits.py
        test_weighting.py
        test_rw.py
        test_latent_states.py
        test_termination_models.py
        test_gru_sequences.py
        test_model_recovery.py

    results/
```

Refactor existing code into this structure only if doing so does not break existing scripts.

Backward compatibility is preferred.

Do not delete:

```text
compare_behavioral_models.py
persistence_latent_state.py
```

Existing entrypoints can call shared utilities after refactoring.

---

# 27. Behavioral dataset export

Create a records-only dataset so future computational analysis does not require loading the large activation tensors.

Suggested outputs:

```text
artifacts/computational_modeling/
    bandit_records.parquet
    foraging_records.parquet
    solvability_records.parquet
```

or CSV if parquet dependencies are undesirable.

Each state should include:

```text
task
episode_id
pair_id
state_id
round

semantic_choice
continue

p_continue
p_disengage
persistence_logit

all observable current-state variables
all observable history variables

oracle-only variables, clearly prefixed or tagged
split
```

Do not duplicate full conversation strings unless necessary.

Create a manifest documenting every variable.

---

# 28. Feature schema

Implement an explicit registry such as:

```python
FEATURE_SCHEMA = {
    "bandit": {
        "observable": [...],
        "oracle": [...]
    },
    "foraging": {
        "observable": [...],
        "oracle": [...]
    },
    "solvability": {
        "observable": [...],
        "oracle": [...]
    }
}
```

No model should pull arbitrary numeric columns from the dataframe.

This avoids accidental leakage.

---

# 29. Reproducibility

Every run must save:

```text
git commit
run ID
config
random seed
task list
model list
split hashes
input artifact hashes
feature schema
hyperparameter grid
selected hyperparameters
package/environment metadata if practical
```

Use existing runtime/provenance infrastructure where possible.

---

# 30. Configuration

Create:

```text
config/computational_model_zoo.yaml
```

Suggested skeleton:

```yaml
protocol_version: persistence_computational_model_zoo_v1
analysis_role: exploratory_model_discovery

seed: 95026

tasks:
  - bandit
  - foraging
  - solvability

targets:
  primary: persistence_logit
  secondary: sampled_continue

information_sets:
  primary: observable
  secondary:
    - oracle

weighting:
  episode_balanced: true
  task_balanced: true

models:
  intercept: true
  time: true
  previous_outcome: true
  streak: true
  choice_inertia: true
  finite_history: true
  rescorla_wagner: true
  bayesian: true
  value_history_hybrid: true
  termination_advantage: true
  sticky_termination: true
  decomposed_meta_control: true
  mvt_like: true
  disengagement_accumulator: true
  latent_commitment: true
  generic_latent_value: true
  flexible_linear: true
  mlp: true
  gru: true

gru:
  hidden_sizes: [32, 64]
  learning_rates: [0.001, 0.0003]
  dropout: [0.0, 0.1]
  max_epochs: 100
  early_stopping_patience: 10

bootstrap:
  development_samples: 2000
  final_samples: 10000

model_recovery:
  enabled: true
```

---

# 31. Required outputs

Run output:

```text
artifacts/computational_modeling/<run_id>/
```

Must include:

```text
config.yaml
run_metadata.json

dataset_manifest.json
feature_schema.json

model_metrics.csv
taskwise_metrics.csv
normalized_performance.csv
model_comparisons.csv
bootstrap_intervals.csv

selected_hyperparameters.json

latent_states/
    <model>_states.csv

model_recovery/
    recovery_matrix.csv
    recovery_summary.json

figures/
    persistence_r2.png
    choice_log_loss.png
    fraction_gru_explained.png
    taskwise_performance.png
    recovery_matrix.png

report.md
```

---

# 32. Latent-state output requirement

For every interpretable dynamic model save state-level latent variables.

Example:

```text
episode_id
state_id
task
round

observed_persistence_logit
predicted_persistence_logit

termination_advantage
commitment_state
choice_kernel
estimated_continue_value
estimated_outside_value
cost_component
progress_component
```

Only save columns appropriate to that model.

This artifact is what we will later align with the L21/L22 neural representation.

---

# 33. Primary report

`report.md` should automatically answer:

### Behavioral predictability

* How much of \(D_t\) can the observable GRU explain?
* How stochastic are sampled actions relative to the oracle policy?

### Best interpretable model

* Which interpretable model best predicts held-out persistence logits?
* What fraction of GRU performance does it capture?
* Is its advantage statistically reliable?

### Recurrence

Compare:

```text
MLP vs GRU
finite-history vs recurrent latent model
```

Does persistent hidden state matter?

### Value versus commitment/history

Compare:

```text
RW / Bayesian
history heuristic
generic latent value
commitment
sticky termination
```

### Cross-task structure

* Does one architecture work well across all three tasks?
* Are common dynamics sufficient with task-specific input mappings?
* Does a fully shared model work?

### Identifiability

* Which candidate models can synthetic recovery distinguish?
* Which remain confusable?

### Conclusion

Use appropriately limited wording, e.g.:

> “The strongest behavioral account was a sticky termination model that captured 87% of the observable GRU's held-out persistence-logit variance.”

Not:

> “The LLM implements sticky termination.”

The latter requires neural/mechanistic evidence.

---

# 34. Scientific decision rule for moving forward

The purpose of this stage is to identify variables worth testing neurally.

Proceed to computational-variable ↔ L21/L22 analysis if:

### Criterion 1

At least one interpretable computational model substantially outperforms simple:

```text
time
recent outcome
choice inertia
```

baselines.

### Criterion 2

It captures a meaningful fraction of flexible GRU performance.

Suggested descriptive benchmark:

$$
\geq 70\%
$$

of GRU held-out \(R^2\).

This is not a statistical significance threshold.

### Criterion 3

Synthetic recovery indicates the winning architecture is distinguishable from major alternatives.

### Criterion 4

The model performs meaningfully in multiple tasks, rather than winning solely because of one task.

If these fail, **do not force a mechanistic interpretation**.

Instead report:

> Behavioral persistence is predictable but the current task suite does not uniquely identify an interpretable computational architecture.

That is a useful result.

---

# 35. Explicit non-goals

Do not in this PRD:

* retrain the LLM;
* collect new LLM trajectories unless required fields are genuinely absent;
* train new persistence probes;
* steer activations;
* patch activations;
* claim the candidate L21/L22 representation implements a computational model;
* search arbitrary neural features for model fit;
* optimize an enormous neural predictor;
* use test data for model discovery;
* treat environment-private state as model-observable;
* treat sampled action accuracy as the sole behavioral target.

---

# 36. Implementation order

The coding agent should proceed in this order.

### Phase 0 — regression

Run the existing test suite.

Record baseline failures before changing code.

---

### Phase 1 — records-only data layer

Export and validate Bandit, Foraging, and Solvability behavioral records.

Implement observable/oracle feature schemas.

**Stop if necessary fields are missing.**

Do not silently invent them.

---

### Phase 2 — unify existing interpretable models

Move/reuse:

* Bandit heuristic;
* RW;
* Bayesian;
* choice inertia;
* latent commitment;
* generic latent value.

Confirm reproduced metrics from existing analyses to numerical tolerance.

This is an important regression check.

---

### Phase 3 — add new interpretable models

Implement:

* finite-history;
* termination advantage;
* sticky termination;
* decomposed meta-control;
* disengagement accumulator;
* MVT-like Foraging model.

---

### Phase 4 — flexible ceilings

Implement:

* flexible linear;
* MLP;
* GRU;
* oracle choice ceiling.

---

### Phase 5 — synthetic recovery

Run model-confusion analysis before interpreting real data.

---

### Phase 6 — held-out model zoo

Freeze:

* features;
* candidate models;
* hyperparameters;
* evaluation code.

Then run final held-out evaluation.

---

### Phase 7 — report and latent export

Generate tables, figures, report, and model-state artifacts.

Do not proceed to neural analysis automatically.

---

# 37. Cluster/runtime requirements

GPU is only needed for MLP/GRU training, and even these should be small.

The core interpretable analysis should run CPU-only.

Avoid loading activation tensors into GPU memory during this project.

Behavioral records should be exported once so computational-model runs do not repeatedly deserialize the full activation banks.

---

# 38. Coding-agent instructions

Treat the existing repository as authoritative.

Before writing code:

1. inspect the existing implementations;
2. identify reusable functions;
3. inspect actual field names in each task record;
4. inspect persisted splits;
5. write RED tests;
6. only then implement.

Do not infer a field exists from this PRD if it is absent from the repository. If a required construct cannot be recovered from existing data, document the gap rather than silently substituting something else.

Prefer small reusable pure functions over monolithic scripts.

Do not overwrite prior artifacts.

Do not alter previously reported Track A/B results.

---

# 39. First deliverable from the coding agent

Before launching the full model zoo, the agent should report:

```text
1. files changed
2. tests added
3. existing model results successfully reproduced
4. behavioral state counts per task
5. episode counts per task
6. observable feature list per task
7. oracle-only feature list per task
8. missing-data issues
9. GRU input tensor dimensions
10. exact command proposed for the full run
```

I would review that checkpoint before spending compute on the full analysis.

---

# 40. Expected scientific endpoint

A successful analysis should leave us with something like:

```text
Best flexible model:
GRU, held-out R² = .XX

Best interpretable model:
Sticky termination / latent commitment / etc.
held-out R² = .YY
fraction of GRU = ZZ%

Architecture:
shared recurrent persistence state
+ task-specific evidence integration

Key inferred variables:
commitment state w_t
termination advantage A_t
choice kernel K_t
```

Those variables then become targets for the subsequent neural analysis:

$$
\text{L21/L22 persistence subspace}
\longleftrightarrow
\{w_t,A_t,K_t,\ldots\}.
$$

That is the point where the project moves from **“there is some persistence-related geometry”** to a potential **computational-mechanistic account of what that geometry represents**.
