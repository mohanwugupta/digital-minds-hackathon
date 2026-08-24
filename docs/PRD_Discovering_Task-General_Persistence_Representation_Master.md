# PRD: Discovering a Task-General Persistence Representation

## 1. Objective

The next phase should determine whether Qwen3.5-4B contains a task-general internal representation specifically related to **continued goal engagement**, rather than the generic decision/readout signal identified in Track B.

Track B showed that a shared Bandit + Foraging direction could strongly predict held-out Solvability behavior, but that same direction also predicted arbitrary binary decisions and rule-determined terminality. The existing absolute-choice target therefore does not isolate persistence specifically.

The project should now pursue two complementary hypotheses:

1. **Shared persistence-pressure representation:** different manipulations that make continued pursuit more or less warranted induce a common internal representational change.
2. **Shared latent policy-commitment state:** the model maintains a continuously varying internal state reflecting how strongly it is committed to continued pursuit versus disengagement.

The central scientific question is:

> **Does the model contain a task-general representation or low-dimensional computation that tracks continued goal engagement across different tasks, causes, and surface action formats, above and beyond generic value, binary choice, terminality, and token geometry?**

This phase is primarily an **exploratory discovery phase** using existing data and infrastructure. Any successful candidate must later be frozen and tested confirmatorily on an untouched fourth task.

---

## 2. Scientific motivation

The original persistence probe was trained on absolute behavioral readouts such as:

\[
D_{\text{bandit}}
=
\log \left(e^{z_A}+e^{z_B}\right)-z_C
\]

with analogous semantic choice contrasts in Foraging and Solvability.

This successfully identified a late decision-aligned signal, but Track B showed that cross-task transfer of this representation was not persistence-specific. The same shared direction also generalized to non-persistence decision controls.

Therefore:

\[
\text{absolute choice representation}
\neq
\text{persistence-specific representation}.
\]

The next search should instead use experimental dissociations that separate the construct from its surface realization.

The first methodological model is **matched causal contrasts**: hold nuisance structure fixed while changing whether continued pursuit is more or less warranted.

The second methodological model is **latent policy-state inference**: infer an underlying commitment state from sequential behavior and then ask whether internal activity tracks that state and its transitions rather than merely the next action.

These approaches are complementary:

\[
\boxed{\text{persistence-relevant inputs}}
\rightarrow
\boxed{\text{internal persistence computation}}
\rightarrow
\boxed{\text{latent policy commitment}}
\rightarrow
\boxed{\text{task-specific behavior}}.
\]

---

## 3. Core construct

Persistence should be distinguished at three levels.

### 3.1 Persistence-relevant inputs

Examples include:

- continuation value;
- stopping/outside-option value;
- continuation cost;
- perceived progress;
- solvability;
- recent outcomes;
- elapsed effort;
- uncertainty.

These are candidate **causes or inputs** to persistence.

### 3.2 Latent policy commitment

The hypothesized construct of interest:

\[
w_t.
\]

Conceptually:

> **How strongly is the model currently committed to continued pursuit of the current objective rather than disengagement?**

This should not be defined solely by the current output choice.

### 3.3 Observable persistence behavior

Examples include:

- A/B rather than C;
- STAY rather than LEAVE;
- TRY AGAIN rather than GIVE UP;
- number of future attempts allocated;
- future run length;
- stopping hazard.

These are behavioral **readouts** of persistence.

---

## 4. Primary hypotheses

### Hypothesis A — Shared persistence-pressure representation

Different persistence-promoting and persistence-discouraging manipulations induce a common representational change:

\[
\Delta h_{\text{persist}}.
\]

This representation may be:

1. a shared linear direction;
2. a small low-dimensional subspace;
3. or a shared layerwise transformation.

The initial search should compare:

\[
k=1,\;2,\;4
\]

and both:

\[
h_l
\]

and:

\[
\delta h_l=h_{l+1}-h_l.
\]

Do **not** initially search rank \(>4\).

### Hypothesis B — Shared latent policy-commitment state

Persistence may instead be represented as a continuous state:

\[
w_t
\]

that integrates persistence-relevant evidence over time and governs continued engagement.

Conceptually:

\[
\text{value / cost / progress / solvability / history}
\rightarrow
w_t
\rightarrow
\text{task-specific continuation/disengagement behavior}.
\]

A useful \(w_t\) must explain something beyond immediate choice, generic value, or simple behavioral inertia.

---

## 5. Critical implementation requirement: leverage the existing codebase

This work must **extend the existing repository**, not create a parallel experimental framework.

Before adding new code, audit and reuse existing components for:

- Qwen3.5-4B model loading;
- Hugging Face chat templates;
- hidden-state capture;
- existing Bandit activation bank;
- existing STOP × CONTINUE factorial;
- existing Foraging activation bank;
- existing Solvability activation bank;
- all-layer activation storage;
- task-specific semantic action mapping;
- label counterbalancing;
- episode-safe and pair-safe splits;
- ridge probe fitting;
- activation normalization;
- state-fixed-effects analysis;
- clustered confidence intervals;
- episode/pair bootstrapping;
- matched random directions;
- steering calibration;
- manifests and run metadata;
- Slurm execution;
- existing Track A and Track B analyses.

Do not duplicate functionality unless the current abstraction cannot support the new analysis.

All existing Track A and Track B results must continue to reproduce after modification.

---

## 6. Scope

### In scope

1. Audit existing task manipulations and activation coverage.
2. Build matched persistence contrasts.
3. Run behavioral validity gates.
4. Search all 32 layers.
5. Compare static and layer-to-layer transformation features.
6. Compare 1D, 2D, and 4D representations.
7. Test cross-manipulation generalization.
8. Test cross-task generalization.
9. Test nuisance specificity.
10. Add one minimal generic-value control if necessary.
11. Fit and validate a minimal latent commitment-state model.
12. Test whether latent commitment predicts future behavior beyond current choice.
13. Search for internal representations of the latent state.
14. Integrate the contrast-derived and latent-state-derived candidates.
15. Decide whether any candidate merits causal testing and a fresh Task 4.

### Explicitly out of scope for now

Do not yet implement:

- Distributed Alignment Search;
- forked-futures architecture discovery;
- sparse autoencoder discovery;
- head-level circuit localization;
- large-scale activation patching;
- new model families;
- large new task suites;
- rank \(>4\) shared subspaces.

These become conditional follow-ups only if the current phase identifies a credible candidate.

---

# TRACK C0 — MATCHED PERSISTENCE-CONTRAST DISCOVERY

## 7. Build a persistence-contrast bank

For every usable manipulation, construct paired observations:

\[
(s^{-},s^{+})
\]

where \(s^{+}\) makes continued pursuit more attractive or warranted than \(s^{-}\).

For layer \(l\):

\[
\Delta h_l = h_l(s^{+})-h_l(s^{-}).
\]

The sign convention must always be:

\[
+\Delta = \text{more persistence-promoting}.
\]

The central search question is:

> **What internal changes recur when different manipulations make continued goal pursuit more warranted?**

---

## 8. Bandit contrasts

Use the existing STOP × CONTINUE factorial.

### 8.1 CONTINUE incentive

Higher CONTINUE bonus versus lower CONTINUE bonus while holding fixed:

- underlying state;
- STOP payoff;
- interaction history.

### 8.2 STOP/outside-option incentive

Lower STOP attractiveness versus higher STOP attractiveness while holding fixed:

- underlying state;
- CONTINUE bonus;
- interaction history.

Both must be oriented so positive \(\Delta h\) means greater pressure to persist.

Do not collapse these manipulations immediately. Generalization between them is itself a test of abstraction.

---

## 9. Foraging contrasts

Use the existing Foraging design.

### 9.1 Search-cost contrast

Lower search cost versus higher search cost, matched on all remaining state variables wherever possible.

### 9.2 Outside-option contrast

Less attractive leaving option versus more attractive leaving option.

Orient both so:

\[
+\Delta h=\text{greater warrant for STAY}.
\]

Label mapping must be held fixed within each pair or averaged over exact counterbalanced semantic states.

---

## 10. Solvability contrast

Audit the existing Solvability dataset before defining the contrast.

Identify the manipulation that most directly changes:

> **evidence that continuing work on the current goal is worthwhile.**

Prefer:

- solvable versus impossible;
- evidence of progress versus futility;
- or an equivalent task-specific continuation-prospect manipulation.

Document:

1. available experimental factors;
2. which factor qualifies as persistence-relevant;
3. whether valid matched pairs can be formed;
4. which nuisance variables remain unmatched.

If no valid matched contrast exists, mark the existing data as unsuitable for this analysis and implement the **smallest possible matched counterfactual replay** needed to create one.

Do not invent a target that the task does not support.

---

## 11. Behavioral validity gate

A contrast is eligible for representational analysis only if it produces the expected behavioral effect.

For manipulation \(m\):

\[
\Delta D_m = D(s^{+})-D(s^{-}).
\]

Require:

\[
E[\Delta D_m]>0
\]

with clustered uncertainty supporting the expected direction.

Always report effect magnitude.

A manipulation failing the behavioral gate must not be used as evidence for or against a persistence representation.

---

## 12. Nuisance-control bank

### Control A — Label identity

Construct exact matched-label contrasts where semantic state is unchanged but response-token mapping changes.

Question:

> Does the candidate track token/output geometry?

### Control B — Arbitrary binary choice

Reuse the existing non-persistence binary-choice control.

Question:

> Is the candidate a generic binary-decision signal?

### Control C — Rule-determined terminality

Reuse the existing externally specified CONTINUE/END control.

Question:

> Is the candidate merely a generic terminal-versus-nonterminal representation?

### Control D — Generic one-shot value comparison

Add one minimal new control only if no equivalent dataset exists.

Requirements:

- two one-shot alternatives;
- differing values;
- no ongoing goal;
- no persistence/termination semantics;
- counterbalanced labels.

Question:

> Is the candidate merely a generic relative-value signal?

---

## 13. Primary representational search

For each layer \(l\), build a matrix of persistence contrast vectors:

\[
\Delta H_l.
\]

Fit:

### Model 1 — Rank-1 direction

\[
d_l=E[\Delta h_l].
\]

### Model 2 — Rank-2 shared subspace

Use PCA/SVD or equivalent training-only low-rank decomposition.

### Model 3 — Rank-4 shared subspace

Same procedure with \(k=4\).

No \(k>4\) search during the initial phase.

---

## 14. Task and manipulation balancing

Prevent high-state-count tasks or manipulation families from dominating.

Use:

- equal aggregate task weighting;
- equal aggregate manipulation-family weighting.

Do not simply weight every state equally across unequal datasets.

---

## 15. Cross-manipulation generalization

This is a primary test.

Perform leave-one-manipulation-family-out evaluation.

Examples:

\[
\text{Bandit CONTINUE + Foraging cost}
\rightarrow
\text{Foraging outside option}
\]

or:

\[
\text{Bandit incentive + Foraging outside option}
\rightarrow
\text{Solvability/progress}.
\]

Scientific question:

> **Does a representation discovered from one reason to persist generalize to a qualitatively different reason to persist?**

This is required to distinguish persistence from:

- reward;
- cost;
- outside-option value;
- progress;
- solvability;
- or any other individual manipulation.

---

## 16. Cross-task generalization

Run leave-one-task-out analyses:

\[
B+F\rightarrow S
\]

\[
B+S\rightarrow F
\]

\[
F+S\rightarrow B.
\]

These are exploratory because all three datasets have already influenced method development.

For each fold:

1. fit only on source-task training data;
2. select layer/rank only on source-task validation data;
3. freeze the representation;
4. evaluate held-out task contrasts.

No held-out-task optimization is allowed.

---

## 17. Static-state versus depth-transformation search

Track A showed a strongly non-monotonic trajectory across model depth.

Repeat the entire contrast search using:

\[
\delta h_l=h_{l+1}-h_l
\]

and:

\[
\Delta \delta h_l
=
\delta h_l(s^+)-\delta h_l(s^-).
\]

Compare:

### Static representation

\[
\Delta h_l
\]

versus

### Layerwise computation

\[
\Delta\delta h_l.
\]

Scientific question:

> **Is the generalizable object a state the model occupies or a transformation the model performs?**

Do not fit more elaborate trajectory models yet.

---

## 18. Layer search

Evaluate all eligible layers.

Do not privilege layer 31.

Track A gives exploratory reasons to inspect:

- layer 15;
- layers 18–22;
- layer 23 onward;

but all layers must be evaluated under the same procedure.

Layer selection must use discovery/validation data only.

---

## 19. Candidate scoring

For every candidate \((\text{layer},\text{feature type},k)\), report separately:

- persistence sensitivity;
- cross-manipulation transfer;
- cross-task transfer;
- label sensitivity;
- arbitrary-choice sensitivity;
- terminality sensitivity;
- generic-value sensitivity.

Do not collapse these into one opaque score.

The ideal candidate has:

\[
\text{high persistence transfer}
\]

and:

\[
\text{low nuisance transfer}.
\]

---

## 20. Persistence-specific candidate criterion

A candidate is scientifically interesting only if:

1. persistence-promoting manipulations project in a consistent direction;
2. transfer occurs across multiple manipulation families;
3. transfer occurs across multiple tasks;
4. label sensitivity is substantially weaker than persistence sensitivity;
5. arbitrary binary-choice sensitivity is substantially weaker;
6. terminality sensitivity is substantially weaker;
7. generic one-shot value sensitivity is substantially weaker.

Do not repeatedly tune thresholds until a candidate passes.

---

## 21. Important alternative hypothesis

Explicitly allow:

\[
\text{persistence behavior}
=
\text{generic relative-value / decision computation}
\]

rather than a dedicated persistence variable.

If every cross-task candidate that predicts persistence also predicts generic value or decision contrasts, report this as evidence favoring a **domain-general decision/value architecture**.

Do not continue indefinitely searching for a persistence-specific vector.

---

## 22. Exploratory nuisance residualization

Only after primary unresidualized analyses are complete, optionally project out nuisance spaces such as:

- token/label direction;
- arbitrary binary-choice subspace;
- terminality direction;
- generic-value direction.

Repeat the search and label outputs:

`exploratory_nuisance_residualization`

Do not treat these results as confirmatory evidence.

---

# TRACK C1 — LATENT POLICY-COMMITMENT SEARCH

## 23. Scientific question

> **Can sequential behavior across the existing tasks be explained by a low-dimensional latent commitment state that integrates persistence-relevant inputs and predicts future engagement beyond the immediate choice?**

This track treats persistence as a **policy state**, not a next-token action.

---

## 24. Minimal initial latent-state model

Start with the smallest interpretable one-dimensional state-space model.

For example:

\[
w_t
=
\rho w_{t-1}
+
\beta^\top X_t
+
\epsilon_t
\]

where \(X_t\) contains task-appropriate persistence inputs such as:

- relative continuation incentive;
- outside-option value;
- search cost;
- progress/solvability evidence;
- recent outcomes;
- elapsed effort.

Behavioral emission:

\[
P(\text{continue}_t)
=
\sigma(
\alpha_{\text{task}}
+
\lambda_{\text{task}}w_t
).
\]

Allow task-specific:

\[
\alpha_{\text{task}},\lambda_{\text{task}}
\]

while orienting \(w_t\) consistently:

\[
w_t\uparrow
\Rightarrow
\text{greater continued engagement}.
\]

Do not initially require input coefficients to be identical across tasks.

---

## 25. Critical validity condition

The latent state is not useful if it is merely:

- a smoothed current choice logit;
- choice inertia;
- generic latent value.

Therefore require evidence that:

\[
w_t
\]

contains information beyond:

\[
D_t=\text{current continue-versus-disengage logit}.
\]

---

## 26. Future-behavior validation

Test whether \(w_t\) predicts:

- remaining episode length;
- probability of continuing for at least \(k\) additional decisions;
- stopping hazard over the next \(k\) states;
- future run length;
- later disengagement.

Conceptually:

\[
\text{future persistence}
=
\beta_0+\beta_1D_t+\beta_2w_t.
\]

A useful latent commitment state should contribute:

\[
\beta_2\neq0
\]

beyond the immediate choice variable.

---

## 27. Optional repeated-rollout commitment estimate

If existing trajectories are insufficient to identify \(w_t\), use limited repeated rollouts from a subset of stored states.

Estimate quantities such as:

\[
P(\text{persist}\geq2\text{ more steps})
\]

\[
P(\text{persist}\geq5\text{ more steps})
\]

\[
E[\text{remaining decisions}]
\]

or a full stopping-hazard curve.

This creates a future behavioral commitment signature less directly tied to the immediate output token.

Only incur this new inference cost if required by model identifiability.

---

## 28. Synthetic model-recovery gate

Before interpreting an inferred latent state, validate the pipeline on synthetic data.

Generate trajectories from known latent states:

\[
w_t^*
\]

with known:

- state persistence;
- input effects;
- noise;
- task-specific emission mappings.

Fit the full model blind to the ground truth.

Test recovery of:

\[
\operatorname{corr}(\hat w_t,w_t^*)
\]

as well as:

- temporal ordering;
- transition timing;
- sign/orientation;
- \(\rho\);
- major input coefficients.

Do not interpret real-data \(w_t\) if recovery is poor.

---

## 29. Architecture confusion tests

Generate and fit at minimum:

### Model 0 — Immediate decision only

\[
P(C_t)=f(X_t).
\]

### Model 1 — Choice-history/inertia model

\[
P(C_t)=f(X_t,C_{t-1},C_{t-2},...).
\]

### Model 2 — Latent commitment state

\[
w_t=\rho w_{t-1}+\beta^\top X_t+\epsilon_t
\]

\[
P(C_t)=g(w_t).
\]

### Model 3 — Generic latent value state

A persistent state explicitly tied to relative option value with no independent commitment process.

The pipeline must demonstrate that it can distinguish these major architectures in synthetic data before real-data interpretation.

---

## 30. Real-data model comparison

Compare the same candidate behavioral models using held-out prediction or an appropriate information criterion.

Scientific question:

> **Does a latent commitment state improve explanation of persistence beyond current values, task variables, and simple behavioral autocorrelation?**

If not, do not force a latent-state interpretation.

---

## 31. Representational search for \(w_t\)

If the latent-state model passes recovery and behavioral validity gates, search for its internal representation.

For every layer:

\[
h_l\rightarrow\hat w_t.
\]

Evaluate:

### Within-task decoding

Can \(\hat w_t\) be decoded within each task?

### Cross-task decoding

Can a representation trained on:

\[
B+F
\]

predict:

\[
w_t^S
\]

without retraining?

Repeat leave-one-task-out.

### Immediate-choice residual test

Test representation of:

\[
w_t
\]

after controlling for:

\[
D_t.
\]

The primary target is the component of commitment that is **not reducible to current choice geometry**.

---

## 32. Behavioral-time transition analysis

Define transitions toward disengagement using validated latent-state or behavioral criteria.

Examples:

\[
w_t>\theta_H
\rightarrow
w_{t+k}<\theta_L
\]

or:

\[
\text{sustained pursuit}\rightarrow\text{STOP}.
\]

Align internal states around these events.

Ask:

> **Is there a common representational trajectory preceding disengagement across tasks?**

Keep two forms of dynamics conceptually distinct:

### Across transformer depth

\[
l\rightarrow l+1
\]

### Across behavioral time

\[
t\rightarrow t+1.
\]

Do not collapse them.

---

## 33. Residual meta-control test

First model behavior from measured persistence-relevant causes:

\[
\text{persistence}
=
f(
Q_{\text{continue}},
Q_{\text{stop}},
\text{history},
t,
\text{progress},
\text{solvability},
...
).
\]

Then test whether a candidate representation contributes additional prediction of:

- latent commitment;
- future run length;
- stopping hazard;
- impending disengagement.

Conceptually:

\[
\text{future disengagement}
=
f(X_t)+\gamma z_t.
\]

A candidate meta-control signal should retain:

\[
\gamma\neq0
\]

after measured task/value/context variables are included.

This helps distinguish:

\[
\boxed{\text{value/context representation}}
\]

from:

\[
\boxed{\text{policy commitment/meta-control representation}}.
\]

---

## 34. Candidate taxonomy

Every candidate representation should be classified as one of:

### A. Input/value representation

Tracks reward, cost, outside option, solvability, or related evidence but does not independently predict commitment.

### B. Immediate decision representation

Tracks current CONTINUE/STOP preference and nuisance binary-choice controls.

### C. Latent policy-state representation

Tracks future engagement, run length, or stopping hazard and generalizes across tasks beyond immediate choice.

### D. Meta-control / transition representation

Specifically predicts or causally influences changes in \(w_t\) or impending disengagement.

---

# TRACK C2 — INTEGRATION

## 35. Convergence between contrast and latent-state approaches

If Track C0 and C1 each identify candidate representations, test whether they converge.

Let:

\[
z_{\Delta P}
\]

be the contrast-derived representation and:

\[
z_w
\]

the latent-state-derived representation.

Test:

- cosine/subspace overlap;
- cross-decoding;
- whether persistence manipulations move \(z_w\);
- whether \(z_{\Delta P}\) predicts \(w_t\);
- whether both localize to similar layers or depth transformations.

A particularly compelling chain would be:

\[
\text{persistence-promoting manipulation}
\]

\[
\Downarrow
\]

\[
\text{contrast-derived internal shift}
\]

\[
\Downarrow
\]

\[
w_t\uparrow
\]

\[
\Downarrow
\]

\[
\text{future persistence increases}.
\]

This would connect experimental cause, internal representation, latent policy state, and behavior.

---

## 36. Causal gate

Do **not** automatically steer the highest-performing exploratory representation.

A candidate becomes eligible for causal intervention only if it satisfies the relevant specificity requirements.

At minimum:

1. generalizes across multiple persistence manipulation families;
2. transfers across tasks;
3. survives label controls;
4. clearly outperforms arbitrary-decision and terminality controls;
5. is not reducible to generic one-shot value;
6. if latent-state-based, predicts future persistence beyond immediate choice.

If no candidate passes:

> **stop the causal pipeline.**

Do not steer a nonspecific direction because it has a large effect.

---

## 37. Conditional causal follow-up

If a candidate passes the gate, perform causal tests first on existing tasks.

Prefer:

- matched activation patching between \(P^+\) and \(P^-\) states;
- subspace patching;
- carefully calibrated steering.

For \(k>1\), intervene on the full subspace rather than collapsing it post hoc to one arbitrary direction.

Primary causal question:

> **Does manipulating the candidate representation selectively alter future persistence/commitment without producing comparable changes in nuisance decisions?**

For latent-state candidates, test the predicted chain:

\[
\text{intervention}
\rightarrow
w_t
\rightarrow
\text{future persistence}.
\]

---

## 38. Future confirmatory Task 4

Any candidate derived from Bandit, Foraging, and Solvability is exploratory because all three tasks have influenced method development.

A successful candidate must ultimately be frozen and tested on an untouched fourth task.

Prefer a task with a response format that differs from binary continue/stop.

Example:

> **Allocate additional future effort**

with choices such as:

\[
0,\;1,\;2,\;4
\]

additional attempts.

A fresh Task 4 should test:

1. zero-shot persistence-contrast transfer;
2. zero-shot latent-state prediction where applicable;
3. nuisance specificity;
4. calibrated causal intervention.

Do not design or run Task 4 until the present discovery phase identifies a credible candidate.

---

# TDD AND ENGINEERING REQUIREMENTS

## 39. TDD requirement

All implementation must follow:

# RED → GREEN → REFACTOR

For every meaningful unit of functionality:

### RED

1. Write the test first.
2. Run it.
3. Confirm it fails for the expected reason.
4. Record the failure in the persistent project scratchpad.

### GREEN

5. Implement the minimum code needed.
6. Run the new test.
7. Confirm it passes.
8. Run the full regression suite.

### REFACTOR

9. Refactor only after GREEN.
10. Rerun all relevant unit, integration, and baseline regression tests.
11. Do not merge code while tests are failing.

Do not back-fill tests after core analysis implementation.

---

## 40. Required RED → GREEN tests: contrast pipeline

### Contrast construction

**RED:** provide mismatched pairs that differ in state variables meant to be held fixed.

**GREEN:** validator rejects them.

### Contrast orientation

**RED:** reverse persistence-promoting and persistence-discouraging conditions.

**GREEN:** sign-check detects the reversal.

### Label invariance

**RED:** pass label-only pairs as persistence contrasts.

**GREEN:** validator classifies them as nuisance contrasts.

### Task balancing

**RED:** one task has 10× more observations.

**GREEN:** equal aggregate task weight is preserved.

### Manipulation balancing

**RED:** one manipulation dominates observation count.

**GREEN:** equal manipulation-family weighting is preserved.

### Leave-one-task-out leakage

**RED:** held-out task enters fitting or layer selection.

**GREEN:** pipeline raises an error.

### Leave-one-manipulation-out leakage

**RED:** held-out manipulation enters discovery.

**GREEN:** pipeline raises an error.

### Rank constraint

**RED:** request \(k=8\).

**GREEN:** initial search config rejects it.

### Displacement features

**RED:** use incorrect layer indexing.

**GREEN:** synthetic activation sequence recovers expected:

\[
h_{l+1}-h_l.
\]

### Low-rank recovery

**RED:** synthetic dataset contains known 2D shared signal plus nuisance.

**GREEN:** rank-2 search recovers the shared signal better than rank-1 and nuisance controls.

### Nuisance specificity

**RED:** synthetic candidate responds only to label swaps.

**GREEN:** candidate is not classified as persistence-specific.

### No-signal case

**RED:** synthetic data contain only generic decision structure.

**GREEN:** pipeline returns:

`no_persistence_specific_candidate`

rather than forcing selection.

---

## 41. Required RED → GREEN tests: latent-state pipeline

### Synthetic state recovery

**RED:** fit an intentionally misspecified model to known \(w_t^*\).

**GREEN:** correct model recovers latent ordering and trajectory within predefined tolerance.

### No-latent-state control

**RED:** generate actions directly from current task variables.

**GREEN:** latent-state model is not spuriously strongly preferred.

### Choice-inertia control

**RED:** generate trajectories using only autoregressive choice history.

**GREEN:** pipeline identifies history/inertia rather than latent commitment.

### Generic-value control

**RED:** generate a persistent latent relative-value process with no commitment variable.

**GREEN:** pipeline does not classify it as persistence-specific.

### Future-prediction test

**RED:** synthetic \(w_t\) predicts only current action.

**GREEN:** future-behavior validation fails.

### Task-specific emission recovery

**RED:** use different action mappings/scales across simulated tasks.

**GREEN:** shared latent ordering is recovered despite task-specific emissions.

---

## 42. Regression protection

Before new development, verify that existing analyses still reproduce:

- Track A factorial dataset dimensions;
- Track A layer-31 STOP and CONTINUE effects;
- Track A non-monotonic trajectory;
- Track B primary Solvability transfer;
- Track B arbitrary-choice control;
- Track B terminality control;
- existing split assignments;
- pair identifiers;
- stored model/output hashes where applicable.

New generalized code must not silently alter prior results.

---

## 43. Persistent project scratchpad

Maintain a repository scratchpad containing:

- scientific question;
- hypothesis;
- code/config changed;
- RED test written;
- expected RED failure;
- observed RED failure;
- GREEN implementation;
- tests run;
- datasets accessed;
- exploratory decisions;
- unexpected findings;
- decision about escalation.

This is required because layer, rank, contrast, and model searches create substantial researcher degrees of freedom.

---

## 44. Suggested code organization

Prefer extensions of existing modules.

```text
analysis/
    persistence_contrasts.py
    persistence_cross_manipulation.py
    persistence_cross_task.py
    persistence_subspace.py
    persistence_displacements.py
    persistence_specificity.py
    persistence_latent_state.py
    persistence_future_behavior.py
    persistence_transition_analysis.py
    persistence_integration.py
    persistence_candidate_report.py

experiments/
    generic_value_control.py

config/
    persistence_discovery.yaml
    persistence_latent_state.yaml

tests/
    test_persistence_contrasts.py
    test_cross_manipulation_isolation.py
    test_cross_task_isolation.py
    test_subspace_recovery.py
    test_displacement_features.py
    test_persistence_specificity.py
    test_latent_state_recovery.py
    test_latent_model_confusion.py
    test_future_behavior_prediction.py
```

Adapt names to repository conventions rather than duplicating existing utilities.

---

# OUTPUTS AND DECISION RULES

## 45. Primary outputs

### A. Contrast inventory

Machine-readable table containing:

- task;
- manipulation;
- state pair;
- persistence orientation;
- behavioral effect;
- matched variables;
- label mapping;
- episode/pair cluster.

### B. Layerwise transfer maps

For every layer:

- within-manipulation performance;
- cross-manipulation performance;
- cross-task performance;
- nuisance-control performance.

### C. Rank comparison

Compare:

\[
k=1,2,4.
\]

### D. Static-versus-displacement comparison

Compare:

\[
\Delta h_l
\]

with:

\[
\Delta\delta h_l.
\]

### E. Specificity matrix

Rows:

- persistence manipulations;
- label;
- arbitrary choice;
- terminality;
- generic value.

Columns:

- candidate layer/rank/feature type.

### F. Latent-state report

Include:

- model-recovery performance;
- model-comparison results;
- inferred \(w_t\) diagnostics;
- future-behavior prediction beyond current choice;
- cross-task \(w_t\) decoding;
- transition-aligned analyses.

### G. Integration report

For strongest candidates:

- overlap between contrast and latent-state representations;
- shared layer/subspace evidence;
- whether persistence manipulations shift \(w_t\);
- whether candidate representation predicts future persistence;
- causal-gate status.

---

## 46. Decision tree

### Result A — Task-general, persistence-specific candidate

Persistence contrasts generalize across manipulation and task while nuisance sensitivity remains weak.

**Action:** proceed to causal intervention and prepare untouched Task 4.

### Result B — Rank-2/4 succeeds but rank-1 fails

**Interpretation:** persistence may be multidimensional.

**Action:** retain the minimal successful rank and use causal subspace tests.

### Result C — Displacement succeeds but static representation fails

**Interpretation:** the task-general object may be a shared computation/trajectory rather than a stable representational state.

**Action:** shift mechanistic work toward identified depth transitions.

### Result D — Latent \(w_t\) adds predictive value beyond current choice

**Interpretation:** evidence supports an ongoing policy-commitment state.

**Action:** search for its cross-task internal representation and relation to contrast-derived candidates.

### Result E — Latent model collapses to choice inertia or value

**Interpretation:** no evidence for a distinct commitment state.

**Action:** do not treat inferred \(w_t\) mechanistically.

### Result F — Persistence and generic value remain inseparable

**Interpretation:** persistence may be implemented through generic value comparison.

**Action:** reframe the scientific question around how generic value/control computations govern disengagement.

### Result G — All cross-task candidates remain dominated by label/decision/terminality structure

**Interpretation:** no evidence for a unified task-general persistence representation in this model under the tested paradigms.

**Action:** stop before DAS/forked-futures/circuit localization unless a new independently motivated hypothesis emerges.

---

# EXECUTION ORDER

## 47. Phase 0 — Baseline protection

1. Audit the existing repository.
2. Verify existing Track A and Track B reproduction.
3. Add missing regression tests before changing shared code.

## 48. Phase 1 — Contrast feasibility and construction

4. Audit usable Bandit manipulations.
5. Audit usable Foraging manipulations.
6. Audit Solvability for a valid persistence contrast.
7. Verify activation coverage.
8. Build matched contrast bank.
9. Run behavioral validity gates.
10. Build nuisance contrasts.
11. Add minimal generic-value control only if necessary.

## 49. Phase 2 — Wide-net contrast search

12. Run all-layer rank-1 static search.
13. Run all-layer rank-2 static search.
14. Run all-layer rank-4 static search.
15. Repeat using displacement features.
16. Run leave-one-manipulation-out tests.
17. Run leave-one-task-out tests.
18. Run nuisance specificity battery.
19. Produce candidate ranking.

## 50. Phase 3 — Latent-state feasibility

In parallel where practical:

20. Audit sequential trajectory structure in all tasks.
21. Implement synthetic state-recovery tests.
22. Implement architecture confusion tests.
23. Fit minimal immediate-choice, history, latent-commitment, and generic-value models.
24. Determine whether a latent commitment state adds explanatory value.
25. Test future persistence beyond current choice.

If the latent-state model fails these gates, stop Track C1.

## 51. Phase 4 — Internal latent-state search

Only if Phase 3 succeeds:

26. Decode \(w_t\) across all layers.
27. Test cross-task transfer.
28. Test residual \(w_t\) after controlling current choice.
29. Analyze behavioral-time transitions toward disengagement.
30. Run residual meta-control tests.

## 52. Phase 5 — Integration and gate

31. Compare contrast-derived and latent-state-derived representations.
32. Produce specificity/transfer matrix.
33. Decide whether any candidate passes the causal gate.

## 53. Phase 6 — Conditional escalation

Only if a candidate passes:

34. run targeted causal subspace intervention;
35. freeze candidate and analysis;
36. design untouched Task 4;
37. run confirmatory cross-task and causal validation.

---

# 54. Immediate scientific deliverable

The immediate deliverable is **not** “the persistence vector.”

It is an answer to two linked questions:

> **When we isolate changes in how warranted continued goal pursuit is, do those changes share a common internal representation across different causes and tasks, above and beyond generic value, decision, terminality, and token geometry?**

and:

> **Does the model maintain a latent policy-commitment state that predicts sustained future engagement beyond the immediate choice, and can that state be identified internally across tasks?**

A strong positive result would support a mechanistic architecture like:

\[
\boxed{\text{value / cost / progress / solvability / history}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{shared persistence-relevant computation}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{latent policy commitment }w_t}
\]

\[
\Downarrow
\]

\[
\boxed{\text{task-specific continuation/disengagement behavior}}.
\]

A strong negative result is also scientifically valuable. It would suggest that persistence may not exist as a unified latent variable in this model and may instead emerge from more general value, decision, or control computations.

The project should stop or reframe rather than force a persistence-specific interpretation when the specificity gates fail.
