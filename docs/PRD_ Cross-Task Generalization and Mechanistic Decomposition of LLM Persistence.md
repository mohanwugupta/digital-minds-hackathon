# Product Requirements Document

## Project title

**Cross-Task Generalization and Mechanistic Decomposition of LLM Persistence**

## 1. Objective

The sprint established that Qwen3.5-4B contains a late persistence-aligned representation that closely tracks the model’s CONTINUE-versus-STOP preference and can causally shift that preference when steered. However, because the strongest persistence probe occurs at layer 31 and was trained directly against the output persistence logit, the current evidence does not establish that this direction is an abstract, task-general representation of persistence rather than a task-specific or output-adjacent decision representation.

The next stage of the project should answer two questions in sequence:

1. **Does the identified persistence representation generalize across tasks?**
2. **If so, what upstream information and computations construct it?**

The immediate priority is therefore **not** to launch a large new mechanistic experiment. It is to cheaply test whether the object we are trying to explain generalizes, while simultaneously extracting additional mechanistic information from the existing dataset.

This PRD defines:

- **Track A:** Reanalysis of the existing STOP × CONTINUE factorial to determine how persistence-related information emerges across layers.
- **Track B:** A cross-task generalization checkpoint using a new foraging-style persistence task, including token controls, a non-persistence negative-control task, a within-task ceiling, and causal transfer.
- **Track C:** Conditional deeper mechanistic decomposition if cross-task generalization is supported.
- **Track D:** A later solvability/give-up paradigm as the higher-value test of generalization to persistence on difficult or impossible tasks.

---

# 2. Core scientific question

The long-term target is:

> **What determines whether an LLM continues pursuing a goal or disengages, how are those determinants combined internally, and is the resulting persistence computation shared across tasks?**

Conceptually:

\[
X_1,X_2,\ldots,X_n
\rightarrow
f(X_1,X_2,\ldots,X_n)
\rightarrow
P_{\text{persist}}
\rightarrow
\text{CONTINUE/STOP}
\]

where candidate inputs may include:

- continuation value;
- stopping/outside-option value;
- recent outcomes;
- elapsed effort/time;
- perceived progress;
- uncertainty;
- expected solvability;
- implicit costs of continuing;
- baseline continuation bias.

The sprint primarily characterized the **right-hand side** of this pathway. The follow-up should work backward toward the upstream computation.

---

# 3. Key distinction: three levels of generalization

Cross-task persistence should not be treated as a binary “generalizes / does not generalize” claim.

The project should explicitly distinguish:

### Level 1: Behavioral generalization

**Question**

> Does the same model exhibit meaningful persistence behavior in a structurally different task?

This establishes that persistence is not solely a behavioral property of the original bandit.

### Level 2: Representational generalization

**Question**

> Does a persistence direction identified entirely in the bandit predict continuation preference in a new task without retraining?

This tests whether the model contains a shared internal dimension associated with continuing versus disengaging.

### Level 3: Causal generalization

**Question**

> Does steering the persistence direction identified in the bandit causally alter persistence in another task?

This is substantially stronger evidence for a shared persistence controller or decision variable.

These three questions must be reported separately.

---

# 4. Critical implementation requirement: leverage the existing codebase

**This project must extend the existing sprint repository rather than build a parallel experimental framework.**

Before implementing new functionality, inspect the existing codebase and identify reusable components for:

- Qwen3.5-4B model loading;
- Hugging Face chat-template construction;
- next-token logit extraction;
- A/B/C action handling;
- episode-level train/validation/test splitting;
- activation capture;
- activation-bank storage;
- ridge probe fitting;
- training-set activation standardization;
- layerwise evaluation;
- steering-vector construction;
- steering calibration;
- random-direction generation;
- episode-level bootstrapping;
- clustered regression;
- random seeds;
- Slurm execution;
- dataset integrity checks;
- output manifests;
- plotting and result tables.

**Do not duplicate existing functionality unless the existing abstraction cannot support the new experiment.**

Where possible:

- generalize existing task-runner abstractions rather than creating new runners;
- parameterize action labels rather than hard-code new ones;
- reuse the existing activation and steering pipeline;
- retain existing data schemas so old and new experiments can be analyzed together;
- add new configuration files rather than branching core logic unnecessarily.

The existing bandit analysis must continue to run unchanged after modifications.

---

# 5. Phase 0 — Reproduce and protect the existing baseline

Before any new analysis or experiment, establish regression tests around the current results.

## Scientific question

None. This is an infrastructure requirement.

## Purpose

The new code will modify shared experimental machinery. We need to know immediately if a change accidentally alters:

- prompt construction;
- token IDs;
- model outputs;
- activation locations;
- standardization;
- probe predictions;
- steering;
- random seeds;
- split assignments.

## Required baseline regression checks

Using stored data where possible, verify reproduction of key existing outputs within numerical tolerance, including:

- persistence-probe layer selection;
- future-return probe layer selection;
- held-out persistence-probe performance;
- factorial dataset dimensions;
- STOP and CONTINUE manipulation signs;
- baseline steering output under \(\lambda=0\);
- exact episode split membership.

The goal is not to hard-code every manuscript statistic. It is to catch accidental behavioral or analysis drift.

---

# 6. Track A — Trace the existing incentive effect across layers

This should be run immediately because the required data already exist.

## Specific scientific question

> **When experimentally changing the relative value of CONTINUE and STOP changes behavior, at what point in the network does that manipulation become expressed along the independently defined persistence representation?**

## Relation to the larger question

We already know that changing current incentives strongly changes persistence and that the layer-31 persistence projection closely mirrors this effect.

What we do not know is whether the final persistence preference:

- develops progressively across many layers;
- appears abruptly late;
- appears early but is transformed repeatedly;
- or follows some other trajectory.

This analysis addresses **where the decision-aligned persistence representation begins to emerge**, not yet what all of its inputs are.

## Design

Use the persistence probes already trained independently on the original bandit dataset.

**Do not retrain persistence probes on the factorial dataset.**

For each layer \(l\):

1. load the frozen probe;
2. project all 12 factorial conditions for each held-out state onto that direction;
3. estimate within-state effects of:
   - STOP payoff;
   - CONTINUE bonus;
   - relative incentive \(C-S\);
4. calculate within-state variance explained by relative incentive.

Estimate:

\[
P^{(l)}_{is}
=
\beta^{(l)}_S S_{is}
+
\beta^{(l)}_C C_{is}
+
\alpha_i
+
\epsilon_{is}
\]

where \(\alpha_i\) is a state fixed effect.

## Primary outputs

Produce layerwise plots of:

\[
\beta^{(l)}_{\text{STOP}}
\]

\[
\beta^{(l)}_{\text{CONTINUE}}
\]

and:

\[
R^{2(l)}_{\text{relative incentive}}.
\]

Also plot effects normalized relative to the final behavioral persistence-logit effect.

## Interpretation

This is a **representational trajectory**, not proof that a particular layer computes persistence.

The appropriate claim is:

> “At these layers, the incentive-induced behavioral difference becomes increasingly expressed along the independently identified persistence representation.”

Do **not** describe the first detectable layer as necessarily “where persistence is computed.”

---

# 7. Track B — Cross-task persistence checkpoint

Track B should be implemented in parallel with Track A.

Its purpose is to determine whether the persistence direction is sufficiently task-general to justify a much deeper mechanistic program.

---

# 8. Task 2 — Foraging-style persistence environment

## Scientific question

> **Does persistence behavior, and the internal representation associated with it, generalize to a structurally different continue-versus-leave problem?**

## Task requirements

Build a simple repeated foraging environment in which the model decides whether to:

- continue exploiting/searching within the current patch; or
- leave/end the current episode.

The task should differ meaningfully from the original two-armed bandit while remaining cheap enough to generate a substantial activation dataset.

Potential manipulated variables include:

- current patch quality;
- patch depletion;
- recent returns;
- outside-option value;
- cost of remaining.

The exact behavioral ecology can be finalized during implementation, but the task must generate substantial within-condition variation in persistence.

---

# 9. Control token/output geometry

A major threat is that the layer-31 bandit direction may encode something closer to:

> “terminal action versus non-terminal action”

or even generic output-token geometry rather than persistence.

Therefore, the new task must **not simply reuse A/B versus C = STOP**.

## Required label counterbalancing

Use arbitrary single-token response labels and counterbalance their semantic mappings.

For example:

Condition 1:

- X = STAY
- Y = LEAVE

Condition 2:

- X = LEAVE
- Y = STAY

Counterbalancing must occur across otherwise comparable episodes.

Compatibility tests must confirm that all selected labels are distinct single-token continuations under the exact applied chat template.

## Scientific question

> Does the bandit-derived persistence direction predict the abstract decision to continue rather than merely a specific output token?

A real persistence signal should preserve its relationship with the semantic choice after label mappings reverse.

---

# 10. Negative-control task

Label counterbalancing is necessary but insufficient.

The bandit direction could still reflect a generic binary decision axis rather than persistence.

## Scientific question

> **Does the bandit persistence direction specifically predict persistence decisions, or does it predict arbitrary binary choices more generally?**

## Design

Add a simple binary task with:

- the same model;
- the same activation-capture procedure;
- comparable response-token structure;
- counterbalanced response labels;
- **no continue/quit or terminal/non-terminal semantics**.

A simple perceptual/classification or rule-based binary judgment is sufficient.

The key property is:

\[
\text{choice 1} \neq \text{continue}
\]

and:

\[
\text{choice 2} \neq \text{stop}.
\]

## Interpretation

If the bandit persistence direction predicts arbitrary binary choices as strongly as it predicts foraging persistence, that argues against interpreting it as a specific persistence representation.

---

# 11. Same-task foraging ceiling

Zero-shot transfer will almost certainly be lower than within-task decoding because of distribution shift.

A weak zero-shot result is therefore uninterpretable without knowing how much persistence information is available in the new task at all.

## Scientific question

> **How much persistence information is linearly available in the foraging task, and what fraction of that information is captured by the bandit-derived direction?**

## Design

Train a separate foraging-specific ridge persistence probe using the same methodology as the original bandit probe:

- episode-level train/validation/test splits;
- training-set standardization only;
- regularization selected on validation;
- layer selected using validation only;
- held-out test performance reported once.

This is a **ceiling/comparison condition**, not the primary generalization result.

Report:

\[
R^2_{\text{bandit zero-shot}}
\]

and:

\[
R^2_{\text{foraging-specific}}.
\]

Also report a descriptive transfer ratio:

\[
T =
\frac{R^2_{\text{zero-shot}}}
{R^2_{\text{foraging ceiling}}}
\]

when both values are positive.

Do not treat this ratio alone as evidence of transfer.

---

# 12. Strict zero-shot representational transfer

## Primary scientific question

> **Does the frozen bandit persistence direction predict persistence in the foraging task without learning a new direction?**

## Primary analysis

Apply:

- the original bandit feature standardization;
- the original frozen bandit probe weights;
- no fitted foraging parameters.

This is the strict zero-shot test.

## Secondary calibration-only analysis

As a secondary diagnostic, allow only an affine transformation of the frozen bandit projection:

\[
P_{\text{foraging}}
=
a+bP_{\text{bandit direction}}.
\]

Do not modify the direction itself.

This helps distinguish:

- failure of the underlying axis to transfer;
- from simple task-specific scaling or offset changes.

The strict zero-shot result remains primary.

---

# 13. Pre-register representational transfer criteria

Transfer criteria must be specified **before examining test-set results**.

Classify the representational result as:

### Strong transfer

All of the following hold:

1. The frozen bandit direction predicts foraging persistence in the expected direction on held-out episodes.
2. The effect is consistent across reversed action-label mappings.
3. The target direction exceeds the 95th percentile of matched random/control directions.
4. The effect is absent or substantially weaker in the non-persistence negative-control task.
5. Zero-shot performance reaches at least 50% of the foraging-specific probe ceiling.

### Partial transfer

Criteria 1–4 hold, but the zero-shot direction reaches less than 50% of the within-task ceiling.

### No convincing transfer

The frozen bandit direction:

- fails to exceed matched controls;
- reverses with label mapping;
- or performs similarly on the non-persistence control task.

The 50% ceiling criterion is a classification aid, not a magic theoretical threshold; raw effect sizes and confidence intervals must always be reported.

---

# 14. Causal cross-task transfer

Representational transfer alone is insufficient.

## Scientific question

> **Does the persistence direction identified in the bandit causally influence persistence in the foraging task?**

## Critical calibration requirement

Do **not** reuse the bandit-calibrated steering magnitude directly.

A null causal-transfer result is uninterpretable unless the intervention has been verified to move the relevant projection in the new task.

Redo calibration using **foraging validation states only**.

Reuse the existing calibration framework:

1. candidate decoded shifts;
2. verify expected probe ordering;
3. enforce activation-standardized RMS limits;
4. freeze \(\lambda\);
5. evaluate once on held-out foraging test episodes.

The original study used approximately one decoded standard-deviation movement with RMS constraints; preserve this logic unless implementation diagnostics show that a small methodological change is necessary.

## Primary causal estimand

\[
\Delta_{\text{causal}}
=
P_{\text{persist}}(+\lambda d_{\text{bandit}})
-
P_{\text{persist}}(-\lambda d_{\text{bandit}})
\]

where persistence is defined according to the semantic STAY/LEAVE mapping, not raw token identity.

## Required controls

- zero steering reproduces baseline exactly;
- reversed label mappings;
- matched random directions;
- non-persistence control task;
- intervention actually shifts the frozen decoded quantity;
- episode-level bootstrap uncertainty.

## Causal transfer criterion

Call the result causal transfer only if:

1. steering successfully moves the frozen decoded quantity in the intended ordering;
2. persistence changes monotonically with \(-\lambda,0,+\lambda\);
3. the behavioral effect exceeds the matched random-direction distribution;
4. the sign survives response-label reversal.

If calibration fails, report the causal test as **invalid/inconclusive**, not null.

---

# 15. Decision matrix after Track B

Track B determines what kind of object the original persistence direction appears to be.

## Outcome A — Representational + causal transfer

Interpretation:

> Evidence supports a task-general persistence-related causal direction.

Proceed aggressively to Track C.

## Outcome B — Representational transfer, causal transfer fails after successful calibration

Interpretation:

> Tasks may share a persistence representation without sharing the same downstream controller.

Track C should focus on task-specific routing from the shared representation to behavior.

## Outcome C — Within-task probes succeed but zero-shot transfer fails

Interpretation:

> Persistence may be a common behavioral category implemented by task-specific internal representations.

The research question shifts toward identifying multiple persistence mechanisms.

## Outcome D — Bandit direction transfers to arbitrary binary decisions

Interpretation:

> The current layer-31 direction may reflect generic decision/output geometry rather than persistence.

Downgrade the current mechanistic interpretation before investing in deep pathway tracing.

---

# 16. Track C — Conditional mechanistic decomposition

Only launch the expensive parts of Track C after Track B has clarified what kind of persistence representation exists.

---

# 17. Controlled-history decomposition

## Scientific question

> **Which aspects of previous experience causally feed into the persistence decision?**

Candidate factors include:

- reward recency;
- loss streak;
- cumulative reward;
- elapsed trials;
- action switching;
- estimated continuation value.

Use transcripts constructed entirely from the **existing original message format**.

Do not add explicit statements such as:

- “You have lost three times in a row”;
- “You are on trial 20”;
- “Your expected value is X.”

The goal is to manipulate the model-visible history without introducing new framing instructions.

### Example

\[
H_A=(+3,+3,-2,-2,-2)
\]

versus:

\[
H_B=(-2,-2,-2,+3,+3)
\]

These can hold constant:

- number of trials;
- cumulative reward;
- number of successes;
- number of failures;

while manipulating outcome recency.

### Critical action-sequence control

The action pattern must also be controlled.

For example, if one history is:

\[
A,A,B,B,B
\]

the comparison history should use the same action sequence unless action switching/repetition is itself intentionally manipulated.

Otherwise, apparent reward-recency effects could actually reflect:

- arm switching;
- arm repetition;
- win-stay;
- lose-shift;
- inferred arm quality.

Action history should either be:

1. held exactly fixed; or
2. introduced as an explicit experimental factor.

---

# 18. Layerwise tracing of controlled inputs

For every experimentally validated determinant of persistence, ask:

> **At what layers does manipulating this variable become expressed along the persistence trajectory?**

Repeat the Track A logic for:

- recency;
- elapsed time;
- continuation value;
- outside-option value;
- other validated determinants.

The goal is eventually to determine whether multiple variables exhibit:

\[
X_1 \searrow
\]

\[
X_2
\rightarrow
\boxed{\text{shared persistence pathway}}
\rightarrow
P_{\text{persist}}
\]

\[
X_3 \nearrow
\]

or remain separate until the final layers.

---

# 19. Activation patching

## Scientific question

> **At what locations does transferring the representation induced by a persistence manipulation become sufficient to transfer part of the behavioral effect?**

Start with coarse layer intervals based on Track A.

Example:

- layer 4;
- layer 8;
- layer 12;
- layer 16;
- layer 20;
- layer 24;
- layer 28;
- layer 31.

Then refine around transition regions.

Use matched state pairs that differ in only one controlled variable.

Do not interpret successful patching as proving that the patched layer is where the computation originated.

Appropriate interpretation:

> The manipulation-induced representation at this location is sufficient to transfer downstream persistence behavior.

---

# 20. Computational modeling

Run behavioral computational-model comparison in parallel once controlled manipulations become available.

Candidate models should include at minimum:

### Relative value

\[
D_t =
Q_{\text{continue},t}
-
Q_{\text{stop},t}
\]

### Value plus elapsed cost

\[
D_t =
Q_{\text{continue},t}
-
Q_{\text{stop},t}
-
c(t)
\]

### Value plus outcome recency

\[
D_t =
Q_{\text{continue},t}
-
Q_{\text{stop},t}
+
\gamma R_{\text{recent},t}
\]

### Multi-input persistence model

\[
D_t =
w_1Q_{\text{continue}}
-
w_2Q_{\text{stop}}
+
w_3R_{\text{recent}}
-
w_4t
+
b_{\text{continue}}.
\]

The project should not end at behavioral model fit.

For the best-supported latent variable \(D_t\), test:

1. whether \(D_t\) becomes increasingly decodable near the layer range identified in Track A/patching;
2. whether its representational trajectory generalizes across tasks;
3. whether manipulating upstream components changes \(D_t\), the persistence representation, and behavior in the predicted sequence.

The desired endpoint is:

\[
\boxed{\text{experimental determinants}}
\rightarrow
\boxed{\text{computational rule}}
\rightarrow
\boxed{\text{internal pathway}}
\rightarrow
\boxed{\text{behavior}}.
\]

---

# 21. Track D — Solvability / give-up paradigm

Foraging is a diagnostic checkpoint, not the final generalization target.

If cross-task evidence is promising, build a third task more closely connected to the motivating safety problem.

## Scientific question

> **Does the persistence representation predict whether an agent continues working on a difficult goal versus appropriately giving up when the task is impossible or no longer worth pursuing?**

The task should manipulate:

- solvable versus impossible problems;
- evidence of progress;
- cost of additional attempts;
- explicit or implicit outside options;
- task horizon.

Eventually this can be extended to tool-using agents, but the first implementation should remain simple enough for controlled causal intervention.

A positive result here would be much stronger evidence that the persistence mechanism relates to meaningful long-horizon goal pursuit rather than repeated exit-option tasks alone.

---

# 22. Test-driven development requirement

All implementation must follow **TDD with explicit RED → GREEN testing**.

This applies not only to generic software infrastructure but also to experiment integrity and analysis logic.

## Required development cycle

For every meaningful unit of functionality:

### RED

1. Write the test first.
2. Run it.
3. Confirm that it fails for the expected reason.
4. Record the failure in the development log.

### GREEN

5. Implement the minimum code required to satisfy the test.
6. Run the new test.
7. Confirm that it passes.
8. Run the complete regression suite.

### REFACTOR

9. Clean or generalize the implementation.
10. Rerun the full test suite.
11. Do not merge/refactor further unless all tests remain green.

**Never write implementation first and back-fill tests afterward for core experimental logic.**

---

# 23. Required RED → GREEN test cases

At minimum, implement the following tests.

| Component | RED test | GREEN requirement |
|---|---|---|
| Existing bandit runner | Existing known state fails reproduction test before correct loader/config is wired | Reproduce stored logits within tolerance |
| Episode splitting | Construct episode IDs that accidentally cross splits | Assert no episode occurs in multiple partitions |
| Frozen probe loading | Load wrong layer/probe metadata | Test fails on layer or dimensionality mismatch |
| Probe standardization | Apply validation/test moments accidentally | Only stored training moments accepted |
| Layerwise factorial projection | Synthetic data with known STOP/CONTINUE slopes | Recover expected slope signs/magnitudes |
| State fixed effects | Add arbitrary state intercepts | Within-state coefficient remains unchanged |
| Foraging task | Malformed episode without valid terminal semantics | Integrity audit rejects it |
| Label counterbalancing | Swap X/Y meanings | Semantic persistence score must reverse appropriately while raw token score does not |
| Token compatibility | Use a multi-token action label | Test rejects configuration |
| Negative control | Accidentally mark a control choice as terminal | Test fails |
| Zero-shot probe | Attempt to fit coefficients on foraging test set | Analysis pipeline rejects fitting |
| Same-task ceiling | Put states from same episode across splits | Split audit fails |
| Steering zero condition | Apply \(\lambda=0\) | Output logits must exactly reproduce baseline |
| Steering calibration | Candidate shift fails ordering or RMS constraint | Candidate rejected |
| Causal null validation | Decoded quantity does not move | Result automatically classified as inconclusive rather than null |
| Random directions | Random vector changes norm/RMS unexpectedly | Matching test fails |
| Action-history control | Controlled reward histories differ in A/B sequence unintentionally | Pair audit fails |
| Seed reproducibility | Same seed executed twice | Exact episode trajectories/data IDs reproduced |
| Output manifests | Missing config/hash/seed metadata | Run marked incomplete |

These tests should be added incrementally before the corresponding functionality is implemented.

---

# 24. Integration tests

In addition to unit tests, maintain several end-to-end tests using very small datasets.

Examples:

### Mini factorial test

- 2 states;
- 2 STOP values;
- 2 CONTINUE values;
- all 32 layers;
- verifies loading → activation projection → state-centering → coefficient calculation → output file.

### Mini foraging run

- 4 episodes;
- both label mappings;
- deterministic seed;
- verifies prompt → logits → action semantics → activation storage → episode termination.

### Mini transfer test

- load frozen bandit direction;
- run on tiny foraging dataset;
- produce zero-shot projection;
- no training allowed.

### Mini steering test

- 2 states;
- \(-\lambda,0,+\lambda\);
- verify exact zero baseline;
- verify projection ordering after calibration.

These should run quickly enough to execute before every cluster submission.

---

# 25. Persistent experiment log / scratchpad

Maintain a persistent project scratchpad in the repository.

For every experiment or coding session, record:

- goal;
- hypothesis;
- code/config changed;
- RED test written;
- observed RED failure;
- GREEN implementation;
- tests run;
- result;
- unexpected behavior;
- decision about next step.

This is especially important because multiple transfer outcomes will be scientifically interpretable, and decisions should not be reconstructed after seeing the final result.

---

# 26. Reproducibility requirements

Every run must store:

- git commit hash;
- model checkpoint;
- Python/package environment;
- experiment config;
- exact prompt template;
- action-token IDs;
- seed;
- train/validation/test assignment;
- activation layers;
- probe ID;
- standardization metadata;
- steering calibration;
- random-control seeds;
- output dataset checksum where practical.

Do not overwrite previous run directories.

---

# 27. Analysis discipline

## No test-set iteration

Test episodes should not be inspected during:

- prompt tuning;
- probe selection;
- layer selection;
- steering calibration;
- threshold selection.

## Pre-register transfer criteria

Write the criteria in Section 13 and the causal criteria in Section 14 into the analysis config or preregistration document before running held-out tests.

## Separate confirmatory and exploratory analyses

Every output should be labeled:

- `confirmatory`
- or `exploratory`.

Unexpected patterns can motivate future experiments but should not silently become primary tests.

---

# 28. Deliverables

## Track A

- Layerwise factorial projection dataset.
- STOP-effect-by-layer figure.
- CONTINUE-effect-by-layer figure.
- Relative-incentive-\(R^2\)-by-layer figure.
- Machine-readable analysis table.
- Brief interpretation memo.

## Track B

- Foraging task implementation.
- Behavioral validation report.
- Counterbalanced-label audit.
- Negative-control task.
- Strict zero-shot transfer analysis.
- Calibration-only secondary transfer analysis.
- Foraging-specific persistence probe ceiling.
- Cross-task steering analysis.
- Random-control comparison.
- Pre-registered outcome classification.
- Summary figure comparing:
  - bandit within-task;
  - foraging zero-shot;
  - foraging within-task ceiling;
  - negative control.

## Conditional Track C

- Controlled-history generator.
- Action-sequence matching audit.
- Layerwise trajectories for candidate determinants.
- Coarse-to-fine activation patching.
- Computational-model comparison.
- Internal latent-variable analysis.

## Conditional Track D

- Give-up/solvability task specification and pilot.

---

# 29. Initial execution order

### First

Protect the existing codebase with baseline regression tests.

### Second

Run **Track A** using the existing factorial activations and frozen probes.

This should require minimal or no new model inference.

### Third

Build and behaviorally validate the foraging environment plus label counterbalancing and negative-control task.

### Fourth

Collect the foraging activation dataset.

### Fifth

Run:

1. strict zero-shot transfer;
2. negative-control transfer;
3. within-foraging ceiling probe.

### Sixth

Only after representational results are frozen, calibrate and run causal cross-task steering.

### Seventh

Use the predefined outcome matrix to decide whether to:

- proceed to full mechanistic decomposition;
- investigate task-specific persistence mechanisms;
- or reinterpret the original layer-31 direction as generic decision/output geometry.

### Eighth

If warranted, proceed to controlled-history decomposition and eventually the solvability/give-up paradigm.

---

# 30. Success criterion

The project is successful even if the persistence direction does **not** generalize.

The primary objective is to discriminate between competing explanations of the sprint result.

The highest-value positive outcome would be evidence for:

\[
\boxed{\text{shared behavioral persistence}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{shared internal persistence direction}}
\]

\[
\Downarrow
\]

\[
\boxed{\text{shared causal influence on continuation}}
\]

followed by identification of the upstream quantities that construct that signal.

The highest-value negative outcome would be clear evidence that the current layer-31 direction is predominantly a task-specific or generic output-decision representation. Knowing this early prevents substantial effort from being spent mechanistically dissecting an artifact with limited generality.

The immediate engineering and scientific priority is therefore:

> **Use the existing code and data to characterize how the known incentive effect develops across layers, while independently testing whether the identified persistence direction transfers behaviorally, representationally, and causally to a different task. Only then invest in a deeper decomposition of what constructs the signal.**