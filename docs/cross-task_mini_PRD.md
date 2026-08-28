## Mini PRD: Cross-Task Persistence Computation in Counterfactual Change Space

### Objective

Test whether the previously identified persistence-related L21→L22 rank-4 subspace contains a **task-general representation of changes in persistence-relevant computational quantities**, even though absolute neural states do not generalize cleanly across tasks.

The previous analysis showed that absolute projections into the rank-4 space contain substantial task identity, generic choice, terminality, and value information, and computational decoders fail leave-one-task-out transfer. 

However, the subspace was originally discovered using **matched persistence contrasts**, not absolute states.

The new question is therefore:

> **When persistence-relevant evidence changes while the underlying state is matched, do the corresponding neural changes encode a shared computational transformation across Bandit, Foraging, and Solvability?**

---

# 1. Core hypothesis

Instead of analyzing:

$$
z_t=W^\top h_t,
$$

analyze matched differences:

$$
\Delta z
=
W^\top
\left[
h(P^+)-h(P^-)
\right].
$$

Then ask whether:

$$
\Delta z
$$

systematically represents the corresponding change in computational variables:

$$
\Delta X
=
X(P^+)-X(P^-).
$$

Candidate variables include:

* continuation value;
* outside-option value;
* cost pressure;
* progress / solvability evidence;
* behavioral model-predicted persistence;
* optionally GRU-predicted persistence.

The key test is whether the **mapping from computational change to neural change transfers across tasks**.

---

# 2. Why use differences

The absolute-state analysis showed very high task identity and nuisance decoding.

Matched differences should remove much of:

* task-level baseline geometry;
* stable token/readout differences;
* episode-specific offsets;
* generic state identity;
* fixed task identity.

Conceptually:

$$
h_{t,+}=b_t+p_{t,+}
$$

$$
h_{t,-}=b_t+p_{t,-}
$$

so:

$$
h_{t,+}-h_{t,-}
=
p_{t,+}-p_{t,-}.
$$

This is the construct the original persistence-contrast analysis was designed to isolate.

---

# 3. Reuse existing data only

Do not recollect activations unless a required matched contrast genuinely does not exist.

Reuse:

* existing persistence contrast bank;
* Bandit factorial contrasts;
* Foraging matched contrasts;
* Solvability matched progress contrasts;
* frozen `displacement-L21-k4` candidate;
* static L21 and L22 candidates where available;
* existing behavioral computational-model outputs;
* existing train/validation/test splits.

Do not refit the persistence subspace.

---

# 4. Contrast families

At minimum include the existing persistence manipulations.

### Bandit

* higher CONTINUE incentive;
* lower STOP / outside-option attractiveness.

### Foraging

* lower search cost;
* lower outside-option attractiveness.

### Solvability

* stronger progress / solvability evidence.

Orient every contrast so:

$$
\Delta>0
$$

means:

> more evidence favoring persistence.

Maintain this semantic orientation everywhere.

---

# 5. Pair integrity audit

Before analysis, verify each contrast is genuinely matched.

For every pair require equality, where applicable, in:

* semantic history;
* action history;
* outcome history;
* decision number;
* environment state;
* label mapping or appropriately counterbalanced mapping;
* all variables except the intended manipulation.

Produce:

```text
contrast_pair_audit.csv
```

with:

```text
task
contrast_family
pair_id
matched
unmatched_fields
orientation_valid
```

Do not include invalid pairs.

---

# 6. Neural change representations

For each matched pair compute:

### Static L21 change

$$
\Delta h_{21}
=
h_{21}^{+}-h_{21}^{-}
$$

### L21→22 computational-change contrast

First:

$$
d_{21}
=
h_{22}-h_{21}.
$$

Then:

$$
\Delta d_{21}
=
d_{21}^{+}-d_{21}^{-}.
$$

### Static L22 change

$$
\Delta h_{22}
=
h_{22}^{+}-h_{22}^{-}.
$$

Project through the frozen candidate:

$$
\Delta z=W^\top\Delta h
$$

or:

$$
\Delta z=W^\top\Delta d.
$$

---

# 7. Computational change targets

For each pair compute differences in the existing behavioral quantities.

Primary targets:

### Persistence-policy change

$$
\Delta D
=
D^+-D^-.
$$

### Flexible behavioral-model change

$$
\Delta \hat D_{\mathrm{GRU}}
=
\hat D^+_{\mathrm{GRU}}
-
\hat D^-_{\mathrm{GRU}}.
$$

### Interpretable behavioral-model change

Prefer the best relevant cross-task interpretable prediction:

$$
\Delta\hat D_{\mathrm{history}}.
$$

### Component changes

Where experimentally manipulated:

$$
\Delta\text{cost}
$$

$$
\Delta\text{progress}
$$

$$
\Delta\text{outside option}
$$

$$
\Delta\text{continuation value}.
$$

Do not manufacture values for quantities that do not vary within a contrast family.

---

# 8. Primary analysis: computational-change decoding

Fit:

$$
\Delta z \rightarrow \Delta X.
$$

Use linear models first.

For each target report:

* held-out \(R^2\);
* Pearson \(r\);
* MSE;
* sign accuracy:

$$
P(\operatorname{sign}(\widehat{\Delta X})
=
\operatorname{sign}(\Delta X)).
$$

Because contrasts are semantically oriented, sign consistency is scientifically meaningful.

---

# 9. Cross-manipulation transfer

Train on all but one persistence manipulation and test on the held-out manipulation.

Examples:

```text
train:
Bandit CONTINUE incentive
Bandit STOP value
Foraging cost
Foraging outside option

test:
Solvability progress
```

and rotate.

Question:

> Does the same neural change code predict persistence-relevant computational change when the causal source of that change differs?

This is a central test.

---

# 10. Cross-task transfer

Run strict leave-one-task-out:

$$
B+F\rightarrow S
$$

$$
B+S\rightarrow F
$$

$$
F+S\rightarrow B.
$$

No:

* target-task calibration;
* target-task rotation;
* target-task normalization fitting;
* target-task decoder fitting.

Training normalization must be frozen before evaluating the target task.

Primary targets:

$$
\Delta D
$$

and:

$$
\Delta\hat D_{\mathrm{GRU}}.
$$

These are most naturally defined across all tasks.

---

# 11. Compare absolute-state versus difference-space transfer

Directly compare:

$$
R^2_{\mathrm{LOTO,absolute}}
$$

with:

$$
R^2_{\mathrm{LOTO,difference}}.
$$

Calculate:

$$
\Delta R^2_{\text{contrast benefit}}
=
R^2_{\Delta z}
-
R^2_{z}.
$$

The core hypothesis predicts:

$$
R^2_{\Delta z}
>
R^2_z.
$$

This is one of the clearest tests of whether task-specific offsets were obscuring a shared computation.

---

# 12. Random-subspace control

Repeat the full analysis using at least:

```yaml
matched_random_subspaces: 100
```

rank-4 random subspaces from the same layer/stage.

Question:

> Is computational-change information unusually concentrated in the persistence candidate?

Report empirical percentile and:

$$
p_{\mathrm{empirical}}.
$$

This control is mandatory because absolute-state decoding previously failed it.

---

# 13. Nuisance-change controls

Construct matched nuisance differences where existing data permit.

### Label swap

$$
\Delta h_{\mathrm{label}}
$$

with persistence semantics unchanged.

### Arbitrary binary choice

Matched changes in ordinary binary-decision evidence.

### Terminality

Matched changes in continue/end evidence unrelated to goal persistence.

### Generic value

Matched changes in one-shot relative value.

Compare:

$$
\text{persistence-change decoding}
$$

against:

$$
\text{nuisance-change decoding}.
$$

A candidate should be stronger for persistence changes than these controls.

---

# 14. Direction/subspace consistency

For each task \(t\), manipulation \(m\), and stage \(l\), calculate:

$$
d_{t,m,l}
=
E[\Delta z_{t,m,l}].
$$

Then compare:

* cosine similarity;
* principal angles;
* subspace overlap;
* sign agreement.

Do not require identical vectors.

Test three architectures:

### A. Shared direction

All:

$$
d_{t,m}
$$

approximately collinear.

### B. Shared low-dimensional subspace

Task-specific directions differ but occupy common low-D geometry.

### C. Task-specific geometry

Little cross-task alignment.

---

# 15. L21→L22 transformation test

Explicitly test whether cross-task alignment increases across:

$$
\Delta h_{21}
$$

$$
\Delta(h_{22}-h_{21})
$$

$$
\Delta h_{22}.
$$

Possible result:

```text
L21:
task-specific

L21→22 displacement:
shared persistence change emerges

L22:
shared change retained
```

would be strong evidence for a common transformation.

Conversely, no improvement supports task-specific computation.

---

# 16. Statistical uncertainty

Use pair-clustered bootstrap.

Minimum:

```yaml
bootstrap_samples: 2000
```

Report 95% CIs for:

* decoding \(R^2\);
* correlations;
* cross-task transfer;
* cross-manipulation transfer;
* absolute-vs-difference improvement;
* persistence-vs-nuisance differences.

Do not treat contrast rows as independent when they share an underlying state/pair.

---

# 17. TDD requirements

Use RED → GREEN → REFACTOR.

Mandatory tests:

### Exact pair matching

Intentional unmatched history should fail.

### Orientation

Reversing \(P^+/P^-\) must reverse:

$$
\Delta z
$$

and:

$$
\Delta X.
$$

### Difference construction

Synthetic data must verify:

$$
\Delta(h_{22}-h_{21})
=
(h_{22}^+-h_{21}^+)
-
(h_{22}^--h_{21}^-).
$$

### Frozen subspace

Any attempt to refit \(W\) must fail.

### LOTO leakage

Held-out-task values must never enter fitting or normalization.

### Null contrast

Random paired differences should yield chance decoding.

### Shared-change recovery

Synthetic tasks with large task-specific offsets but one shared change direction should:

* fail absolute-state transfer;
* pass difference-space transfer.

This synthetic test is especially important.

---

# 18. Required outputs

Save under:

```text
artifacts/persistence_change_geometry/<run_id>/
```

Required:

```text
contrast_pair_audit.csv

change_decoding.csv
cross_task_change_transfer.csv
cross_manipulation_transfer.csv

absolute_vs_change.csv
random_subspace_controls.csv
nuisance_change_controls.csv

direction_alignment.csv
stage_transition.csv

figures/
    change_decoding.png
    absolute_vs_change_transfer.png
    cross_task_change_transfer.png
    cross_manipulation_transfer.png
    persistence_vs_nuisance_change.png
    direction_alignment.png

report.md
```

---

# 19. Decision rule

### Outcome A — Strong shared difference-space representation

Require qualitatively:

* positive LOTO transfer;
* positive cross-manipulation transfer;
* better than absolute-state transfer;
* above matched random rank-4 controls;
* persistence changes stronger than nuisance changes.

Interpretation:

> **Different tasks may occupy distinct baseline geometries, while persistence-relevant interventions induce a shared neural transformation.**

This would justify targeted causal testing.

### Outcome B — Cross-manipulation but not cross-task

Interpretation:

> Persistence computations are consistent within tasks but implemented in task-specific coordinates.

Do not claim task-general representation.

### Outcome C — Difference-space does not beat random/control geometry

Interpretation:

> The previous persistence candidate does not support a specific shared computational representation.

Stop searching this candidate for a universal persistence code.

### Outcome D — Generic value/decision changes transfer equally well

Interpretation:

> The shared transformation is better characterized as generic value/control computation than persistence.

Reframe accordingly.

---

# 20. Immediate scientific deliverable

The analysis should answer one narrow question:

> **Did the earlier matched-contrast analysis identify a genuinely shared persistence-related transformation that was obscured by task-specific absolute-state geometry, or was the apparent commonality still a byproduct of broader decision/value representations?**

This is the cleanest remaining falsification test before deciding whether the task-general persistence-representation hypothesis should be retained or abandoned.
