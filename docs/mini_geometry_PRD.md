## Mini PRD: Linking Behavioral Persistence Computations to the L21/L22 Persistence Subspace

### Objective

Test whether the previously identified **rank-4 persistence-specific representation around layers 21–22** carries the computational ingredients that best predict persistence behavior across Bandit, Foraging, and Solvability.

The behavioral model-zoo follow-up found that the shared GRU predicts persistence well (\(R^2=.897\)), while the strongest cross-task interpretable model is finite history (\(R^2=.704\)). Feature-family ablations identified **history**, **time/effort**, and **cost** as the strongest cross-task ingredients; progress/solvability contributed primarily in Solvability. 

The next question is:

> **Does the L21/L22 persistence subspace represent these computational ingredients, and do they combine within that subspace to explain the model’s persistence decision?**

Do not assume the subspace represents a single scalar “persistence variable.”

---

# 1. Core hypotheses

### H1 — Computational-variable representation

The L21/L22 persistence subspace encodes one or more of:

$$
\text{history},\quad
\text{time/effort},\quad
\text{cost},\quad
\text{progress/solvability}.
$$

### H2 — Multidimensional integration

Different dimensions of the rank-4 subspace carry partially distinct computational quantities rather than four redundant copies of the same signal.

### H3 — Integration before decision

The L21/L22 representation explains persistence beyond the individual behavioral inputs, and later layers become increasingly aligned with the final persistence logit.

Conceptually:

$$
\boxed{\text{task evidence}}
\rightarrow
\boxed{\text{L21/L22 computational subspace}}
\rightarrow
\boxed{\text{persistence decision}}.
$$

### Alternative

The L21/L22 candidate could still be:

* generic history representation;
* generic effort/time representation;
* task identity;
* generic value/state information;
* or an arbitrary subspace correlated with persistence.

The analysis must test these alternatives.

---

# 2. Reuse existing artifacts

Do not recollect activations.

Reuse:

* existing Bandit activation bank;
* existing Foraging activation bank;
* existing Solvability activation bank;
* persisted episode/pair-safe splits;
* existing persistence contrast bank;
* previously identified L21/L22 rank-4 candidate;
* behavioral model-zoo records;
* feature-group definitions from `model_zoo_mac_v2`.

The behavioral feature schema currently groups predictors into:

* `history`
* `time_effort`
* `cost`
* `progress_solvability`
* `continuation_value`
* `outside_option`
* `derived_termination`

with task indicators treated as nuisance features. 

---

# 3. Define neural representations

For every state, extract:

### Static L21

$$
h_{21}
$$

### Static L22

$$
h_{22}
$$

### L21→L22 displacement

$$
\delta h_{21}
=
h_{22}-h_{21}.
$$

Project each into the previously discovered rank-4 persistence subspace:

$$
z_t=W^\top h_t
$$

or for the displacement candidate:

$$
z_t=W^\top(h_{22}-h_{21}),
$$

where:

$$
z_t\in\mathbb{R}^4.
$$

**Freeze \(W\).**

Do not refit the persistence subspace using the computational targets.

---

# 4. Primary analysis: decode computational variables from the subspace

For each computational feature family, construct a target from the existing behavioral model-zoo records.

Primary targets:

### History

Prefer a compact, behaviorally validated quantity rather than every raw lag independently.

Test both:

1. finite-history model prediction;
2. summarized history variables such as recent outcomes/choices.

### Time / effort

Use:

* `log_round`
* `normalized_time`

or a prespecified composite.

### Cost

Use harmonized:

$$
\text{cost_pressure}.
$$

### Progress / solvability

Use:

* `progress_evidence`
* `cumulative_progress`

where defined.

Also test secondary controls:

* continuation value;
* outside-option value;
* derived termination advantage.

For each target \(y\):

$$
z_t\rightarrow y_t.
$$

Use train episodes only for decoder fitting.

Evaluate on held-out episodes.

Report:

$$
R^2,\quad r,\quad MSE.
$$

---

# 5. Compare full hidden state versus persistence subspace

For each target compare:

### A. Rank-4 persistence subspace

$$
z_t\in\mathbb R^4
$$

### B. Matched random rank-4 subspaces

Sample at least:

```yaml
matched_random_subspaces: 100
```

### C. Full hidden state with capacity-controlled decoder

Use a regularized linear decoder with hyperparameters chosen on validation data.

Scientific question:

> How much computational information is specifically concentrated in the persistence subspace rather than merely available somewhere in the layer?

A useful statistic:

$$
\text{concentration ratio}
=
\frac{R^2_{\text{rank4 persistence}}}
{R^2_{\text{full hidden}}}.
$$

---

# 6. Dimension-level analysis

Do not automatically interpret the four basis vectors produced by the original subspace algorithm.

Instead ask whether the **subspace as a whole** supports separable computational variables.

Fit a multivariate mapping:

$$
z_t
\rightarrow
\begin{bmatrix}
H_t\\
T_t\\
C_t\\
P_t
\end{bmatrix}.
$$

Then assess:

* rank of the fitted mapping;
* canonical correlations;
* cross-validated variance explained for each target;
* principal angles between target-specific directions inside the rank-4 space.

The important question is:

> Are history, effort, cost, and progress encoded along partially separable axes?

Avoid naming individual raw PCs as “history neuron,” etc.

---

# 7. Cross-task generalization

This is a primary test.

For each computational target that exists meaningfully across tasks, run leave-one-task-out:

$$
B+F\rightarrow S
$$

$$
B+S\rightarrow F
$$

$$
F+S\rightarrow B.
$$

Freeze:

* subspace;
* normalization;
* decoder;
* hyperparameters.

Then evaluate the held-out task.

This tests:

> **Does the same neural geometry represent the same computational quantity across tasks?**

A variable can be task-general even if its behavioral coefficient differs by task.

---

# 8. Compare L21, displacement, and L22

Run the same analyses for:

$$
h_{21}
$$

$$
h_{22}-h_{21}
$$

$$
h_{22}.
$$

This directly tests whether the computational information is:

* already present at L21;
* introduced/transformed between L21 and L22;
* or expressed more strongly at L22.

Expected interpretation examples:

```text
L21 weak → displacement strong → L22 strong
```

suggests the computation emerges across the transition.

```text
L21 strong → displacement weak → L22 strong
```

suggests the quantity is largely maintained.

---

# 9. Does the subspace mediate the behavioral ingredients?

After identifying encoded quantities, fit:

### Behavioral-only model

$$
D_t=f(X_t)
$$

where \(X_t\) contains the major behavioral ingredients.

### Neural-only model

$$
D_t=g(z_t).
$$

### Combined model

$$
D_t=f(X_t)+g(z_t).
$$

Ask:

$$
\Delta R^2_{\text{neural}\mid\text{behavior}}
$$

and:

$$
\Delta R^2_{\text{behavior}\mid\text{neural}}.
$$

Interpretation:

* If \(z_t\) adds nothing beyond \(X_t\), it may primarily encode those variables.
* If \(z_t\) explains variance beyond \(X_t\), it may contain additional integration/computation relevant to persistence.
* If \(X_t\)'s predictive contribution collapses after adding \(z_t\), that is consistent with—but does **not prove**—mediation.

Do not make causal mediation claims from prediction alone.

---

# 10. Specificity controls

Run the same computational decoders on existing nuisance tasks where meaningful:

### Arbitrary binary choice

Does the L21/L22 subspace encode the same history-like structure in a task with no persistence semantics?

### Terminality

Does the subspace merely track proximity to episode ending?

### Generic value

Does it simply encode relative value?

Also test task identity:

$$
z_t\rightarrow\text{task}.
$$

A persistence-specific subspace should not owe its computational decoding primarily to task separation.

---

# 11. GRU-state comparison

The GRU is now a validated flexible behavioral predictor and reaches:

$$
R^2=.897.
$$



Save the GRU hidden state at each decision:

$$
g_t.
$$

Exploratorily test:

$$
z_t\leftrightarrow g_t.
$$

Use:

* linear CCA;
* low-rank regression;
* cross-task transfer.

Question:

> Does the L21/L22 persistence subspace resemble the flexible recurrent state that best predicts behavior?

This is secondary to the interpretable-variable analysis.

Do not treat the GRU state itself as a cognitive construct.

---

# 12. Behavioral ablation targets

Prioritize variables according to the behavioral ablation results:

### Priority 1

History:

$$
\Delta R^2\approx .150
$$

and supported across all three tasks.

### Priority 2

Time/effort:

$$
\Delta R^2\approx .063.
$$

### Priority 3

Cost:

$$
\Delta R^2\approx .047.
$$

### Priority 4

Progress/solvability.

Task-specific but important in Solvability.

Continuation value, outside-option value, and derived termination advantage should remain controls because their unique behavioral contributions were weak. 

---

# 13. Leakage requirements

Use the existing persisted train/validation/test splits.

Never:

* fit target decoders on test episodes;
* redefine the rank-4 subspace using computational labels;
* rotate the subspace using held-out task data;
* normalize with test statistics;
* choose targets based on test performance.

All rotations/CCA/canonical mappings must be fit on training data only.

---

# 14. TDD requirements

Use RED → GREEN → REFACTOR.

Minimum tests:

### Frozen subspace

Attempt to refit persistence basis during computational decoding.

Expected:

```text
ERROR: persistence subspace is frozen
```

### Episode leakage

Place same episode in train/test.

Expected rejection.

### Random-subspace recovery

Synthetic rank-4 data with known signal should outperform matched random subspaces.

### Cross-task leakage

Held-out task target values must never enter decoder fitting.

### Displacement indexing

Synthetic layers verify exactly:

$$
h_{22}-h_{21}.
$$

### Null target

Randomized computational target should yield chance-level held-out decoding.

### Task-identity confound

Synthetic data where target is entirely task-determined should be flagged as non-generalizing under leave-one-task-out.

---

# 15. Required outputs

Save to:

```text
artifacts/persistence_computational_neural/model_zoo_mac_v2/
```

Required files:

```text
computational_decoding.csv
cross_task_decoding.csv
random_subspace_controls.csv
full_hidden_comparison.csv

dimension_geometry.csv
behavior_neural_incremental.csv
specificity_controls.csv

gru_state_alignment.csv

figures/
    computational_decoding_by_target.png
    computational_decoding_by_layer_stage.png
    cross_task_transfer.png
    subspace_geometry.png
    behavioral_vs_neural_incremental.png

report.md
```

---

# 16. Primary report questions

The report must answer:

1. Which computational ingredients are represented in the L21/L22 persistence subspace?
2. Are they concentrated there relative to matched random subspaces?
3. Are they separable within the rank-4 space?
4. Do they generalize across tasks?
5. Does the L21→L22 transformation specifically increase any of them?
6. Does the subspace explain persistence beyond the explicit behavioral predictors?
7. Is the representation specific relative to generic choice, terminality, and value controls?

---

# 17. Decision criteria

### Outcome A — Shared multidimensional computational representation

History, cost/effort, progress, etc. decode reliably and transfer across tasks.

Interpretation:

> The candidate persistence subspace integrates multiple task-general computational ingredients relevant to continued goal pursuit.

Proceed to targeted causal tests.

### Outcome B — History dominates everything

Only recent choice/outcome history generalizes.

Interpretation:

> The persistence representation may primarily encode behavioral state/history rather than an abstract meta-control variable.

Shift mechanistic work toward history integration.

### Outcome C — Variables are task-specific

Within-task decoding succeeds but leave-one-task-out fails.

Interpretation:

> Similar persistence behavior may arise through different task-specific computations.

Do not claim task-general computational implementation.

### Outcome D — Neural subspace adds substantial information beyond behavioral variables

Interpretation:

> The behavioral feature set is incomplete; the L21/L22 state may contain an additional integrated control signal.

Use the residual component as a candidate for further characterization.

### Outcome E — Nothing meaningful decodes

Interpretation:

> The rank-4 persistence contrast representation is not straightforwardly aligned with the computational variables tested.

Do not force interpretation.

---

# 18. Immediate deliverable

The immediate deliverable should **not** be:

> “dimension 1 = history, dimension 2 = cost…”

The intended result is something like:

> **The L21→L22 persistence subspace contains a low-dimensional combination of recent-history, effort, and task-specific cost/progress information that generalizes across persistence tasks and predicts the downstream persistence decision.**

Or, if unsupported:

> **The candidate subspace predicts persistence but does not correspond cleanly to the computational ingredients identified behaviorally.**

Either result meaningfully narrows the mechanistic hypothesis.
