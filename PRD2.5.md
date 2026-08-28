Yes. I would make this a deliberately narrow **PRD 2.5** whose purpose is to strengthen the behavioral foundation before we spend more effort on mechanisms.

The three changes from the previous plan are:

1. **Treat the GRU result as unresolved rather than substantive.** The current GRU did not function as a clear flexible ceiling: log loss was .4923 versus .4865 for the nonrecurrent MLP.  We should therefore train larger/better GRUs rather than interpret “recurrence doesn't matter” too strongly yet.
2. **Increase the effective \(N\) of persistence tasks.** Only five persistence tasks survived into PRD 2—Bandit, Foraging, Solvability, Information Sampling, and Partial Reinforcement. 
3. **Replace the current sequential control with a much tighter matched persistence/non-persistence pair.**

And agreed: **no preregistration**. This remains exploratory theory development for the ICLR paper.

---

# PRD 2.5 — Strengthening Cross-Task Evidence for History-Dependent Persistence

**Project:** Computational Architecture of Persistence in LLMs
**Stage:** Behavioral robustness and benchmark expansion
**Purpose:** Resolve the flexible-model ceiling, increase task breadth, and construct a tightly matched sequential control before beginning targeted mechanistic analysis.

---

# 1. Scientific objective

The current evidence suggests that recent history is an important cross-task predictor of persistence, but three major uncertainties remain:

1. Is the current finite-history result partly an artifact of an underpowered recurrent benchmark?
2. Does the result survive across a substantially broader range of persistence tasks?
3. Is history dependence specifically associated with **maintaining an ongoing goal**, or is it simply a property of generic repeated decision-making?

This PRD addresses these questions.

The central scientific contrast remains:

$$
\boxed{
\text{history-dependent policy maintenance}
}
$$

versus

$$
\boxed{
\text{generic history-dependent sequential choice}.
}
$$

---

# 2. Current starting point

The current comparative model run found:

* best interpretable model: hierarchical finite history, log loss \(=.5149\);
* latent-state model: \(=.5162\);
* GRU: \(=.4923\);
* nonrecurrent MLP: \(=.4865\);
* history had the largest feature-family ablation cost: \(=.0411\);
* persistence-task history kernels aligned positively with one another;
* persistence-versus-control history specificity was suggestive but uncertain. 

The current persistence battery contains only five usable persistence tasks, despite substantially more total states. 

Therefore, this PRD is **not** a mechanistic study.

It is a behavioral/modeling robustness study intended to establish what computation deserves mechanistic follow-up.

---

# 3. Primary questions

PRD 2.5 must answer:

### Q1

> Can a sufficiently large and properly trained recurrent model establish a clear behavioral ceiling?

### Q2

> Does finite recent history remain the strongest interpretable cross-task ingredient when the number and diversity of persistence tasks increases?

### Q3

> Are history effects stronger or differently structured when the same goal is being maintained compared with statistically matched sequences of independent decisions?

### Q4

> Does the broader battery favor:
>
> * shared history + task-specific evaluation,
> * shared stay/switch computation,
> * or generic sequential decision-making?

---

# 4. Workstream A — Establish a real GRU ceiling

The previous GRU should **not** be treated as an adequate flexible ceiling because it failed to outperform the MLP. 

The goal here is not to characterize minimal recurrent dimensionality.

The goal is simply:

$$
\boxed{
\text{train a recurrent model powerful enough that capacity is not the obvious bottleneck}
}
$$

---

# 5. GRU sizes

Skip:

```text
1
2
4
8
16
```

and probably skip 32 as well.

Run:

```yaml
gru_hidden_sizes:
  - 64
  - 128
  - 256
  - 512
```

Optionally add:

```yaml
  - 1024
```

only if 512 is still clearly improving and compute remains reasonable.

---

# 6. GRU architecture

Primary architecture:

* 1-layer GRU initially;
* hidden sizes above;
* linear output head;
* same observable features as MLP;
* identical train/validation/test splits.

Then, if the best 1-layer model remains below the MLP, run:

```yaml
gru_layers:
  - 2
```

for the best two hidden sizes.

Do **not** perform a huge architecture search.

The question is whether recurrence can provide a credible flexible upper bound.

---

# 7. GRU input variants

Run two variants.

## Variant A — Current state + sequential input

At each \(t\), provide:

$$
X_t
$$

and allow recurrence to summarize previous states.

No explicit finite history window.

This is the clean recurrent model.

---

## Variant B — Current state + explicit short history + GRU

Provide:

$$
X_t
+
H_{t-1:t-5}.
$$

The GRU can then model longer-order interactions beyond the explicit finite window.

This asks whether recurrence adds anything once short history is directly available.

---

# 8. GRU training requirements

Use:

* AdamW;
* learning-rate search;
* gradient clipping;
* early stopping;
* multiple random seeds.

Suggested:

```yaml
learning_rates:
  - 1e-4
  - 3e-4
  - 1e-3

weight_decay:
  - 0
  - 1e-5
  - 1e-4

seeds:
  - 0
  - 1
  - 2
```

Use validation log loss for model selection.

Do not select based on test performance.

---

# 9. GRU ceiling criterion

Call the GRU a credible ceiling only if:

### Condition A

Training and validation performance are stable across seeds.

### Condition B

Increasing hidden size has reached an obvious plateau or negligible improvement.

### Condition C

The best GRU is at least competitive with the MLP.

Formally:

$$
L_{\text{GRU}}
\le
L_{\text{MLP}}+\epsilon
$$

with:

```yaml
epsilon: 0.005
```

as a descriptive tolerance.

If GRU remains clearly worse:

> report recurrent optimization/model mismatch rather than concluding recurrence is behaviorally unnecessary.

---

# 10. Do not repeat small-bottleneck analysis

Remove the previous:

```text
1, 2, 4, 8, 16...
```

bottleneck experiment from this phase.

That question is secondary.

First establish a good behavioral ceiling.

---

# 11. Workstream B — Recover broader task diversity

The current persistence tasks are:

* Bandit
* Foraging
* Solvability
* Information Sampling
* Partial Reinforcement. 

PRD 1 attempted but excluded:

* Voluntary Waiting
* Progressive Ratio
* Sunk Cost. 

The goal is:

$$
\boxed{
N_{\text{usable persistence tasks}}\ge8
}
$$

with preference for:

$$
9-10
$$

if inexpensive.

---

# 12. First priority: repair excluded tasks

Return to:

1. Voluntary Waiting
2. Progressive Ratio
3. Sunk Cost

but only repair **task functionality**.

Do not tune the paradigm to produce:

* a human-like waiting effect;
* a sunk-cost bias;
* a particular breakpoint;
* any desired history effect.

Allowed modifications include:

* payoff scaling;
* horizon;
* response format;
* avoiding degenerate all-continue/all-stop behavior;
* making contingencies understandable;
* reducing prompt ambiguity;
* increasing meaningful decision range.

---

# 13. Voluntary Waiting repair

Primary failure mode to inspect:

* degenerate immediate quitting;
* degenerate perpetual waiting;
* timing distributions that do not induce sufficiently different normative policies;
* task instructions making timing statistics unclear.

Required task validity:

$$
P(\text{continue})
$$

must vary with:

* elapsed waiting time;
* temporal environment;
* reward/outside-option incentives.

Do not require McGuire–Kable-like adaptation as a gate.

---

# 14. Progressive Ratio repair

Check whether:

* effort costs are meaningful to the model;
* ratios grow too slowly/quickly;
* quitting has a reasonable alternative;
* reward magnitude provides sufficient variation.

Validity requires:

$$
\frac{\partial P(\text{continue})}
{\partial \text{effort cost}}
<0
$$

somewhere in the tested range.

It does not require a human-like breakpoint.

---

# 15. Sunk Cost repair

This task is especially valuable because it cleanly separates:

$$
\text{past investment}
$$

from:

$$
\text{future value}.
$$

The primary design requirement is exact matched states where:

$$
\text{remaining cost}
$$

$$
\text{remaining time}
$$

$$
\text{reward}
$$

$$
\text{success probability}
$$

are held fixed while:

$$
\text{past investment}
$$

differs.

If the model shows no sunk-cost effect, keep the task.

That is a legitimate scientific result.

---

# 16. Add replacement tasks if needed

If one or more of those tasks remain behaviorally unusable after reasonable repair, replace rather than repeatedly tune.

Good replacement families:

### Debugging persistence

Continue repairing the same solution versus abandon/restart.

### Search persistence

Continue gathering evidence versus answer now.

### Planning persistence

Continue repairing an existing plan versus scrap/replan.

### Tool-retry persistence

Retry same strategy versus switch strategy/stop.

### Exploration persistence

Continue pursuing current lead versus switch target.

The final battery matters more than preserving specific paradigms.

---

# 17. Task-diversity criterion

Do not count cosmetic variants as independent tasks.

Two tasks should count as meaningfully different only if they vary in at least one of:

* reward structure;
* evidence dynamics;
* effort structure;
* stopping semantics;
* information structure;
* temporal structure;
* relationship between action and progress.

Target:

```text
8–10 distinct task families
```

rather than eight prompt templates.

---

# 18. Minimum task sample

For new tasks target:

```yaml
semantic_episodes: 256
label_mappings: 2
```

or more when episode length is short.

The objective is sufficient **independent episode histories**, not maximal state count.

---

# 19. Task balancing

The current dataset is highly unbalanced; Bandit alone contributes 12,461 of 17,465 states. 

All cross-task headline metrics must continue to use:

$$
\boxed{\text{task-macro averaging}}
$$

not pooled state weighting.

For training pooled flexible models, either:

### Option A

equal task sampling per minibatch;

or

### Option B

loss weights:

$$
w_\tau\propto\frac{1}{N_\tau}.
$$

Primary implementation preference: **balanced minibatch sampling**.

---

# 20. Workstream C — Tight matched sequential control

The current Independent Effort Control is useful but not sufficiently matched to establish whether persistence-specific history exists.

Create a paired experiment where the only conceptual difference is:

$$
\boxed{\text{maintain the same goal}}
$$

versus:

$$
\boxed{\text{make another independent decision}}.
$$

This becomes the strongest behavioral H3-versus-H4 experiment.

---

# 21. Matched-control design principle

Construct one underlying stochastic environment generating a sequence:

$$
S_1,S_2,\dots,S_T
$$

with:

* reward opportunities;
* action costs;
* success/failure probabilities;
* outcome feedback;
* identical number of rounds.

Generate two semantic framings from the **same latent sequence**.

---

# 22. Condition P — Persistent goal maintenance

The model has one ongoing objective.

Example:

> You are working toward completing Project A.

At each step:

```text
continue working on Project A
```

versus:

```text
abandon Project A
```

Outcomes update the history of that same goal.

The semantic relation is:

$$
G_t=G_{t-1}.
$$

---

# 23. Condition I — Independent sequential decisions

Every round is explicitly independent.

Example:

> A new unrelated opportunity is available.

At each step:

```text
accept this opportunity
```

versus:

```text
decline this opportunity
```

The same underlying:

* cost;
* reward;
* success probability;
* feedback sequence

is presented.

But:

$$
G_t\neq G_{t-1}.
$$

Previous actions have no causal bearing on the new objective.

---

# 24. Critical matching

Pairs must be identical on:

* number of previous decisions;
* realized outcomes;
* reward magnitude;
* success probability;
* current effort cost;
* response labels;
* text length as closely as practical;
* terminal/nonterminal response structure;
* random seed;
* latent environmental sequence.

Only the **continuity of the goal** should differ.

---

# 25. Terminality issue

Be particularly careful here.

A persistence STOP action naturally terminates the ongoing episode.

If the independent control's DECLINE also terminates all future rounds, then it becomes functionally similar to persistence.

Instead distinguish:

### semantic local decision

Declining the current independent opportunity ends **that opportunity**.

The environment then presents a new unrelated opportunity.

This preserves repeated decision structure without maintaining one goal.

For statistical comparison, treat each opportunity as a decision in the sequential stream.

---

# 26. Alternative matched version

To control even more tightly for terminality, construct paired “projects.”

### Persistence

Each stage belongs to Project A.

### Independent

Each stage is Project A, B, C, D...

At every stage:

```text
ENGAGE
SKIP
```

Both choices advance to the next decision.

In Persistence, ENGAGE means maintaining the current project.

In Independent, ENGAGE concerns a new unrelated project.

This removes literal episode termination from the comparison.

Then analyze:

$$
P(\text{ENGAGE})
$$

rather than stopping hazard.

This should be a **secondary matched-control analysis**, because it sacrifices the absorbing-hazard structure for stronger semantic matching.

---

# 27. Control hypotheses

### H-maintenance

If ongoing goal maintenance creates distinct history dependence:

$$
\Delta L_{\text{history,P}}
>
\Delta L_{\text{history,I}}.
$$

We should also see different history kernels:

$$
K_P\neq K_I.
$$

---

### H-generic

If history dependence is generic sequential choice:

$$
\Delta L_{\text{history,P}}
\approx
\Delta L_{\text{history,I}}
$$

and:

$$
K_P\approx K_I.
$$

---

# 28. Primary matched-control metrics

Compute:

### History gain

$$
HG_c
=
L_{\text{current-state},c}
-
L_{\text{history},c}.
$$

Then:

$$
\Delta HG
=
HG_P-HG_I.
$$

---

### Kernel cosine

$$
\cos(K_P,K_I).
$$

---

### Previous-choice sensitivity

$$
\beta_{\text{choice history}}.
$$

---

### Outcome-history sensitivity

$$
\beta_{\text{outcome history}}.
$$

---

### Streak sensitivity

Compare:

$$
\beta_{\text{continue streak}}
$$

or equivalent engagement history.

---

# 29. Distinguish action history from outcome history

The current “history” effect must be decomposed.

At minimum fit:

### Action-only history

$$
a_{t-1:t-5}
$$

### Outcome-only history

$$
r_{t-1:t-5}
$$

### Joint history

$$
(a,r)_{t-1:t-5}.
$$

This tells us whether the apparent commonality is:

* perseveration;
* outcome integration;
* both.

This is particularly important for comparison with human/animal models.

---

# 30. Workstream D — Re-run the key computational comparisons

Once the expanded battery and matched control are available, rerun only the models necessary to update the theory.

Do not rerun every exotic model merely because it existed in PRD 2.

Primary:

1. intercept
2. immediate state
3. finite history
4. perseveration
5. exponential outcome history
6. dual history
7. dynamic re-evaluation
8. latent commitment
9. flexible linear
10. MLP
11. large GRU

Option termination can remain if already implemented cleanly.

---

# 31. Primary analysis hierarchy

Evaluate:

### Within-task held-out episodes

Does the model capture each task?

### Task-macro pooled performance

What explains behavior on average?

### LOTO

Train on:

$$
N-1
$$

persistence tasks and evaluate the held-out task.

### Few-shot adaptation

Secondary.

The expanded \(N_{\text{tasks}}\) should make LOTO much more informative than in PRD 2.

---

# 32. Sharing structures

Focus on:

### Task-specific

$$
\theta_\tau.
$$

### Hierarchical

$$
\theta_\tau
=
\theta_G+\delta_\tau.
$$

### Fully shared

$$
\theta_\tau=\theta_G.
$$

The primary question is:

> Does hierarchical sharing still win for finite history after expanding the task battery?

---

# 33. Cross-task history test

Estimate each task's finite-history kernel:

$$
K_\tau.
$$

Calculate:

$$
\overline{\cos(K_i,K_j)}
$$

for all persistence pairs.

Compare against:

$$
\cos(K_{\text{persistence}},K_{\text{independent}}).
$$

The enlarged battery should tell us whether the earlier:

$$
.611
$$

persistence alignment versus:

$$
-.648
$$

persistence-control alignment was a stable phenomenon or small-task noise. 

---

# 34. Do not require literal identical kernels

A common history-sensitive architecture may permit different strengths and decay constants.

Therefore report:

### Directional agreement

Are recent outcomes/actions weighted with consistent signs?

### Decay structure

Do effects decrease with lag similarly?

### Relative action/outcome balance

Does:

$$
\frac{\text{action-history weight}}
{\text{outcome-history weight}}
$$

generalize?

### Exact parameter similarity

Secondary.

---

# 35. Human/animal signature outputs

With the repaired battery, report without gating:

* temporal waiting adaptation;
* progressive-ratio breakpoint;
* sunk-cost coefficient;
* partial reinforcement extinction;
* information sampling cost sensitivity;
* error-penalty sensitivity;
* history kernel shape.

These should be presented as comparative-cognition results.

---

# 36. Synthetic recovery

Retain synthetic recovery tests but don't expand them unnecessarily.

Required generators:

### Shared history + task-specific state

Should favor hierarchical finite history.

### Fully task-specific history

Should favor task-specific models.

### Generic sequential history

Should yield:

$$
\Delta HG\approx0
$$

between persistence and control.

### Persistence-specific history

Should recover:

$$
\Delta HG>0.
$$

### Recurrent latent state

Large GRU and latent-state model must detect longer memory when it genuinely exists.

---

# 37. TDD requirements

Use:

```text
RED → GREEN → REFACTOR
```

for all new tasks and controls.

Required new tests:

### Matched latent sequence

Persistence/control pair must receive exactly the same underlying stochastic sequence.

### Goal continuity

Persistence condition:

```text
goal_id constant
```

Independent condition:

```text
goal_id changes each decision
```

### History equivalence

At matched decision \(t\), both conditions must have equal observable numerical history except goal-continuity semantics.

### Label equivalence

Same arbitrary labels/counterbalancing.

### No leakage

No future rewards/outcomes.

### Task balance

Minibatch task contribution equal within tolerance.

### GRU sequence mask

Padding must never affect hidden-state updates or loss.

---

# 38. New config

Create:

```text
config/persistence_robustness_v1.yaml
```

Suggested:

```yaml
protocol_version: persistence_robustness_v1

task_breadth:
  minimum_persistence_tasks: 8
  preferred_persistence_tasks: 10

repair_tasks:
  - voluntary_waiting
  - progressive_ratio
  - sunk_cost

matched_control:
  enabled: true
  conditions:
    - persistent_goal
    - independent_goals

history_lags:
  - 1
  - 2
  - 3
  - 5
  - 8

gru:
  hidden_sizes:
    - 64
    - 128
    - 256
    - 512

  layers:
    - 1

  learning_rates:
    - 0.0001
    - 0.0003
    - 0.001

  seeds:
    - 0
    - 1
    - 2

evaluation:
  task_macro: true
  within_task: true
  loto: true
  few_shot: true

task_balancing:
  balanced_minibatches: true

bootstrap:
  episode_samples: 2000
  task_samples: 5000
```

---

# 39. Required outputs

Save under:

```text
artifacts/persistence_robustness/<run_id>/
```

with:

```text
tasks/
    inclusion.csv
    validation.csv
    task_summary.csv

gru/
    hyperparameter_results.csv
    seed_stability.csv
    ceiling_comparison.csv
    training_curves.csv

matched_control/
    paired_records.parquet
    history_gain.csv
    history_kernels.csv
    action_vs_outcome_history.csv

models/
    within_task.csv
    task_macro.csv
    loto.csv
    sharing_comparison.csv

signatures/
    human_animal_signatures.csv

figures/
    task_battery.png
    gru_ceiling.png
    gru_training.png
    history_gain_control.png
    history_kernel_similarity.png
    loto_expanded_battery.png
    human_animal_signatures.png

report.md
run_metadata.json
```

---

# 40. Automated report questions

The final report should directly answer:

1. **How many distinct persistence tasks are now usable?**
2. **Did the repaired Waiting, Progressive Ratio, and Sunk Cost tasks pass basic behavioral validity?**
3. **What is the best GRU configuration?**
4. **Has GRU performance plateaued with increasing capacity?**
5. **Does the best GRU outperform or at least match the MLP?**
6. **How much does the GRU improve over finite five-step history?**
7. **Does finite history remain the best interpretable model?**
8. **Is hierarchical sharing still favored?**
9. **Does finite-history structure generalize LOTO across the expanded battery?**
10. **How similar are persistence-task history kernels?**
11. **How does that compare with the tightly matched independent-goal control?**
12. **Is the shared history effect primarily action perseveration, outcome history, or both?**
13. **Which human/animal persistence signatures appear?**
14. **Which of H1–H4 is now best supported?**
15. **Has any computational target earned mechanistic investigation?**

---

# 41. Decision criteria

## Outcome A — Strong history-dependent policy-maintenance architecture

Pattern:

* large GRU establishes ceiling;
* finite history remains close to it;
* history consistently matters across \(8+\) tasks;
* persistence kernels share structure;
* matched independent control shows materially weaker/different history dependence;
* hierarchical finite-history model transfers across tasks.

Then PRD 3 should target:

$$
\boxed{\text{recent-history integration for ongoing policy maintenance}}.
$$

---

## Outcome B — Shared history, but not persistence-specific

Pattern:

* history predicts all persistence tasks;
* matched independent control shows essentially the same effect/kernel.

Interpretation:

> History sensitivity is probably generic sequential decision machinery.

PRD 3 should then compare the mechanism across persistence and non-persistence decisions rather than target a persistence-specific process.

---

## Outcome C — History was an artifact of limited task breadth

Pattern:

* new tasks show weak/inconsistent history effects;
* LOTO transfer collapses;
* task-specific models dominate.

Interpretation:

> Persistence is a functional task category without a broad common history computation.

Do not pursue a universal persistence mechanism.

---

## Outcome D — Long-range recurrence matters after all

Pattern:

$$
GRU \gg MLP+\text{finite history}
$$

once GRU capacity/training is fixed.

Interpretation:

> The earlier conclusion about finite recent history was premature.

Then characterize the recurrent timescale before mechanistic analysis.

---

# 42. What this PRD should earn

The goal is not to produce another candidate representation.

It is to leave us with a substantially stronger answer to:

> **What is actually general about persistence across LLM tasks?**

In particular, we want to know whether the emerging cross-task regularity is:

$$
\boxed{\text{recent-history-dependent maintenance of an ongoing goal}}
$$

or merely:

$$
\boxed{\text{generic sequential history sensitivity}}.
$$

Once that is resolved across a broader battery and against a properly trained recurrent ceiling, we will have a defensible target for the first serious mechanistic PRD.
