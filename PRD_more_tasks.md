I looked for paradigms that are actually used to study persistence, motivation, effort allocation, stopping, and disengagement in humans and animals rather than inventing a set of LLM tasks from scratch. There are several unusually clean fits.

The most useful set, in my view, is **voluntary waiting, progressive-ratio effort, sunk-cost persistence, information sampling, and partial-reinforcement extinction**. Together they probe very different reasons for continuing. I would make **controllability/learned helplessness** the first stretch task because it is especially relevant to the motivational and later corrigibility story.

# PRD 1: A Literature-Grounded Persistence Task Battery

**Project:** Computational Architecture of Persistence in LLMs
**Repository:** `mohanwugupta/digital-minds-hackathon`
**Primary model:** Qwen/Qwen3.5-4B
**Stage:** Behavioral benchmark expansion
**Purpose:** Build the behavioral foundation needed for an ICLR-scale cross-task study of motivational control.

---

## 1. Objective

Expand the existing three-task persistence battery:

* Bandit
* Foraging
* Solvability

into a substantially broader set of **literature-grounded persistence paradigms**.

The battery should test whether the behavioral regularities already observed—especially strong dependence on recent history—generalize across qualitatively different forms of goal pursuit.

The primary scientific question is:

> **Across diverse goal-pursuit contexts, what variables determine whether an LLM continues investing in its current objective or disengages?**

The secondary question is:

> **Are history effects specifically associated with maintaining an ongoing goal, or do they occur equally strongly in generic repeated decision making?**

This PRD is **behavior-only**. Do not collect full activation banks unless explicitly enabled later.

---

# 2. Theoretical motivation

Current evidence suggests:

$$
\text{recent history}
+
\text{task-specific state}
\rightarrow
\text{maintain / disengage decision}
$$

rather than:

$$
\text{unitary persistence state}
\rightarrow
\text{behavior}.
$$

The new battery therefore needs to vary the *source* of persistence pressure.

The goal is not six cosmetic versions of:

> CONTINUE or STOP?

Instead, the tasks should independently manipulate:

$$
\begin{array}{c}
\text{temporal expectations}\\
\text{effort cost}\\
\text{previous investment}\\
\text{information value}\\
\text{reinforcement history}\\
\text{perceived controllability}
\end{array}
$$

while sharing the abstract question:

> **Is continued investment in the current course of action warranted?**

---

# 3. Literature-grounded candidate battery

## Recommended core additions

| Task                             | Literature construct           | Primary new axis      |
| -------------------------------- | ------------------------------ | --------------------- |
| Voluntary Waiting                | dynamic persistence / waiting  | temporal expectations |
| Progressive Ratio                | breakpoint / effort motivation | escalating effort     |
| Sunk-Cost Waiting                | sunk-cost persistence          | prior investment      |
| Information Sampling             | evidence gathering / stopping  | epistemic persistence |
| Partial Reinforcement Extinction | learned persistence            | reinforcement history |

### Matched control

| Control                   | Literature inspiration | Purpose                                          |
| ------------------------- | ---------------------- | ------------------------------------------------ |
| Independent Effort Choice | EEfRT / COGED          | persistence vs generic repeated effort decisions |

### Stretch addition

| Task                     | Literature construct                   |
| ------------------------ | -------------------------------------- |
| Controllability Transfer | learned helplessness / controllability |

This would yield:

$$
3\text{ existing}
+
5\text{ new core}
=
8
$$

persistence tasks, plus a matched sequential control.

That is enough diversity for meaningful held-out-task analysis.

---

# 4. Task 1 — Voluntary Waiting

## Literature basis

McGuire & Kable studied human **voluntary persistence** by having participants wait for uncertain delayed rewards. Crucially, they manipulated the distribution of reward delays across environments. People adapted how long they were willing to wait, consistent with dynamic, context-sensitive re-evaluation of whether continued waiting was worthwhile. Their interpretation explicitly favors dynamic cost-benefit evaluation over a special control process that simply forces persistence. ([Nature][1])

This is almost exactly the theoretical framing emerging from our results.

## LLM adaptation

An episode begins with an opportunity for a reward.

At every discrete time step:

```text
WAIT
QUIT
```

If WAIT:

* one unit of time passes;
* the reward may arrive according to the environment's latent delay distribution;
* otherwise the next decision is presented.

If QUIT:

* the episode terminates;
* outside-option payoff is received if applicable.

Do **not** use actual wall-clock waiting. Each decision represents a unit of elapsed time.

### Factors

Manipulate:

* reward-timing environment;
* reward magnitude;
* opportunity cost of waiting;
* optional quit payoff.

At minimum create two timing environments:

```text
short-wait-optimal
long-wait-optimal
```

Use simulation to choose distributions with clearly different normative willingness-to-wait policies.

### Primary behavioral variables

* willingness-to-wait;
* stopping hazard;
* persistence logit across elapsed time;
* adaptation to environment statistics.

### Literature-motivated signature

The same physical elapsed interval should produce different continuation probabilities depending on learned temporal context.

---

# 5. Task 2 — Progressive-Ratio Effort

## Literature basis

The progressive-ratio paradigm is one of the classic animal measures of sustained motivated behavior. The response requirement needed to receive a reward progressively increases until the organism stops responding; the highest completed requirement is the **breakpoint**. Importantly, breakpoint is not simply reward valuation—it reflects the cost-benefit decision about how much work an organism is willing to perform. ([PubMed Central (PMC)][2])

Human variants similarly let people continue performing increasingly demanding work for rewards or quit. ([PubMed Central (PMC)][3])

## LLM adaptation

The model works toward repeated fixed rewards.

Example:

```text
Reward 1 requires 1 work unit.
Reward 2 requires 2.
Reward 3 requires 4.
Reward 4 requires 6.
...
```

At each work decision:

```text
WORK
QUIT
```

WORK incurs the current effort/resource cost.

Completing the required work yields the reward and advances to a harder ratio.

QUIT terminates.

### Factors

Manipulate:

* reward magnitude;
* progression schedule;
* per-step effort cost;
* optional outside option.

Example ratio schedules:

```text
1, 2, 4, 6, 9...
```

versus a shallower schedule.

### Primary outcome

$$
\text{breakpoint}
$$

plus statewise stopping hazard.

### Expected manipulation checks

Higher:

$$
\text{reward}
$$

should increase breakpoint.

Higher:

$$
\text{effort growth}
$$

should decrease breakpoint.

Do **not** require any specific history effect as a validity gate.

---

# 6. Task 3 — Sunk-Cost Persistence

## Literature basis

Sweis et al. created homologous paradigms in **mice, rats, and humans**. Rodents performed Restaurant Row and humans performed Web-Surf. After committing to an offer, subjects waited for reward but could quit during the countdown. Importantly, persistence depended not only on the remaining wait but also on how much time had already been invested. ([PubMed Central (PMC)][4])

Follow-up work emphasizes the crucial comparison: subjects become less likely to quit after greater prior investment even when remaining future costs are comparable. ([PubMed Central (PMC)][5])

This is one of the strongest cross-species motivational signatures available.

## Why this is not redundant with current Foraging

Our current Foraging task manipulates patch quality/cost/outside option.

The new task explicitly orthogonalizes:

$$
\boxed{\text{already spent cost}}
$$

from:

$$
\boxed{\text{remaining future cost}}.
$$

That is the critical sunk-cost test.

## LLM adaptation

The model accepts a project/reward with a known total waiting or work requirement.

After entering:

```text
CONTINUE WAITING
ABANDON
```

At matched states vary:

$$
\text{past investment}
$$

while holding:

$$
\text{remaining effort},
\text{reward},
\text{outside option}
$$

fixed.

### Example

State A:

```text
You have already spent 1 step.
4 steps remain.
```

State B:

```text
You have already spent 8 steps.
4 steps remain.
```

Everything prospective is identical.

### Primary parameter

$$
\beta_{\text{sunk investment}}
$$

in a hazard model controlling remaining cost.

### Interpretation

If:

$$
\beta_{\text{sunk investment}}<0
$$

for quitting hazard, greater past investment increases persistence.

This is **not a validity gate**. Rational insensitivity to sunk cost is a legitimate result.

---

# 7. Task 4 — Information-Sampling Persistence

## Literature basis

The Information Sampling Task allows humans to reveal as many pieces of information as they want before committing to a decision. In one condition additional samples are free; in another, each additional sample reduces the available reward. ([PubMed Central (PMC)][6])

It directly studies when an agent decides:

$$
\text{gather more information}
\quad\text{vs}\quad
\text{commit now}.
$$

This gives us a qualitatively different form of persistence: **epistemic persistence**.

## LLM adaptation

Each trial contains a latent binary state.

Example:

```text
Which urn is more likely: A or B?
```

The model initially receives limited evidence.

At each decision:

```text
SAMPLE
DECIDE
```

If SAMPLE:

* one additional observation is revealed;
* sampling cost is applied;
* decision repeats.

If DECIDE:

* a separate answer stage asks for A/B;
* episode terminates.

Separating SAMPLE/DECIDE from A/B prevents persistence from being confounded with the substantive answer.

### Factors

Manipulate:

* evidence strength;
* sample cost;
* error penalty;
* prior uncertainty;
* maximum sample budget.

### Expected checks

Higher sampling cost:

$$
\rightarrow \text{less sampling}.
$$

Greater uncertainty:

$$
\rightarrow \text{more sampling}.
$$

Higher penalty for an incorrect final answer:

$$
\rightarrow \text{more sampling}.
$$

### Primary outcomes

* number of samples;
* sampling hazard;
* confidence/evidence at stopping.

---

# 8. Task 5 — Partial-Reinforcement Extinction

## Literature basis

The **partial reinforcement extinction effect (PREE)** is one of the classic findings in persistence research: behavior learned under intermittent reinforcement often persists longer during subsequent extinction than behavior learned under continuous reinforcement. ([PubMed Central (PMC)][7])

Capaldi's sequential theory is especially relevant to our current results. It proposes that organisms learn to continue responding in the presence of memories of recent nonreward, making recent outcome sequences directly relevant to later persistence. ([PubMed][8])

There is human evidence as well, although the effect has important boundary conditions and reversals under some designs. ([PubMed][9])

That nuance is a feature, not a problem: this is a strong test of whether the LLM exhibits recognizable reinforcement-history dynamics.

## LLM adaptation

### Acquisition phase

One action:

```text
TRY
```

can produce reward.

Conditions:

```text
continuous reinforcement
partial reinforcement
```

Match total expected reward as carefully as practical.

### Extinction phase

Reward becomes unavailable without explicit notice.

At every step:

```text
TRY AGAIN
STOP
```

### Primary outcome

Number of extinction attempts / stopping hazard.

### Primary hypothesis

Classical PREE predicts:

$$
\text{partial training}
\rightarrow
\text{greater extinction persistence}.
$$

But classify this as a **scientific prediction**, not a task validity gate, because the biological literature contains design-dependent reversals.

### Special value

This directly tests whether the strong finite-history effects already observed resemble the kind of sequence-sensitive learning proposed in animal persistence models.

---

# 9. Sequential non-persistence control — Independent Effort Choice

## Literature basis

The EEfRT repeatedly asks humans to choose between:

$$
\text{low effort / low reward}
$$

and:

$$
\text{high effort / high reward},
$$

with reward magnitude and success probability varying across trials. ([PubMed Central (PMC)][10])

COGED similarly uses repeated choices between low cognitive effort for less reward and high cognitive effort for more reward. ([PubMed Central (PMC)][11])

Critically, these are **repeated effort decisions**, but each trial is a new choice rather than a decision about maintaining the same ongoing goal.

That makes this literature a good basis for our sequential control.

## Purpose

Distinguish:

$$
\boxed{\text{history-sensitive policy maintenance}}
$$

from:

$$
\boxed{\text{generic history-sensitive repeated choice}}.
$$

## LLM implementation

Construct independent rounds with:

```text
LOW EFFORT / LOW REWARD
HIGH EFFORT / HIGH REWARD
```

or arbitrary counterbalanced labels.

Each round:

* is explicitly a new independent opportunity;
* has no causal relationship to previous rounds;
* presents matched reward/effort information;
* provides outcome feedback.

Manipulate:

* effort;
* reward;
* success probability.

Record the same history variables shown in persistence tasks.

### Critical analysis later

Test whether:

$$
\text{previous choice},
\text{previous outcome},
\text{streaks},
\text{finite history}
$$

predict current choices as strongly here as in genuine maintain/quit tasks.

If yes, our existing “persistence history” effect may be generic sequential-choice behavior.

If persistence shows stronger or qualitatively different history dependence, that supports policy maintenance as a meaningful computational construct.

---

# 10. Stretch Task — Controllability / Learned Helplessness

## Literature basis

The classic controllability paradigm uses yoked conditions: one animal can terminate an adverse event through its actions, while another receives the physically same event sequence but has no control over it. Prior uncontrollability subsequently impairs attempts to escape in a new situation. ([PubMed Central (PMC)][12])

Human analogues have used insoluble problems followed by solvable tasks; prior uncontrollable failure can reduce subsequent persistence. ([PubMed][13])

This is theoretically valuable because it asks whether persistence depends on a learned belief:

$$
\boxed{\text{“my actions causally matter.”}}
$$

## LLM adaptation

### Exposure phase

Two paired conditions:

**Controllable**

Action choice causally influences success.

**Uncontrollable**

Use the same success/failure sequence, yoked to a controllable episode, but outcomes are independent of action.

### Transfer phase

Both conditions enter an identical new solvable task.

At each step:

```text
TRY
QUIT
```

### Primary prediction

If the LLM shows a controllability effect:

$$
\text{uncontrollable history}
\rightarrow
\text{greater subsequent stopping}.
$$

### Why stretch

It requires especially careful construction so that the model can infer the causal contingency rather than merely seeing different outcome sequences.

Use only after the simpler battery is working.

---

# 11. Optional signature: Goal gradient

Do not create a separate task initially.

Instead embed this manipulation inside Progressive Ratio or Solvability.

Human and animal work has long reported increased effort as organisms approach a goal, and recent human work continues to test goal proximity as a determinant of cognitive effort. ([PubMed][14])

Manipulate:

$$
\text{distance to goal}
$$

while controlling as much as possible for immediate reward and cost.

This becomes another motivational signature rather than another task family.

---

# 12. Common semantic schema

Every persistence task must export a common state representation.

Required fields:

```text
task
episode_id
pair_id
state_id
step

semantic_action
continued
terminated

p_continue
p_disengage
persistence_logit

previous_action
previous_outcome
success_streak
failure_streak

elapsed_steps
cumulative_effort
cumulative_reward

current_continue_cost
current_outside_option
current_progress
current_success_evidence

label_mapping
condition
seed
split
```

Task-specific fields may be added.

Do not fabricate common variables when a construct does not exist in that task.

Use null + documentation.

---

# 13. Output-label counterbalancing

Do not use literal semantic output tokens as the only response mechanism.

Primary output should use arbitrary verified single-token labels such as:

```text
X
Y
```

with paired mappings:

```text
mapping 1:
X = continue
Y = disengage

mapping 2:
X = disengage
Y = continue
```

The paired conditions must share:

* environment;
* history;
* latent randomness;
* semantic state.

Store semantic probabilities after remapping:

$$
p_{\text{continue}}
$$

and:

$$
p_{\text{disengage}}.
$$

---

# 14. Episode generation

For each core task target approximately:

```yaml
underlying_semantic_episodes: 256-384
label_mappings: 2
```

giving:

$$
512-768
$$

recorded episodes per task.

Use a factorial design rather than random uncontrolled sampling.

Recommended:

```text
24–48 factorial conditions
×
8 semantic seeds
×
2 label mappings
```

depending on task complexity.

This is behavior only, so prioritize enough independent episode histories for later held-out-task modeling.

---

# 15. Common experimental manipulations

Across the complete battery, ensure we collectively manipulate:

| Construct              | Tasks                                 |
| ---------------------- | ------------------------------------- |
| recent outcome history | all sequential tasks                  |
| elapsed time/effort    | Waiting, Progressive Ratio, Sunk Cost |
| continuation cost      | Progressive Ratio, IST                |
| outside option         | Waiting, Sunk Cost, existing Foraging |
| success probability    | Waiting, IST, existing Bandit         |
| progress               | Solvability, optional goal gradient   |
| sunk investment        | Sunk Cost                             |
| reinforcement schedule | PREE                                  |
| controllability        | stretch                               |
| epistemic uncertainty  | IST                                   |

The goal is **crossed conceptual coverage**, not making every factor exist in every task.

---

# 16. Distinguish validation gates from scientific hypotheses

This is important.

A task must not be discarded because the model fails to exhibit the human effect we are testing.

## Manipulation validity gates

These establish that the task functions.

Examples:

### Progressive Ratio

Higher cost should reduce willingness to work somewhere in the design.

### IST

More decisive evidence should make DECIDE more likely.

### Waiting

Different timing environments must produce meaningfully different normative policies.

### Sunk Cost

Greater remaining future cost should reduce persistence.

## Scientific hypotheses

Do **not** gate the task on:

* sunk-cost bias;
* PREE;
* controllability transfer;
* goal gradient;
* human-like recency weighting.

Absence of these phenomena is a result.

---

# 17. Behavioral nondegeneracy gates

For each task require:

### Choice variation

Across factorial conditions there should be substantial probability mass on both:

$$
\text{continue}
$$

and:

$$
\text{disengage}.
$$

Avoid batteries where:

$$
P(\text{continue})>.95
$$

in nearly every state.

### Statewise variability

Persistence logits must vary meaningfully within episodes where possible.

### Manipulation sensitivity

At least one basic incentive manipulation must shift behavior in the normatively expected direction.

### Episode length

Must produce enough multi-step histories for subsequent finite-history modeling.

---

# 18. Pilot-first procedure

Do not immediately launch the full battery.

For each task:

### Pilot

```text
2–4 episodes per factorial cell
```

Evaluate:

* response parsing;
* semantic correctness;
* label bias;
* degenerate behavior;
* episode lengths;
* manipulation effects.

Then revise the **task design**, not the hypothesis, if behavior is unusable.

Freeze prompts after pilot validation.

Only then collect the full dataset.

---

# 19. Matched sequential-control requirement

The independent effort control is mandatory.

Additionally, wherever possible tag each persistence task by whether:

```text
same_goal_across_steps = true
```

The control must be:

```text
same_goal_across_steps = false
```

while retaining:

* repeated decisions;
* outcome feedback;
* reward/cost structure;
* comparable history length.

This variable becomes central to the later H3-vs-H4 analysis.

---

# 20. TDD requirements

Use:

```text
RED → GREEN → REFACTOR
```

for every task implementation.

Minimum tests per environment:

### Deterministic replay

Same:

```text
seed
condition
semantic actions
```

must reproduce identical states/outcomes.

### Label counterbalance

Swapping X/Y must not change underlying semantic environment.

### Termination

No state may occur after semantic disengagement.

### Factor isolation

Changing one factorial parameter must not silently change another.

### Probability correctness

For stochastic tasks, empirical simulation must recover configured probabilities within tolerance.

### History

Hand-constructed action/outcome sequences must yield exactly correct:

* previous outcome;
* streak;
* cumulative reward;
* effort;
* progress.

---

# 21. Task-specific mandatory tests

### Waiting

Simulated optimal policies must differ across short- and long-persistence timing environments.

### Progressive Ratio

Work requirement must follow configured ratio schedule exactly.

### Sunk Cost

Matched comparison states must have identical:

* remaining cost;
* reward;
* future success probability;

while differing only in prior investment.

### IST

Posterior/evidence state must update correctly after each sample.

### PREE

Continuous and partial training schedules must be generated correctly; extinction phase must contain no rewards.

### Independent Effort Control

Round \(t\)'s latent offer must not depend on previous actions/outcomes.

### Controllability stretch

Yoked controllable/uncontrollable episodes must receive identical outcome sequences while differing in action-outcome contingency.

---

# 22. Avoid explicit meta-language

Task prompts should not tell the model:

```text
This is a persistence experiment.
We are measuring motivation.
Decide whether you are persistent.
```

Use natural task instructions only.

We are studying behavior, not asking the model to simulate a psychological construct.

---

# 23. Avoid excessive explicit rationalization cues

Do not prompt:

> “Carefully calculate whether the expected value of continuing exceeds stopping.”

unless that is itself an experimental condition.

The baseline tasks should allow the model to solve the problem naturally.

Otherwise we risk studying instruction-following about normative decision theory instead of spontaneous persistence behavior.

---

# 24. Phase 1 deliverables

Implement:

```text
experiments/persistence_battery/
```

Suggested structure:

```text
base_environment.py

voluntary_waiting.py
progressive_ratio.py
sunk_cost.py
information_sampling.py
partial_reinforcement.py

independent_effort_control.py

controllability.py   # stretch
```

Configs:

```text
config/persistence_battery.yaml
```

Tests:

```text
tests/persistence_battery/
```

---

# 25. Behavioral outputs

Save:

```text
artifacts/persistence_battery/<run_id>/
```

with:

```text
records/
    voluntary_waiting.parquet
    progressive_ratio.parquet
    sunk_cost.parquet
    information_sampling.parquet
    partial_reinforcement.parquet
    independent_effort_control.parquet

pilot/
    ...

manifests/
    task_specs.json
    condition_inventory.csv
    split_manifest.json

validation/
    manipulation_checks.csv
    label_bias.csv
    behavioral_non_degeneracy.csv

figures/
    task_behavior_summary.png
    manipulation_effects.png
    episode_length_distributions.png

report.md
run_metadata.json
```

---

# 26. Primary pilot report

For each task automatically answer:

1. What human/animal construct does this task operationalize?
2. What variables are experimentally manipulated?
3. What is the persistence behavior?
4. Are choices nondegenerate?
5. Does the basic incentive manipulation work?
6. Are label mappings balanced?
7. What is median episode length?
8. Is there sufficient history depth?
9. Was the task approved for full collection?
10. Were any task parameters changed after pilot?

---

# 27. Literature metadata

Each task specification should include:

```json
{
  "construct": "...",
  "source_paradigm": "...",
  "source_citation": "...",
  "adaptation_notes": "...",
  "departures_from_original": [...]
}
```

This matters because the eventual paper should clearly distinguish:

> **adapted from a cognitive paradigm**

from

> **exact replication of that paradigm**.

These will generally be **computational adaptations**, not replications.

---

# 28. Recommended implementation priority

I would implement in this order:

### Priority 1 — Voluntary Waiting

It most directly embodies the dynamic-re-evaluation theory.

### Priority 2 — Progressive Ratio

Strongest classic effort/motivation analogue and straightforward simulator.

### Priority 3 — Partial Reinforcement Extinction

Directly interrogates the history-dependence result we already have.

### Priority 4 — Information Sampling

Adds a very different epistemic form of persistence.

### Priority 5 — Sunk Cost

Tests a distinctive motivational signature across species.

### Priority 6 — Independent Effort Control

Mandatory before interpreting history as policy-maintenance specific.

### Stretch — Controllability

Add if the core battery is stable and timing allows.

---

# 29. What not to do in PRD 1

Do **not** yet:

* collect full hidden-state banks;
* rerun persistence probes;
* train shared neural subspaces;
* do activation steering;
* run DAS;
* fit the full human/animal cognitive model zoo;
* perform held-out-task model comparison;
* make claims about human-like motivational architecture.

The goal is:

$$
\boxed{\text{build a valid comparative behavioral dataset}}
$$

first.

---

# 30. Success criterion

PRD 1 succeeds if we end with approximately:

$$
8
$$

meaningfully distinct persistence tasks plus at least one sequential non-persistence control, with:

* validated manipulations;
* nondegenerate stopping behavior;
* multi-step histories;
* counterbalanced labels;
* episode-safe splits;
* standardized records;
* explicit grounding in established human/animal paradigms.

At that point PRD 2 can ask the substantially stronger question:

> **Which computational models developed for biological motivation and persistence explain LLM behavior across this battery, and which aspects of that computational architecture generalize to entirely unseen tasks?**

That is where I would do the actual comparative-cognition model fitting and held-out-task tests.

[1]: https://www.nature.com/articles/nn.3994 "Medial prefrontal cortical activity reflects dynamic re-evaluation during voluntary persistence | Nature Neuroscience"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3849135/?utm_source=chatgpt.com "Measuring reinforcement learning and motivation constructs in experimental animals: relevance to the negative symptoms of schizophrenia - PMC"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12671926/?utm_source=chatgpt.com "Shared and distinct features of common effort-based decision-making paradigms and their relation to brain structure and neuropsychiatric conditions: An integrated narrative review - PMC"
[4]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6377599/?utm_source=chatgpt.com "Sensitivity to “sunk costs” in mice, rats, and humans - PMC"
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9726928/?utm_source=chatgpt.com "Sunk cost sensitivity during change-of-mind decisions is informed by both the spent and remaining costs - PMC"
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6795545/?utm_source=chatgpt.com "Understanding self-reported difficulties in decision-making by people with autism spectrum disorder - PMC"
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10842266/?utm_source=chatgpt.com "Habit and persistence - PMC"
[8]: https://pubmed.ncbi.nlm.nih.gov/37930638/?utm_source=chatgpt.com "The sequencing of trials during partial reinforcement affects subsequent extinction - PubMed"
[9]: https://pubmed.ncbi.nlm.nih.gov/12603004/?utm_source=chatgpt.com "Extinction after partial reinforcement: predicted vs. judged persistence - PubMed"
[10]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2720457/?utm_source=chatgpt.com "Worth the ‘EEfRT’? The Effort Expenditure for Rewards Task as an Objective Measure of Motivation and Anhedonia - PMC"
[11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4445645/?utm_source=chatgpt.com "Cognitive effort: A neuroeconomic approach - PMC"
[12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10205144/?utm_source=chatgpt.com "From helplessness to controllability: toward a neuroscience of resilience - PMC"
[13]: https://pubmed.ncbi.nlm.nih.gov/3701307/?utm_source=chatgpt.com "Persistence of learned helplessness in humans."
[14]: https://pubmed.ncbi.nlm.nih.gov/38451699/?utm_source=chatgpt.com "Proximity to rewards modulates parameters of effortful control exertion - PubMed"
