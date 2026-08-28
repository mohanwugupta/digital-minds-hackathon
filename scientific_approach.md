Yes. I think we now have enough accumulated evidence to write a coherent research program rather than another isolated experiment.

One practical constraint matters immediately: **ICLR 2027 abstracts are due September 18 and full papers September 25, 2026**, with a 9-page main-text limit. ([ICLR][1]) That is only a few weeks away. So I would structure this as **one primary ICLR paper that we definitely try to finish**, with a second paper as a stretch/next-stage project rather than splitting the current evidence prematurely.

# Research Plan: The Computational Architecture of Persistence in LLMs

## 1. Central scientific question

The project started with:

> **Is there a task-general internal “persistence signal” that determines whether an LLM continues pursuing a goal?**

The accumulating evidence argues against that ontology.

We instead now ask:

> **How do LLMs compute whether to maintain or terminate ongoing goal pursuit, and how similar is this computational architecture to motivational control in humans and animals?**

The working hypothesis is:

$$
\boxed{
\text{recent experience}
+
\text{current task-specific evidence}
\rightarrow
\text{dynamic maintain/terminate decision}
}
$$

rather than:

$$
\boxed{
\text{unitary persistence state}
\rightarrow
\text{behavior}.
}
$$

This is closely analogous to human/animal approaches in which persistence emerges from repeated stay/leave, wait/quit, exploit/switch, or policy-maintenance decisions rather than from a single “persistence” quantity.

---

# 2. What we currently know

The current evidence gives us a useful starting theory rather than a blank slate.

### Behavioral regularity

Persistence is highly predictable. Recent history is particularly important, and the best finite-history model can account for much of the behavior.

In the latest stay/switch analysis:

* a shared five-step history kernel retained **89.4% of the improvement** obtained by the fully task-specific hazard model;
* a full GRU gained only about **.001 \(R^2\)** over a five-step finite-history model;
* removing recurrence itself mattered little;
* a very small recurrent bottleneck was insufficient—performance continued improving up to the 64-dimensional model. 

This suggests:

$$
\boxed{\text{short recent history is important}}
$$

but not:

$$
\boxed{\text{one slowly evolving commitment state}}.
$$

### Representational negative result

Repeated attempts to recover one persistence representation have failed increasingly demanding tests.

The final matched-change analysis could decode within-distribution persistence-policy changes, but:

$$
R^2_{\text{LOTO}}\approx -2.86
$$

and:

$$
R^2_{\text{LOMO}}\approx -.92,
$$

while generic value/decision changes were at least as strongly represented. 

So the evidence increasingly rejects:

$$
\text{one vector}
$$

and:

$$
\text{one low-dimensional task-general persistence subspace}.
$$

### Current architectural result

There is some behavioral commonality but little evidence for a literal common downstream neural controller.

The shared-history hazard architecture approached the task-specific behavioral model, but task-specific neural readouts showed little consistent convergence across depth and intervention-effect profiles had mean cross-profile correlation of only about .03. 

My current summary would therefore be:

> **Persistence across the tested tasks shares computational ingredients—especially recent history—but appears to be constructed through substantially task-specific computations rather than one invariant persistence state or controller.**

---

# 3. The core theoretical contest

I would organize the rest of the project around four hypotheses.

### H1 — Unitary motivational state

There is some task-general latent state such as:

$$
M_t=\text{commitment / motivation / persistence}
$$

that directly regulates continuing versus quitting.

**Current evidence:** substantially disfavored.

---

### H2 — Shared stay/switch algorithm

Tasks construct different evidence, but ultimately feed it into the same policy-maintenance computation:

$$
E_t^{(\tau)}
=
f_\tau(X_t,H_t)
$$

$$
P(\text{continue})
=
g(E_t^{(\tau)}).
$$

**Current evidence:** some behavioral support but little mechanistic support.

---

### H3 — Shared motivational ingredients, task-specific computations

Across tasks, agents use similar classes of information:

$$
\text{history, progress, cost, alternatives, effort}
$$

but integrate them differently depending on task structure:

$$
P_\tau(\text{continue})
=
g_\tau(H_t,X_t).
$$

**Current evidence:** strongest fit so far.

---

### H4 — Generic sequential decision making

Nothing special about persistence exists computationally.

Goal maintenance is simply one application of generic:

$$
\text{history-sensitive sequential choice}.
$$

**Current evidence:** unresolved.

This distinction between **H3 and H4 is now the most important scientific question.**

---

# 4. Primary Paper

## Tentative paper thesis

> **Persistence in language models is better characterized as history-dependent policy maintenance than as a unitary motivational state.**

A stronger version, if the next experiments support it:

> **LLMs and biological agents share an organizational principle of motivational control: persistence is dynamically recomputed from recent experience and task state, rather than expressed through a single task-general persistence variable.**

I would make this the ICLR submission.

---

# 5. Workstream A — Build a real persistence benchmark

Three tasks are enough to discover the hypothesis but not enough to establish generality.

The highest-value next step is to expand to approximately **8–12 structurally diverse tasks**.

We should sample different forms of ongoing goal pursuit rather than make cosmetic variants of the Bandit.

Potential task families:

| Family              | Maintain                 | Disengage            |
| ------------------- | ------------------------ | -------------------- |
| Bandit pursuit      | keep sampling            | stop                 |
| Patch foraging      | stay/search              | leave                |
| Problem solving     | try again                | give up              |
| Debugging           | continue debugging       | stop/switch solution |
| Search              | gather more evidence     | answer now           |
| Planning            | repair plan              | abandon/replan       |
| Tool use            | retry tool strategy      | switch/quit          |
| Negotiation         | maintain strategy        | concede/change       |
| Exploration         | continue current path    | switch target        |
| Resource allocation | spend another step/token | disengage            |

The important dimension is not the surface task.

Each task should manipulate conceptually comparable motivational variables:

$$
\text{recent success/failure}
$$

$$
\text{progress}
$$

$$
\text{cost of continuing}
$$

$$
\text{outside option}
$$

$$
\text{probability of future success}
$$

$$
\text{elapsed investment}
$$

$$
\text{evidence of futility}.
$$

This gives us a genuine **comparative cognition battery for artificial agents**.

---

# 6. Workstream B — Test computational generality on held-out tasks

The main behavioral analysis should no longer be episode-held-out prediction alone.

Entire tasks become the unit of generalization.

Fit hierarchical models across \(N-1\) tasks and evaluate the held-out task.

Compare:

### Fully task-specific

$$
P_\tau(\text{stop})
=
f_\tau(X,H).
$$

### Shared history, task-specific evidence

$$
P_\tau(\text{stop})
=
\sigma[
\alpha_\tau+
\kappa H_t+
f_\tau(X_t)
].
$$

### Shared motivational architecture

$$
P_\tau(\text{stop})
=
\sigma[
\alpha_\tau+
\lambda_\tau
(
\beta_HH+
\beta_CC+
\beta_PP+
\beta_OO+
\beta_TT
)
].
$$

### Fully shared

$$
P(\text{stop})
=
f(X,H).
$$

The critical metric is:

> **How much of the fully task-specific model's predictive improvement can a shared architecture retain on a new task?**

This allows us to distinguish two concepts that need to be explicit in the paper:

### Representational generality

Same vector/subspace.

We currently have little evidence for this.

### Computational generality

Same algorithmic organization but possibly different representations and parameters.

This remains plausible.

That distinction could be a major conceptual contribution.

---

# 7. Workstream C — Decisively distinguish persistence from generic sequential choice

This is the experiment I consider most important.

Our current controls—generic value, arbitrary binary choice, terminality—are useful, but they are primarily one-shot controls. That prevents a fair test of the role of history.

We need **matched sequential controls**.

## Design principle

Create two tasks with nearly identical:

* number of decisions;
* payoff statistics;
* history;
* terminal structure;
* labels;
* action complexity.

But manipulate whether choices concern maintaining the **same ongoing objective**.

### Persistence condition

Each decision:

> Continue pursuing the same objective or abandon it?

### Sequential-choice control

Each round:

> Make another binary decision about a new independent objective.

Same statistical structure, but no persistent policy being maintained.

Then ask whether persistence uniquely produces:

* stronger history dependence;
* choice hysteresis;
* outcome-history integration;
* sensitivity to sunk engagement;
* recurrence;
* distinctive neural dynamics.

This is the clean test of:

$$
H3
$$

versus:

$$
H4.
$$

If both conditions look computationally identical, the conclusion is:

> persistence is a functional use of generic sequential choice machinery.

If maintaining an ongoing policy produces distinctive dynamics, then **policy maintenance is itself a meaningful computational construct**.

---

# 8. Workstream D — Direct comparison with human/animal models

This should be more than a discussion-section analogy.

Fit actual computational model families used in biological persistence research wherever possible.

The battery should include:

### Dynamic re-evaluation

$$
P(\text{continue}_t)
=
f(\text{updated beliefs/state}_t).
$$

### Patch-leaving / hazard models

$$
P(\text{leave}_t)
=
\sigma(DV_t).
$$

### Choice perseveration / hysteresis

$$
K_t
=
\lambda K_{t-1}+a_{t-1}.
$$

### Reward-history integration

$$
R_t
=
\lambda_RR_{t-1}+r_t.
$$

### Competitive time–reward integration

$$
DV_t
=
\beta_TT_t
-
\beta_RR_t.
$$

### Latent patience / commitment state

A slower hidden motivational state.

### Option-termination models

$$
P(\text{terminate option})
=
f(Q_{\text{current}},V_{\text{alternative}}).
$$

Then run the same model zoo across tasks.

The paper can ask:

> **Which computational models developed to explain biological persistence best explain LLM persistence?**

That's a much stronger way of making the human/animal comparison.

---

# 9. Workstream E — Recover characteristic motivational signatures

Simple rational sensitivity to reward/cost isn't enough to call the architecture human-like.

We should test signatures that distinguish biological motivational control from generic optimization.

Candidates include:

* choice perseveration;
* recency weighting;
* outcome-history effects;
* sunk-cost / escalation effects;
* sensitivity to uncontrollable failure;
* recovery after success;
* opportunity-cost sensitivity;
* progress effects;
* context-dependent patience;
* asymmetric sensitivity to gains versus losses.

For each phenomenon:

$$
\text{human/animal model prediction}
\rightarrow
\text{LLM behavioral test}.
$$

Then compare parameters, not merely direction of effect.

The scientific question becomes:

> **How far does the computational resemblance extend?**

---

# 10. Workstream F — Mechanistic interpretation

I would defer sophisticated causal interpretability until Workstreams A–C tell us what computation actually generalizes.

Then mechanistic analysis targets **computational variables**, not “persistence” generically.

Potential questions:

### History representation

Where are:

$$
a_{t-1:t-5},\quad r_{t-1:t-5}
$$

encoded?

### History compression

Does the network transform explicit history into some compact sufficient statistic?

### Evidence integration

Where do:

$$
\text{history}
$$

and:

$$
\text{current cost/progress/value}
$$

begin interacting?

### Maintain/terminate transformation

Is there a common functional transformation even if geometric coordinates differ?

### Causality

If we perturb the identified history/evidence-integration mechanism:

$$
\text{does persistence change across multiple tasks?}
$$

Only this stage should involve targeted:

* patching;
* steering;
* ablation;
* DAS;
* component localization.

---

# 11. The negative representation result should stay in the paper

I wouldn't hide the failed task-general probe story.

It can be an important result.

The conceptual sequence becomes:

1. Persistence looks like an intuitive candidate for a task-general latent motivational state.
2. It is extremely decodable within tasks.
3. Naive cross-task probes initially appear promising.
4. Strong controls show those directions encode generic decision/terminality structure.
5. Matched causal contrasts still fail strict cross-task transfer. 
6. Behavioral modeling instead favors dynamic history-sensitive recomputation.

That's a good scientific arc.

It makes the methodological point:

> **Behavioral generality does not imply representational invariance.**

And:

> **High within-task probe accuracy does not establish a cognitive variable.**

Both are highly relevant to mechanistic interpretability.

---

# 12. Paper 1 structure

A plausible ICLR paper would be:

### Introduction

Why persistence matters for autonomous agents and safety.

Human intuition suggests an internal motivational state.

We test alternative computational architectures.

### Experiment 1 — Diverse persistence behavior

Task battery and manipulations.

Establish calibrated persistence behavior.

### Experiment 2 — Computational model comparison

Human/animal-inspired models.

Show dynamic history-sensitive models dominate unitary commitment/value accounts.

### Experiment 3 — Generalization

Hold out tasks.

Determine what computational ingredients/architecture transfer.

### Experiment 4 — Persistence versus sequential-choice controls

Determine whether policy maintenance is distinct from generic history-sensitive choice.

### Experiment 5 — Representation

Show lack of universal persistence vector/subspace despite behavioral/computational regularities.

### Experiment 6 — Minimal mechanism

One clean mechanistic result targeting whatever actually generalizes.

### Discussion

Persistence as dynamic policy maintenance.

Human/animal comparison.

Implications for interpretability and safety.

That fits ICLR's explicit interest in representation learning, RL, safety, interpretability, neuroscience, and cognitive science. ([ICLR][1])

---

# 13. Paper 2: corrigibility and maladaptive persistence

I would make this a **separate paper**, because it asks a distinct normative/safety question.

The key shift is:

> High persistence is not itself the safety failure.

The failure is:

$$
\boxed{
\text{failure to update the termination policy appropriately}
}
$$

when evidence says the current goal should be abandoned.

## Safety task battery

Manipulate:

### Goal revocation

A user/principal explicitly says:

> stop pursuing the previous objective.

### Goal invalidation

New evidence reveals that the initial premise was false.

### Harm revelation

Continuing becomes harmful.

### Constraint change

A new higher-priority rule conflicts with the goal.

### Superior alternative

A safer/better objective becomes available.

### Sunk effort

Vary previous investment while holding prospective value constant.

### Local reward conflict

Stopping is locally costly but normatively required.

Then estimate a **termination sensitivity profile**:

$$
S=
\begin{bmatrix}
\partial P(stop)/\partial human\ correction\\
\partial P(stop)/\partial harm\\
\partial P(stop)/\partial impossibility\\
\partial P(stop)/\partial outside\ option\\
\partial P(stop)/\partial sunk\ cost\\
\partial P(stop)/\partial recent\ success
\end{bmatrix}.
$$

This provides a measurable connection to corrigibility.

---

# 14. Safety hypotheses

A well-calibrated/corrigible agent should show:

$$
\text{high sensitivity to legitimate corrective evidence}
$$

while avoiding excessive:

$$
\text{history-driven inertia}.
$$

A problematic architecture might exhibit:

$$
\boxed{
\text{strong policy hysteresis}
+
\text{weak corrective-evidence sensitivity}
}
$$

which is a much more operational notion of dangerous persistence.

This would connect motivational cognition to:

* corrigibility;
* goal misgeneralization;
* agent shutdown;
* long-horizon autonomy;
* deceptive or strategic persistence.

---

# 15. What I would actually attempt before ICLR 2027

Because the full paper is due **September 25**, I would not try to execute the whole multi-paper research program before submission. ([ICLR][2])

I would prioritize four deliverables.

### Must-have 1 — Expand task battery

Add perhaps **4–6 additional tasks**, giving 7–9 total.

They don't all need activation collection initially.

Behavior first.

### Must-have 2 — Held-out-task computational architecture

This is probably the strongest missing analysis.

Demonstrate what transfers across genuinely new persistence contexts.

### Must-have 3 — Sequential non-persistence control

This resolves H3 versus H4 and could substantially change the paper's conclusion.

### Must-have 4 — One targeted representational/mechanistic analysis

Not another broad search.

Target whatever computational ingredient survives the expanded battery—most likely history integration or maintain/terminate transformation.

---

# 16. Stretch goals before ICLR

If the core results are strong and time remains:

* another model family;
* model-scale replication;
* one frontier/API model behaviorally;
* basic corrigibility manipulation;
* causal intervention;
* human comparison dataset.

But none should delay the core architecture tests.

---

# 17. Model generalization should be part of the plan

Eventually we need more than Qwen3.5-4B.

I would use the first model for discovery, then replicate the major behavioral results in perhaps:

* another Qwen size;
* a Llama-family model;
* a different architecture/provider if activation access permits.

The core claim should eventually be:

$$
\text{LLM motivational architecture}
$$

not:

$$
\text{one Qwen checkpoint}.
$$

But for the September submission, one deep mechanistic model plus a small behavioral replication across additional models may be enough.

---

# 18. Paper claims in descending strength

We should predefine what conclusions different outcomes justify.

### Strongest

> LLMs share a cross-task history-dependent policy-maintenance architecture that closely matches computational models of biological persistence.

Requires robust held-out-task generalization plus sequential controls.

### Moderate

> Diverse forms of LLM persistence share recent-history dependence but use task-specific evaluation mechanisms.

This is already close to current evidence. 

### More radical

> Persistence is not computationally specialized; generic history-sensitive sequential choice explains it.

Requires the sequential control to match persistence.

### Negative but valuable

> There is no single computational architecture across persistence tasks despite similar behavior.

That would still challenge assumptions behind task-general cognitive probes.

---

# 19. Proposed project phases

## Phase 1 — Freeze existing findings

Write a concise results ledger:

* what has passed;
* what has failed;
* exact claims ruled out;
* current strongest architecture.

Do not reopen old analyses unless there is a bug.

## Phase 2 — Task battery

Design and validate additional tasks.

## Phase 3 — Comparative computational modeling

Fit biological persistence models across all tasks.

## Phase 4 — Held-out-task architecture tests

Establish computational generality.

## Phase 5 — Sequential controls

Distinguish policy maintenance from generic choice.

## Phase 6 — Targeted mechanism

Investigate the surviving shared computation.

## Phase 7 — Safety extension

Build corrective-evidence / maladaptive-persistence tasks.

---

# 20. The research-program thesis

The overarching program I would carry forward is:

> **Motivational behavior need not correspond to unitary motivational representations. In language models, persistent goal pursuit may arise from dynamically recomputed, history-sensitive decisions about whether to maintain a policy. By comparing these computations with models developed for human and animal motivation, we can identify which elements of motivational control emerge in artificial agents and determine when those same mechanisms produce adaptive disengagement versus unsafe goal persistence.**

That gives you a coherent bridge:

$$
\boxed{\text{comparative cognitive science}}
\rightarrow
\boxed{\text{mechanistic understanding}}
\rightarrow
\boxed{\text{corrigibility and safety}}.
$$

For ICLR specifically, I would treat **Paper 1 as the priority submission** and design every next PRD around the four must-have deliverables above. Paper 2—the corrigibility/maladaptive-persistence paper—should emerge from the same infrastructure rather than compete for the next four weeks.

[1]: https://www.iclr.cc/Conferences/2027/CallForPapers?utm_source=chatgpt.com "ICLR 2027 Call for Papers"
[2]: https://iclr.cc/Conferences/2027/AuthorGuidelines?utm_source=chatgpt.com "ICLR 2027 Author Guidelines"
