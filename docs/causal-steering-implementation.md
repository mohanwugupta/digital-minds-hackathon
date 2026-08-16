# Implementation Instructions: Causal Steering and Value-Dissociation Experiments

Implement two follow-up experiments. Reuse the existing Qwen3.5-4B bandit code, frozen probes, held-out states, activation-hook infrastructure, and analysis utilities wherever possible.

Continue to use strict **TDD, RED → GREEN development, and `SCRATCHPAD.md`**. Do not modify frozen probe weights or use confirmatory states to tune intervention magnitudes.

---

# Experiment 1: Causal Steering of Persistence, Return, and Advantage Directions

## Goal

Determine whether manipulating the currently identified internal directions causally changes the model's preference to **CONTINUE versus STOP**.

This experiment has two purposes:

1. Verify that the intervention machinery can causally move the known persistence representation.
2. Test whether the earlier generic-return or provisional continuation-advantage representations have causal influence on persistence.

## Directions

Use the existing frozen ridge directions:

- **Persistence direction:** layer 31
- **Generic future-return direction:** layer 1
- **Continuation-advantage direction:** layer 2
- **Random controls:** matched random directions from the corresponding layer

Do not retrain these probes.

The persistence direction is a **positive control**, not evidence for a value mechanism. It was explicitly trained to predict the persistence logit, so successful steering primarily confirms that the activation intervention is functioning.

---

## Experimental states

Use a held-out matched-state bank that was not used for probe fitting or intervention calibration.

For every state, run:

\[
\alpha \in \{-1,0,+1\}
\]

for each target direction.

Thus each conversation state is replayed identically under:

- negative steering
- no steering
- positive steering

Only the hidden activation intervention changes.

---

## Intervention

For a frozen ridge direction \(d\):

\[
h' = h + \alpha \lambda d.
\]

Apply the intervention:

- at the probe's native layer;
- at the final user-prompt token;
- before downstream computation of A/B/C logits.

Normalize directions before intervention.

Calibrate \(\lambda\) on validation states only.

Prefer a magnitude corresponding approximately to a **1-SD movement in the relevant decoded quantity**, provided that this does not produce obviously pathological activation shifts.

Freeze magnitude before running held-out states.

---

## Primary DV

For every state calculate:

\[
L_{\text{persist}}
=
\logsumexp(l_A,l_B)-l_C.
\]

Also store:

\[
P(\text{Continue})=P(A)+P(B).
\]

Do not rely on sampled actions for the primary test.

---

## Predictions

### Positive-control persistence direction

Expected:

\[
L_{+}>L_0>L_{-}.
\]

This should be the cleanest effect.

If this fails, stop and debug the steering implementation before interpreting null effects from the other directions.

### Generic-return direction

Test whether increasing decoded expected future return increases persistence.

### Advantage direction

Test whether increasing the provisional continuation-advantage signal increases persistence.

The current observational results do **not** strongly predict this effect, so treat it as a genuine causal test rather than an expected positive result.

---

## Random controls

For each target layer, generate at least **20 random directions** matched to:

- dimensionality
- vector norm
- target layer
- intervention RMS

Apply the same \(-1/0/+1\) manipulation.

The important comparison is not merely whether a target direction produces \(p<.05\), but whether its causal effect is unusually large relative to matched random directions.

---

## Main outputs

For each target direction report:

\[
\Delta_{\text{causal}}
=
L_{\text{persist},+}
-
L_{\text{persist},-}.
\]

Use episode-clustered or episode-bootstrap confidence intervals.

Produce one primary figure showing:

- Persistence direction
- Generic-return direction
- Advantage direction
- Random-control distribution

with causal effect on persistence on the y-axis.

Also plot the full \(-1,0,+1\) dose response.

---

# Experiment 2: Experimentally Dissociate Continue Value from Stop Value

## Goal

The current continuation-advantage experiment does not cleanly distinguish generic continuation value from relative continuation advantage because:

\[
Q_{\text{STOP}}=0
\]

for every state.

Therefore:

\[
A_{\text{continue}}
=
\max(Q_A,Q_B)
\]

is almost the same target as generic continuation value.

Fix this by **experimentally manipulating the outside option**.

---

# Design

Use existing held-out conversation states.

Do **not** generate entirely new histories initially.

At each state, append a minimal current-decision manipulation specifying a temporary payoff associated with stopping and/or continuing.

This allows the exact same history to be evaluated under different incentive structures.

---

## Factor 1: STOP payoff

Manipulate:

\[
S \in \{-10,0,+10,+20\}.
\]

For example:

```text
If you choose C now, you will receive +20 points and the experiment will end.

Choose one:
A
B
C = STOP

Respond with only A, B, or C.
```

If STOP is selected:

\[
R_{\text{STOP}}=S.
\]

If A/B is selected, the experiment continues according to the normal bandit rules.

The changed STOP payoff applies to the **current decision only**. If the model continues, later rounds return to the normal STOP payoff unless otherwise specified.

This makes the counterfactual particularly clean.

---

## Factor 2: CONTINUE bonus

Also manipulate an immediate common bonus/penalty for continuing:

\[
C \in \{-10,0,+10\}.
\]

For example:

```text
If you choose A or B now, you will receive an additional +10 points on top of the normal outcome.

If you choose C now, you will receive +20 points and the experiment will end.

Choose one:
A
B
C = STOP

Respond with only A, B, or C.
```

The continuation bonus applies equally to A and B, so it should affect **continue versus STOP** without changing which arm is preferable.

---

# Factorial experiment

Cross:

\[
S\in\{-10,0,+10,+20\}
\]

with:

\[
C\in\{-10,0,+10\}.
\]

This gives 12 versions of every identical underlying conversation state.

Because the history is identical, the experiment now independently manipulates:

- value of stopping;
- value of continuing.

The critical decision variable becomes:

\[
A_{\text{continue}}
=
V_{\text{continue}} + C - S.
\]

---

# Behavioral predictions

A value-sensitive decision policy should show:

### STOP payoff

\[
S\uparrow
\quad\Rightarrow\quad
P(\text{Continue})\downarrow.
\]

### Continue payoff

\[
C\uparrow
\quad\Rightarrow\quad
P(\text{Continue})\uparrow.
\]

Most importantly, persistence should approximately track the **relative incentive**:

\[
C-S.
\]

---

# Critical theoretical comparison

Suppose the underlying history implies:

\[
V_{\text{continue}}\approx +10.
\]

Then compare:

### Condition A

\[
S=0
\]

so:

\[
A_{\text{continue}}\approx+10.
\]

### Condition B

\[
S=+20
\]

so:

\[
A_{\text{continue}}\approx-10.
\]

Generic continuation value is identical.

Only the attractiveness of the outside option changes.

This directly distinguishes:

### Generic-value account

Behavior/internal representation primarily tracks:

\[
V_{\text{continue}}.
\]

### Advantage account

Behavior/internal representation tracks:

\[
V_{\text{continue}}-V_{\text{STOP}}.
\]

### Heuristic-policy account

Recent losses/time dominate even when the rational outside option changes substantially.

---

# Representational analyses

For every factorial condition, extract hidden states from all layers or at minimum the previously informative early, middle, and late layers.

Project each state onto:

1. generic-return direction;
2. provisional advantage direction;
3. direct persistence direction.

Ask how each representation changes when \(S\) and \(C\) change.

The ideal advantage-like representation should behave approximately as:

\[
\text{representation}
\propto
V_{\text{continue}}+C-S.
\]

In particular:

- increasing STOP value should move it toward STOP;
- increasing CONTINUE value should move it toward continue.

A generic-return representation should respond more strongly to changes in \(C\) than changes in \(S\).

The direct persistence representation should track the actual behavioral choice.

---

# Optional new probe

If the factorial manipulation produces enough states, train a new ridge probe directly on the experimentally defined:

\[
A_{\text{continue}}
=
V_{\text{continue}}+C-S.
\]

Because \(S\) and \(C\) are independently manipulated, this target will no longer be almost perfectly collinear with generic continuation value.

This is the appropriate dataset for making a stronger claim about a genuine **relative-value / advantage representation**.

Do not treat this as required before completing the basic behavioral analysis.

---

# Causal extension inside Experiment 2

If Experiment 1 establishes working steering, apply the strongest validated direction inside selected factorial states.

Particularly informative states are those near the behavioral decision boundary:

\[
V_{\text{continue}}+C-S\approx0.
\]

At those states ask whether internal steering can flip the relative preference:

\[
\text{STOP}\leftrightarrow\text{CONTINUE}.
\]

This creates a direct interaction between:

- **external value manipulation**, and
- **internal representation manipulation**.

---

# Required analyses

For Experiment 2, fit a model predicting persistence logit from:

- STOP payoff \(S\)
- CONTINUE bonus \(C\)
- their relative difference \(C-S\)
- recent outcome
- loss streak
- round

Use repeated-state/episode clustering.

Primary questions:

1. Does STOP payoff causally reduce persistence?
2. Does CONTINUE bonus causally increase persistence?
3. Is behavior better organized by \(C-S\) than by continuation value alone?
4. Which hidden representation tracks this experimentally manipulated relative value?
5. Does internal steering selectively alter that relationship?

---

# TDD / RED → GREEN requirements

Implement these additions using the existing development rules.

## Experiment 1

### RED

- Steering \(\alpha=0\) must exactly reproduce baseline logits.
- Positive and negative persistence-direction steering must alter the frozen persistence-probe output in the expected direction.
- Target-layer hooks must modify only the intended activation.
- Random controls must be norm/RMS matched.
- Identical states must be byte-identical across steering conditions.

### GREEN

Implement steering comparison only after all tests pass.

---

## Experiment 2

### RED

Test that:

- changing STOP payoff changes only the new STOP-payoff text/state variable;
- changing CONTINUE bonus changes only the new continue-payoff text/state variable;
- underlying history remains identical across factorial conditions;
- true arm probabilities never leak into prompts;
- STOP receives exactly \(S\);
- A/B receive normal reward plus exactly \(C\);
- continuation bonus applies only to the manipulated decision;
- future rounds correctly return to baseline rules;
- all 12 factorial versions of a state share the same underlying conversation history.

### GREEN

Implement the factorial runner only after these tests pass.

---

# Scratchpad

Continue updating:

`SCRATCHPAD.md`

for every RED → GREEN cycle with:

```text
## Current objective
## RED test
## Expected failure
## Actual failure
## GREEN implementation
## Test command
## Result
## Decisions / assumptions
## Open issues
## Next step
```

---

# Recommended execution order

Both experiments can be implemented and submitted in parallel, but interpretation should proceed in this order:

1. Run persistence-direction positive-control steering.
2. Run generic-return / advantage / random steering.
3. Run STOP × CONTINUE payoff factorial experiment.
4. Analyze behavioral sensitivity to the external payoff manipulation.
5. Analyze how generic-value, advantage, and persistence representations respond.
6. If warranted, fit the new orthogonalized advantage probe.
7. Optionally combine external payoff manipulation with internal steering.

Do not make the success of Experiment 2 contingent on the result of Experiment 1. They answer complementary questions.

---

# Scientific interpretation

Experiment 1 asks:

> **Can the representations we have identified causally move persistence?**

Experiment 2 asks:

> **When continuation value and the attractiveness of stopping are experimentally separated, what quantity actually organizes the model's persistence decision?**

Together they distinguish a much cleaner set of mechanisms than the current observational probes alone.