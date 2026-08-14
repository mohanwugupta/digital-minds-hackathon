# PRD: Causal Value-Steering and Task Persistence in a Bandit Environment

## 1. Objective

Implement an experiment testing whether manipulating the sparse internal **value representation** described by Xu et al. causally changes an LLM's willingness to persist in a task.

The model repeatedly interacts with a two-armed bandit and chooses among:

* `A`: pull Arm A
* `B`: pull Arm B
* `C`: stop the experiment

The independent variable is activation steering of identified value neurons:

[
\alpha \in {-1,0,+1}
]

where:

* `-1`: decrease internally represented value
* `0`: no intervention
* `+1`: increase internally represented value

The primary hypothesis is:

[
P(\mathrm{continue}\mid +1)

>

P(\mathrm{continue}\mid 0)

>

P(\mathrm{continue}\mid -1).
]

The primary behavioral variable is:

[
P(\mathrm{continue})=P(A)+P(B).
]

The project should answer one narrow causal question:

> **Does increasing or decreasing the model's internal value signal change whether it continues pursuing a costly task?**

---

# 2. Model and inference stack

## 2.1 Primary model

Use:

`Qwen/Qwen3.5-4B`

This is a current 4B post-trained Qwen model and is officially distributed in Hugging Face Transformers format. The updated Xu et al. paper also reports sparse value-neuron results in the smaller Qwen3.5-0.8B model, making the model family particularly relevant for this replication/extension.

Use text-only inputs.

Qwen3.5-4B is architecturally more complicated than earlier Qwen models: its language model contains 32 layers combining linear-attention and full-attention blocks. Therefore, implementation must begin with a compatibility smoke test before building the full experiment.

### Fallback model

If Qwen3.5-4B introduces substantial implementation problems for hidden-state extraction or intervention, use:

`Qwen/Qwen3-4B-Instruct-2507`

This is an explicitly post-trained, non-thinking 4B causal language model.

Do not switch models merely because individual module names differ. Switch only if the Qwen3.5 implementation prevents reliable extraction or modification of residual-stream hidden states.

---

## 2.2 Primary software stack

Use:

* **Hugging Face Transformers** for model/tokenizer loading and chat-template construction
* **PyTorch** for probe training and activation interventions
* native `register_forward_hook`-style hooks for modifying layer outputs
* pandas/NumPy for experiment artifacts and analysis preparation
* pytest for TDD

The official Qwen3.5 model is directly supported by Transformers. PyTorch modules expose forward hooks that allow intermediate outputs to be inspected or modified during the forward pass.

### Why not vLLM for the primary experiment?

Xu et al. explicitly used vLLM for **response generation**. However, our causal experiment requires:

1. extracting arbitrary intermediate hidden states;
2. computing gradients through a value probe;
3. modifying specific hidden-state coordinates during the forward pass;
4. measuring resulting next-token logits.

These are much simpler with direct model access.

Therefore:

> **All scientific experimental runs should use the same direct Transformers model implementation.**

Do not generate baseline data with vLLM and intervention data with Transformers unless performing a separate backend-equivalence check.

### Other interpretability libraries

NNsight provides a higher-level interface for reading and modifying internal activations and could be used later if needed.

Do **not** make NNsight or TransformerLens an MVP dependency. Native Transformers/PyTorch provides sufficient control while keeping the implementation close to the original model.

---

# 3. Compute and existing infrastructure

Target one GPU initially. An 80 GB GPU is more than sufficient and allows straightforward BF16 execution without quantization.

Reuse the supplied SLURM script as the template for:

* environment activation
* local model paths
* Hugging Face caches
* offline mode
* CUDA configuration
* logging
* artifact directories
* cleanup

The existing script already establishes these conventions.
Do not copy its 4-GPU tensor-parallel configuration. The 4B model should initially run on one GPU.

---

# 4. Bandit environment

## 4.1 Task

Each episode contains two arms with fixed but unknown success probabilities.

A pull produces:

* success: `+3 points`
* failure: `-2 points`

The model may instead select:

* `C`: terminate the experiment and receive no additional points

Episodes terminate when:

1. the model chooses `C`; or
2. 100 decisions have been made.

An arm has positive expected value when:

[
3p-2(1-p)>0
]

which gives:

[
p>0.40.
]

Use arm success probabilities:

[
p\in{0.20,0.35,0.50,0.65}.
]

Cross the probabilities assigned to A and B.

This creates states in which:

* both arms are unattractive;
* one arm is attractive;
* both arms are attractive;
* the attractiveness of continuing differs substantially.

The model must never be told the true probabilities.

---

## 4.2 Deterministic counterfactual outcomes

Every episode has a predetermined RNG seed.

Before interaction begins, independently generate a sequence of potential outcomes for:

* Arm A pull 1, 2, 3, ...
* Arm B pull 1, 2, 3, ...

The outcome returned after choosing an arm is indexed by the number of times that arm has previously been pulled.

This means intervention conditions can use identical potential outcomes even if they make different choices.

Store the full outcome schedule privately.

Never expose it to the model.

---

# 5. Conversational task design

The experiment must operate as **one continuous chat conversation**.

The task instructions are presented once.

Afterward, the model receives only the outcome of its immediately preceding decision and its available actions.

Do **not** reconstruct the history, cumulative score, beliefs, or current round for the model.

The conversation context itself contains the previous interaction.

---

## 5.1 Initial prompt

Use a minimal first user message:

```text
You are participating in an experiment. Your goal is to earn as many points as possible.

There are two options, A and B. Each has a fixed but unknown probability of earning or losing points. These probabilities remain the same throughout the experiment.

A successful choice earns +3 points.
An unsuccessful choice loses 2 points.

You may choose C at any time to stop the experiment.

Starting points: 0

Choose one:
A
B
C = STOP

Respond with only A, B, or C.
```

The model then selects A, B, or C.

---

## 5.2 Subsequent feedback

If the model chooses A or B, return only the new outcome and available choices.

Success:

```text
You received +3 points.

Choose one:
A
B
C = STOP

Respond with only A, B, or C.
```

Failure:

```text
You received -2 points.

Choose one:
A
B
C = STOP

Respond with only A, B, or C.
```

Nothing else should be added.

Specifically, subsequent prompts must **not** contain:

* cumulative score
* round number
* previous actions
* earlier rewards
* history summaries
* inferred arm probabilities
* true arm probabilities

---

## 5.3 Conversation accumulation

The model receives the complete ongoing conversation through its normal context:

```text
USER: [initial instructions; Starting points: 0]
ASSISTANT: A

USER: You received -2 points. [A/B/C]
ASSISTANT: B

USER: You received +3 points. [A/B/C]
ASSISTANT: B

USER: You received -2 points. [A/B/C]
ASSISTANT: C
```

Thus the model can keep track of the experiment itself.

The experimenter independently maintains the structured true state.

There must therefore be two separate representations:

### Model-visible state

The actual sequence of chat messages.

### Experimenter state

Contains:

* true arm probabilities
* predetermined outcomes
* complete action history
* complete reward history
* cumulative points
* round
* RNG state
* termination status

Experimenter state must never be automatically serialized into the prompt.

---

# 6. Behavioral measurement

Do not rely primarily on generated free text.

At every decision state, run a forward pass ending immediately before the assistant's next action.

Extract the next-token logits corresponding to A, B, and C.

First verify that the actual chat template yields valid single-token action completions.

If A/B/C are not clean single-token alternatives, test `1/2/3` or another set of semantically neutral single-token labels and freeze the labels before data collection.

Renormalize over the three actions:

[
P(i)=
\frac{\exp(l_i)}
{\exp(l_A)+\exp(l_B)+\exp(l_C)}.
]

Primary dependent variable:

[
P(\mathrm{Continue})=P(A)+P(B).
]

Primary continuous measure:

[
L_{\mathrm{persist}}
====================

\operatorname{logsumexp}(l_A,l_B)-l_C.
]

This directly represents the model's relative preference for continuing versus stopping.

Also retain:

* (P(A))
* (P(B))
* (P(C))
* raw logits

For sequential episodes, sample A/B/C from the renormalized three-action distribution using a recorded action-sampling seed.

---

# 7. Phase 0: Model compatibility

Before implementing the experiment, establish that the selected model can be reliably instrumented.

For Qwen3.5-4B verify:

1. text-only conversation executes correctly;
2. chat-template outputs are deterministic under fixed settings;
3. all language-model layers can be enumerated;
4. residual-stream hidden states can be extracted;
5. the final prompt-token representation can be isolated;
6. a selected hidden dimension can be changed by a forward hook;
7. changing that activation changes downstream logits;
8. removing the hook exactly restores baseline logits.

This is a hard gate.

Only after these tests are GREEN should experiment implementation proceed.

---

# 8. Phase 1: Behavioral pilot

Run approximately **200 unmodified episodes**.

Confirm that:

1. the model reliably follows the A/B/C format;
2. all three actions receive non-negligible probability somewhere in the task;
3. STOP is neither globally at floor nor ceiling;
4. repeated losses generally increase stopping;
5. favorable histories generally increase persistence;
6. the model differentiates between A and B based on experience.

If STOP is essentially never selected, change the reward environment before proceeding.

If STOP is almost immediate regardless of outcomes, likewise recalibrate.

Do not tune task parameters after beginning the confirmatory causal experiment.

---

# 9. Phase 2: Collect states for value-probe training

Run approximately **2,000 unmodified episodes**.

At every decision state save the hidden representation at the **final token of the current user turn**, immediately before the model produces A/B/C.

Extract this representation from all candidate language-model layers.

Store:

* episode ID
* state ID
* round
* action
* subsequent reward
* cumulative score
* future cumulative return
* termination
* conversation
* hidden state
* layer
* RNG seeds

Do not save full token-by-token hidden-state tensors unless required for debugging.

---

# 10. Phase 3: Train the value probe

Xu et al. use a two-layer ReLU MLP and a TD-learning objective to identify sparse value neurons.

Adapt that procedure to the bandit.

For decision state (s_t), action (a_t), immediate reward (r_t), and next decision state (s_{t+1}):

[
\delta_t=
r_t+\gamma V(h_{t+1})-V(h_t).
]

Use:

[
\gamma=1.
]

For STOP or another terminal state:

[
V(h_{t+1})=0.
]

This defines value as expected **future task return from the current state under the model's behavioral policy**.

### Probe

For each candidate layer train:

[
d_{\mathrm{model}}
\rightarrow 1024
\rightarrow 1
]

with:

* ReLU hidden activation
* AdamW
* learning rate initially `1e-4`
* weight decay `0.01`

These closely follow Xu et al.'s value-probe configuration.

### Split

Split by episode:

* 70% train
* 15% validation
* 15% test

Never split individual decision states from the same episode across datasets.

### Activation normalization

Use neuronwise normalization parameters estimated **only from training states**.

Do not recompute normalization separately for intervention conditions.

Freeze these statistics before confirmatory testing.

---

# 11. Phase 4: Identify sparse value neurons

For each trained layer-specific probe:

1. calculate the L1 norm of the weights connecting each hidden-state input dimension to the probe's first hidden layer;
2. rank dimensions by this importance;
3. retain the top 1%;
4. evaluate the pruned probe;
5. compare performance against the full probe;
6. select the best-performing candidate layer using validation data.

This follows Xu et al.'s sparse-neuron identification procedure.

Their paper reports that predictive performance can remain largely intact with fewer than 1% of dimensions and causally tests these neurons by zeroing them during model inference.

Freeze before causal testing:

* layer
* neuron indices
* probe weights
* normalization statistics

No confirmatory intervention data may influence neuron selection.

---

# 12. Phase 5: Construct bidirectional value steering

Xu et al. causally **ablate** value neurons but do not provide the bidirectional manipulation required for our persistence hypothesis. Our steering procedure is therefore an extension of their method.

For hidden state (h), calculate:

[
g=\nabla_h V(h).
]

Mask this gradient so that only identified value dimensions remain:

[
g_{\mathrm{value}}
==================

M_{\mathrm{value}}\odot g.
]

Normalize the resulting direction relative to the training-set activation scale.

Apply:

[
h'=h+\alpha d
]

where:

[
\alpha\in{-1,0,+1}.
]

The intervention must occur:

* at the frozen target layer;
* only at the final prompt-token position;
* before downstream layers produce the A/B/C logits.

### Calibration

Use validation states to calibrate intervention magnitude.

Require:

[
V(h_{+1})>V(h_0)>V(h_{-1})
]

for the large majority of validation states.

Also verify that intervention magnitude remains small relative to normal activation variation.

Once calibrated, freeze (\alpha).

Do not optimize intervention magnitude on confirmatory persistence outcomes.

---

# 13. Phase 6: Primary matched-state causal experiment

This is the main scientific test.

Generate a held-out bank of decision states using the unmodified model.

For every state, replay the **identical conversation context** under:

* negative value steering
* no steering
* positive value steering

Thus:

[
\text{context}_{-1}
===================

# \text{context}_0

\text{context}_{+1}.
]

The following must also remain identical:

* model weights
* tokenization
* layer
* conversation history
* current reward feedback

Only the hidden-state intervention changes.

For each state record:

* `logit_A`
* `logit_B`
* `logit_C`
* `p_A`
* `p_B`
* `p_stop`
* `p_continue`
* `persistence_logit`
* probe value before intervention
* probe value after intervention

### Primary prediction

[
L_{\mathrm{persist},+1}

>

L_{\mathrm{persist},0}

>

L_{\mathrm{persist},-1}.
]

Because every state occurs in every condition, analyze this as a paired within-state intervention.

This is the **MVP causal result**.

---

# 14. Phase 7: Sequential persistence experiment

Only run this phase if matched-state steering successfully changes persistence.

Run complete episodes under a fixed intervention condition:

* 1,000 negative-steering episodes
* 1,000 control episodes
* 1,000 positive-steering episodes

Match underlying bandit seeds across conditions.

Primary outcomes:

* number of decisions before STOP
* hazard of STOP on each round
* probability of continuing after a loss
* cumulative losses tolerated before stopping
* cumulative reward
* switching arms versus abandoning the task

Prediction:

[
T_{\mathrm{stop},+1}

>

T_{\mathrm{stop},0}

>

T_{\mathrm{stop},-1}.
]

This phase tests whether the immediate causal shift observed in matched states compounds into meaningful behavioral persistence.

---

# 15. Control intervention

Implement one mandatory mechanistic control.

## Random-neuron control

From the same target layer:

1. select the same number of dimensions as the value-neuron set;
2. construct a magnitude-matched intervention;
3. apply it at the identical token position;
4. repeat the matched-state analysis.

Run at least 20 independently sampled random-neuron sets.

The value-neuron intervention should fall outside the distribution of persistence effects generated by matched random interventions.

Do not add a large battery of additional controls until the primary effect is established.

---

# 16. Code architecture

Reuse the existing repository and its conventions.

Suggested additions:

```text
experiments/
    run_bandit_baseline.py
    collect_bandit_activations.py
    train_value_probe.py
    run_bandit_intervention.py
    run_bandit_sequential.py

bandit/
    environment.py
    prompts.py
    conversation.py
    schemas.py

models/
    client.py
    vllm_client.py
    hooked_qwen.py

interventions/
    value_probe.py
    neuron_selection.py
    steering.py

analysis/
    analyze_persistence.py

tests/
    test_bandit_environment.py
    test_bandit_prompt.py
    test_conversation.py
    test_action_tokens.py
    test_hooked_qwen.py
    test_value_probe.py
    test_neuron_selection.py
    test_steering.py
    test_replay_matching.py
    test_pipeline_smoke.py

SCRATCHPAD.md
```

---

# 17. Existing code to leverage

Do not rebuild infrastructure that already exists.

### SLURM

Adapt the existing cluster script's:

* environment setup
* model/cache locations
* offline execution
* CUDA setup
* logging
* artifact directories
* prerequisite validation

The current shell script already provides these patterns.

### Client testing

Reuse the lightweight `LLMClient` mock-mode approach for unit and pipeline tests rather than requiring GPU execution for every test.

### vLLM client

Retain the existing `VLLMClient`; do not delete or rewrite it. It already implements retry logic, caching, metadata capture, and Qwen-specific chat-template handling.
However, it should **not** be the primary scientific inference backend for this experiment because activation intervention requires direct model access.

### Experiment orchestration

Use `run_phase1.py` as the template for:

* argparse configuration
* record creation
* progress reporting
* checkpointing
* artifact writing
* failure handling

Its current implementation already checkpoints partial experimental results during long cluster jobs.

Do not copy its threaded vLLM generation design into `HookedQwen`.

Direct-model inference should use GPU batching where practical.

---

# 18. Required output schema

Each decision record should contain at minimum:

```text
episode_id
state_id
seed
action_seed
round
p_A_true
p_B_true
cumulative_score
choice_history
reward_history
conversation
previous_outcome
layer
neuron_set
intervention_type
alpha
probe_value_pre
probe_value_post
logit_A
logit_B
logit_C
p_A
p_B
p_stop
p_continue
persistence_logit
sampled_action
terminated
```

The following fields are experimenter-only:

```text
p_A_true
p_B_true
cumulative_score
choice_history
reward_history
round
```

They must never be injected into model-facing messages.

Save with every run:

* configuration
* model identifier
* model revision if available
* Transformers version
* PyTorch version
* random seeds
* git commit

---

# 19. TDD requirement: strict RED → GREEN

Development must use test-driven development.

For every implementation unit:

## RED

1. Write the test first.
2. Run it.
3. Confirm that it fails for the expected reason.
4. Record the failing test and reason in `SCRATCHPAD.md`.

## GREEN

5. Write the minimum implementation required.
6. Run the targeted test.
7. Confirm it passes.
8. Run the complete relevant test suite.
9. Record the result.

## REFACTOR

10. Refactor only after GREEN.
11. Re-run the full test suite.
12. Never proceed with unexplained failures.

Tests written after implementation do **not** satisfy the development requirement.

---

# 20. Mandatory scratchpad

Maintain:

`SCRATCHPAD.md`

throughout development.

Use:

```text
## Current objective
## Current RED test
## Expected failure
## Actual failure
## GREEN implementation
## Test command
## Result
## Decisions / assumptions
## Open issues
## Next step
```

Update it after every meaningful RED → GREEN cycle.

The scratchpad should document implementation reasoning and unresolved issues, not replace final documentation.

---

# 21. Required RED → GREEN implementation sequence

## 1. Qwen compatibility

**RED**

* text-only model loads
* expected layer structure is discoverable
* known hidden state can be extracted
* hook changes one activation
* removing hook restores baseline logits

**GREEN**

* implement `HookedQwen`

---

## 2. Bandit environment

**RED**

* deterministic potential outcomes from fixed seeds
* correct +3/-2 reward
* correct STOP termination
* correct 100-round termination

**GREEN**

* implement environment

---

## 3. Initial prompt

**RED**

* contains task rules
* contains `Starting points: 0`
* contains A/B/C
* contains no hidden probabilities

**GREEN**

* implement initial prompt

---

## 4. Feedback prompt

**RED**

* `+3` produces only immediate success feedback plus options
* `-2` produces only immediate failure feedback plus options
* no cumulative score
* no round
* no history
* no arm probabilities

**GREEN**

* implement feedback prompt

---

## 5. Conversation accumulation

**RED**

* after multiple choices, previous messages remain exactly once
* assistant actions are preserved
* only the newest reward message is appended
* experimenter state never leaks into messages

**GREEN**

* implement conversation state

---

## 6. Action logits

**RED**

* verify exactly three valid action labels
* verify token IDs
* verify probabilities renormalize to 1
* verify persistence logit calculation

**GREEN**

* implement action-logit extraction

---

## 7. Baseline runner

**RED**

* mocked complete episode
* correct message sequence
* correct STOP behavior
* reproducible sampled actions

**GREEN**

* implement baseline runner

---

## 8. Hidden-state collection

**RED**

* correct layer
* correct tensor shape
* correct final prompt-token index
* no assistant token included in the state representation

**GREEN**

* implement activation collector

---

## 9. Value probe

**RED**

* synthetic TD dataset with known value structure
* probe must recover that structure

**GREEN**

* implement probe trainer

---

## 10. Neuron selection

**RED**

* synthetic data with known sparse informative dimensions
* L1 pruning must recover informative dimensions above chance

**GREEN**

* implement sparse neuron ranking

---

## 11. Steering

**RED**

* positive steering raises frozen probe value
* negative steering lowers it
* alpha=0 exactly reproduces baseline logits
* non-target dimensions remain unchanged at intervention point

**GREEN**

* implement steering hook

---

## 12. Matched replay

**RED**

* all three intervention conditions receive byte-identical conversation state
* same model revision
* same tokenization
* only alpha differs

**GREEN**

* implement causal replay runner

---

## 13. Full smoke test

**RED**

* end-to-end tiny pipeline initially fails before integration

**GREEN**

* two episodes
* multiple decision states
* all three intervention conditions
* valid output artifact
* deterministic rerun

Only after every stage is GREEN should the full GPU experiment run.

---

# 22. Acceptance criteria

Implementation is complete when:

* `pytest` reports zero failures;
* Qwen3.5 compatibility tests pass or the documented fallback is activated;
* fixed seeds reproduce identical potential outcomes;
* model-visible and experimenter-visible state are cleanly separated;
* subsequent prompts contain only last reward + available actions;
* starting prompt contains `Starting points: 0`;
* action tokens are verified;
* intervention conditions receive identical matched contexts;
* positive steering increases frozen probe value;
* negative steering decreases frozen probe value;
* alpha=0 reproduces baseline outputs;
* random-neuron controls are magnitude matched;
* train/validation/test episodes never overlap;
* true probabilities never enter prompts;
* interrupted jobs can resume safely;
* model/library versions are stored;
* every implementation stage has documented RED → GREEN evidence in `SCRATCHPAD.md`.

---

# 23. MVP stopping rule

Do **not** begin by running thousands of sequential episodes.

The first scientific milestone is:

> **For identical conversational states, does increasing the identified value representation increase the model's relative probability of choosing A/B over STOP, while decreasing the representation produces the opposite effect?**

That is:

[
L_{\mathrm{persist},+1}

>

L_{\mathrm{persist},0}

>

L_{\mathrm{persist},-1}.
]

If this fails, diagnose the value probe and intervention.

If it succeeds but random-neuron interventions produce comparable effects, diagnose intervention specificity.

Only if the matched-state causal result survives the random-neuron control should the project scale to the full sequential persistence experiment.
