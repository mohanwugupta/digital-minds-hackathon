# Development scratchpad

## Current objective

Deliver the PRD as a modular, resumable Transformers/PyTorch experiment while
preserving the old vLLM starter files and keeping private state out of prompts.

## Current RED test

The following tests were created before their corresponding implementation
units:

1. `test_hooked_qwen.py` and `test_action_tokens.py`: layer discovery, hidden
   extraction, action-token validity, logit changes under a hook, and exact
   restoration after hook removal.
2. `test_bandit_environment.py`: fixed-seed schedules, per-arm indexing,
   +3/-2 rewards, STOP, and horizon termination.
3. `test_bandit_prompt.py`: exact initial/feedback constraints and numeric-label
   fallback.
4. `test_conversation.py`: cumulative message order and private-state isolation.
5. `test_baseline_runner.py`: deterministic sampling, complete mocked episode,
   and clean STOP behavior.
6. `test_value_probe.py`: terminal TD targets, frozen normalization, and recovery
   of a known synthetic value function.
7. `test_neuron_selection.py`: recovery of known sparse input dimensions.
8. `test_steering.py`: bidirectional probe-value changes, target-only changes,
   exact alpha-zero identity, 20 random sets, exclusion, and magnitude matching.
9. `test_replay_matching.py`: byte-identical context across all alphas.
10. `test_pipeline_smoke.py`: deterministic two-episode, multi-state, three-alpha
    integration.

## Expected failure

Before implementation, collection should fail because `bandit`,
`models.hooked_qwen`, `interventions`, and the new experiment runners did not
exist. After implementation, the CPU tests should pass in the declared Conda
environment; the real Qwen test remains a separate GPU hard gate.

## Actual failure

The first requested RED command was:

```text
python3 -m pytest tests/test_bandit_environment.py tests/test_bandit_prompt.py
tests/test_conversation.py tests/test_action_tokens.py tests/test_schemas.py
```

The host stopped before collection with:

```text
/usr/bin/python3: No module named pytest
```

An environment inventory confirmed that pytest, NumPy, pandas, PyTorch, and
Transformers are all absent. Creating a temporary venv also failed because the
host lacks `ensurepip`. This is an execution-environment failure, not a test
result, and is not recorded as scientific GREEN evidence.

## GREEN implementation

- `HookedQwen` discovers Qwen3.5 and Qwen3 decoder layouts, verifies completions
  under the real chat template, extracts every final-position residual state,
  modifies only `[:, -1, :]`, and always removes hook handles in `finally`.
- The primary checkpoint's full `qwen3_5` config is loaded with the official
  conditional-generation class using text-only token inputs; the fallback's
  ordinary causal-LM config uses `AutoModelForCausalLM`.
- The environment pre-generates independent per-arm potential outcomes and
  maintains private structured state separately from `BanditConversation`.
- The default prompts exactly reproduce the PRD. If A/B/C are not clean actual
  chat-template tokens, 1/2/3 are tested and frozen consistently.
- Baseline action probabilities are renormalized over three logits and sampled
  with an independent recorded seed.
- Activation collection saves one atomic episode shard with only final-position
  vectors for all candidate layers.
- Value probes use frozen train-only neuronwise statistics, a 1024-unit ReLU
  hidden layer by default, AdamW, and semi-gradient TD(0) with gamma 1.
- Sparse neurons are first-layer input dimensions ranked by outgoing L1 norm;
  the top 1% is evaluated and the layer is selected using pruned validation TD
  MSE.
- Steering uses the frozen probe gradient, value-dimension masking, and a
  training-activation-scale norm. Calibration is validation-only.
- Confirmatory replay hashes canonical contexts, rejects overlap with every
  probe split, and evaluates value plus at least 20 excluded random-neuron sets.
- Sequential episodes are blocked unless both matched-state gates pass.
- Long jobs resume from complete CSV episodes or per-episode atomic shards.
- Configuration, model/revision, library versions, all seeds, and Git commit are
  saved alongside runs.

## Test command

Intended complete command in the project environment:

```bash
pytest -q
```

Commands actually available and run on this host:

```bash
python3 -m compileall -q bandit models interventions experiments analysis tests
bash -n run_qwen35_bandit.sh
python3 - <<'PY'
# deterministic stdlib smoke assertions for environment, prompts,
# conversation, episode splitting, action metrics, baseline, and replay
PY
```

The cluster compatibility command is:

```bash
python -m experiments.check_qwen_compatibility --model /path/to/Qwen3.5-4B
```

## Result

- Full source compilation: PASS.
- SLURM shell syntax: PASS.
- Dependency-free mocked core smoke checks: PASS (four decisions through STOP,
  deterministic rerun).
- A later expanded smoke command initially failed because its hand-written
  expected numeric `p_continue` constant was incorrect. Replacing that check
  with the defining normalization invariant made the same command pass; no
  implementation change was needed.
- pytest suite: NOT RUN on this host because pytest and scientific dependencies
  are absent.
- Qwen3.5 GPU compatibility: NOT RUN; local weights/GPU are absent. It remains a
  hard gate in code and documentation, with fallback attempted only after a real
  primary-model instrumentation failure.

## Decisions / assumptions

- A decision includes A, B, or C; the 100-decision cap therefore counts the
  terminating decision if it occurs at that boundary.
- STOP has reward zero in private reward history but never creates another user
  feedback message.
- Potential-outcome streams are deterministically derived from the episode seed
  and are independent between arms.
- Future cumulative return includes the current transition reward through the
  terminal transition.
- Layer selection uses pruned validation TD MSE because the causal intervention
  uses the pruned dimensions.
- Alpha zero deliberately runs with no hook, giving an exact baseline rather
  than relying on a mathematically zero mutation.
- Random-neuron controls exclude selected value neurons, use deterministic signs,
  and match whitened intervention norm per state.
- Confirmatory state-bank seeds must differ from all probe-bank seeds; code also
  validates IDs against the frozen episode split.
- Chat-template generation markers are part of the prompt. No generated
  assistant action token is present when the state is extracted.

## Open issues

- The declared Conda environment must be created before `pytest -q` can provide
  the acceptance-criterion result.
- Qwen3.5-4B model weights and an 80 GB GPU are needed for the mandatory real
  compatibility gate.
- Pilot task calibration is a scientific decision and must be made from the
  200-episode pilot before probe or confirmatory data collection.

## Next step

On the cluster, create/activate `value-steering-bandit`, run `pytest -q`, then
submit `PHASE=compatibility sbatch run_qwen35_bandit.sh`. Proceed to the pilot
only if both are GREEN.

---

## Reward-design and cleanup cycle

### Current objective

Make the reward calculation explicit, balance/randomize probability conditions,
add condition-specific stopping diagnostics, remove unrelated starter-project
files, and add a short cluster smoke job.

### Current RED test

- `test_condition_values_are_explicit_relative_to_stop` was added before the
  expected-reward and condition-class helpers.
- `test_probability_cells_are_balanced_and_seeded_in_random_order` was added
  before changing condition scheduling.

### Expected failure

The first test would fail to import the new helpers. The second would fail the
random-order assertion under the original lexicographic schedule.

### Actual failure

Both targeted RED commands stopped before collection because `/usr/bin/python3`
still has no pytest. During the available stdlib verification, an exact float
comparison for `5 * 0.2 - 2` exposed ordinary binary rounding; the pytest
assertions were correctly changed to `pytest.approx`.

### GREEN implementation

- Reward constants and `E[r|p] = 5p - 2` are explicit in
  `bandit/environment.py`.
- The configured probabilities map to expected one-step rewards of -1.00,
  -0.25, +0.50, and +1.25 relative to STOP=0.
- The 16 ordered probability cells are balanced to within one episode and their
  order is deterministically shuffled by the run seed.
- Pilot analysis reports STOP/persistence and near-boundary coverage separately
  for both-negative, mixed, and both-positive conditions.
- `smoke_qwen35.slurm` runs pytest, the real model/hook compatibility gate, and
  two end-to-end episodes capped at three decisions.
- Removed the unrelated prompt-controller registry, YAMLs, phase-one runner,
  and Llama SLURM script. Retained `client.py` and `vllm_client.py` because the
  PRD explicitly requires preserving them. Removed unused dependencies while
  retaining `openai` for the legacy vLLM client.

### Test command

```bash
bash -n smoke_qwen35.slurm
bash -n run_qwen35_bandit.sh
python3 -m compileall -q bandit models interventions experiments analysis tests
```

Plus stdlib assertions for expected rewards, condition labels, deterministic
condition scheduling, balance, and YAML loading.

### Result

Shell syntax, Python compilation, YAML parsing, reward assertions, and condition
schedule assertions: PASS. Full pytest and GPU smoke remain cluster-only.

### Decisions / assumptions

- The reward grid is documented as provisional until pilot acceptance; it was
  not changed merely to mimic a forced-choice bandit from the literature.
- Classic Daw-style bandits motivate learning/exploration analyses. A retirement
  bandit or patch-foraging model is the closer formal reference for STOP.
- Depleting rewards would intentionally cause exits but would change the task
  from persistence under stationary value to adaptation under depletion, so it
  is reserved for a separate robustness experiment.

### Open issues

- Pre-register numerical pilot acceptance thresholds before using pilot results
  to revise the reward grid.
- Run `sbatch smoke_qwen35.slurm` on the GPU cluster.

### Next step

Review pilot acceptance criteria, then run the smoke job and 200-episode pilot.

---

## Cluster import errors and runtime instrumentation

### Current objective

Resolve the cluster's 12 pytest collection errors and remove unnecessary
activation-capture overhead from behavioral runs.

### Current RED test

`errors.txt` records all 12 test modules failing during collection with
`ModuleNotFoundError` for project-local packages. The hook test was extended to
require logits-only decisions by default and hidden states only when explicitly
requested.

### Expected failure

Without repository-root path configuration, pytest cannot import `bandit`,
`models`, `experiments`, or `interventions`. Before the capture refactor,
logits-only decisions also returned all-layer states.

### Actual failure

Cluster pytest stopped during collection before any test ran. The local host
still lacks pytest, so the attached cluster output is the authoritative RED
result for path configuration.

### GREEN implementation

- Added `pyproject.toml` with setuptools package discovery and pytest
  `pythonpath = ["."]`, making imports independent of whether pytest is invoked
  through its console script or `python -m pytest`.
- `HookedQwen.decision()` is now logits-only by default.
- Activation collection, compatibility, matched replay, and sequential steering
  explicitly request hidden states.
- All-layer capture now clones final-position vectors on the GPU and performs
  CPU conversion after the forward pass, avoiding 32 separate per-layer CPU
  transfers/synchronizations.
- Baseline, activation collection, and probe training now print actual elapsed
  time and throughput to their SLURM logs.

### Test command

```bash
python -m pytest -q
python -m compileall -q bandit models interventions experiments analysis tests
bash -n smoke_qwen35.slurm
```

### Result

Python compilation and shell syntax are locally checkable; cluster pytest must
be rerun after pulling this change.

### Decisions / assumptions

- No wall-time claim is made before observing cluster throughput. Runtime scales
  with the realized number and length of decision states.
- Behavioral pilot needs logits only; collecting all layers there was wasteful
  and scientifically unnecessary.

### Open issues

- Confirm pytest collection and the real model adapter on the cluster.
- Use logged states/second and per-layer probe time to choose later SLURM limits.

### Next step

Pull on the cluster and resubmit `smoke_qwen35.slurm`.

---

## Hugging Face model-download cycle

### Current objective

Provide a resumable Python utility that downloads the primary Qwen3.5-4B model
and Qwen3-4B-Instruct-2507 fallback into the exact directories used by the
existing offline SLURM jobs.

### Current RED test

`tests/test_download_qwen_models.py` was added first to freeze the two official
repository IDs, deterministic local directory names, and single-model
selection behavior.

### Actual failure

The requested targeted RED command again stopped before collection because the
host Python has no pytest:

```text
/usr/bin/python3: No module named pytest
```

### GREEN implementation

- `scripts/download_qwen_models.py` resolves `main` to a concrete Hugging Face
  commit, downloads with resumable local metadata, verifies config/tokenizer
  files and safetensor weights, and records the commit in a manifest.
- The Python utility defaults to both checkpoints under
  `/scratch/gpfs/JORDANAT/mg9965/models` and is intended to run directly on an
  internet-connected login node.
- `--selection primary` and `--selection fallback` support smaller one-model
  downloads without changing the script.

### Test command

```bash
python3 -m compileall -q scripts tests/test_download_qwen_models.py
```

### Result

Python compilation, download-plan assertions, and a mocked
resolve/download/verify/manifest cycle: PASS. The SLURM wrapper was removed
because cluster compute nodes cannot reach Hugging Face. No network download
was started; the actual model transfer and pytest remain cluster tasks.

---

## Conda activation under Bash nounset

### Current objective

Allow cluster jobs to activate the Conda environment without weakening strict
undefined-variable checking during the experiment itself.

### Actual failure

SLURM job `12376964` exited during environment setup with
`environment: _CE_M: unbound variable`. No tests or model code ran. Conda's
shell hook references optional `_CE_*` variables and is not compatible with
enabling Bash `nounset` before activation on this cluster.

### GREEN implementation

Both SLURM launchers now enable `errexit` and `pipefail` initially, activate
Conda, and then enable `nounset` for all subsequent setup and experiment code.

### Test command

```bash
bash -n smoke_qwen35.slurm run_qwen35_bandit.sh
```

### Result

Shell syntax: PASS. Cluster environment activation must be confirmed by
resubmitting the smoke job.

---

## Transformers BatchEncoding compatibility

### Current objective

Normalize current Transformers chat-template output before deterministic token
comparison and model forwarding.

### Current RED test

The real GPU smoke job loaded both checkpoints, then failed each compatibility
gate because `torch.equal` received a `BatchEncoding` instead of a tensor. A
regression test now supplies a mapping-style `BatchEncoding` stand-in and
requires `HookedQwen.tokenize()` to return its tensor fields directly.

### Actual failure

SLURM job `12377100` reported:

```text
TypeError: equal(): argument 'input' (position 1) must be Tensor, not BatchEncoding
```

The preceding CPU gate passed all 25 tests, and both model weight sets loaded.

### GREEN implementation

`HookedQwen.tokenize()` now recognizes any mapping-compatible tokenizer output
and converts it to a plain input dictionary. Tensor-only output remains wrapped
under `input_ids` as before.

### Result

Local Python compilation and dependency-free structural assertions pass. The
real compatibility and hook checks must be rerun on the GPU cluster.

---

## Behavioral-pilot integrity and timing audit

### Current objective

Verify the uploaded 200-episode pilot at the row, episode, environment, prompt,
and action-sampling levels, then characterize when the model quits rather than
relying on the state-weighted mean STOP probability.

### Verification

`analysis/analyze_pilot_detailed.py` replays every episode from its environment
and action seeds and reconstructs every model-visible conversation. Across
3,706 states it found no mismatches in state IDs, rounds, rewards, histories,
cumulative scores, future returns, probability normalization, sampled actions,
termination, or conversations. The 16 probability cells contained 12 or 13
episodes each; the three value classes contained 50/100/50 episodes.

### Descriptive result

- 180/200 episodes quit and 20 were right-censored at the 100-decision horizon.
- Median quit decision was 5 (IQR 4-9); 142/180 quits occurred by decision 10.
- The observed per-state quit hazard peaked at 20.8% during decisions 4-5.
- STOP occurred on 1.9%, 3.9%, 9.4%, and 13.6% of states following loss
  streaks of 0, 1, 2, and 3+, respectively.
- Mean model P(STOP) rose from 8.1% two decisions before quitting to 14.2% one
  decision before and 22.9% at the sampled quit decision.
- Better-arm choice was at chance initially and rose above 60% after decision
  10, but the initial prompt induced a strong A-label prior (P(A)=0.911).

### Outputs

The detailed JSON, Markdown report, and two dependency-free SVG figures are in
`artifacts/pilot_diagnostics/`. These are exploratory pilot summaries; later
uncertainty estimates must cluster or resample at the episode level.

---

## Integrated-value probe diagnostic

### Current objective

Distinguish a probe that integrates reward history into graded task value from
one that merely recovers the immediately preceding reward or a simple STOP
heuristic.

### Current RED test

`tests/test_probe_mechanism.py` was added before implementation. Synthetic
tests require an integrated-history probe to add held-out persistence signal
beyond recent-history controls, a latest-reward-only probe to add none, and a
strong cumulative-score control to explain a deliberately score-equivalent
synthetic probe.

### Actual failure

The targeted RED command again stopped before collection because the local host
does not contain pytest. The real cluster environment is expected to collect
and execute these tests.

### GREEN implementation

- The best layer and sparse neurons remain selected exclusively on validation
  TD error.
- Only after freezing that selection, untouched test episodes are evaluated.
- The primary nested regression controls previous outcome, an initial-state
  indicator, loss streak, and nonlinear round terms before adding sparse probe
  value.
- Episode-clustered sandwich standard errors account for repeated states.
- Diagnostics repeat within last-loss and last-gain states, add cumulative
  score as a stronger history baseline, compare the full probe, and test whether
  sparse probe value itself encodes cumulative history.
- JSON, Markdown, state-level CSV, and SVG outputs are written alongside the
  frozen probe. These associations diagnose mechanism; steering remains the
  causal test.

### Verification

Python compilation, shell syntax, configuration assertions, and whitespace
checks pass locally. The torch-based synthetic tests must run in the cluster
environment.

### Next step

Run the expanded smoke job before probe collection. It now checks the synthetic
probe-mechanism regression and report pipeline, real-model hidden-state
compatibility, short behavioral episodes, and two capped real activation
shards. The sprint probe bank is 512 episodes (resumable), not the original
2,000-episode target.

---

## Sprint confirmatory causal design

### Decision made before confirmatory data collection

- Use 48 held-out episodes, exactly three per ordered arm-probability cell.
- Preserve all three matched alpha conditions and all 20 magnitude-matched
  random-neuron sets.
- Treat episodes as the independent sampling units. Average paired state effects
  within episode, then estimate equal-episode-weighted effects, t confidence
  intervals, and p-values across episodes.
- The primary gate requires monotonic adjacent means and a positive-versus-
  negative 95% confidence interval strictly above zero.
- The specificity gate retains the corrected 20-control empirical test; the
  value effect must exceed all 20 random-set effects to attain p=1/21.
- Full 1,000-per-alpha sequential episodes remain out of sprint scope unless
  both matched-state gates pass.

### Verification

`tests/test_analyze_persistence.py` constructs repeated states within episodes
and requires the analysis to report six—not eighteen—independent units, a
positive episode-level confidence interval, and the exact 1/21 random-control
empirical p-value. Local syntax and whitespace checks pass; pandas/scipy pytest
execution remains cluster-only.

---

## Probe variance, exact matching, and Monte Carlo follow-up

### Current-data result

- Report probe-only, recent-history-only, and joint persistence models. Sparse
  probe R² is .240 and full probe R² is .440, but unique increments beyond the
  .753 history model are only .0027 and .0003.
- Exact matching on round, previous outcome, and loss streak retains 1,180 test
  states in 147 strata from 55 episodes. Older cumulative history predicts the
  full probe (standardized beta .440, p=1.3e-5) but not persistence; neither
  probe predicts persistence within these exact strata.
- This distinguishes “the probe contains no information” from the supported
  conclusion: the unpruned hidden-state-derived probe contains accumulated-
  history information, but its persistence prediction is shared with recent
  history/round and does not survive exact matching.

### Exploratory follow-up

`experiments/train_monte_carlo_probe.py` directly predicts stored realized
future cumulative return, avoiding the moving TD bootstrap target. It freezes
the existing episode split, evaluates constant and recent-history baselines,
selects dimensions from a full probe using validation performance, refits the
sparse probe on only those dimensions, and repeats mechanism and exact-match
analyses. Because this redesign followed inspection of the TD test result, its
reuse of the original test split is explicitly labeled exploratory.

Counterfactual continuation advantage remains deferred: it requires defining
whose expectation is estimated and repeated policy rollouts after forced A/B
actions. It should only be built if the stable generic future-return probe is
still insufficient.

The `train_mc_probe` SLURM phase stops after analysis so an exploratory probe is
not automatically promoted into a steering direction. Use
`PHASE=calibrate_mc_probe` only after inspecting its held-out target prediction,
three-model variance decomposition, and exact-match result.

### Monte Carlo follow-up result

- Layer 2 was selected. Full and sparse validation/test future-return R² were
  .150/.251 and .152/.179, versus .020 for the frozen recent-history baseline.
  The selected checkpoints occurred at epochs 9 and 4, so this training did not
  reproduce the TD probe's epoch-one collapse.
- In 1,180 states exactly matched on round, previous outcome, and loss streak,
  the sparse and full probes predicted reconstructed future return (standardized
  beta .456 and .479; episode-clustered p=1.0e-5 and 2.8e-6). Enumerating all
  2^11 possible final rewards for capped episodes kept both effects positive.
- Those same probe outputs did not predict persistence within matched states
  (beta -.070 and -.063; p=.480 and .564). Recent-state controls explained
  75.3% of persistence variance, with essentially zero unique probe increment.
- Adjudication: a decodable integrated future-return signal exists, rescuing the
  representational part of the value account and showing the TD procedure was
  inadequate. The observed STOP policy is nevertheless better described by a
  recent-state heuristic. A distinct continuation-advantage representation is
  plausible but remains untested; generic value steering is not yet a validated
  causal persistence manipulation.

### Linear-to-advantage mechanism sequence

- `PHASE=linear_probes` fits ridge-linear realized-return and direct-persistence
  probes at every layer using the frozen train/validation/test episode split.
  Ridge alpha and layer are validation-selected. Raw activation-space direction
  cosine and top-1% dimension overlap are reported at each layer.
- `PHASE=collect_advantage` restores each stored conversation/history state,
  forces A or B under paired fresh outcome schedules and action-sampling seeds,
  then follows the unmodified policy to STOP or the 100-decision horizon.
  Q(STOP)=0 and continuation advantage is max(mean Q_A, mean Q_B).
- Counterfactual collection is resumable and shardable. The sprint default is
  128 stratified states per episode split and 20 paired rollouts per state; using
  zero states-per-split requests all 12,461 states and is expected to be much
  more expensive.
- `PHASE=train_advantage` fits ridge probes across layers, repeats held-out and
  exact matched-state tests, and compares advantage directions with the direct
  persistence directions. The max of noisy Q estimates is acknowledged as
  upward biased, so raw per-arm returns and standard errors are preserved.
