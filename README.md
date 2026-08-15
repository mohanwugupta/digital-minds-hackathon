# Causal value steering in a bandit environment

This repository implements the experiment in [PRD.md](PRD.md): identify a
sparse internal value representation in Qwen3.5-4B, steer it up and down at a
single residual-stream location, and measure the causal change in preference
for continuing a costly two-armed bandit rather than stopping.

The scientific path uses Hugging Face Transformers and native PyTorch hooks.
The supplied vLLM starter client is retained for the older project but is not
used for baseline or intervention data.

## Layout

- `bandit/`: deterministic counterfactual environment, exact prompts,
  conversation state, artifact schema, and episode-level splitting.
- `models/hooked_qwen.py`: model loading, actual-chat-template action-token
  validation, layer discovery, final-position states, action logits, and hooks.
- `interventions/`: TD value probe, sparse selection, frozen artifacts,
  calibration, value directions, and magnitude-matched random controls.
- `experiments/`: resumable command-line programs for each experimental phase.
- `analysis/`: pilot, matched-state, and sequential summaries.
- `tests/`: CPU unit/integration tests plus a tiny mocked pipeline.
- `config/bandit_experiment.yaml`: frozen design defaults.
- `smoke_qwen35.slurm`: unit/integration, real-model compatibility, behavioral,
  and activation-capture gates.
- `scripts/download_qwen_models.py`: resumable Hugging Face model downloader.

Model-visible state is represented only by `BanditConversation`. True arm
probabilities, score, schedules, seeds, round, and structured histories live in
`BanditEnvironment` and `DecisionRecord`; no code path serializes them back into
messages.

The action vocabulary defaults to `A/B/C`. If those are not clean single-token
continuations under the selected model's actual chat template, the adapter
automatically tests and freezes `1/2/3`; prompts and stored assistant actions use
that frozen vocabulary while experimenter-side actions remain `A/B/C`.

## Reward and stopping design

Each A/B pull is an independent Bernoulli potential outcome generated before
the episode. A success adds 3 points and a failure subtracts 2, so an arm with
success probability `p` has one-step expected reward

```text
E[r | p] = 3p - 2(1-p) = 5p - 2.
```

STOP adds zero and ends the episode. The configured probabilities therefore
have one-step values `-1.00, -0.25, +0.50, +1.25`; zero is crossed at `p=0.40`.
This one-step calculation excludes the information value of sampling an
uncertain arm, which is important early in a finite episode.

The 16 ordered A/B probability pairs are balanced to within one episode and
their presentation order is deterministically randomized from the run seed.
They form four both-negative, eight mixed, and four both-positive cells. The
pilot analysis reports STOP probability and the share
of decision states near the choice boundary separately for those three classes.
The probability grid is provisional until the pilot is accepted and must be
frozen before probe/confirmatory collection.

Classic forced-choice bandits are the right reference for learning and
exploration, but patch-foraging and bandit-retirement models are the closer
reference for the STOP decision. Depleting rewards or an explicit opportunity
cost should not be introduced into the primary MVP without changing the causal
question: they would make stopping easier to elicit, but would also turn
persistence into adaptation to a changing environment. A depleting-patch task
is better treated as a preregistered robustness study.

## Installation

Use the cluster Conda environment:

```bash
conda env create -f environment.yml
conda activate value-steering-bandit
pytest -q
```

Qwen3.5 support requires Transformers 5.12 or newer. The published 4B checkpoint
uses the full Qwen3.5 configuration, so the adapter loads the official
conditional-generation class but sends text tokens only and instruments its
nested language-model backbone. Model loading is
offline by default. Pass `--online` only when downloads are intentional.

## Downloading the model checkpoints

Run the downloader directly from an internet-connected cluster login node
before running the offline GPU smoke test:

```bash
conda activate value-steering-bandit
python scripts/download_qwen_models.py
```

It downloads both public Hugging Face checkpoints into these directories:

```text
/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen3.5-4B
/scratch/gpfs/JORDANAT/mg9965/models/Qwen--Qwen3-4B-Instruct-2507
```

The first is the primary model and the second is the compatibility fallback.
Downloads are resumable: if the process is interrupted, run the same command
again. Each completed directory contains `.download_manifest.json` with the
resolved Hugging Face commit. The models are public, so a token is normally not
needed; if authenticated Hub access is required, export `HF_TOKEN` before
running the command.

To download only one checkpoint, use `--selection primary` or
`--selection fallback`:

```bash
python scripts/download_qwen_models.py --selection primary
python scripts/download_qwen_models.py --selection fallback
```

## Experimental sequence

All commands are run from the repository root. Do not continue past a failed
gate.

1. Run the hard compatibility gate:

   ```bash
   python -m experiments.check_qwen_compatibility \
     --model /path/to/Qwen3.5-4B
   ```

   It tests deterministic text tokenization, layer enumeration, all-layer
   hidden states, final-position extraction, hook modification, downstream
   logits, and exact restoration after hook removal. It tries the fallback only
   if the primary model fails these checks.

2. Run and inspect the 200-episode pilot:

   ```bash
   python -m experiments.run_bandit_baseline \
     --model /path/to/Qwen3.5-4B --episodes 200
   python -m analysis.analyze_pilot
   ```

3. Collect 512 fresh baseline episodes for research-sprint probe development:

   ```bash
   python -m experiments.collect_bandit_activations \
     --model /path/to/Qwen3.5-4B --episodes 512
   ```

   Each episode is atomically checkpointed as a separate `.pt` shard. Existing
   shards are skipped on restart. Only final prompt-position vectors are saved,
   with shape `decision_states x layers x d_model`. The 512-episode target is
   sized for the sprint and can be extended if held-out estimates are too noisy.

4. Train layer probes, select the top 1%, and calibrate on validation states:

   ```bash
   python -m experiments.train_value_probe
   python -m experiments.calibrate_steering
   ```

   `episode_split.json`, per-layer probes, selection metrics,
   `frozen_best.pt`, and `steering_calibration.json` freeze every quantity used
   by confirmatory runs. After validation-only layer/neuron selection, probe
   training also runs a mechanism diagnostic on untouched test episodes. It
   tests whether sparse probe value predicts the persistence logit beyond
   previous outcome, loss streak, and nonlinear round terms; repeats the test
   within last-loss and last-gain states; and adds cumulative score as a
   stronger history baseline. Results are written to
   `probe_mechanism.json`, `probe_mechanism_report.md`,
   `probe_mechanism_test_states.csv`, and `probe_mechanism.svg` under
   `artifacts/value_probes/`.

   This diagnostic distinguishes a history-integrating value representation
   from a latest-reward heuristic, but remains associational. The subsequent
   activation-steering experiment is the causal test.

   If the bootstrapped TD objective is unstable, run the prespecified
   exploratory Monte Carlo follow-up on the same activation bank:

   ```bash
   sbatch --export=ALL,PHASE=train_mc_probe run_qwen35_bandit.sh
   ```

   This directly regresses hidden states onto stored realized future cumulative
   return, compares against constant and recent-history baselines, refits the
   validation-selected sparse dimensions rather than merely masking a full
   probe, and repeats the held-out mechanism diagnostic. Because this follow-up
   was motivated by inspecting the TD test result, it is labeled exploratory;
   realized future return under the observed policy is also not the same target
   as a counterfactual continuation advantage.

   This phase deliberately stops after training and analysis. Inspect
   `artifacts/mc_value_probes/publication/monte_carlo_probe_report.md` before
   promoting the probe to a steering direction. If it predicts held-out future
   return and passes the mechanism diagnostics, calibrate it separately:

   ```bash
   sbatch --export=ALL,PHASE=calibrate_mc_probe run_qwen35_bandit.sh
   ```

   The next mechanism sequence uses the same frozen episode split. First train
   ridge-linear probes for both realized future return and the model's actual
   persistence logit:

   ```bash
   sbatch --export=ALL,PHASE=linear_probes run_qwen35_bandit.sh
   ```

   This writes layer-by-layer held-out R², frozen ridge directions, exact-match
   diagnostics, and return/persistence direction overlap under
   `artifacts/linear_probes/`. The persistence target is
   `logsumexp(logit_A, logit_B) - logit_C`; it localizes the decision variable
   but is not itself evidence for value.

   Continuation advantage requires new model rollouts. At each selected stored
   conversation state, the collector forces A and B in paired counterfactual
   worlds, follows the unmodified sampled policy, and defines
   `A_continue = max(mean_return_A, mean_return_B) - 0`. Raw rollout returns and
   arm-specific standard errors are retained. A small end-to-end check is:

   ```bash
   sbatch --time=02:00:00 \
     --export=ALL,PHASE=advantage_pipeline,ADVANTAGE_ROLLOUTS=2,ADVANTAGE_STATES_PER_SPLIT=8,ADVANTAGE_TARGET_DIR=artifacts/advantage_targets_smoke,ADVANTAGE_PROBE_DIR=artifacts/advantage_probes_smoke \
     run_qwen35_bandit.sh
   ```

   The sprint default uses 20 rollouts per forced action and 128 stratified
   states from each split (384 states total). It deliberately selects blocks
   from common round/outcome/loss-streak strata so the held-out exact-matching
   test remains possible. Because every rollout may require many additional
   model forwards, collection is resumable and supports SLURM arrays:

   ```bash
   sbatch --array=0-3 --time=12:00:00 \
     --export=ALL,PHASE=collect_advantage,ADVANTAGE_NUM_SHARDS=4,ADVANTAGE_ROLLOUTS=20,ADVANTAGE_STATES_PER_SPLIT=128 \
     run_qwen35_bandit.sh
   ```

   After every array task finishes, train and analyze the advantage probe:

   ```bash
   sbatch --export=ALL,PHASE=train_advantage run_qwen35_bandit.sh
   ```

   Do not mix an unsharded `targets.csv` with `targets_shard_*.csv` in the same
   target directory; duplicate state IDs are rejected rather than silently
   averaged. The decisive report is
   `artifacts/advantage_probes/publication/advantage_probe_report.md`.

   On SLURM, collection and training can share one allocation. The shell script
   exits immediately if collection, training, mechanism analysis, or
   calibration fails:

   ```bash
   sbatch --export=ALL,PHASE=probe_pipeline,PROBE_EPISODES=512 \
     run_qwen35_bandit.sh
   ```

   Keep confirmatory steering separate so the frozen probe and its held-out
   mechanism diagnostic can be inspected before causal data collection.

5. Create a separate held-out state bank with a new output directory and seed,
   then run matched replay:

   ```bash
   python -m experiments.collect_bandit_activations \
     --model /path/to/Qwen3.5-4B --episodes 48 \
     --seed 52026 --output-dir artifacts/confirmatory_state_bank
   python -m experiments.run_bandit_intervention \
     --model /path/to/Qwen3.5-4B \
     --state-bank artifacts/confirmatory_state_bank
   python -m analysis.analyze_persistence
   ```

   Replay rejects episode IDs found in any probe split. Every context is
   canonicalized and hashed, and each state is evaluated at alpha `-1, 0, +1`
   for the value neurons and at least 20 independently sampled random-neuron
   sets. Alpha zero uses the unhooked baseline forward pass exactly.

   The sprint confirmatory bank uses 48 episodes, placing exactly three in each
   ordered arm-probability cell. Matched effects and confidence intervals use
   episodes—not decision states—as the independent sampling unit, with equal
   episode weighting so longer episodes cannot dominate inference.
   The analysis writes `matched_analysis.json`, a concise
   `matched_analysis.md` report, and `matched_analysis.svg` with the steering
   ordering, episode-level confidence intervals, and random-control comparison.

   After reviewing the frozen probe outputs, the complete sprint causal phase
   can run in one allocation. Six hours allows headroom for roughly 43 forward
   passes per held-out state (baseline plus bidirectional value and 20 random-
   set interventions):

   ```bash
   sbatch --time=06:00:00 \
     --export=ALL,PHASE=causal_pipeline,CONFIRMATORY_EPISODES=48,MATCHED_RANDOM_SETS=20 \
     run_qwen35_bandit.sh
   ```

6. Only after `matched_analysis.json` reports both gates passing, run sequential
   episodes:

   ```bash
   python -m experiments.run_bandit_sequential \
     --model /path/to/Qwen3.5-4B \
     --matched-analysis artifacts/matched_analysis.json
   python -m analysis.analyze_sequential
   ```

7. Run the frozen ridge-direction causal comparison. Calibration reads only the
   existing validation episodes and freezes a native-layer intervention near a
   one-SD decoded shift subject to the activation-RMS safety bound:

   ```bash
   # Only needed once if artifacts/confirmatory_state_bank is not already present.
   sbatch --export=ALL,PHASE=collect_confirmatory run_qwen35_bandit.sh

   sbatch --export=ALL,PHASE=causal_calibrate run_qwen35_bandit.sh
   sbatch --export=ALL,PHASE=causal_positive_control run_qwen35_bandit.sh
   ```

   Inspect
   `artifacts/causal_steering/publication_positive_control/causal_steering_report.md`.
   Do not interpret generic-return or advantage null effects unless the frozen
   layer-31 persistence direction has a monotonic `-1,0,+1` dose response and a
   positive episode-bootstrap confidence interval.

   The full comparison uses the layer-31 persistence direction, layer-1 generic
   future-return direction, layer-2 continuation-advantage direction, and 20
   sign-randomized controls per native layer. Sign randomization preserves every
   coordinate magnitude, hence both Euclidean norm and activation-standardized
   RMS. Collection is resumable and can be sharded:

   ```bash
   sbatch --array=0-3 --time=06:00:00 \
     --export=ALL,PHASE=causal_steering_collect,CAUSAL_NUM_SHARDS=4 \
     run_qwen35_bandit.sh
   sbatch --export=ALL,PHASE=causal_steering_analyze run_qwen35_bandit.sh
   ```

   `alpha=0` always reuses the unhooked baseline result. The primary outcome is
   `logsumexp(logit_A, logit_B) - logit_C`; sampled actions are not used for the
   causal test. The analysis resamples episodes and compares each target's
   positive-minus-negative effect with its matched random-direction distribution.

8. Independently run the external-value dissociation. Each held-out conversation
   history is replayed under all 12 combinations of STOP payoff
   `[-10, 0, +10, +20]` and common CONTINUE bonus `[-10, 0, +10]`. Temporary
   payoffs apply only to the current decision, and no new histories or private arm
   probabilities enter the prompt.

   ```bash
   sbatch --array=0-3 --time=03:00:00 \
     --export=ALL,PHASE=value_dissociation_collect,DISSOCIATION_NUM_SHARDS=4 \
     run_qwen35_bandit.sh
   sbatch --export=ALL,PHASE=value_dissociation_analyze run_qwen35_bandit.sh
   ```

   Every factorial forward pass records persistence probability/logit and native-
   layer projections onto the frozen generic-return, provisional advantage, and
   direct-persistence directions. All-layer float16 activation tensors are also
   saved as one atomic shard per state under
   `artifacts/value_dissociation/activations/`, preserving the optional
   orthogonalized-probe analysis without putting tensors in CSV. The primary analysis uses state fixed effects
   with episode-clustered uncertainty. Because `C-S` is exactly collinear with
   `C` and `S`, the report fits equivalent `S+C` and relative-plus-common-value
   parameterizations rather than putting all three variables in a singular model.

The sequential runner shares bandit and action-sampling seeds across all three
conditions. `--force` exists for diagnostics but should not be used for the
confirmatory workflow.

## Artifacts and reproducibility

Long-running phases write incremental episode shards or append completed CSV
records. Metadata sidecars include the full CLI configuration, model ID and
revision, Python/PyTorch/Transformers versions, seeds, timestamp, and Git commit.
CSV conversation and history fields are canonical JSON strings. Private outcome
schedules are held by the environment and are never included in model prompts.

`SCRATCHPAD.md` records the required RED-to-GREEN development history and the
local verification limitation. A real Qwen compatibility result cannot be
claimed until the GPU smoke test has run against the local model weights.

For the complete preflight in one short SLURM job:

```bash
MODEL_PATH=/path/to/Qwen3.5-4B sbatch smoke_qwen35.slurm
```

This runs the full test suite first and loads the model only if those tests
pass. It then exercises real chat tokenization, all-layer state capture, a
mid-layer intervention, downstream logit change, and exact restoration after
hook removal. Finally, it runs two three-decision end-to-end bandit episodes.
It also writes two three-decision activation shards, checking the exact input
format used by probe training. Smoke artifacts use the SLURM job ID and are
diagnostic, not scientific data.
