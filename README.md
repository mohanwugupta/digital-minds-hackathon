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
- `smoke_qwen35.slurm`: short unit-test plus real-model compatibility gate.
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

3. Collect 2,000 baseline episodes for probe development:

   ```bash
   python -m experiments.collect_bandit_activations \
     --model /path/to/Qwen3.5-4B --episodes 2000
   ```

   Each episode is atomically checkpointed as a separate `.pt` shard. Existing
   shards are skipped on restart. Only final prompt-position vectors are saved,
   with shape `decision_states x layers x d_model`.

4. Train layer probes, select the top 1%, and calibrate on validation states:

   ```bash
   python -m experiments.train_value_probe
   python -m experiments.calibrate_steering
   ```

   `episode_split.json`, per-layer probes, selection metrics,
   `frozen_best.pt`, and `steering_calibration.json` freeze every quantity used
   by confirmatory runs.

5. Create a separate held-out state bank with a new output directory and seed,
   then run matched replay:

   ```bash
   python -m experiments.collect_bandit_activations \
     --model /path/to/Qwen3.5-4B --episodes 200 \
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

6. Only after `matched_analysis.json` reports both gates passing, run sequential
   episodes:

   ```bash
   python -m experiments.run_bandit_sequential \
     --model /path/to/Qwen3.5-4B \
     --matched-analysis artifacts/matched_analysis.json
   python -m analysis.analyze_sequential
   ```

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
Smoke artifacts use the SLURM job ID and are diagnostic, not scientific data.
