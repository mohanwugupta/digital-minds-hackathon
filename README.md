# Value representations and persistence in Qwen3.5-4B

This repository contains the experiments and analyses reported in the Digital
Minds Research Sprint paper. It studies whether an internal representation of
expected future return causally governs a model's decision to continue or stop
in a sequential two-armed bandit.

The paper is available at
[`docs/Digital_Minds_Research_Sprint.docx`](docs/Digital_Minds_Research_Sprint.docx).
The original requirements and research log remain in `PRD.md` and
`SCRATCHPAD.md`.

## Repository layout

- `bandit/`: deterministic environment, prompts, conversation state, schemas,
  and episode-level splitting.
- `models/hooked_qwen.py`: Qwen loading, action-token validation, hidden-state
  capture, and final-position activation hooks.
- `interventions/`: TD, nonlinear, and ridge probes plus calibrated ridge
  steering.
- `experiments/`: resumable collection, probe training, factorial replay, and
  causal-steering programs.
- `analysis/`: behavioral, probe, factorial, and causal analyses. The R figure
  source is retained as-is in `analysis/digital_minds_sprint_analysis.Rmd`.
- `tests/`: CPU unit and integration tests.
- `scripts/`: model download and support utilities.
- `artifacts/`: retained source data, frozen probes, metrics, and publication
  summaries. See `artifacts/README.md`.
- `docs/`: paper and implementation notes.
- `run_qwen35_bandit.sh`: main Slurm phase dispatcher.
- `smoke_qwen35.slurm`: CPU and real-model preflight.

## Task

At each decision the model chooses `A`, `B`, or `C = STOP`. A successful pull
earns `+3`; an unsuccessful pull earns `-2`; STOP ends the episode with no
additional reward. For arm success probability `p`, expected immediate reward
is

```text
E[r | p] = 3p - 2(1-p) = 5p - 2.
```

The probability grid `[0.20, 0.35, 0.50, 0.65]` gives immediate expected
rewards `[-1.00, -0.25, +0.50, +1.25]`. Experimenter-only variables such as
true arm probabilities and random schedules are never included in the
model-visible conversation.

## Environment and model

```bash
conda env create -f environment.yml
conda activate value-steering-bandit
pytest -q
```

The reported run used Python 3.10.20, Transformers 5.15.0, PyTorch
2.13.0+cu130, CUDA 13.0, and bfloat16 inference. Runtime metadata sidecars
preserve the observed software and checkpoint information for each collection
phase.

Download Qwen from an internet-connected login node before starting an offline
GPU job:

```bash
python scripts/download_qwen_models.py --selection primary
```

The default cluster path is
`/scratch/gpfs/JORDANAT/$USER/models/Qwen--Qwen3.5-4B`.

## Reproducing the paper

Run commands from the repository root. Long GPU phases can be submitted through
`run_qwen35_bandit.sh` by setting `PHASE`.

### 1. Compatibility and behavioral pilot

```bash
sbatch --export=ALL,PHASE=compatibility run_qwen35_bandit.sh
sbatch --export=ALL,PHASE=pilot run_qwen35_bandit.sh
```

The pilot runs 200 episodes and writes the detailed integrity and stopping
diagnostics reported in Appendix A.

### 2. Activation bank and probe analyses

```bash
sbatch --export=ALL,PHASE=collect_probe,PROBE_EPISODES=512 run_qwen35_bandit.sh

# Appendix B: initial TD probe
sbatch --export=ALL,PHASE=train_probe run_qwen35_bandit.sh

# Appendix C.1: nonlinear Monte Carlo future-return probe
sbatch --export=ALL,PHASE=train_mc_probe run_qwen35_bandit.sh

# Main linear future-return and persistence probes
sbatch --export=ALL,PHASE=linear_probes run_qwen35_bandit.sh
```

Per-layer metrics are retained in JSON. Redundant per-layer neural checkpoints
can be recreated from the activation bank, so only validation-selected frozen
neural probes are retained.

### 3. Provisional continuation advantage

Collect 10 paired forced-action rollouts for 384 states. Collection is
resumable and may be sharded:

```bash
sbatch --array=0-3 --time=12:00:00 \
  --export=ALL,PHASE=collect_advantage,ADVANTAGE_NUM_SHARDS=4,ADVANTAGE_ROLLOUTS=10,ADVANTAGE_STATES_PER_SPLIT=128 \
  run_qwen35_bandit.sh

sbatch --export=ALL,PHASE=train_advantage run_qwen35_bandit.sh
```

### 4. Independent confirmatory state bank

```bash
sbatch --export=ALL,PHASE=collect_confirmatory,CONFIRMATORY_EPISODES=48 \
  run_qwen35_bandit.sh
```

These episodes are disjoint from probe fitting and are reused for factorial and
causal replay.

### 5. STOP-payoff × CONTINUE-bonus factorial

```bash
sbatch --array=0-3 --time=03:00:00 \
  --export=ALL,PHASE=value_dissociation_collect,DISSOCIATION_NUM_SHARDS=4 \
  run_qwen35_bandit.sh

sbatch --export=ALL,PHASE=value_dissociation_analyze run_qwen35_bandit.sh
```

The default retains behavioral observations and the three frozen probe
projections used by the paper. Optional all-layer tensors are omitted; pass
`--save-activations` directly to `experiments.run_value_dissociation` only for
new analyses that require them.

### 6. Causal steering

```bash
sbatch --export=ALL,PHASE=causal_calibrate run_qwen35_bandit.sh

sbatch --array=0-7 --time=06:00:00 \
  --export=ALL,PHASE=causal_steering_collect,CAUSAL_NUM_SHARDS=8 \
  run_qwen35_bandit.sh

sbatch --export=ALL,PHASE=causal_steering_analyze run_qwen35_bandit.sh
```

This replays confirmatory states under the future-return, provisional-advantage,
and persistence directions plus 20 layer-matched random controls. Alpha zero
reuses the exact unhooked baseline. Inference is clustered or bootstrapped at
the episode level as described in the paper.

## Smoke test

```bash
MODEL_PATH=/path/to/Qwen3.5-4B sbatch smoke_qwen35.slurm
```

The smoke job runs CPU tests, validates the checkpoint and chat template, and
executes tiny behavioral, activation, counterfactual-rollout, factorial, and
causal checks. Its outputs are temporary diagnostics and should not be
committed.

## Artifact policy

Source code, compact result tables, metadata, selected frozen probes, and
publication summaries belong in the repository. Smoke outputs, redundant
per-layer neural checkpoints, and optional all-layer factorial tensors do not.
The activation and confirmatory banks are retained because they are the inputs
needed to refit probes and replay the reported experiments.
