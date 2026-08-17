# Track A and Track B implementation guide

This extension reuses the original Qwen loader, final-token residual capture,
ridge probes, episode-level splitting, steering calibration, random controls,
metadata, and Slurm dispatcher. The A/B/C bandit path is unchanged.

## Baseline gate

Run this before and after generating new data:

```bash
python -m analysis.check_baseline_regression
```

The gate checks selected probe layers, held-out performance, the exact split
hash, factorial dimensions and effect signs, and alpha-zero/positive-control
causal results against `config/baseline_regression.json`.

## Track A: existing factorial trajectory

The repository retained all 32 independently trained bandit persistence probes
and all factorial prompts/results, but not the optional multi-gigabyte
factorial activation tensors. The projection command therefore replays each
retained prompt once and immediately saves only 32 scalar probe outputs. It
stops if the replayed persistence logit differs from the retained baseline
beyond tolerance.

Submit the guarded test → smoke → full replay array → analysis dependency chain
from the cluster login node:

```bash
TRACK_A_RUN_ID=track_a_v1 TRACK_A_SHARDS=8 \
  bash scripts/submit_track_a.sh
```

The helper uses `run_qwen35_bandit.sh` for every job. The full replay retains
the float16 condition × layer × width activation tensors under
`artifacts/value_dissociation/activations/` and writes compact scalar
projections per replay shard. The final analysis runs only if every replay
array task succeeds.

To submit the phases manually, use:

```bash
sbatch --export=ALL,PHASE=track_a_tests run_qwen35_bandit.sh
sbatch --export=ALL,PHASE=track_a_smoke run_qwen35_bandit.sh
sbatch --array=0-7 --time=08:00:00 \
  --export=ALL,PHASE=factorial_layerwise_project,LAYERWISE_NUM_SHARDS=8,TRACK_A_RUN_ID=track_a_v1 \
  run_qwen35_bandit.sh
sbatch --export=ALL,PHASE=factorial_layerwise_analyze,TRACK_A_RUN_ID=track_a_v1 \
  run_qwen35_bandit.sh
```

If optional tensors already exist, pass `--activation-dir` directly to
`experiments.project_factorial_layers` to avoid model replay. The analyzer
fits state-fixed-effect STOP and CONTINUE models and the relative-incentive
model at every layer. It writes JSON, Markdown, CSV, and SVG outputs under
`artifacts/value_dissociation/layerwise_publication_<run-id>/`. The output is
explicitly described as a representational trajectory, not a computation
location.

## Track B: behavioral and activation collection

First validate X/Y under the exact rendered chat templates:

```bash
sbatch --export=ALL,PHASE=cross_task_compatibility run_qwen35_bandit.sh
```

Then collect paired tasks. Each pair shares its task condition and stochastic
seed while reversing X/Y semantics. Pair IDs, not individual episodes, are the
unit assigned to train, validation, or test.

```bash
sbatch --array=0-3 \
  --export=ALL,PHASE=cross_task_collect_foraging,CROSS_TASK_NUM_SHARDS=4 \
  run_qwen35_bandit.sh

sbatch --array=0-3 \
  --export=ALL,PHASE=cross_task_collect_control,CROSS_TASK_NUM_SHARDS=4 \
  run_qwen35_bandit.sh
```

Foraging uses a depleting patch with varying initial quality, depletion,
outside option, and search cost. The control is a one-shot integer comparison
with the same X/Y response structure and no continue/quit meaning.

## Representational checkpoint

Fit the within-foraging ceiling and evaluate the frozen bandit probe:

```bash
sbatch --export=ALL,PHASE=cross_task_train_ceiling run_qwen35_bandit.sh
sbatch --export=ALL,PHASE=cross_task_representational run_qwen35_bandit.sh
```

The primary result applies the original bandit feature standardization and
weights without fitting a foraging parameter. A validation-only affine fit is
secondary. The report includes mapping-specific effects, matched
sign-randomized directions, the non-persistence control, episode-bootstrap
intervals, the within-task ceiling, the transfer ratio, and the predeclared
strong/partial/no-transfer classification.

## Causal checkpoint

Calibration uses foraging validation activations only. Test states are touched
only after the magnitude is frozen.

```bash
sbatch --export=ALL,PHASE=cross_task_causal_calibrate run_qwen35_bandit.sh

sbatch --array=0-7 \
  --export=ALL,PHASE=cross_task_causal_foraging,CROSS_TASK_NUM_SHARDS=8 \
  run_qwen35_bandit.sh

sbatch --array=0-7 \
  --export=ALL,PHASE=cross_task_causal_control,CROSS_TASK_NUM_SHARDS=8 \
  run_qwen35_bandit.sh

sbatch --export=ALL,PHASE=cross_task_causal_analyze run_qwen35_bandit.sh
```

Alpha zero reuses the exact unhooked result. The causal report separately
checks decoded projection ordering, monotonic semantic STAY probability,
matched random directions, label reversal, and the non-persistence task. A
failed decoded calibration is classified as invalid/inconclusive, not null.
