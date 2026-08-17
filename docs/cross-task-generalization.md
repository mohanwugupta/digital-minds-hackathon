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

## Track B: construct-level shared persistence test

The current protocol supersedes the original assumption that the layer-31
Bandit A/B-versus-C probe is itself the task-general construct. It treats that
direction as a potentially output-adjacent, task-specific diagnostic. The
primary question is instead whether a direction learned jointly from two
different semantic operationalizations predicts a third, never-used
operationalization.

The tasks and raw labels are:

- Bandit: A/B versus C/STOP (retained data).
- Foraging: STAY versus LEAVE, counterbalanced X/Y.
- Solvability: TRY AGAIN versus GIVE UP, counterbalanced M/N.
- Negative control: one-shot integer comparison, counterbalanced M/N so its
  output tokens match the primary held-out task.
- Terminality control: an external even/odd rule determines PROCEED versus END,
  counterbalanced M/N, with no judgment about whether continued pursuit is
  worthwhile.

Within each discovery task, the semantic persistence logit is standardized
using that task's training episodes only. A single ridge loss gives Bandit and
Foraging equal weight regardless of state count. Layer, ridge penalty,
activation moments, target moments, and weights are selected only from those
two tasks. The primary artifact is then frozen before its strict scale-free
correlation with held-out Solvability is evaluated.

All three leave-one-task-out folds are also frozen before analysis. They are a
robustness analysis; the prospectively primary result remains
Bandit+Foraging → Solvability.

### Behavioral and activation collection

Submit the guarded tests → real-model smoke → collection → development-only
behavioral gates → task-specific ceilings + shared discovery → held-out test
dependency chain from the cluster login node:

```bash
TRACK_B_RUN_ID=track_b_shared_v3 TRACK_B_SHARDS=4 \
  bash scripts/submit_track_b.sh
```

Every artifact is isolated under `artifacts/cross_task/<run-id>/`; choose a new
run ID rather than overwriting a previous experiment. The helper first checks
each task's labels under the exact rendered chat templates, then collects
Foraging, Solvability, the binary control, and the terminality control. Each
pair shares its task
condition and stochastic seed while reversing the raw-label semantics. Pair
IDs, not individual episodes, are assigned to train, validation, or test.
Before full collection, its real-model smoke collects 16 tiny episodes per new
task and executes all three one-layer LOTO fits, probe serialization, a held-out
projection, matched random directions, and both control projections. The
smoke summary is explicitly non-scientific.

The equivalent manual collection phases are:

```bash
sbatch --array=0-3 \
  --export=ALL,PHASE=cross_task_collect_foraging,CROSS_TASK_NUM_SHARDS=4,TRACK_B_RUN_ID=track_b_shared_v3 \
  run_qwen35_bandit.sh

sbatch --array=0-3 \
  --export=ALL,PHASE=cross_task_collect_solvability,CROSS_TASK_NUM_SHARDS=4,TRACK_B_RUN_ID=track_b_shared_v3 \
  run_qwen35_bandit.sh

sbatch --array=0-3 \
  --export=ALL,PHASE=cross_task_collect_control,CROSS_TASK_NUM_SHARDS=4,TRACK_B_RUN_ID=track_b_shared_v3 \
  run_qwen35_bandit.sh

sbatch --array=0-3 \
  --export=ALL,PHASE=cross_task_collect_terminality,CROSS_TASK_NUM_SHARDS=4,TRACK_B_RUN_ID=track_b_shared_v3 \
  run_qwen35_bandit.sh
```

Foraging uses a depleting patch with varying initial quality, depletion,
outside option, and search cost. Solvability uses repeated diagnostic attempts
with varying progress evidence, attempt cost, and give-up value. The control
is a one-shot integer comparison with the same M/N labels as Solvability and
no continue/quit meaning. The second control asks the model to obey an external
even/odd PROCEED/END rule, separating goal persistence from generic
terminality. The frozen design collects 768 episodes (384 counterbalanced
pairs) per new task. The prospective report distinguishes all 384 pairs from
the approximately 59 independent clusters in the final 15% test partition.

Before any held-out probe result is opened, the helper runs:

```bash
sbatch --export=ALL,PHASE=cross_task_behavioral_validate,TRACK_B_RUN_ID=track_b_shared_v3 \
  run_qwen35_bandit.sh
```

This audits exact inverse mappings, paired conditions, terminal semantics,
episode/state completeness, and probability geometry in all four new banks.
Its behavioral criteria use only train+validation episodes and separately
require meaningful Foraging and Solvability behavior. A failed gate writes a
report and exits nonzero; it cannot proceed to held-out transfer testing.

After that gate, the helper runs `cross_task_matched_label_foraging` and
`cross_task_matched_label_solvability` as sharded jobs. Each reconstructs one
canonical held-out semantic trajectory per counterbalanced pair and renders
every state twice, once under each inverse mapping. The final analyzer checks
complete source-state coverage and refuses partial replay banks.

### Representational checkpoint

The dependency helper runs these logical phases:

```bash
sbatch --export=ALL,PHASE=cross_task_train_foraging_ceiling,TRACK_B_RUN_ID=track_b_shared_v3 run_qwen35_bandit.sh
sbatch --export=ALL,PHASE=cross_task_train_solvability_ceiling,TRACK_B_RUN_ID=track_b_shared_v3 run_qwen35_bandit.sh
sbatch --export=ALL,PHASE=cross_task_train_shared,TRACK_B_RUN_ID=track_b_shared_v3 run_qwen35_bandit.sh
sbatch --export=ALL,PHASE=cross_task_shared_representational,TRACK_B_RUN_ID=track_b_shared_v3 run_qwen35_bandit.sh
```

The shared training phase never evaluates held-out test targets. The analyzer
reports the strict held-out correlation and correlation-squared association,
counterbalanced-pair-clustered intervals, mapping-specific intervals, 100 matched
sign-randomized directions, both M/N controls, the Solvability-specific
ceiling, all LOTO folds, and layerwise cosine alignment among independently
trained task probes. It also requires positive validation performance above
matched-random controls in each discovery task separately. Exact held-out
semantic histories are replayed under both inverse label mappings; organic
paired trajectories are not used as the label-geometry clearance test. A
validation-only affine fit is retained as a scale
diagnostic and cannot affect clearance.

Foraging and Solvability ceiling jobs use `--defer-test`: they select and
freeze their layers from validation only. Their test performance is first
computed inside the final shared analyzer, after all shared artifacts are
frozen.

The historical exact Bandit→Foraging probe test runs only after this primary
analysis and writes to `transfer/`. It is diagnostic and is not consulted by
the causal submission gate.

### Causal checkpoint

Only a `strong_shared_transfer` or `partial_shared_transfer` primary result can
unlock the causal helper. It calibrates the shared Bandit+Foraging direction on
Solvability validation activations; test states are touched only after the
magnitude is frozen.

```bash
TRACK_B_RUN_ID=track_b_shared_v3 TRACK_B_CAUSAL_SHARDS=8 \
  bash scripts/submit_track_b_causal.sh
```

The helper and calibration program independently enforce this gate.

Alpha zero reuses the exact unhooked result. The causal report separately
checks decoded projection ordering, monotonic semantic TRY-AGAIN probability,
pair-cluster bootstrap uncertainty, full held-out coverage for all 20 matched random
directions, label reversal, and specificity against both the non-persistence
binary task and rule-determined terminality.
A failed decoded calibration is classified as invalid/inconclusive, not null.
Neither submission helper launches Track C.
