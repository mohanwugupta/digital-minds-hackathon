# Task-general persistence discovery (Track C)

This implementation extends the existing Track A/B repository. It reuses the
model adapter, activation formats, episode splits, ridge fitters, bootstrap
routines, provenance helpers, and Slurm runner.

## Implemented analyses

Track C0 builds five positively oriented persistence families:

- Bandit higher CONTINUE incentive;
- Bandit lower STOP/outside-option value;
- Foraging lower search cost;
- Foraging lower leaving value;
- Solvability stronger displayed progress/solvability evidence.

Foraging and Solvability pairs hold exact semantic history, round, response
mapping, and all nonmanipulated task factors fixed. Bandit uses the existing
state-fixed STOP x CONTINUE factorial. Every pair is constructed independently
inside train, validation, or test; pair validation rejects any mismatch.

The nuisance battery contains exact label remappings, the arbitrary integer
comparison, rule-determined PROCEED/END, and a new counterbalanced one-shot
voucher-value choice with no ongoing goal.

The wide search evaluates all eligible layers using static contrast features
and layer-to-layer displacement features, with ranks 1, 2, and 4 only. It
reports persistence sensitivity, leave-one-manipulation-out transfer,
leave-one-task-out transfer, every nuisance sensitivity, and matched-random
subspaces separately. Strict folds select only on source validation data.

Track C1 implements synthetic recovery/confusion gates for immediate choice,
choice inertia, recurrent commitment, and generic latent value. The real-data
state is interpreted only if the latent architecture wins held-out episode
prediction and improves held-out future-persistence prediction beyond the
current continuation logit. All-layer residual decoding and behavioral-time
transition alignment are conditional on those gates.

Integration compares candidate subspaces, tests whether matched persistence
manipulations shift inferred commitment, and applies a conservative causal
gate. The submission script never launches causal work or Task 4.

## Required existing artifacts

Track B banks under `artifacts/cross_task/track_b_shared_v3/` are reused
directly. Track C also requires the all-layer Bandit factorial tensors under:

```text
artifacts/value_dissociation/activations/
```

Track A metadata shows these were written on the cluster, although they may be
absent from a lightweight local artifact download. If absent on the cluster,
recover them without retraining probes:

```bash
LAYERWISE_NUM_SHARDS=8 \
sbatch --array=0-7 \
  --export=ALL,PHASE=factorial_layerwise_project,LAYERWISE_NUM_SHARDS=8 \
  run_qwen35_bandit.sh
```

The replay is resumable and writes state tensors alongside the existing
layerwise projections.

## Cluster execution

From the repository root:

```bash
bash scripts/submit_persistence_discovery.sh
```

The dependency graph is:

```text
tests -> smoke -> generic-value collection -> contrast bank -> contrast search
               \-> latent search ----------------------------/ -> integration
```

Useful overrides are `PERSISTENCE_SHARDS`, `PERSISTENCE_COLLECTION_TIME`,
`MODEL_PATH`, and `CONDA_ENV`.

Every phase can be restarted independently:

```bash
PHASE=persistence_tests sbatch run_qwen35_bandit.sh
PHASE=persistence_smoke sbatch run_qwen35_bandit.sh
PHASE=persistence_contrast sbatch run_qwen35_bandit.sh
PHASE=persistence_search sbatch run_qwen35_bandit.sh
PHASE=persistence_latent sbatch run_qwen35_bandit.sh
PHASE=persistence_integration sbatch run_qwen35_bandit.sh
```

Generic-value collection is episode-sharded and resumes from existing atomic
files. A four-shard restart is:

```bash
sbatch --array=0-3 \
  --export=ALL,PHASE=persistence_collect_generic_value,CROSS_TASK_NUM_SHARDS=4 \
  run_qwen35_bandit.sh
```

## Outputs

The contrast phase writes `contrast_bank.pt`, `contrast_inventory.csv`, and
`contrast_audit.json` under `artifacts/persistence_discovery/`.

The search directory contains the complete candidate/fold JSON, frozen
subspaces, a layerwise transfer CSV, a static/displacement SVG, and a Markdown
report. Track C1 writes synthetic recovery, behavioral comparison, inferred
states, future behavior, transition results, and conditional residual probes
under `artifacts/persistence_discovery/latent_state/`. The final decision is
under `artifacts/persistence_discovery/integration/`.

All B/F/S results are exploratory. A large transfer effect is insufficient if
label, arbitrary-choice, terminality, or generic-value sensitivity is
comparable. If integration says `stop_causal_pipeline`, do not steer the
candidate and do not design Task 4.

