# Cross-task persistence computation in matched change space

This follow-up tests the frozen `displacement-L21-k4` candidate in the space it
was originally designed for: semantically oriented matched differences. It
never recollects model behavior or refits/rotates the candidate basis.

## Design

The retained contrast inventory defines five persistence families:

- Bandit higher CONTINUE incentive;
- Bandit lower STOP/outside-option value;
- Foraging lower search cost;
- Foraging lower outside-option value;
- Solvability stronger progress evidence.

Every contrast is oriented as `P+ - P-`, where `P+` is the condition that
promotes persistence. The runner independently rechecks the source endpoint
fields declared by each contrast definition. Invalid pairs remain visible in
`contrast_pair_audit.csv` and never enter a fit.

Only endpoint layers 21 and 22 are streamed. The three tested representations
are `Δh21`, `Δ(h22 - h21)`, and `Δh22`, projected through the read-only frozen
rank-4 basis. The runner applies the already validation-selected model-zoo GRU
and finite-history architecture to the contrast endpoints; test targets never
enter model selection or normalization.

Primary decoding reports held-out R², Pearson correlation, MSE, and semantic
sign accuracy. Strict leave-one-task-out and leave-one-manipulation-out folds
fit and normalize on source train data, select ridge regularization on source
validation data, and evaluate only the held-out test group. Pair-clustered
bootstraps use 2,000 draws.

The specificity battery includes 100 same-stage random rank-4 subspaces,
matched label replays, balanced arbitrary choice, unrelated terminality, and
one-shot generic-value changes. Direction cosines, task-subspace principal
angles/overlap, and stage-transition summaries distinguish a shared direction,
a shared low-dimensional subspace, and task-specific geometry.

## Required source artifacts

The analysis uses the retained Track B and nuisance banks, model-zoo records,
contrast inventory, and frozen candidate. It also needs the optional all-layer
Bandit factorial tensors:

```text
artifacts/value_dissociation/activations/
```

These tensors exist in the full cluster workspace but are intentionally absent
from lightweight local downloads. Sync them to run on a laptop. If they are
missing on the cluster, regenerate only this pre-existing replay artifact:

```bash
LAYERWISE_NUM_SHARDS=8 \
sbatch --array=0-7 \
  --export=ALL,PHASE=factorial_layerwise_project,LAYERWISE_NUM_SHARDS=8 \
  run_qwen35_bandit.sh
```

## Running

From the repository root:

```bash
python -u -m analysis.run_persistence_change_geometry \
  --config config/persistence_change_geometry.yaml \
  --phase all \
  --run-id model_zoo_mac_v2
```

Append `--resume` after interruption. `--smoke` uses smaller pair, bootstrap,
and random-control counts while preserving every analysis path.

Cluster phase:

```bash
sbatch --export=ALL,PHASE=persistence_change_geometry run_qwen35_bandit.sh
```

## Outputs

Runs are written under `artifacts/persistence_change_geometry/<run_id>/` with
the exact tables and six figures listed in `cross-task_mini_PRD.md`. Durable
`progress.jsonl` and `timings.json` sidecars expose section progress and elapsed
time. The report applies Outcomes A–D conservatively and explicitly distinguishes
exploratory representation from causal mediation.
