# Literature-grounded persistence battery

This subsystem implements `PRD_more_tasks.md` as a behavior-only pilot-first
benchmark. It does not collect hidden states, train probes, fit neural
subspaces, or perform interventions.

## Included paradigms

The five new persistence tasks are voluntary waiting, progressive-ratio
effort, sunk-cost waiting, information sampling, and partial-reinforcement
extinction. Independent effort choice supplies the mandatory sequential
non-persistence control. A yoked controllability-transfer task is implemented
but disabled by default under `stretch.controllability_enabled`.

Every task specification records its construct, source paradigm, citation,
adaptation notes, and departures from the original paradigm. These are
computational adaptations, not claims of exact human or animal replication.

## Counterbalancing and replay

Each semantic pair uses two verified single-token mappings:

```text
X = positive/continue semantic action, Y = negative/disengage action
Y = positive/continue semantic action, X = negative/disengage action
```

The primary mapping supplies a sampled semantic trajectory. The reverse
mapping is scored on an exact replay of the same semantic actions, environment
seed, outcomes, and history. Consequently, label-bias comparisons are matched
state comparisons rather than two potentially divergent episodes.

The independent-effort control has repeated offers and feedback but marks
`same_goal_across_steps=false`. Its persistence-specific probability and logit
fields are null; generic choice fields remain available.

## Pilot-first workflow

Run a model-free implementation smoke:

```bash
python -u -m analysis.run_persistence_battery \
  --phase pilot \
  --run-id battery_smoke_v1 \
  --smoke \
  --model-free
```

This smoke can never approve a task and is ignored by Git.

Run the real pilot:

```bash
python -u -m analysis.run_persistence_battery \
  --phase pilot \
  --run-id battery_pilot_v1 \
  --model /path/to/Qwen--Qwen3.5-4B
```

Inspect `report.md` and the three files under `validation/`. If a task fails,
revise only parameters needed to make the task usable, record the change under
`pilot_adjustments`, and use a new run ID. Sunk-cost bias, PREE,
controllability transfer, goal gradients, and human-like recency are explicitly
non-gating scientific hypotheses.

After every requested task is approved, launch full collection in the same run:

```bash
python -u -m analysis.run_persistence_battery \
  --phase full \
  --run-id battery_pilot_v1 \
  --model /path/to/Qwen--Qwen3.5-4B \
  --resume
```

The full phase refuses to start when the real pilot approval is missing,
model-free, or negative for any requested task.

## Sharded cluster collection

Pairs—not mappings—are the sharding unit, so both label mappings always remain
together. Submit pilot jobs with, for example:

```bash
sbatch --array=0-3 \
  --export=ALL,PHASE=persistence_battery_pilot,PERSISTENCE_BATTERY_RUN_ID=battery_pilot_v1,PERSISTENCE_BATTERY_NUM_SHARDS=4,PERSISTENCE_BATTERY_RESUME=1 \
  run_qwen35_bandit.sh
```

After every shard finishes, consolidate without loading Qwen:

```bash
sbatch --export=ALL,PHASE=persistence_battery_finalize,PERSISTENCE_BATTERY_RUN_ID=battery_pilot_v1,PERSISTENCE_BATTERY_DATASET=pilot \
  run_qwen35_bandit.sh
```

The same finalize command also recovers a single-job run that completed model
inference but stopped while writing Parquet. Raw semantic-pair caches are
validated before reuse, and a fully cached resumed pilot skips model loading
and inference entirely.

Use `PHASE=persistence_battery_full` only after approval. Full collection can
be sharded with the same environment variables and finalized with
`PERSISTENCE_BATTERY_DATASET=full`.

## Outputs

Pilot records live under `pilot/records/`; approved full records live under
`records/`. Parquet is preferred. If neither `pyarrow` nor `fastparquet` is
installed, finalization writes lossless compressed CSV files instead and
records the selected format in `records_manifest.json`. Validation and the
downstream comparative-modeling pipeline transparently read either format.
Root-level manifests document task specifications, factorial
conditions, and pair-safe splits. Validation files cover manipulation checks,
label bias, response parsing, nondegeneracy, and episode length. The three
figures, ten-question task report, progress log, timing log, and run metadata
are also written at the run root.

Raw per-pair JSON used for resumability lives under `cache/`. It is regenerable,
ignored by Git, and should not be pushed or added to LFS.
