# Data manifest

The computational-model comparison uses the repository's existing pilot data:

```text
artifacts/bandit_pilot.csv
```

The file is intentionally not duplicated here. The analysis reads the original
artifact without modifying it. Required columns are:

- `episode_id`
- `round`
- `sampled_action`
- `subsequent_reward`

The committed derived outputs in `../results/` were generated from 200 episodes
and 3,706 decision states in that file.

The cross-task model zoo additionally creates scalar-only CSV exports under
`artifacts/computational_modeling/records/`. These are streamed from the
retained activation banks one episode at a time; activation tensors and full
conversation strings are excluded. `dataset_manifest.json` documents every
column, task/episode/state counts, persisted split hashes, and behavioral input
hashes. `feature_schema.json` is the authoritative allow-list separating
prompt-observable features from explicitly prefixed oracle-state features.
