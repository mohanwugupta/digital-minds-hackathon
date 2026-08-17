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
