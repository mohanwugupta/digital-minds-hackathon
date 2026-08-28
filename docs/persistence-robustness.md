# PRD 2.5 persistence robustness

This subsystem strengthens the behavioral benchmark before any new mechanistic
claim. It repairs the waiting, progressive-ratio, and sunk-cost tasks; adds a
controllability task and a debugging-persistence fallback; constructs an exact
yoked same-goal versus independent-goal control; and establishes a larger GRU
ceiling over the expanded data.

Smoke and pilot inclusion are deliberately distinct. A smoke run may exercise
candidate tasks, but it never counts them toward the task-breadth gate. An
extension task becomes scientifically usable only after the real Qwen pilot
passes all functional validity checks and its full dataset contains at least
256 independent semantic histories (512 recorded counterbalanced episodes).
Human/animal signatures are reported but never used as inclusion gates.

## What runs where

The battery and matched-control collection phases call Qwen and should run on
the cluster. They retain behavioral probabilities and prompts, not hidden
states. The linear analyses can run on a laptop. The full 64/128/256/512 GRU
search is most practical on a GPU, although the reduced `--smoke` analysis is
laptop-safe.

All phases sharing data must use the same run ID. Raw JSON pair caches support
resume and sharding, live below the run-local `cache/`, and are ignored by Git.
Final behavioral records use Parquet when an engine is available and otherwise
fall back losslessly to compressed CSV.

## Local plumbing smoke

This runs without a model checkpoint and produces no scientific evidence:

```bash
python -u -m analysis.run_persistence_battery \
  --config config/persistence_robustness_v1.yaml \
  --phase pilot \
  --tasks voluntary_waiting,progressive_ratio,sunk_cost,controllability,debugging_persistence \
  --run-id robustness_smoke_v1 \
  --smoke --model-free

python -u -m experiments.collect_matched_goal_control \
  --config config/persistence_robustness_v1.yaml \
  --phase collect --dataset pilot \
  --run-id robustness_smoke_v1 \
  --smoke --model-free --resume

python -u -m analysis.run_persistence_robustness \
  --config config/persistence_robustness_v1.yaml \
  --phase all \
  --run-id robustness_smoke_v1 \
  --smoke --resume
```

## Cluster collection

First run the tests and the functional pilot. Semantic pairs, not individual
label mappings, are the sharding unit:

```bash
sbatch --export=ALL,PHASE=persistence_robustness_tests \
  run_qwen35_bandit.sh

sbatch --array=0-3 \
  --export=ALL,PHASE=persistence_robustness_pilot,PERSISTENCE_ROBUSTNESS_RUN_ID=robustness_v1,PERSISTENCE_ROBUSTNESS_NUM_SHARDS=4 \
  run_qwen35_bandit.sh

sbatch \
  --export=ALL,PHASE=persistence_robustness_battery_finalize,PERSISTENCE_ROBUSTNESS_RUN_ID=robustness_v1,PERSISTENCE_ROBUSTNESS_DATASET=pilot \
  run_qwen35_bandit.sh
```

Inspect `artifacts/persistence_robustness/robustness_v1/report.md` and
`validation/pilot_approval.json`. Then collect only the approved task names by
setting `PERSISTENCE_ROBUSTNESS_TASKS`. For example, if all five pass:

```bash
sbatch --array=0-3 \
  --export=ALL,PHASE=persistence_robustness_full,PERSISTENCE_ROBUSTNESS_RUN_ID=robustness_v1,PERSISTENCE_ROBUSTNESS_NUM_SHARDS=4 \
  run_qwen35_bandit.sh

sbatch \
  --export=ALL,PHASE=persistence_robustness_battery_finalize,PERSISTENCE_ROBUSTNESS_RUN_ID=robustness_v1,PERSISTENCE_ROBUSTNESS_DATASET=full \
  run_qwen35_bandit.sh
```

For fast exploratory iteration, full collection can be started directly from
a fresh run ID. This is an explicit override recorded in `run_metadata.json`;
the resulting full records still undergo the same validation, and only tasks
that pass those checks count as scientifically usable:

```bash
sbatch --array=0-3 \
  --export=ALL,PHASE=persistence_robustness_full,PERSISTENCE_ROBUSTNESS_RUN_ID=robustness_full_v1,PERSISTENCE_ROBUSTNESS_NUM_SHARDS=4,PERSISTENCE_ROBUSTNESS_SKIP_PILOT_APPROVAL=1 \
  run_qwen35_bandit.sh
```

After all array jobs finish, run the full battery finalizer shown above with
`PERSISTENCE_ROBUSTNESS_RUN_ID=robustness_full_v1`. The matched-control full
collection can use that same run ID and run concurrently.

The matched primary and secondary controls use their own 256 yoked latent
sequences. They may be collected while the approved full battery is running:

```bash
sbatch --array=0-3 \
  --export=ALL,PHASE=persistence_robustness_matched,PERSISTENCE_ROBUSTNESS_RUN_ID=robustness_v1,PERSISTENCE_ROBUSTNESS_DATASET=full,PERSISTENCE_ROBUSTNESS_NUM_SHARDS=4 \
  run_qwen35_bandit.sh

sbatch \
  --export=ALL,PHASE=persistence_robustness_matched_finalize,PERSISTENCE_ROBUSTNESS_RUN_ID=robustness_v1,PERSISTENCE_ROBUSTNESS_DATASET=full \
  run_qwen35_bandit.sh
```

Finally run the expanded model comparison and report. This phase does not load
Qwen:

```bash
sbatch \
  --export=ALL,PHASE=persistence_robustness_analysis,PERSISTENCE_ROBUSTNESS_RUN_ID=robustness_v1 \
  run_qwen35_bandit.sh
```

`PERSISTENCE_ROBUSTNESS_PHASE` can resume one of `prepare`, `gru`,
`matched_control`, `models`, `signatures`, `synthetic`, or `report`.
`PERSISTENCE_ROBUSTNESS_MODELS` accepts a comma-separated reduced model list.

## Analysis and outputs

The GRU search uses task-balanced episode minibatches, packed causal sequences,
AdamW, gradient clipping, early stopping, validation-only hyperparameter and
depth selection, and seeds 0/1/2. It tests current-state-only and explicit
short-history inputs. A ceiling is called credible only when training is
stable, increasing capacity plateaus, and held-out performance is within the
predefined tolerance of the MLP.

Results are written below `artifacts/persistence_robustness/<run_id>/` in
`tasks/`, `gru/`, `matched_control/`, `models/`, `signatures/`, `synthetic/`,
and `figures/`. `report.md` directly answers all 15 PRD questions. The matched
control reports history gain, persistent-minus-independent history gain,
action/outcome/streak coefficients, and kernel similarity for both the primary
absorbing and secondary advancing designs. It models choice probabilities at
exact yoked states and is not inserted into the absorbing-hazard risk set.
