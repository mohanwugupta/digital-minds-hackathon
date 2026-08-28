# Comparative computational models of persistence

This subsystem implements `PRD_computational_model_zoo.md`. It is a behavioral
analysis over existing records: it never loads Qwen, collects trajectories,
or reads hidden-state caches.

## Frozen task inventory

The original Bandit, Foraging, and Solvability records are combined with tasks
that passed the frozen real-pilot usability gates. In the current retained
pilot, information sampling, partial reinforcement, and the independent-effort
sequential control passed. Voluntary waiting, progressive ratio, and sunk cost
remain excluded because they failed basic pilot gates. Their scientific
signatures were not used to decide inclusion.

Each task is converted to an absorbing discrete-time termination-hazard risk
set. Horizon completion is right-censored; states after termination are
forbidden. Semantic values use frozen environment-spec scales rather than
statistics estimated from target-task behavior. Missing constructs are encoded
with explicit presence indicators, and future outcomes are blocked from model
features.

## Model and evaluation coverage

The registry includes M0–M14: null/time/immediate baselines, finite and
exponential history, perseveration, dual history, dynamic reevaluation and its
separately labeled oracle ceiling, option termination, competitive time/reward,
latent commitment, sunk-cost extension, flexible linear, MLP, and causal GRU.
Relevant models run under task-specific, fully shared, and hierarchical
sharing assumptions.

The primary score is equal-task macro log loss. Secondary outputs include
Brier score, AUC, calibration, deviance explained, and state-weighted log loss.
Within-task intervals resample complete episodes; cross-task intervals resample
task identities. Strict source-only LOTO and LOFO, pair-disjoint few-shot
adaptation, architecture transfer, history/control PSH, feature ablations,
human/animal signature checks, GRU bottlenecks, and H1–H4 synthetic recovery
are included.

## Local commands

First run the integrity and recovery tests:

```bash
python -m pytest -q tests/comparative_persistence
```

Run the reduced smoke analysis:

```bash
python -u -m analysis.run_comparative_persistence \
  --config config/comparative_persistence.yaml \
  --phase all \
  --run-id comparative_smoke_v1 \
  --smoke \
  --skip-neural
```

Run the complete analysis:

```bash
python -u -m analysis.run_comparative_persistence \
  --config config/comparative_persistence.yaml \
  --phase all \
  --run-id comparative_v1
```

The smoke run is suitable for a laptop. The full linear zoo can also run on a
MacBook with `--skip-neural`; the full MLP/GRU grids are faster on a GPU.
Use a new run ID unless resuming an existing output. `--resume` reuses the
serialized modeling dataset and supports restarting an individual `--phase`
(`prepare`, `models`, `generalization`, `history`, `features`, `synthetic`, or
`report`). A phase that depends on earlier tables expects those phases to have
completed in the same run directory.

To run only selected architectures, pass a comma-separated registry list:

```bash
python -u -m analysis.run_comparative_persistence \
  --run-id comparative_history_v1 \
  --models immediate_state,finite_history,dual_history,latent_commitment,mlp,gru
```

## Cluster commands

The shared Slurm dispatcher does not require the Qwen checkpoint for these
analysis-only phases. Run the tests with:

```bash
sbatch --export=ALL,PHASE=comparative_persistence_tests run_qwen35_bandit.sh
```

Run the full zoo with:

```bash
sbatch \
  --export=ALL,PHASE=comparative_persistence,COMPARATIVE_PERSISTENCE_RUN_ID=comparative_v1 \
  run_qwen35_bandit.sh
```

Optional environment variables are
`COMPARATIVE_PERSISTENCE_PHASE`, `COMPARATIVE_PERSISTENCE_RESUME=1`,
`COMPARATIVE_PERSISTENCE_SMOKE=1`, `COMPARATIVE_PERSISTENCE_SKIP_NEURAL=1`,
`COMPARATIVE_PERSISTENCE_MODELS`, and `COMPARATIVE_PERSISTENCE_CONFIG`.

## Outputs and interpretation

Results are written under `artifacts/comparative_persistence/<run_id>/`.
`report.md` starts with direct answers to all ten PRD questions. The report
labels smoke output as plumbing-only. A real-data ranking should be interpreted
only after inspecting `synthetic/recovery.csv` and `synthetic/confusion_matrix.csv`;
candidate families that are not distinguishable at realistic sample sizes do
not support a strong empirical ranking claim.

The dataset, held-out predictions, and generated tables are analysis artifacts.
No regenerable neural activation or memmap cache is created by this pipeline.
