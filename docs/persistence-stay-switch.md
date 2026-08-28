# Persistence stay/switch pivot

This pipeline tests whether Bandit, Foraging, and Solvability share a
history-dependent policy-maintenance computation even when their literal
neural directions differ. It is an analysis-only pipeline: it reuses existing
behavior, activation, manipulation, control, split, and model-zoo artifacts and
never invokes Qwen inference.

## Analyses

- Discrete-time absorbing hazard models compare task-specific, shared-history,
  shared stay/switch-rule, and fully shared architectures on untouched episodes.
- Finite and exponential causal history kernels are selected on validation data.
- GRU recurrence, causal-window, feature-family, bottleneck, and distillation
  analyses reuse the model zoo's selected architecture and observables.
- Independent task readouts are fit at all 32 layers with train-only
  normalization, validation-selected ridge, equal projection capacity, random
  targets, mapping controls, literal geometry, cross-prediction, and semantic
  history/current-evidence decoding.
- Existing matched manipulations are projected onto each task's own readout to
  estimate onset, peak, area, late/early gain, and cross-profile similarity.
- Existing arbitrary-choice, terminality, and generic-value banks receive the
  same all-layer readout capacity. They remain explicitly one-shot: no history
  or recurrent sequence is fabricated.

## Commands

From the repository root, run a quick real-artifact integration check:

```bash
python -u -m analysis.run_persistence_stay_switch \
  --config config/persistence_stay_switch.yaml \
  --phase all \
  --run-id stay_switch_smoke_v1 \
  --smoke
```

Run the complete protocol locally or on the cluster login/compute node:

```bash
python -u -m analysis.run_persistence_stay_switch \
  --config config/persistence_stay_switch.yaml \
  --phase all \
  --run-id stay_switch_v1
```

Resume a run after interruption by adding `--resume`. Individual phases are
available through `--phase behavior`, `gru`, `neural`, `interventions`,
`controls`, or `report`. A report-only retry therefore uses both `--phase
report` and `--resume`.

The Slurm dispatcher also supports:

```bash
sbatch --export=ALL,PHASE=persistence_stay_switch,PERSISTENCE_STAY_SWITCH_RUN_ID=stay_switch_v1 \
  run_qwen35_bandit.sh
```

## Artifact and cache policy

Results are written under `artifacts/persistence_stay_switch/<run_id>/` without
overwriting earlier analysis directories. Progress is flushed to
`progress.jsonl`, section durations to `timings.json`, and source hashes and
leakage safeguards to `run_metadata.json`.

The run-local `cache/` contains aligned float16 memmaps and metadata that can be
regenerated from the retained activation banks. It is ignored by Git and should
remain local rather than being pushed or placed in LFS.

Bandit persistence interventions use the existing factorial CSVs plus
`artifacts/value_dissociation/activations/` when those tensors are available.
The lightweight laptop artifact bundle intentionally omits that directory; the
runner logs the omission and still evaluates organic Foraging/Solvability
profiles. It automatically includes Bandit on the cluster where those tensors
are materialized.

## Interpretation

The generated report answers the eight questions in `pivot_PRD.md` and assigns
one of the preregistered outcomes only for a full run. Smoke runs are clearly
labelled as pipeline validation and do not receive a scientific outcome. These
analyses remain correlational; the report preserves the PRD's causal gate.
