# Computational modeling of persistence

## Cross-task computational model zoo

The model-zoo extension exports records-only Bandit, Foraging, and Solvability
datasets, enforces explicit observable/oracle information sets, compares M0--M18
on persisted episode/pair-safe splits, runs synthetic architecture recovery,
and saves interpretable latent states for later neural alignment. It does not
load activation tensors after export and does not perform the future L21/L22
analysis.

Run the required pre-compute checkpoint first:

```bash
python -m computational_modeling.analysis.run_model_zoo \
  --config config/computational_model_zoo.yaml \
  --phase checkpoint \
  --run-id implementation_checkpoint_v1
```

After reviewing `implementation_checkpoint.json`, launch the full exploratory
analysis with a fresh run ID:

```bash
python -m computational_modeling.analysis.run_model_zoo \
  --config config/computational_model_zoo.yaml \
  --phase all \
  --run-id model_zoo_mac_v2
```

Add `--final-bootstrap` only for the final 10,000-sample interval run. The
default uses 2,000 development bootstrap samples. Existing run directories are
never overwritten unless `--resume` is passed explicitly.

The primary ranking is observable-only and enforces exact three-task coverage;
one-task candidates are ranked only within their supported task. The full
flexible linear model is repeated with oracle state, which can access only
fields carrying the `oracle_` prefix in the feature schema. MLP and GRU models
are intentionally small flexible candidates, with all architecture and
early-stopping choices made on validation episodes. Neither is assumed to be a
ceiling: both must first pass the saved synthetic `y = X beta + epsilon`
linear-recovery gate, and `best_flexible_model` is selected from flexible
linear, MLP, and GRU by held-out macro R².

Before any L21/L22 work, the runner also refits the strongest validation-chosen
observable flexible-linear architecture after removing each preregistered
feature family and with each family alone. Task indicators remain nuisance
controls. Feature families come from `data/feature_schema.py`; the exact
`relative_value`/`termination_advantage` alias is deliberately not duplicated.
All ablation differences receive pair/episode-clustered bootstrap intervals.

Outputs are written beneath `artifacts/computational_modeling/<run_id>/` and
include the full required tables, clustered intervals, recovery matrix,
figures, report, provenance, and state-level latent-variable exports.
The follow-up's fixed deliverables are
`cross_task_model_ranking.csv`, `task_specific_model_ranking.csv`,
`flexible_model_comparison.csv`, `neural_ceiling_sanity.json`,
`feature_group_ablation.csv`, `feature_group_only.csv`,
`feature_group_bootstrap.csv`, the three corresponding figures, and
`report.md`.
Every phase emits flushed `[model-zoo +elapsed] [section]` progress lines. The
same events are appended to `progress.jsonl`, while completed/failed section
durations are summarized in `timings.json`, making long Slurm jobs inspectable
without waiting for completion.

## Original Bandit comparison

This folder is an additive, self-contained supplement. It does not modify the
existing experimental or analysis pipeline. Run it from the repository root:

```bash
python -m computational_modeling.analysis.compare_behavioral_models \
  --input artifacts/bandit_pilot.csv \
  --output-dir computational_modeling/results
```

The source dataset remains at `artifacts/bandit_pilot.csv`; see
[`data/README.md`](data/README.md) for the manifest. Derived tables, predictions,
and the generated report are stored in [`results/`](results/).

## Question

Does a dynamically learned arm value improve prediction of STOP beyond recent
reward and time? This is the behavioral counterpart of the repository's causal
target, the STOP-versus-CONTINUE persistence logit. It is not evidence by
itself that the transformer's hidden states implement a particular algorithm.

## Latent learning models

For Rescorla--Wagner (RW), both arms start at the expected reward under a
uniform success prior, `Q_A(0) = Q_B(0) = 0.5`. After choosing arm `a` and
observing reward `r`:

```text
prediction error:  delta_t = r_t - Q_a(t)
update:             Q_a(t+1) = Q_a(t) + alpha * delta_t
unchosen arm:       Q_b(t+1) = Q_b(t)
```

The policy receives the best current value and absolute A/B value gap. Learning
rate `alpha` is selected using training episodes only in each outer fold.

The Bayesian alternative gives each arm an independent `Beta(1,1)` prior over
success probability. Success/failure counts update for the selected arm, and
posterior mean probability is converted to points as `5p - 2`. The posterior
mean alone does not represent the option value of exploration.

## Choice models

All candidates predict the sampled binary action `STOP` versus `CONTINUE` with
a regularized logistic policy:

- **time:** `log(1 + round)`;
- **heuristic:** time, previous outcome, and consecutive-loss streak;
- **RW:** time, best RW value, and RW A/B gap;
- **RW hybrid:** heuristic plus RW states;
- **Bayesian:** time, best Bayesian posterior value, and posterior A/B gap;
- **Bayesian hybrid:** heuristic plus Bayesian states.

Five-fold evaluation holds out whole episodes. Predictors are standardized
using training statistics only. Every episode has equal total weight, because
long episodes are an outcome of stopping rather than independent extra
participants. Primary fit is held-out log loss; AUC and Brier score are
secondary. Uncertainty is obtained by resampling episode-mean losses.

## Pilot result

The analysis contains 3,706 decision states from 200 episodes, including 180
sampled STOP actions.

| Model | Held-out log loss | AUC | Brier |
|---|---:|---:|---:|
| Bayesian hybrid | 0.4527 | 0.802 | 0.1429 |
| Heuristic | 0.4553 | 0.805 | 0.1445 |
| RW hybrid | 0.4612 | 0.787 | 0.1457 |
| Bayesian | 0.4625 | 0.746 | 0.1453 |
| RW | 0.4711 | 0.737 | 0.1492 |
| Time only | 0.4990 | 0.753 | 0.1600 |

Relative to the heuristic, positive episode-level log-loss differences favor
the candidate:

| Candidate | Difference | 95% bootstrap CI | p |
|---|---:|---:|---:|
| Bayesian hybrid | +0.0026 | [-0.0032, +0.0083] | .379 |
| RW hybrid | -0.0059 | [-0.0108, -0.0011] | .014 |
| Bayesian | -0.0071 | [-0.0188, +0.0047] | .244 |
| RW | -0.0157 | [-0.0278, -0.0031] | .016 |

RW therefore does **not** explain stopping well relative to recent history. Its
values carry predictive information, but they do not improve held-out STOP
prediction after loss streak, previous reward, and time are included. The
Bayesian hybrid's small improvement is not distinguishable from zero.

This agrees with the probe results: future return is decodable, but decoded
value does not explain stopping after recent-history matching. The model may
represent how the episode is going without using that integrated value as the
main policy variable for quitting.

## Limits and next steps

This establishes predictive adequacy on held-out pilot episodes. It does not
identify a neural RW implementation, prove a verbal loss-streak rule, or show
that value is causally irrelevant. The causal steering and external-payoff
experiments address the latter question.

Before treating this as a final computational account:

1. preregister the candidates and evaluate them on a new behavioral sample;
2. fit A versus B conditional on continuing, separating arm learning from STOP;
3. perform parameter and model recovery on synthetic episodes;
4. test a finite-horizon retirement-bandit model with information value;
5. use a hierarchical model across checkpoints, prompts, or temperatures.

## References

- Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian conditioning:
  Variations in the effectiveness of reinforcement and nonreinforcement.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press.
- Daw, N. D. (2011). Trial-by-trial data analysis using computational models.
  In *Decision Making, Affect, and Learning*.
