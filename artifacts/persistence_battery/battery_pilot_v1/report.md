# Literature-grounded persistence task battery

Collection stage: **pilot**. This is a behavior-only run; no hidden activations were collected.

## Voluntary Waiting

1. **Construct:** dynamic persistence / voluntary waiting; adapted from uncertain delayed-reward voluntary persistence (McGuire & Kable, Nature Neuroscience (2015), doi:10.1038/nn.3994).
2. **Manipulated variables:** timing_environment, reward_magnitude, opportunity_cost, quit_payoff.
3. **Persistence behavior:** WAIT probability mean=0.757; sampled rate=0.667.
4. **Nondegenerate choices:** yes (positive-probability SD=0.182; within-episode choice-logit SD=0.737; valid top-token rate=1.000).
5. **Basic incentive manipulation:** lower opportunity cost increases waiting; signed effect=0.086; passed.
6. **Label mappings balanced:** yes (paired correlation=0.967, mean absolute gap=0.067).
7. **Median episode length:** 1.0 decision states.
8. **Sufficient history depth:** no.
9. **Approved for full collection:** no — revise task parameters or parsing before full collection.
10. **Parameters changed after pilot:** none recorded.

## Progressive Ratio

1. **Construct:** breakpoint / effort motivation; adapted from progressive-ratio reinforcement schedule (Markou et al., Schizophrenia Bulletin (2013), PMCID: PMC3849135).
2. **Manipulated variables:** ratio_schedule, reward_magnitude, effort_cost, outside_option.
3. **Persistence behavior:** WORK probability mean=0.995; sampled rate=0.994.
4. **Nondegenerate choices:** no (positive-probability SD=0.025; within-episode choice-logit SD=2.110; valid top-token rate=1.000).
5. **Basic incentive manipulation:** shallower effort growth increases breakpoint; signed effect=1.188; passed.
6. **Label mappings balanced:** yes (paired correlation=0.979, mean absolute gap=0.003).
7. **Median episode length:** 36.0 decision states.
8. **Sufficient history depth:** yes.
9. **Approved for full collection:** no — revise task parameters or parsing before full collection.
10. **Parameters changed after pilot:** none recorded.

## Sunk Cost

1. **Construct:** sunk-cost persistence; adapted from Restaurant Row / Web-Surf change-of-mind waiting (Sweis et al., Science (2018), PMCID: PMC6377599).
2. **Manipulated variables:** prior_investment, remaining_steps, reward_magnitude, outside_option, step_cost.
3. **Persistence behavior:** CONTINUE_WAITING probability mean=0.772; sampled rate=0.792.
4. **Nondegenerate choices:** yes (positive-probability SD=0.185; within-episode choice-logit SD=0.806; valid top-token rate=1.000).
5. **Basic incentive manipulation:** lower remaining cost increases continuation; signed effect=0.047; passed.
6. **Label mappings balanced:** no (paired correlation=0.623, mean absolute gap=0.159).
7. **Median episode length:** 2.0 decision states.
8. **Sufficient history depth:** yes.
9. **Approved for full collection:** no — revise task parameters or parsing before full collection.
10. **Parameters changed after pilot:** none recorded.

## Information Sampling

1. **Construct:** epistemic persistence / information sampling; adapted from Information Sampling Task (Clark et al. paradigm; adaptation overview PMCID: PMC6795545).
2. **Manipulated variables:** evidence_accuracy, sample_cost, error_penalty, prior_a, true_state.
3. **Persistence behavior:** SAMPLE probability mean=0.603; sampled rate=0.648.
4. **Nondegenerate choices:** yes (positive-probability SD=0.165; within-episode choice-logit SD=0.648; valid top-token rate=1.000).
5. **Basic incentive manipulation:** lower sampling cost increases sampling; signed effect=0.058; passed.
6. **Label mappings balanced:** yes (paired correlation=0.902, mean absolute gap=0.081).
7. **Median episode length:** 2.0 decision states.
8. **Sufficient history depth:** yes.
9. **Approved for full collection:** yes — all pilot gates passed.
10. **Parameters changed after pilot:** none recorded.

## Partial Reinforcement

1. **Construct:** partial-reinforcement extinction persistence; adapted from partial reinforcement extinction effect (PREE) (Capaldi sequential theory; review PMCID: PMC10842266).
2. **Manipulated variables:** reinforcement_schedule, acquisition_trials, partial_probability, extinction_try_cost.
3. **Persistence behavior:** TRY_AGAIN probability mean=0.471; sampled rate=0.467.
4. **Nondegenerate choices:** yes (positive-probability SD=0.385; within-episode choice-logit SD=2.300; valid top-token rate=1.000).
5. **Basic incentive manipulation:** lower extinction cost increases trying; signed effect=0.085; passed.
6. **Label mappings balanced:** yes (paired correlation=0.992, mean absolute gap=0.054).
7. **Median episode length:** 2.0 decision states.
8. **Sufficient history depth:** yes.
9. **Approved for full collection:** yes — all pilot gates passed.
10. **Parameters changed after pilot:** none recorded.

## Independent Effort Control

1. **Construct:** generic repeated effort choice; adapted from EEfRT / COGED (Treadway et al. (2009), PMCID: PMC2720457; Westbrook et al. (2013), PMCID: PMC4445645).
2. **Manipulated variables:** high_reward_bonus, high_effort_cost, high_success_probability.
3. **Persistence behavior:** HIGH_EFFORT probability mean=0.563; sampled rate=0.531.
4. **Nondegenerate choices:** yes (positive-probability SD=0.226; within-episode choice-logit SD=0.816; valid top-token rate=1.000).
5. **Basic incentive manipulation:** high-effort choice tracks offer utility; signed effect=0.268; passed.
6. **Label mappings balanced:** yes (paired correlation=0.957, mean absolute gap=0.053).
7. **Median episode length:** 8.0 decision states.
8. **Sufficient history depth:** yes.
9. **Approved for full collection:** yes — all pilot gates passed.
10. **Parameters changed after pilot:** none recorded.

## Interpretation guardrail

Sunk-cost sensitivity, partial-reinforcement extinction, controllability transfer, goal gradients, and human-like recency are scientific hypotheses, not validity gates. Their absence must be reported rather than designed away.

All counterbalanced pairs share environmental seeds, semantic actions, outcomes, and histories. The independent-effort control is sequential but marks `same_goal_across_steps=false`; persistence probabilities are left null rather than fabricated.
