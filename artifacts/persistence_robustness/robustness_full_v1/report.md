# Literature-grounded persistence task battery

Collection stage: **full**. This is a behavior-only run; no hidden activations were collected.

## Voluntary Waiting

1. **Construct:** dynamic persistence / voluntary waiting; adapted from uncertain delayed-reward voluntary persistence (McGuire & Kable, Nature Neuroscience (2015), doi:10.1038/nn.3994).
2. **Manipulated variables:** timing_environment, reward_magnitude, opportunity_cost, quit_payoff.
3. **Persistence behavior:** WAIT probability mean=0.672; sampled rate=0.681.
4. **Nondegenerate choices:** yes (positive-probability SD=0.211; within-episode choice-logit SD=0.736; valid top-token rate=1.000).
5. **Basic incentive manipulation:** elapsed time changes waiting; signed effect=0.227; passed.
6. **Label mappings balanced:** yes (paired correlation=0.976, mean absolute gap=0.058).
7. **Median episode length:** 2.0 decision states.
8. **Sufficient history depth:** yes.
9. **Approved for full collection:** no — revise task parameters or parsing before full collection.
10. **Parameters changed after pilot:** added moderate early/late hazard profiles with fewer first-step arrivals, crossed reward, wait cost, and quit payoff over a ten-step horizon.

## Progressive Ratio

1. **Construct:** breakpoint / effort motivation; adapted from progressive-ratio reinforcement schedule (Markou et al., Schizophrenia Bulletin (2013), PMCID: PMC3849135).
2. **Manipulated variables:** ratio_schedule, reward_magnitude, effort_cost, outside_option.
3. **Persistence behavior:** WORK probability mean=0.980; sampled rate=0.986.
4. **Nondegenerate choices:** no (positive-probability SD=0.058; within-episode choice-logit SD=2.217; valid top-token rate=1.000).
5. **Basic incentive manipulation:** lower effort cost increases work; signed effect=0.054; passed.
6. **Label mappings balanced:** yes (paired correlation=0.968, mean absolute gap=0.012).
7. **Median episode length:** 28.0 decision states.
8. **Sufficient history depth:** yes.
9. **Approved for full collection:** no — revise task parameters or parsing before full collection.
10. **Parameters changed after pilot:** removed zero-cost cells and crossed materially costly work with outside options, shortened schedules while retaining moderate versus sharp effort growth.

## Sunk Cost

1. **Construct:** sunk-cost persistence; adapted from Restaurant Row / Web-Surf change-of-mind waiting (Sweis et al., Science (2018), PMCID: PMC6377599).
2. **Manipulated variables:** prior_investment, remaining_steps, reward_magnitude, outside_option, step_cost.
3. **Persistence behavior:** CONTINUE_WAITING probability mean=0.618; sampled rate=0.645.
4. **Nondegenerate choices:** yes (positive-probability SD=0.179; within-episode choice-logit SD=0.599; valid top-token rate=1.000).
5. **Basic incentive manipulation:** lower remaining cost increases continuation; signed effect=0.023; passed.
6. **Label mappings balanced:** yes (paired correlation=0.809, mean absolute gap=0.106).
7. **Median episode length:** 2.0 decision states.
8. **Sufficient history depth:** yes.
9. **Approved for full collection:** no — revise task parameters or parsing before full collection.
10. **Parameters changed after pilot:** expanded exact prospective matches across three prior-investment levels, clarified that past investment is irrecoverable and prospectively irrelevant.

## Controllability

1. **Construct:** controllability transfer / learned helplessness; adapted from yoked controllable versus uncontrollable exposure (Maier & Seligman tradition; review PMCID: PMC10205144).
2. **Manipulated variables:** exposure_type, transfer_success_probability, transfer_cost.
3. **Persistence behavior:** TRY probability mean=0.857; sampled rate=0.854.
4. **Nondegenerate choices:** yes (positive-probability SD=0.202; within-episode choice-logit SD=1.494; valid top-token rate=1.000).
5. **Basic incentive manipulation:** lower transfer cost increases trying; signed effect=0.003; did not pass.
6. **Label mappings balanced:** yes (paired correlation=0.927, mean absolute gap=0.054).
7. **Median episode length:** 1.0 decision states.
8. **Sufficient history depth:** no.
9. **Approved for full collection:** no — revise task parameters or parsing before full collection.
10. **Parameters changed after pilot:** none recorded.

## Debugging Persistence

1. **Construct:** debugging / repair persistence; adapted from repeated troubleshooting with accumulating diagnostic evidence (PRD 2.5 replacement-task design).
2. **Manipulated variables:** base_success_probability, clue_increment, attempt_cost, solution_reward, restart_value.
3. **Persistence behavior:** DEBUG probability mean=0.626; sampled rate=0.659.
4. **Nondegenerate choices:** yes (positive-probability SD=0.147; within-episode choice-logit SD=0.480; valid top-token rate=1.000).
5. **Basic incentive manipulation:** lower restart value increases debugging; signed effect=0.058; passed.
6. **Label mappings balanced:** yes (paired correlation=0.952, mean absolute gap=0.073).
7. **Median episode length:** 1.0 decision states.
8. **Sufficient history depth:** no.
9. **Approved for full collection:** no — revise task parameters or parsing before full collection.
10. **Parameters changed after pilot:** none recorded.

## Interpretation guardrail

Sunk-cost sensitivity, partial-reinforcement extinction, controllability transfer, goal gradients, and human-like recency are scientific hypotheses, not validity gates. Their absence must be reported rather than designed away.

All counterbalanced pairs share environmental seeds, semantic actions, outcomes, and histories. The independent-effort control is sequential but marks `same_goal_across_steps=false`; persistence probabilities are left null rather than fabricated.
