# Behavioral pilot audit and descriptive analysis

## Integrity audit

- Audit passed: **True**
- Rows: 3,706; episodes: 200
- Replayed from seeds: rewards, stopping, action sampling, histories, cumulative scores, and future returns all matched.
- Every stored conversation exactly matched the intended model-visible chat; experimenter-only state remained separate.
- All state IDs were unique and each episode had one terminal final row.

## When the model quits

- 180/200 episodes quit; 20 reached the 100-decision horizon.
- Median quit decision: **5** (IQR 4-9; 90th percentile 18).
- Mean P(STOP) was 0.229 on quit states versus 0.035 on continuing states.
- Observed STOP rose from 0.019 with no current loss streak to 0.136 after three or more consecutive losses.
- The discrete quit hazard peaked at decisions 4-5, then fell sharply among surviving episodes.

## Condition differences

| Condition | Episodes | Quit fraction | Mean decisions | Median quit decision |
|---|---:|---:|---:|---:|
| Both negative | 50 | 0.940 | 15.28 | 5.0 |
| One positive | 100 | 0.880 | 19.84 | 5.0 |
| Both positive | 50 | 0.900 | 19.16 | 7.0 |

## Interpretation cautions

- Before receiving evidence, the model assigned P(A)=0.911, P(B)=0.075, and P(STOP)=0.015. This strong A-label prior should be treated as a nuisance effect.
- The overall mean STOP probability understates the timing result: quits are concentrated early and after short loss streaks, while a selected group of persistent episodes survives to the horizon.
- State-level observations are correlated within episodes. Confirmatory uncertainty should resample or cluster by episode.
- Pilot data should remain separate from probe fitting and confirmatory intervention data.
