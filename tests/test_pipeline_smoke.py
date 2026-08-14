from experiments.run_bandit_baseline import run_episode
from experiments.run_bandit_intervention import build_matched_replays


class StopAfterOnePull:
    model_id = "mock/deterministic"

    def __init__(self):
        self.calls = 0

    def decision(self, messages):
        self.calls += 1
        if self.calls % 2:
            probabilities = (1.0, 0.0, 0.0)
        else:
            probabilities = (0.0, 0.0, 1.0)
        return {
            "logit_A": 10.0 if probabilities[0] else -10.0,
            "logit_B": -10.0,
            "logit_C": 10.0 if probabilities[2] else -10.0,
            "p_A": probabilities[0],
            "p_B": probabilities[1],
            "p_stop": probabilities[2],
            "p_continue": probabilities[0] + probabilities[1],
            "persistence_logit": 20.0 if probabilities[0] else -20.0,
        }


def tiny_pipeline():
    rows = []
    replay_rows = []
    for episode in range(2):
        records = run_episode(
            StopAfterOnePull(), 0.2, 0.65, seed=episode, action_seed=episode + 100
        )
        rows.extend(record.to_row() for record in records)
        for record in records:
            replay_rows.extend(build_matched_replays(record.state_id, record.conversation))
    return rows, [(item.state_id, item.alpha, item.context_hash) for item in replay_rows]


def test_two_episode_pipeline_is_deterministic_and_has_all_replays():
    first = tiny_pipeline()
    second = tiny_pipeline()
    assert first == second
    rows, replays = first
    assert len(rows) == 4
    assert len(replays) == 12
    assert {alpha for _, alpha, _ in replays} == {-1.0, 0.0, 1.0}
