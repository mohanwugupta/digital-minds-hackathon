from bandit.conversation import BanditConversation
from experiments.run_bandit_intervention import build_matched_replays


def test_replay_conditions_are_byte_identical_except_alpha():
    conversation = BanditConversation.start().snapshot()
    replays = build_matched_replays("state-1", conversation)
    assert [item.alpha for item in replays] == [-1.0, 0.0, 1.0]
    assert len({item.context_bytes for item in replays}) == 1
    assert len({item.context_hash for item in replays}) == 1

