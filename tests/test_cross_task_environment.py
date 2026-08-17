from math import isclose

from cross_task.common import counterbalanced_mappings, grouped_episode_split
from cross_task.foraging import (
    LEAVE,
    STAY,
    ForagingCondition,
    ForagingEnvironment,
    episode_conditions,
    initial_prompt,
)


def test_foraging_is_deterministic_and_depletes_after_stay():
    condition = ForagingCondition(0.8, 0.2, 2, 1)
    left = ForagingEnvironment(condition, 17, max_decisions=4)
    right = ForagingEnvironment(condition, 17, max_decisions=4)

    assert left.uniform_schedule == right.uniform_schedule
    assert isclose(left.patch_probability(), 0.8)
    first = left.step(STAY)
    assert first.reward in {-1, 3}
    assert isclose(left.patch_probability(), 0.6)
    final = left.step(LEAVE)
    assert final.reward == 2
    assert final.terminated and final.reason == "leave"


def test_foraging_pair_reverses_labels_without_changing_ecology_or_seed():
    episodes = list(episode_conditions(4, 101))
    for first, second in zip(episodes[::2], episodes[1::2]):
        pair_a, condition_a, mapping_a, seed_a, action_seed_a = first
        pair_b, condition_b, mapping_b, seed_b, action_seed_b = second
        assert (pair_a, condition_a, seed_a, action_seed_a) == (
            pair_b,
            condition_b,
            seed_b,
            action_seed_b,
        )
        assert mapping_a.positive_label != mapping_b.positive_label
        assert "STAY" in initial_prompt(condition_a, mapping_a)
        assert str(condition_a.initial_quality) not in initial_prompt(condition_a, mapping_a)


def test_grouped_split_never_separates_counterbalanced_pairs():
    episode_to_pair = {
        f"pair-{pair}-{mapping}": f"pair-{pair}"
        for pair in range(20)
        for mapping in ("x", "y")
    }
    split = grouped_episode_split(episode_to_pair, seed=3)
    membership = {
        episode: name for name, episodes in split.items() for episode in episodes
    }
    for pair in range(20):
        assert membership[f"pair-{pair}-x"] == membership[f"pair-{pair}-y"]


def test_counterbalanced_mapping_is_semantically_reversible():
    first, second = counterbalanced_mappings(STAY, LEAVE)
    assert first.semantic_for(first.label_for(STAY)) == STAY
    assert second.semantic_for(second.label_for(LEAVE)) == LEAVE
    assert first.label_for(STAY) == second.label_for(LEAVE)
