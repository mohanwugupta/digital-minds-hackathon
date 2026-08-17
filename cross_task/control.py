"""Counterbalanced non-persistence binary comparison control."""

import random

from .common import LabelMapping, counterbalanced_mappings, stable_balanced_pairs


LEFT_GREATER = "LEFT_GREATER"
RIGHT_GREATER = "RIGHT_GREATER"


def comparison_prompt(left: int, right: int, mapping: LabelMapping) -> str:
    return (
        "You are participating in a number-comparison experiment. Decide which "
        "displayed integer is larger. This is a single classification judgment; "
        "it does not start or end another activity.\n\n"
        f"Left integer: {left}\n"
        f"Right integer: {right}\n\n"
        "Choose one:\n"
        f"{mapping.label_for(LEFT_GREATER)} = the LEFT integer is larger\n"
        f"{mapping.label_for(RIGHT_GREATER)} = the RIGHT integer is larger\n\n"
        f"Respond with only {mapping.labels[0]} or {mapping.labels[1]}."
    )


def episode_conditions(n_episodes: int, base_seed: int):
    # The placeholder cells are shuffled by the shared pairing helper; numeric
    # stimuli themselves are generated from each pair seed for exact reversal.
    pairs = stable_balanced_pairs(range(max(1, n_episodes // 2)), n_episodes, base_seed)
    mappings = counterbalanced_mappings(LEFT_GREATER, RIGHT_GREATER)
    for pair_index, _cell in pairs:
        pair_seed = base_seed + pair_index
        rng = random.Random(pair_seed * 2 + 113)
        left, right = rng.randint(-999, 999), rng.randint(-999, 999)
        while left == right:
            right = rng.randint(-999, 999)
        pair_id = f"control-pair-{pair_seed:07d}"
        for mapping in mappings:
            yield pair_id, left, right, mapping, pair_seed
