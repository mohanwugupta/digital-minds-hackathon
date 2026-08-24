"""Counterbalanced one-shot relative-value control with no ongoing goal."""

from __future__ import annotations

import random

from .common import LabelMapping, counterbalanced_mappings, stable_balanced_pairs


LEFT_VOUCHER = "LEFT_VOUCHER"
RIGHT_VOUCHER = "RIGHT_VOUCHER"


def voucher_prompt(left_value: int, right_value: int, mapping: LabelMapping) -> str:
    return (
        "You are making a single one-shot choice between two independent "
        "vouchers. Choose the voucher worth more points. This choice is not "
        "part of an ongoing activity and has no later consequences.\n\n"
        f"Left voucher value: {int(left_value):+d} points\n"
        f"Right voucher value: {int(right_value):+d} points\n\n"
        "Choose one:\n"
        f"{mapping.label_for(LEFT_VOUCHER)} = take the LEFT voucher\n"
        f"{mapping.label_for(RIGHT_VOUCHER)} = take the RIGHT voucher\n\n"
        f"Respond with only {mapping.labels[0]} or {mapping.labels[1]}."
    )


def episode_conditions(
    n_episodes: int,
    base_seed: int,
    *,
    labels=("M", "N"),
):
    cells = ((-8, 3), (-3, 8), (1, 5), (5, 1), (8, -3), (3, -8))
    pairs = stable_balanced_pairs(cells, n_episodes, base_seed)
    mappings = counterbalanced_mappings(
        LEFT_VOUCHER, RIGHT_VOUCHER, tuple(labels)
    )
    for pair_index, base_values in pairs:
        pair_seed = int(base_seed) + pair_index
        rng = random.Random(pair_seed * 2 + 947)
        offset = rng.randint(-20, 20)
        left, right = int(base_values[0]) + offset, int(base_values[1]) + offset
        if left == right:
            raise RuntimeError("generic-value construction produced a tied choice")
        pair_id = f"generic-value-pair-{pair_seed:07d}"
        for mapping in mappings:
            yield pair_id, left, right, mapping, pair_seed
