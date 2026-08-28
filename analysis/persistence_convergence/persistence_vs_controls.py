"""Honest comparisons with sequential and one-shot generic decision controls."""

from __future__ import annotations

import pandas as pd


def sequence_support(records):
    lengths = pd.Series([row["episode_id"] for row in records]).value_counts()
    if len(lengths) and int(lengths.max()) == 1:
        return {
            "history": False,
            "recurrence": False,
            "reason": "one_shot_control",
        }
    return {"history": True, "recurrence": True, "reason": "sequential"}


def compare_persistence_controls(persistence, controls):
    reference = (
        persistence.groupby("layer", as_index=False).test_r_squared.mean()
        .rename(columns={"test_r_squared": "persistence_test_r_squared"})
    )
    result = controls.merge(reference, on="layer", validate="many_to_one")
    result = result.rename(columns={"test_r_squared": "control_test_r_squared"})
    result["delta_r_squared_persistence_minus_control"] = (
        result.persistence_test_r_squared - result.control_test_r_squared
    )
    return result

