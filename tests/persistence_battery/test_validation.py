import pandas as pd

from experiments.persistence_battery.validation import nondegeneracy_checks


def _config():
    return {
        "validation": {
            "persistence_probability_bounds": [0.05, 0.95],
            "minimum_persistence_logit_sd": 0.15,
            "minimum_median_episode_length": 2,
            "minimum_top_token_action_rate": 0.90,
        }
    }


def _records():
    return pd.DataFrame(
        {
            "episode_id": ["e1", "e1", "e2", "e2"],
            "pair_id": ["p1", "p1", "p2", "p2"],
            "semantic_action": ["WAIT", "QUIT", "WAIT", "QUIT"],
            "p_positive_semantic": [0.35, 0.65, 0.40, 0.60],
            "choice_logit": [-0.6, 0.6, -0.4, 0.4],
            "top_token_is_action": [True, True, True, True],
        }
    )


def _label_bias():
    return pd.DataFrame([{"task": "voluntary_waiting", "passed": True}])


def test_scientific_hypothesis_is_not_a_pilot_gate():
    manipulations = pd.DataFrame(
        [
            {
                "task": "voluntary_waiting",
                "gate_role": "validity_gate",
                "passed": True,
            },
            {
                "task": "voluntary_waiting",
                "gate_role": "design_gate",
                "passed": True,
            },
            {
                "task": "voluntary_waiting",
                "gate_role": "scientific_hypothesis",
                "passed": False,
            },
        ]
    )
    result = nondegeneracy_checks(
        {"voluntary_waiting": _records()},
        manipulations,
        _label_bias(),
        _config(),
        model_free=False,
    )
    assert bool(result.iloc[0].approved_for_full_collection)


def test_failed_design_check_and_model_free_mode_each_block_approval():
    manipulations = pd.DataFrame(
        [
            {
                "task": "voluntary_waiting",
                "gate_role": "validity_gate",
                "passed": True,
            },
            {
                "task": "voluntary_waiting",
                "gate_role": "design_gate",
                "passed": False,
            },
        ]
    )
    failed_design = nondegeneracy_checks(
        {"voluntary_waiting": _records()},
        manipulations,
        _label_bias(),
        _config(),
        model_free=False,
    )
    assert not bool(failed_design.iloc[0].approved_for_full_collection)

    manipulations.loc[manipulations.gate_role == "design_gate", "passed"] = True
    model_free = nondegeneracy_checks(
        {"voluntary_waiting": _records()},
        manipulations,
        _label_bias(),
        _config(),
        model_free=True,
    )
    assert not bool(model_free.iloc[0].approved_for_full_collection)
    assert model_free.iloc[0].approval_note == "model-free smoke is non-scientific"
