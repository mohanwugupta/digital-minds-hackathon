import numpy as np
import pandas as pd
from types import SimpleNamespace

from analysis.persistence_convergence.intervention_profiles import (
    functional_effect,
    profile_summary,
)
from analysis.persistence_convergence.run_intervention_profiles import (
    run_intervention_profiles,
)


def test_zero_manipulation_and_orientation_reversal():
    readout = np.array([1.0, -2.0])
    state = np.array([3.0, 4.0])
    assert functional_effect(readout, state, state) == 0.0
    forward = functional_effect(readout, np.array([2.0, 1.0]), state)
    reverse = functional_effect(readout, state, np.array([2.0, 1.0]))
    assert reverse == -forward


def test_common_late_functional_stage_recovery():
    profiles = {
        "bandit": np.array([0.0, 0.1, 0.3, 1.0, 1.2]),
        "foraging": np.array([-0.1, 0.0, 0.4, 0.9, 1.1]),
        "solvability": np.array([0.1, -0.1, 0.2, 1.1, 1.3]),
    }
    summary = profile_summary(profiles, onset_fraction=0.5)
    assert summary["mean_profile_correlation"] > 0.9
    assert set(summary["onset_layers"].values()) == {3}


def test_each_intervention_uses_its_own_task_readout():
    class IdentityProjector:
        def transform(self, values):
            return np.asarray(values)

    def dataset(task, positive):
        values = np.asarray([[0.0, 0.0], positive], dtype=np.float16)[:, None, :]
        return SimpleNamespace(
            metadata=pd.DataFrame(
                {"state_id": [f"{task}-negative", f"{task}-positive"]}
            ),
            shape=values.shape,
            open=lambda: values,
        )

    inventory = pd.DataFrame(
        [
            {
                "task": task,
                "contrast_kind": "persistence",
                "manipulation": "synthetic",
                "split": "test",
                "cluster_id": f"{task}-episode",
                "positive_state_id": f"{task}-positive",
                "negative_state_id": f"{task}-negative",
            }
            for task in ("bandit", "foraging")
        ]
    )

    def model(direction):
        return {
            "normalizer_mean": np.zeros(2),
            "normalizer_scale": np.ones(2),
            "coefficient": np.asarray([0.0, *direction]),
        }

    readouts = {
        "models": {
            0: {
                "bandit": model([1.0, 0.0]),
                "foraging": model([0.0, 1.0]),
            }
        },
        "projector": IdentityProjector(),
    }
    profiles, _summary = run_intervention_profiles(
        inventory,
        {
            "bandit": dataset("bandit", [2.0, 0.0]),
            "foraging": dataset("foraging", [0.0, 3.0]),
        },
        readouts,
        {"onset_fraction": 0.5},
    )
    observed = profiles.set_index("task").mean_functional_effect.to_dict()
    assert observed == {"bandit": 2.0, "foraging": 3.0}
    assert (profiles.task == profiles.readout_task).all()
