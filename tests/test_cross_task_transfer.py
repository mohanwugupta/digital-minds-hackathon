import pytest

torch = pytest.importorskip("torch")

from analysis.analyze_cross_task_transfer import (
    affine_fit,
    direction_score,
    evaluate_transfer,
)
from interventions.ridge_probe import RidgeProbe


def _probe(width=16):
    weight = torch.linspace(0.2, 1.0, width)
    return RidgeProbe(
        weight=weight,
        state_mean=torch.zeros(width),
        state_std=torch.ones(width),
        target_mean=0.0,
        target_std=1.0,
        alpha=1.0,
        target="persistence",
    )


def _shard(task, index, state, target, mapping):
    episode = f"{task}-{index}"
    key = "persistence_logit" if task == "foraging" else "choice_logit"
    return {
        "task": task if task == "foraging" else "binary_control",
        "episode_id": episode,
        "pair_id": f"pair-{index // 2}",
        "mapping_id": mapping,
        "records": [
            {
                "episode_id": episode,
                "pair_id": f"pair-{index // 2}",
                "state_id": f"{episode}:0",
                "mapping_id": mapping,
                key: float(target),
            }
        ],
        "activations": state.reshape(1, 1, -1),
    }


def test_strict_transfer_reaches_ceiling_and_rejects_binary_control():
    probe = _probe()
    weight = probe.raw_activation_direction()
    norm_squared = float(weight.square().sum())
    foraging, control = [], []
    for index in range(40):
        target = float((index // 2) % 5 - 2)
        mapping = "stay_x" if index % 2 == 0 else "stay_y"
        state = target * weight / norm_squared
        basis = torch.zeros_like(weight)
        basis[index % len(weight)] = 1.0
        orthogonal_noise = basis - torch.dot(basis, weight) * weight / norm_squared
        state = state + (0.25 + 0.03 * (index % 3)) * orthogonal_noise
        foraging.append(_shard("foraging", index, state, target, mapping))

        control_projection = 1.0 if (index // 2) % 2 == 0 else -1.0
        control_target = 1.0 if (index // 4) % 2 == 0 else -1.0
        control_state = control_projection * weight / norm_squared
        control_mapping = "left_greater_x" if index % 2 == 0 else "left_greater_y"
        control.append(
            _shard("control", index, control_state, control_target, control_mapping)
        )
    split = {
        "train": [f"foraging-{index}" for index in range(20)],
        "validation": [f"foraging-{index}" for index in range(20, 30)],
        "test": [f"foraging-{index}" for index in range(30, 40)],
    }
    control_split = {
        "train": [f"control-{index}" for index in range(20)],
        "validation": [f"control-{index}" for index in range(20, 30)],
        "test": [f"control-{index}" for index in range(30, 40)],
    }
    result = evaluate_transfer(
        foraging_shards=foraging,
        foraging_split=split,
        control_shards=control,
        control_split=control_split,
        bandit_probe=probe,
        bandit_layer=0,
        foraging_probe=probe,
        foraging_layer=0,
        random_directions=20,
        random_seed=7,
        bootstrap_samples=20,
        thresholds={
            "negative_control_max_absolute_correlation": 0.1,
            "negative_control_relative_correlation_fraction": 0.5,
            "strong_transfer_ceiling_fraction": 0.5,
        },
    )

    assert result["strict_zero_shot"]["r_squared"] == pytest.approx(1.0)
    assert result["foraging_specific_ceiling"]["r_squared"] == pytest.approx(1.0)
    assert result["classification"] == "strong_transfer"


def test_affine_diagnostic_and_direction_score_preserve_frozen_axis():
    probe = _probe(4)
    states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    assert torch.allclose(direction_score(states, probe), probe.predict(states))
    calibration = affine_fit([0, 1, 2], [2, 4, 6])
    assert calibration == pytest.approx({"intercept": 2.0, "slope": 2.0})
