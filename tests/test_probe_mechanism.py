import json

import pytest

torch = pytest.importorskip("torch")

from analysis.probe_mechanism import (
    nested_probe_regression,
    run_probe_mechanism_analysis,
)
from interventions.value_probe import ValueProbe


def _synthetic_rows(probe_is_integrated: bool):
    generator = torch.Generator().manual_seed(41)
    rows = []
    for episode in range(40):
        history_value = torch.randn((), generator=generator).item()
        for round_index in range(6):
            previous_outcome = -2 if round_index % 2 else 3
            loss_streak = 1 if previous_outcome == -2 else 0
            if probe_is_integrated:
                probe_value = history_value + 0.05 * torch.randn(
                    (), generator=generator
                ).item()
            else:
                probe_value = float(previous_outcome)
            persistence = (
                0.8 * previous_outcome
                - 0.15 * loss_streak
                - 0.03 * round_index
                + 2.0 * history_value
                + 0.05 * torch.randn((), generator=generator).item()
            )
            rows.append(
                {
                    "episode_id": f"episode-{episode}",
                    "round": round_index,
                    "previous_outcome": previous_outcome,
                    "loss_streak": loss_streak,
                    "cumulative_score": history_value,
                    "persistence_logit": persistence,
                    "probe_value": probe_value,
                }
            )
    return rows


def test_integrated_probe_adds_persistence_signal_beyond_recent_history():
    result = nested_probe_regression(_synthetic_rows(probe_is_integrated=True))

    # The deliberately large immediate-outcome term dilutes the standardized
    # coefficient and incremental fit; the partial-correlation assertion below
    # is the strongest construct-recovery check.
    assert result["probe_standardized_beta"] > 0.4
    assert result["delta_r_squared"] > 0.2
    assert result["partial_correlation"] > 0.8
    assert result["episode_clusters"] == 40


def test_latest_reward_only_probe_adds_no_signal_when_reward_is_controlled():
    result = nested_probe_regression(_synthetic_rows(probe_is_integrated=False))

    assert abs(result["probe_standardized_beta"]) < 0.05
    assert result["delta_r_squared"] < 1e-6


def test_stronger_score_control_can_explain_synthetic_history_probe():
    result = nested_probe_regression(
        _synthetic_rows(probe_is_integrated=True),
        include_cumulative_score=True,
    )

    assert result["delta_r_squared"] < 0.01


def test_probe_mechanism_pipeline_reads_shards_and_writes_artifacts(tmp_path):
    generator = torch.Generator().manual_seed(73)
    shards = []
    episode_ids = []
    for episode in range(20):
        episode_id = f"episode-{episode}"
        episode_ids.append(episode_id)
        integrated_history = (episode - 9.5) / 4
        records = []
        activations = []
        reward_history = []
        cumulative_score = 0
        for round_index in range(6):
            previous_outcome = reward_history[-1] if reward_history else None
            activation_value = integrated_history + 0.04 * round_index
            persistence_logit = (
                1.15 * integrated_history
                + (0 if previous_outcome is None else 0.18 * previous_outcome)
                - 0.03 * round_index
                + 0.02 * torch.randn((), generator=generator).item()
            )
            records.append(
                {
                    "episode_id": episode_id,
                    "state_id": f"{episode_id}:{round_index}",
                    "round": round_index,
                    "reward_history": json.dumps(reward_history),
                    "previous_outcome": previous_outcome,
                    "cumulative_score": cumulative_score,
                    "persistence_logit": persistence_logit,
                    "p_stop": 1 / (1 + torch.exp(torch.tensor(persistence_logit)).item()),
                    "sampled_action": "A",
                }
            )
            activations.append([[activation_value, 0.0]])
            outcome = -2 if round_index % 2 == 0 else 3
            reward_history.append(outcome)
            cumulative_score += outcome
        shards.append(
            {
                "episode_id": episode_id,
                "records": records,
                "activations": torch.tensor(activations, dtype=torch.float32),
            }
        )

    probe = ValueProbe(2, hidden_dim=1)
    with torch.no_grad():
        probe.hidden.weight.copy_(torch.tensor([[1.0, 0.0]]))
        probe.hidden.bias.fill_(10.0)
        probe.output.weight.fill_(1.0)
        probe.output.bias.fill_(-10.0)

    result = run_probe_mechanism_analysis(
        shards,
        probe,
        layer=0,
        neuron_indices=torch.tensor([0]),
        test_episode_ids=episode_ids,
        output_dir=tmp_path,
    )

    assert result["states"] == 120
    assert result["episodes"] == 20
    assert result["primary_pruned_probe"]["delta_r_squared"] > 0.1
    for filename in (
        "probe_mechanism.json",
        "probe_mechanism_report.md",
        "probe_mechanism_test_states.csv",
        "probe_mechanism.svg",
    ):
        assert (tmp_path / filename).is_file()
