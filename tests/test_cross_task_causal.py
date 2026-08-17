import pytest

torch = pytest.importorskip("torch")

from analysis.analyze_cross_task_causal import analyze
from experiments.run_cross_task_steering import evaluate_binary_replays
from interventions.ridge_probe import RidgeProbe
from interventions.ridge_steering import normalized_probe_direction
from models.hooked_qwen import binary_choice_metrics


def _probe():
    return RidgeProbe(
        weight=torch.linspace(0.2, 1.0, 8),
        state_mean=torch.zeros(8),
        state_std=torch.ones(8),
        target_mean=0.0,
        target_std=1.0,
        alpha=1.0,
        target="persistence",
    )


class BinaryReplayModel:
    def __init__(self):
        self.hidden = torch.linspace(-0.2, 0.2, 8)

    def binary_decision(
        self,
        _messages,
        labels,
        *,
        positive_label,
        layer=None,
        transform=None,
        capture_hidden_states=False,
    ):
        hidden = self.hidden.unsqueeze(0)
        if transform is not None:
            assert layer == 0
            hidden = transform(hidden)
        score = float(hidden.sum())
        negative = next(label for label in labels if label != positive_label)
        result = binary_choice_metrics(
            {positive_label: score, negative: -score},
            positive_label=positive_label,
        )
        if capture_hidden_states:
            result["hidden_states"] = [hidden.squeeze(0)]
        return result


def test_binary_causal_replay_reuses_zero_and_orders_frozen_projection():
    probe = _probe()
    model = BinaryReplayModel()
    record = {
        "state_id": "episode-1:0",
        "conversation": [{"role": "user", "content": "X or Y"}],
        "positive_label": "Y",
        "negative_label": "X",
    }
    baseline = model.binary_decision(
        record["conversation"], ("X", "Y"), positive_label="Y", capture_hidden_states=True
    )
    rows = evaluate_binary_replays(
        model,
        record=record,
        layer=0,
        delta=normalized_probe_direction(probe) * 0.2,
        probe=probe,
        baseline=baseline,
    )
    by_alpha = {row["alpha"]: row for row in rows}
    assert by_alpha[0]["choice_logit"] == baseline["choice_logit"]
    assert by_alpha[-1]["probe_value_post"] < by_alpha[0]["probe_value_post"] < by_alpha[1]["probe_value_post"]


def synthetic_causal_rows():
    rows = []
    for episode in range(12):
        mapping = "stay_x" if episode % 2 == 0 else "stay_y"
        state_id = f"episode-{episode}:0"
        for control_type, ids in (
            ("target", ("target",)),
            ("random", tuple(f"random_{index:02d}" for index in range(20))),
        ):
            for control_id in ids:
                effect = 0.1 if control_type == "target" else 0.001
                for alpha in (-1.0, 0.0, 1.0):
                    rows.append(
                        {
                            "episode_id": f"episode-{episode}",
                            "state_id": state_id,
                            "mapping_id": mapping,
                            "control_type": control_type,
                            "control_id": control_id,
                            "alpha": alpha,
                            "context_hash": f"context-{state_id}",
                            "p_positive": 0.5 + alpha * effect,
                            "p_negative": 0.5 - alpha * effect,
                            "choice_logit": alpha * effect * 4,
                            "probe_value_pre": 0.0,
                            "probe_value_post": alpha,
                            "direction_l2_norm": 1.0,
                            "intervention_relative_rms": 0.1,
                            "logit_X": alpha * effect,
                            "logit_Y": -alpha * effect,
                        }
                    )
    return rows


def test_causal_analysis_requires_all_cross_task_controls():
    result = analyze(synthetic_causal_rows(), bootstrap_samples=20)

    assert result["classification"] == "causal_transfer"
    assert all(result["criteria"].values())
    assert result["foraging"]["target"]["mean_probability_difference"] == pytest.approx(0.2)
