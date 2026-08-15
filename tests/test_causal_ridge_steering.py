import pytest

torch = pytest.importorskip("torch")

from interventions.ridge_probe import RidgeProbe
from interventions.ridge_steering import (
    calibrate_ridge_magnitude,
    matched_sign_random_directions,
    normalized_probe_direction,
)
from experiments.run_causal_steering import evaluate_direction_replays


def probe() -> RidgeProbe:
    weights = torch.linspace(0.2, 1.0, 32)
    weights[1::2] *= -1
    return RidgeProbe(
        weight=weights,
        state_mean=torch.zeros(32),
        state_std=torch.full((32,), 2.0),
        target_mean=0.0,
        target_std=3.0,
        alpha=1.0,
        target="persistence",
    )


def test_normalized_direction_increases_probe_and_calibration_targets_one_sd():
    ridge = probe()
    states = torch.zeros(12, 32)
    direction = normalized_probe_direction(ridge)
    calibration = calibrate_ridge_magnitude(
        ridge, states, decoded_sd_candidates=(1.0, 0.5, 0.25)
    )
    delta = direction * calibration.magnitude

    assert torch.allclose(direction.norm(), torch.tensor(1.0))
    assert torch.all(ridge.predict(states + delta) > ridge.predict(states))
    assert calibration.decoded_sd_shift == pytest.approx(1.0)
    assert calibration.relative_rms <= 0.25


def test_random_controls_exactly_match_norm_and_activation_rms():
    ridge = probe()
    delta = normalized_probe_direction(ridge) * 0.2
    controls = matched_sign_random_directions(delta, n_directions=20, seed=9)

    assert len(controls) == 20
    assert all(torch.allclose(item.norm(), delta.norm()) for item in controls)
    reference_rms = ((delta / ridge.state_std) ** 2).mean().sqrt()
    assert all(
        torch.allclose(((item / ridge.state_std) ** 2).mean().sqrt(), reference_rms)
        for item in controls
    )
    assert all(not torch.equal(item, delta) for item in controls)


class TinyReplayModel:
    def __init__(self):
        self.hidden = torch.linspace(-0.75, 0.75, 32)

    def decision(
        self,
        _conversation,
        layer=None,
        transform=None,
        capture_hidden_states=False,
    ):
        hidden = self.hidden.unsqueeze(0)
        if transform is not None:
            assert layer == 0
            hidden = transform(hidden)
        score = float(hidden.sum())
        result = {
            "logit_A": score,
            "logit_B": score - 1,
            "logit_C": -score,
            "persistence_logit": 2 * score,
            "p_continue": 0.75,
        }
        if capture_hidden_states:
            result["hidden_states"] = [hidden.squeeze(0)]
        return result


def test_alpha_zero_exactly_reuses_baseline_and_bidirectional_probe_ordering():
    ridge = probe()
    model = TinyReplayModel()
    conversation = [{"role": "user", "content": "Choose one: A B C"}]
    baseline = model.decision(conversation, capture_hidden_states=True)
    delta = normalized_probe_direction(ridge) * 0.2
    rows = evaluate_direction_replays(
        model,
        state_id="state-1",
        conversation=conversation,
        layer=0,
        delta=delta,
        probe=ridge,
        baseline=baseline,
    )

    control = next(row for row in rows if row["alpha"] == 0)
    assert control["logit_A"] == baseline["logit_A"]
    assert control["logit_B"] == baseline["logit_B"]
    assert control["logit_C"] == baseline["logit_C"]
    assert len({row["context_hash"] for row in rows}) == 1
    by_alpha = {row["alpha"]: row["probe_value_post"] for row in rows}
    assert by_alpha[1.0] > by_alpha[0.0] > by_alpha[-1.0]
