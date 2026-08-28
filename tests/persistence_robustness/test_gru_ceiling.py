import numpy as np
import pytest

from analysis.persistence_robustness.gru_ceiling import (
    SequenceExample,
    balanced_episode_batches,
    collate_sequences,
    make_gru,
)


def _example(task, episode, length, features=3):
    return SequenceExample(
        task,
        episode,
        np.ones((length, features), dtype=np.float32),
        np.zeros(length, dtype=np.float32),
        tuple(f"{episode}:{index}" for index in range(length)),
    )


def test_balanced_batches_have_equal_task_contribution():
    examples = [*[_example("large", f"l-{i}", 2) for i in range(9)], _example("small", "s-1", 2)]
    batches = balanced_episode_batches(examples, episodes_per_task=2, seed=4)
    for batch in batches:
        counts = {task: sum(example.task == task for example in batch) for task in {"large", "small"}}
        assert counts == {"large": 2, "small": 2}


def test_packed_gru_padding_does_not_change_valid_logits():
    torch = pytest.importorskip("torch")
    torch.manual_seed(5)
    model = make_gru(3, 4, 1).eval()
    short = _example("task", "short", 2)
    long = _example("task", "long", 5)
    single = collate_sequences([short], torch.device("cpu"))
    padded = collate_sequences([short, long], torch.device("cpu"))
    with torch.no_grad():
        alone = model(single[0], single[3])[0, :2]
        together = model(padded[0], padded[3])[0, :2]
    assert torch.allclose(alone, together, atol=1e-6)

