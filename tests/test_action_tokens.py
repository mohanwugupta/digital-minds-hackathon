import math

from models.hooked_qwen import action_metrics, verify_action_tokens


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"A": [11], "B": [12], "C": [13]}[text]


def test_action_labels_are_distinct_single_tokens():
    assert verify_action_tokens(TinyTokenizer()) == {"A": 11, "B": 12, "C": 13}


def test_action_metrics_renormalize_and_compute_persistence_logit():
    metrics = action_metrics({"A": 1.0, "B": 2.0, "C": 0.5})
    assert math.isclose(metrics["p_A"] + metrics["p_B"] + metrics["p_stop"], 1.0)
    assert math.isclose(metrics["p_continue"], metrics["p_A"] + metrics["p_B"])
    expected = math.log(math.exp(1.0) + math.exp(2.0)) - 0.5
    assert math.isclose(metrics["persistence_logit"], expected)

