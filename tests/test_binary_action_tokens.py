import math

from models.hooked_qwen import (
    binary_choice_metrics,
    verify_chat_choice_tokens,
    verify_choice_tokens,
)


class BinaryTokenizer:
    def apply_chat_template(self, _messages, **kwargs):
        assert kwargs["add_generation_prompt"]
        return "rendered>"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return {
            "X": [11],
            "Y": [12],
            "rendered>": [1, 2],
            "rendered>X": [1, 2, 11],
            "rendered>Y": [1, 2, 12],
        }[text]


def test_binary_labels_are_validated_under_exact_chat_prompt():
    tokenizer = BinaryTokenizer()
    messages = [{"role": "user", "content": "choose"}]
    assert verify_choice_tokens(tokenizer, ("X", "Y")) == {"X": 11, "Y": 12}
    assert verify_chat_choice_tokens(tokenizer, messages, ("X", "Y")) == {
        "X": 11,
        "Y": 12,
    }


def test_binary_metrics_follow_semantics_after_label_reversal():
    first = binary_choice_metrics({"X": 2.0, "Y": 0.0}, positive_label="X")
    reversed_mapping = binary_choice_metrics(
        {"X": 2.0, "Y": 0.0}, positive_label="Y"
    )

    assert first["choice_logit"] == 2.0
    assert reversed_mapping["choice_logit"] == -2.0
    assert math.isclose(first["p_positive"], reversed_mapping["p_negative"])
