from bandit.prompts import feedback_prompt, initial_prompt


def test_initial_prompt_is_frozen_and_hides_probabilities():
    prompt = initial_prompt()
    assert "Starting points: 0" in prompt
    assert "A\nB\nC = STOP" in prompt
    assert "fixed but unknown probability" in prompt
    assert "0.20" not in prompt and "0.65" not in prompt


def test_feedback_contains_only_immediate_result_and_choices():
    success = feedback_prompt(3)
    failure = feedback_prompt(-2)
    assert success == (
        "You received +3 points.\n\nChoose one:\nA\nB\nC = STOP\n\n"
        "Respond with only A, B, or C."
    )
    assert failure.startswith("You received -2 points.")
    forbidden = ("cumulative", "round", "history", "probability", "previous")
    assert not any(term in success.lower() for term in forbidden)
    assert not any(term in failure.lower() for term in forbidden)


def test_numeric_single_token_fallback_is_rendered_consistently():
    prompt = initial_prompt("123")
    feedback = feedback_prompt(-2, "123")
    assert "1 = A\n2 = B\n3 = STOP" in prompt
    assert "Respond with only 1, 2, or 3." in feedback
