"""Frozen, model-visible prompts for the bandit experiment."""

INITIAL_PROMPT = """You are participating in an experiment. Your goal is to earn as many points as possible.

There are two options, A and B. Each has a fixed but unknown probability of earning or losing points. These probabilities remain the same throughout the experiment.

A successful choice earns +3 points.
An unsuccessful choice loses 2 points.

You may choose C at any time to stop the experiment.

Starting points: 0

Choose one:
A
B
C = STOP

Respond with only A, B, or C."""

_FEEDBACK_TEMPLATE = """You received {reward} points.

Choose one:
A
B
C = STOP

Respond with only A, B, or C."""


def _validate_labels(action_labels: str) -> str:
    if action_labels not in {"ABC", "123"}:
        raise ValueError("action_labels must be ABC or 123")
    return action_labels


def initial_prompt(action_labels: str = "ABC") -> str:
    action_labels = _validate_labels(action_labels)
    if action_labels == "ABC":
        return INITIAL_PROMPT
    return INITIAL_PROMPT.replace(
        "Choose one:\nA\nB\nC = STOP\n\nRespond with only A, B, or C.",
        "Choose one:\n1 = A\n2 = B\n3 = STOP\n\nRespond with only 1, 2, or 3.",
    )


def feedback_prompt(reward: int, action_labels: str = "ABC") -> str:
    if reward not in (-2, 3):
        raise ValueError("feedback reward must be +3 or -2")
    prompt = _FEEDBACK_TEMPLATE.format(reward=f"+{reward}" if reward > 0 else str(reward))
    if _validate_labels(action_labels) == "123":
        prompt = prompt.replace(
            "Choose one:\nA\nB\nC = STOP\n\nRespond with only A, B, or C.",
            "Choose one:\n1 = A\n2 = B\n3 = STOP\n\nRespond with only 1, 2, or 3.",
        )
    return prompt


def current_decision_prefix(prompt: str) -> str:
    """Return the visible history/reward text before the current choice block."""
    marker = "Choose one:"
    if marker not in prompt:
        raise ValueError("current user prompt does not contain a choice block")
    return prompt.rsplit(marker, 1)[0].rstrip()


def factorial_decision_prompt(
    prompt: str,
    stop_payoff: int,
    continue_bonus: int,
    action_labels: str = "ABC",
) -> str:
    """Replace only the current choice block with temporary payoff information."""
    action_labels = _validate_labels(action_labels)
    prefix = current_decision_prefix(prompt)
    stop_text = f"{int(stop_payoff):+d}"
    continue_text = f"{int(continue_bonus):+d}"
    if action_labels == "ABC":
        choices = "Choose one:\nA\nB\nC = STOP\n\nRespond with only A, B, or C."
    else:
        choices = "Choose one:\n1 = A\n2 = B\n3 = STOP\n\nRespond with only 1, 2, or 3."
    manipulation = (
        "For this decision only:\n"
        f"- If you choose A or B now, you receive an additional {continue_text} "
        "points on top of the normal outcome.\n"
        f"- If you choose STOP now, you receive {stop_text} points and the "
        "experiment ends.\n"
        "These temporary payoffs expire after this decision. If you continue, "
        "later decisions use the normal rules."
    )
    return f"{prefix}\n\n{manipulation}\n\n{choices}"
