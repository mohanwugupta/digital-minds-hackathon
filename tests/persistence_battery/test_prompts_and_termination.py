import yaml
import pytest

from cross_task.common import counterbalanced_mappings
from experiments.persistence_battery.registry import TASKS


def _config():
    with open("config/persistence_battery.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.mark.parametrize("task", list(TASKS))
def test_prompts_avoid_explicit_psychological_meta_language(task):
    config = _config()
    definition = TASKS[task]
    condition = definition.conditions(config["tasks"][task])[0]
    environment = definition.environment(condition, 7, config["tasks"][task])
    mapping = counterbalanced_mappings(
        definition.positive_action, definition.negative_action
    )[0]
    prompt = environment.initial_prompt(mapping).lower()
    assert "persistence experiment" not in prompt
    assert "measuring motivation" not in prompt
    assert "whether you are persistent" not in prompt


@pytest.mark.parametrize("task", list(TASKS))
def test_all_environments_reject_states_after_termination(task):
    config = _config()
    definition = TASKS[task]
    condition = definition.conditions(config["tasks"][task])[0]
    environment = definition.environment(condition, 7, config["tasks"][task])
    if definition.persistence:
        environment.step(definition.negative_action)
    else:
        while not environment.terminated:
            environment.step(definition.negative_action)
    with pytest.raises(RuntimeError, match="terminated"):
        environment.step(definition.negative_action)
