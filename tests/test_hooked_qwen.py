import pytest

torch = pytest.importorskip("torch")

from models.hooked_qwen import HookedQwen, discover_layers


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return {"A": [0], "B": [1], "C": [2]}[text]


class Output:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([
            torch.nn.Linear(3, 3, bias=False), torch.nn.Linear(3, 3, bias=False)
        ])
        self.lm_head = torch.nn.Linear(3, 3, bias=False)
        for layer in self.model.layers:
            layer.weight.data.copy_(torch.eye(3))
        self.lm_head.weight.data.copy_(torch.eye(3))

    def forward(self, input_ids=None, output_hidden_states=True, use_cache=False):
        hidden = torch.nn.functional.one_hot(input_ids, num_classes=3).float()
        states = [hidden]
        for layer in self.model.layers:
            hidden = layer(hidden)
            states.append(hidden)
        return Output(self.lm_head(hidden), tuple(states))


def test_layer_discovery_hook_changes_and_restores_logits():
    model = TinyModel()
    wrapper = HookedQwen(model, FakeTokenizer(), "tiny")
    wrapper.tokenize = lambda messages: {"input_ids": torch.tensor([[0]])}
    assert len(discover_layers(model)) == 2
    baseline = wrapper.decision([{"role": "user", "content": "x"}])
    steered = wrapper.decision(
        [{"role": "user", "content": "x"}], layer=0,
        transform=lambda hidden: hidden + torch.tensor([[0.0, 2.0, 0.0]])
    )
    restored = wrapper.decision([{"role": "user", "content": "x"}])
    assert steered["logit_B"] != baseline["logit_B"]
    assert restored["logit_B"] == baseline["logit_B"]

