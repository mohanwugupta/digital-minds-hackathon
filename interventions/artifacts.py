"""Serialization for frozen probes and neuron selections."""

import torch

from .value_probe import ValueProbe


def save_frozen_probe(path: str, probe: ValueProbe, layer: int, neuron_indices, metadata: dict) -> None:
    from experiments.runtime import atomic_torch_save

    atomic_torch_save({
        "layer": int(layer),
        "input_dim": probe.hidden.in_features,
        "hidden_dim": probe.hidden.out_features,
        "state_dict": {key: value.detach().cpu() for key, value in probe.state_dict().items()},
        "neuron_indices": neuron_indices.detach().cpu().long(),
        "metadata": metadata,
    }, path)


def load_frozen_probe(path: str, map_location: str = "cpu"):
    artifact = torch.load(path, map_location=map_location, weights_only=False)
    probe = ValueProbe(artifact["input_dim"], artifact["hidden_dim"])
    probe.load_state_dict(artifact["state_dict"])
    probe.eval()
    return probe, int(artifact["layer"]), artifact["neuron_indices"].long(), artifact
