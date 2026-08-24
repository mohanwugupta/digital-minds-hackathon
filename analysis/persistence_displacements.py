"""Static versus transformer-depth displacement features."""

from __future__ import annotations


def displacement_features(activations):
    """Return h[l+1] - h[l] along the penultimate (layer) axis."""

    if activations.ndim < 2 or int(activations.shape[-2]) < 2:
        raise ValueError("displacement features require at least two layers")
    return activations[..., 1:, :] - activations[..., :-1, :]


def contrast_features(activation_deltas, *, layer: int, feature_type: str):
    if activation_deltas.ndim != 3:
        raise ValueError("contrast tensor must be observations x layers x width")
    if feature_type == "static":
        if not 0 <= layer < int(activation_deltas.shape[1]):
            raise IndexError("static contrast layer is out of range")
        return activation_deltas[:, layer, :]
    if feature_type == "displacement":
        transformed = displacement_features(activation_deltas)
        if not 0 <= layer < int(transformed.shape[1]):
            raise IndexError("displacement transition is out of range")
        return transformed[:, layer, :]
    raise ValueError("feature_type must be 'static' or 'displacement'")

