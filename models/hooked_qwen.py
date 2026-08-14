"""Model-family-tolerant residual-stream extraction and intervention."""

from collections.abc import Mapping as MappingABC
from contextlib import contextmanager
import math
from typing import Callable, Dict, Iterable, List, Mapping, Optional


PRIMARY_MODEL = "Qwen/Qwen3.5-4B"
FALLBACK_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ACTION_LABELS = ("A", "B", "C")


def verify_action_tokens(tokenizer, labels: Iterable[str] = ACTION_LABELS) -> Dict[str, int]:
    token_ids: Dict[str, int] = {}
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"action {label!r} is not a single token: {encoded}")
        token_ids[label] = int(encoded[0])
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError("action labels do not map to distinct tokens")
    if len(token_ids) != 3:
        raise ValueError("exactly three action labels are required")
    return token_ids


def verify_chat_action_tokens(
    tokenizer, messages: List[dict], labels: Iterable[str] = ACTION_LABELS
) -> Dict[str, int]:
    """Verify labels as continuations of the actual generation prompt."""
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        prompt = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **template_kwargs
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    token_ids = {}
    for label in labels:
        completed = tokenizer.encode(prompt + label, add_special_tokens=False)
        if completed[: len(prefix)] != prefix or len(completed) != len(prefix) + 1:
            raise ValueError(
                f"action {label!r} is not a clean single-token completion under the chat template"
            )
        token_ids[label] = int(completed[-1])
    if len(set(token_ids.values())) != 3:
        raise ValueError("chat action labels do not map to distinct tokens")
    return token_ids


def action_metrics(logits: Mapping[str, float]) -> Dict[str, float]:
    if set(logits) != set(ACTION_LABELS):
        raise ValueError("logits must contain exactly A, B, and C")
    maximum = max(float(value) for value in logits.values())
    exponentials = {key: math.exp(float(value) - maximum) for key, value in logits.items()}
    denominator = sum(exponentials.values())
    probabilities = {key: value / denominator for key, value in exponentials.items()}
    continue_lse = max(float(logits["A"]), float(logits["B"]))
    continue_lse += math.log(
        math.exp(float(logits["A"]) - continue_lse)
        + math.exp(float(logits["B"]) - continue_lse)
    )
    return {
        "logit_A": float(logits["A"]),
        "logit_B": float(logits["B"]),
        "logit_C": float(logits["C"]),
        "p_A": probabilities["A"],
        "p_B": probabilities["B"],
        "p_stop": probabilities["C"],
        "p_continue": probabilities["A"] + probabilities["B"],
        "persistence_logit": continue_lse - float(logits["C"]),
    }


def _get_attr_path(root, path: str):
    value = root
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def discover_layers(model) -> List[object]:
    """Find decoder layers across Qwen causal-LM wrapper variants."""
    paths = (
        "model.language_model.layers",  # Qwen3.5 conditional generation
        "model.layers",                 # Qwen3 / common HF causal LM
        "language_model.layers",
        "transformer.h",
    )
    for path in paths:
        layers = _get_attr_path(model, path)
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return list(layers)
    raise RuntimeError("could not discover language-model layers")


def _replace_hidden(output, hidden):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


class HookedQwen:
    def __init__(
        self, model, tokenizer, model_id: str = PRIMARY_MODEL, action_labels: str = "auto"
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.layers = discover_layers(model)
        candidates = ("ABC", "123") if action_labels == "auto" else (action_labels,)
        last_error = None
        for candidate in candidates:
            try:
                display_ids = verify_action_tokens(tokenizer, candidate)
                if hasattr(tokenizer, "apply_chat_template"):
                    from bandit.conversation import BanditConversation

                    messages = BanditConversation.start(candidate).snapshot()
                    display_ids = verify_chat_action_tokens(tokenizer, messages, candidate)
                self.action_labels = candidate
                self.action_token_ids = dict(zip("ABC", display_ids.values()))
                break
            except ValueError as error:
                last_error = error
        else:
            raise ValueError(f"no valid action vocabulary (tried {candidates}): {last_error}")
        self._chat_actions_verified = hasattr(tokenizer, "apply_chat_template")

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = PRIMARY_MODEL,
        *,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        revision: Optional[str] = None,
        local_files_only: bool = True,
    ) -> "HookedQwen":
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype)
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        # The published 4B checkpoint has a full Qwen3.5 multimodal config even
        # for text-only use. Loading that config through AutoModelForCausalLM is
        # not reliable; the official conditional-generation class accepts plain
        # input_ids while exposing the nested text backbone we instrument.
        if getattr(config, "model_type", None) == "qwen3_5" and hasattr(config, "text_config"):
            try:
                from transformers import Qwen3_5ForConditionalGeneration

                model_class = Qwen3_5ForConditionalGeneration
            except ImportError:
                from transformers import AutoModelForMultimodalLM

                model_class = AutoModelForMultimodalLM
        else:
            model_class = AutoModelForCausalLM
        model = model_class.from_pretrained(
            model_name_or_path,
            config=config,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=True,
            dtype=torch_dtype,
            device_map=device_map,
        )
        model.eval()
        return cls(model, tokenizer, model_name_or_path)

    def tokenize(self, messages: List[dict]):
        kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}
        try:
            token_ids = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            token_ids = self.tokenizer.apply_chat_template(messages, **kwargs)
        # Recent Transformers versions return BatchEncoding here. It behaves
        # like a mapping but is not a dict, so wrapping it as input_ids creates
        # a nested BatchEncoding that neither torch.equal nor the model accepts.
        if isinstance(token_ids, MappingABC):
            return dict(token_ids)
        return {"input_ids": token_ids}

    def _to_model_device(self, inputs: dict) -> dict:
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in inputs.items()}

    @contextmanager
    def intervention(self, layer: int, transform: Callable):
        if not 0 <= layer < len(self.layers):
            raise IndexError(f"layer {layer} outside [0, {len(self.layers)})")

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            changed = hidden.clone()
            changed[:, -1, :] = transform(changed[:, -1, :])
            return _replace_hidden(output, changed)

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def capture_final_states(self):
        """Capture one vector per layer without retaining sequence-wide states."""
        captured = [None] * len(self.layers)
        handles = []
        for index, module in enumerate(self.layers):
            def capture(_module, _inputs, output, position=index):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                # Clone only the final vector on-device. A view would retain the
                # full sequence tensor; copying each layer to CPU here would
                # force 32 separate device synchronizations on Qwen3.5.
                captured[position] = hidden[0, -1].detach().clone()

            handles.append(module.register_forward_hook(capture))
        try:
            yield captured
        finally:
            for handle in handles:
                handle.remove()

    def forward(
        self,
        messages: List[dict],
        layer: Optional[int] = None,
        transform=None,
        *,
        capture_hidden_states: bool = False,
    ):
        import torch

        inputs = self._to_model_device(self.tokenize(messages))
        context = self.intervention(layer, transform) if transform is not None else _nullcontext()
        with torch.inference_mode(), context:
            if capture_hidden_states:
                # Hooks retain only one final-position vector per layer.
                with self.capture_final_states() as hidden_states:
                    outputs = self.model(
                        **inputs, output_hidden_states=False, use_cache=False
                    )
                outputs.hidden_states = tuple(
                    state.float().cpu() for state in hidden_states
                )
            else:
                outputs = self.model(**inputs, output_hidden_states=False, use_cache=False)
                # Some test doubles or model wrappers may populate this despite
                # output_hidden_states=False. Keep the public contract explicit.
                outputs.hidden_states = None
        return outputs

    def decision(
        self,
        messages: List[dict],
        layer: Optional[int] = None,
        transform=None,
        *,
        capture_hidden_states: bool = False,
    ) -> dict:
        import torch

        if not self._chat_actions_verified and hasattr(self.tokenizer, "apply_chat_template"):
            display_ids = verify_chat_action_tokens(
                self.tokenizer, messages, self.action_labels
            )
            self.action_token_ids = dict(zip("ABC", display_ids.values()))
            self._chat_actions_verified = True
        outputs = self.forward(
            messages,
            layer=layer,
            transform=transform,
            capture_hidden_states=capture_hidden_states,
        )
        last_logits = outputs.logits[0, -1]
        selected = {
            label: float(last_logits[token_id].detach().float().cpu())
            for label, token_id in self.action_token_ids.items()
        }
        result = action_metrics(selected)
        action_ids = torch.tensor(
            list(self.action_token_ids.values()), device=last_logits.device, dtype=torch.long
        )
        result["p_action_mass_raw"] = float(
            torch.exp(torch.logsumexp(last_logits[action_ids].float(), dim=0)
                      - torch.logsumexp(last_logits.float(), dim=0)).cpu()
        )
        result["top_token_is_action"] = int(last_logits.argmax()) in self.action_token_ids.values()
        if getattr(outputs, "hidden_states", None) is not None:
            result["hidden_states"] = [
                state for state in outputs.hidden_states
            ]
        return result


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False
