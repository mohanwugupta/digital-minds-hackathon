"""Hard-gate smoke test for Qwen text inference and residual hooks."""

import argparse
import json
import os

from bandit.conversation import BanditConversation


def run_checks(wrapper) -> dict:
    import torch

    messages = BanditConversation.start(wrapper.action_labels).snapshot()
    first_tokens = wrapper.tokenize(messages)["input_ids"]
    second_tokens = wrapper.tokenize(messages)["input_ids"]
    deterministic = torch.equal(first_tokens, second_tokens)
    baseline = wrapper.decision(messages, capture_hidden_states=True)
    layer = len(wrapper.layers) // 2
    captured = {}

    def perturb(hidden):
        captured["before"] = hidden.detach().clone()
        changed = hidden.clone()
        changed[..., 0] += 1.0
        captured["after"] = changed.detach().clone()
        return changed

    changed = wrapper.decision(messages, layer=layer, transform=perturb)
    restored = wrapper.decision(messages)
    baseline_logits = torch.tensor([baseline[f"logit_{x}"] for x in "ABC"])
    changed_logits = torch.tensor([changed[f"logit_{x}"] for x in "ABC"])
    restored_logits = torch.tensor([restored[f"logit_{x}"] for x in "ABC"])
    checks = {
        "text_only_executes": all(key in baseline for key in ("logit_A", "logit_B", "logit_C")),
        "chat_template_deterministic": deterministic,
        "layers_discovered": len(wrapper.layers) > 0,
        "all_layer_hidden_states_extracted": len(baseline.get("hidden_states", [])) == len(wrapper.layers),
        "final_prompt_representation_is_vector": all(
            state.ndim == 1 for state in baseline.get("hidden_states", [])
        ),
        "hook_changed_selected_dimension": bool(
            torch.allclose(captured["after"][..., 0], captured["before"][..., 0] + 1.0)
        ),
        "hook_changed_downstream_logits": not torch.equal(changed_logits, baseline_logits),
        "hook_removal_exactly_restores_logits": torch.equal(restored_logits, baseline_logits),
        "action_tokens": wrapper.action_token_ids,
        "action_labels": wrapper.action_labels,
        "layer_count": len(wrapper.layers),
        "target_layer": layer,
    }
    checks["passed"] = all(
        value for key, value in checks.items()
        if key not in {"action_tokens", "action_labels", "layer_count", "target_layer", "passed"}
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--fallback-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", default="artifacts/qwen_compatibility.json")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    from models.hooked_qwen import HookedQwen
    from experiments.runtime import save_run_metadata

    failures = {}
    wrapper = None
    for model_name in (args.model, args.fallback_model):
        try:
            candidate = HookedQwen.from_pretrained(
                model_name, revision=args.revision, local_files_only=not args.online
            )
            result = run_checks(candidate)
            if result["passed"]:
                wrapper = candidate
                result["selected_model"] = model_name
                break
            failures[model_name] = result
        except Exception as error:
            failures[model_name] = {"error": f"{type(error).__name__}: {error}"}
    if wrapper is None:
        raise RuntimeError(f"primary and fallback compatibility gates failed: {failures}")
    result["failed_attempts"] = failures
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    save_run_metadata(args.output + ".metadata.json", vars(args), wrapper)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
