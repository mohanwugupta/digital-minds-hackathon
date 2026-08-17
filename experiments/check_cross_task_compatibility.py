"""Validate X/Y labels under every cross-task prompt mapping."""

import argparse
import json
import os

from cross_task.common import counterbalanced_mappings
from cross_task.control import LEFT_GREATER, RIGHT_GREATER, comparison_prompt
from cross_task.foraging import (
    LEAVE,
    STAY,
    ForagingCondition,
    ForagingStep,
    feedback_prompt,
    initial_prompt,
)
from cross_task.solvability import (
    GIVE_UP,
    TRY_AGAIN,
    SolvabilityCondition,
    SolvabilityStep,
    feedback_prompt as solvability_feedback_prompt,
    initial_prompt as solvability_initial_prompt,
)
from cross_task.terminality import END, PROCEED, rule_prompt
from models.hooked_qwen import verify_chat_choice_tokens
from experiments.runtime import run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--config", default="config/cross_task_experiment.yaml")
    parser.add_argument(
        "--output", default="artifacts/cross_task/compatibility.json"
    )
    args = parser.parse_args()

    import yaml

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    collection = config["collection"]

    from models.hooked_qwen import HookedQwen

    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    checks = []
    condition = ForagingCondition(0.55, 0.05, 2, 1)
    for task, mappings in (
        (
            "foraging",
            counterbalanced_mappings(
                STAY, LEAVE, tuple(collection["foraging_response_labels"])
            ),
        ),
        (
            "solvability",
            counterbalanced_mappings(
                TRY_AGAIN,
                GIVE_UP,
                tuple(collection["solvability_response_labels"]),
            ),
        ),
        (
            "control",
            counterbalanced_mappings(
                LEFT_GREATER,
                RIGHT_GREATER,
                tuple(collection["control_response_labels"]),
            ),
        ),
        (
            "terminality",
            counterbalanced_mappings(
                PROCEED,
                END,
                tuple(collection["terminality_response_labels"]),
            ),
        ),
    ):
        for mapping in mappings:
            if task == "foraging":
                content = initial_prompt(condition, mapping)
            elif task == "solvability":
                content = solvability_initial_prompt(
                    SolvabilityCondition(0.5, 1, 2), mapping
                )
            elif task == "control":
                content = comparison_prompt(17, -4, mapping)
            else:
                content = rule_prompt(18, mapping)
            prompts = [("initial", [{"role": "user", "content": content}])]
            if task == "foraging":
                for found_food, reward in ((True, 3), (False, -1)):
                    step = ForagingStep(
                        choice=STAY,
                        reward=reward,
                        found_food=found_food,
                        patch_probability=0.5,
                        terminated=False,
                        reason=None,
                        decision=1,
                        cumulative_score=reward,
                    )
                    prompts.append(
                        (
                            "feedback_found" if found_food else "feedback_empty",
                            [
                                {"role": "user", "content": content},
                                {
                                    "role": "assistant",
                                    "content": mapping.label_for(STAY),
                                },
                                {
                                    "role": "user",
                                    "content": feedback_prompt(step, mapping),
                                },
                            ],
                        )
                    )
            elif task == "solvability":
                for progress in (True, False):
                    step = SolvabilityStep(
                        choice=TRY_AGAIN,
                        progress_made=progress,
                        terminated=False,
                        reason=None,
                        attempt=1,
                        cumulative_cost=1,
                        progress_count=int(progress),
                    )
                    prompts.append(
                        (
                            "feedback_progress" if progress else "feedback_no_progress",
                            [
                                {"role": "user", "content": content},
                                {
                                    "role": "assistant",
                                    "content": mapping.label_for(TRY_AGAIN),
                                },
                                {
                                    "role": "user",
                                    "content": solvability_feedback_prompt(step, mapping),
                                },
                            ],
                        )
                    )
            for stage, messages in prompts:
                token_ids = verify_chat_choice_tokens(
                    model.tokenizer, messages, mapping.labels
                )
                checks.append(
                    {
                        "task": task,
                        "stage": stage,
                        "mapping_id": mapping.mapping_id,
                        "mapping": mapping.to_dict(),
                        "token_ids": token_ids,
                    }
                )
    result = {
        **run_metadata(
            {
                "model": args.model,
                "revision": args.revision,
                "online": args.online,
                "response_labels": {
                    task: collection[f"{task}_response_labels"]
                    for task in (
                        "foraging",
                        "solvability",
                        "control",
                        "terminality",
                    )
                },
            },
            model,
        ),
        "passed": True,
        "checks": checks,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary = args.output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
