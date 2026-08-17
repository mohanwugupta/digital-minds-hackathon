"""Validate X/Y labels under every cross-task prompt mapping."""

import argparse
import json

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
from models.hooked_qwen import verify_chat_choice_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    from models.hooked_qwen import HookedQwen

    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    checks = []
    condition = ForagingCondition(0.55, 0.05, 2, 1)
    for task, mappings in (
        ("foraging", counterbalanced_mappings(STAY, LEAVE)),
        ("control", counterbalanced_mappings(LEFT_GREATER, RIGHT_GREATER)),
    ):
        for mapping in mappings:
            content = (
                initial_prompt(condition, mapping)
                if task == "foraging"
                else comparison_prompt(17, -4, mapping)
            )
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
    print(json.dumps({"model": model.model_id, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
