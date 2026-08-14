"""Tiny real-model gate for forced-action continuation rollouts."""

import argparse
import glob
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument("--output", default="artifacts/advantage_rollout_smoke.json")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    import torch

    from experiments.collect_continuation_advantage import estimate_state_advantage
    from experiments.runtime import save_run_metadata
    from models.hooked_qwen import HookedQwen

    paths = sorted(glob.glob(os.path.join(args.activation_dir, "episode_*.pt")))
    if not paths:
        raise FileNotFoundError("smoke activation directory has no episode shards")
    shard = torch.load(paths[0], map_location="cpu", weights_only=False)
    record = shard["records"][0]
    model = HookedQwen.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.online
    )
    estimate = estimate_state_advantage(
        model,
        record,
        rollouts=2,
        seed=99026,
        max_decisions=3,
    )
    if not all(
        isinstance(estimate[key], (int, float))
        for key in ("q_A", "q_B", "q_STOP", "continuation_advantage")
    ):
        raise RuntimeError("advantage smoke produced nonnumeric Q estimates")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"state_id": record["state_id"], **estimate}, handle, indent=2)
        handle.write("\n")
    save_run_metadata(args.output + ".metadata.json", vars(args), model)
    print(json.dumps({"state_id": record["state_id"], **estimate}, indent=2))


if __name__ == "__main__":
    main()
