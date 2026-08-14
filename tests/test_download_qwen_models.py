from pathlib import Path

from scripts.download_qwen_models import build_download_plan


def test_download_plan_uses_expected_repositories_and_directories():
    model_root = Path("/scratch/gpfs/JORDANAT/mg9965/models")

    plan = build_download_plan(model_root, selection="all")

    assert [(item.repo_id, item.destination) for item in plan] == [
        (
            "Qwen/Qwen3.5-4B",
            model_root / "Qwen--Qwen3.5-4B",
        ),
        (
            "Qwen/Qwen3-4B-Instruct-2507",
            model_root / "Qwen--Qwen3-4B-Instruct-2507",
        ),
    ]


def test_download_plan_can_select_one_checkpoint():
    model_root = Path("/models")

    primary = build_download_plan(model_root, selection="primary")
    fallback = build_download_plan(model_root, selection="fallback")

    assert [item.repo_id for item in primary] == ["Qwen/Qwen3.5-4B"]
    assert [item.repo_id for item in fallback] == [
        "Qwen/Qwen3-4B-Instruct-2507"
    ]
