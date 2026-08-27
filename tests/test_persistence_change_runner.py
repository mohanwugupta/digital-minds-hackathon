import numpy as np
import pandas as pd

from analysis.persistence_change_data import COMPONENT_TARGET_BY_MANIPULATION
from analysis.persistence_change_geometry import direction_alignment_rows
from analysis.persistence_geometry import FrozenPersistenceSubspace, STAGES
from analysis.run_persistence_change_geometry import (
    generate_figures,
    generate_report,
    run_absolute_comparison,
    run_change_decoding,
    run_group_transfer,
    run_nuisance_controls,
    run_random_controls,
    run_stage_transition,
)


class _Logger:
    def note(self, *_args, **_kwargs):
        pass


def _synthetic_dataset(seed=41):
    rng = np.random.default_rng(seed)
    families = {
        "bandit": ("continue_incentive", "stop_outside_option"),
        "foraging": ("search_cost", "outside_option"),
        "solvability": ("progress_evidence",),
    }
    metadata, target_rows = [], []
    hidden = {stage: [] for stage in STAGES}
    absolute_metadata, absolute_l21, absolute_l22 = [], [], []
    row_index = 0
    for task_index, (task, manipulations) in enumerate(families.items()):
        for manipulation in manipulations:
            for split, count in (("train", 18), ("validation", 8), ("test", 10)):
                for _ in range(count):
                    y = rng.normal()
                    delta = rng.normal(scale=.05, size=8)
                    delta[0] = y + rng.normal(scale=.03)
                    delta[1] = .5 * y + rng.normal(scale=.05)
                    transition = delta + rng.normal(scale=.03, size=8)
                    l22 = delta + transition
                    for stage, values in (("l21", delta), ("displacement", transition), ("l22", l22)):
                        hidden[stage].append(values)
                    cluster = f"cluster-{row_index}"
                    metadata.append(
                        {
                            "task": task,
                            "manipulation": manipulation,
                            "contrast_kind": "persistence",
                            "nuisance_type": "",
                            "contrast_id": f"contrast-{row_index}",
                            "episode_id": cluster,
                            "pair_id": cluster,
                            "split": split,
                            "positive_state_id": f"p-{row_index}",
                            "negative_state_id": f"n-{row_index}",
                        }
                    )
                    targets = {
                        "persistence_policy_change": y,
                        "gru_prediction_change": y + rng.normal(scale=.05),
                        "history_prediction_change": 0.0,
                        "nuisance_policy_change": np.nan,
                        **{name: np.nan for name in set(COMPONENT_TARGET_BY_MANIPULATION.values())},
                    }
                    targets[COMPONENT_TARGET_BY_MANIPULATION[manipulation]] = y
                    target_rows.append(targets)
                    midpoint21 = np.full(8, 30.0 * (task_index + 1))
                    midpoint22 = midpoint21 + rng.normal(scale=.2, size=8)
                    for polarity, sign in (("positive", 1), ("negative", -1)):
                        absolute_metadata.append(
                            {
                                "task": task,
                                "manipulation": manipulation,
                                "contrast_kind": "persistence",
                                "contrast_id": f"contrast-{row_index}",
                                "episode_id": cluster,
                                "pair_id": cluster,
                                "split": split,
                                "state_id": f"{polarity}-{row_index}",
                                "polarity": polarity,
                                "persistence_policy": sign * y / 2,
                                "gru_prediction": sign * y / 2,
                                "history_prediction": 0.0,
                            }
                        )
                        absolute_l21.append(midpoint21 + sign * delta / 2)
                        absolute_l22.append(midpoint22 + sign * l22 / 2)
                    row_index += 1
    for control in ("arbitrary_choice", "terminality", "generic_value"):
        for split, count in (("train", 18), ("validation", 8), ("test", 10)):
            for _ in range(count):
                y = rng.normal()
                values = rng.normal(size=8)
                cluster = f"nuisance-{row_index}"
                metadata.append(
                    {
                        "task": f"{control}_control",
                        "manipulation": control,
                        "contrast_kind": "nuisance",
                        "nuisance_type": control,
                        "contrast_id": f"contrast-{row_index}",
                        "episode_id": cluster,
                        "pair_id": cluster,
                        "split": split,
                        "positive_state_id": f"p-{row_index}",
                        "negative_state_id": f"n-{row_index}",
                    }
                )
                target_rows.append(
                    {
                        "persistence_policy_change": np.nan,
                        "gru_prediction_change": np.nan,
                        "history_prediction_change": np.nan,
                        "nuisance_policy_change": y,
                        **{name: np.nan for name in set(COMPONENT_TARGET_BY_MANIPULATION.values())},
                    }
                )
                for stage in STAGES:
                    hidden[stage].append(values)
                for polarity, sign in (("positive", 1), ("negative", -1)):
                    absolute_metadata.append(
                        {
                            "task": f"{control}_control",
                            "manipulation": control,
                            "contrast_kind": "nuisance",
                            "contrast_id": f"contrast-{row_index}",
                            "episode_id": cluster,
                            "pair_id": cluster,
                            "split": split,
                            "state_id": f"{polarity}-{row_index}",
                            "polarity": polarity,
                            "persistence_policy": sign * y / 2,
                            "gru_prediction": np.nan,
                            "history_prediction": np.nan,
                        }
                    )
                    absolute_l21.append(sign * values / 2)
                    absolute_l22.append(sign * values)
                row_index += 1
    basis = np.eye(8, 4, dtype=np.float32)
    arrays = {stage: np.asarray(values, dtype=np.float32) for stage, values in hidden.items()}
    return {
        "metadata": pd.DataFrame(metadata),
        "targets": pd.DataFrame(target_rows),
        "hidden": arrays,
        "projected": {stage: values @ basis for stage, values in arrays.items()},
        "absolute_metadata": pd.DataFrame(absolute_metadata),
        "absolute_hidden": {
            "l21": np.asarray(absolute_l21, dtype=np.float32),
            "l22": np.asarray(absolute_l22, dtype=np.float32),
        },
    }, FrozenPersistenceSubspace.from_array(basis, source="synthetic")


def test_synthetic_runner_produces_all_core_analyses(tmp_path):
    dataset, frozen = _synthetic_dataset()
    config = {
        "seed": 5,
        "ridge_alphas": [0.01, 1.0],
        "bootstrap_samples": 20,
        "matched_random_subspaces": 2,
    }
    logger = _Logger()
    change, _fits = run_change_decoding(dataset, config, logger)
    cross_task, task_fits = run_group_transfer(
        dataset,
        config,
        group_column="task",
        target_names=("persistence_policy_change", "gru_prediction_change"),
        logger=logger,
    )
    cross_manipulation, _ = run_group_transfer(
        dataset,
        config,
        group_column="manipulation",
        target_names=("persistence_policy_change", "gru_prediction_change"),
        logger=logger,
    )
    absolute = run_absolute_comparison(
        dataset, frozen, task_fits, config, logger
    )
    random = run_random_controls(
        dataset,
        change,
        cross_task,
        cross_manipulation,
        frozen,
        config,
        tmp_path,
        False,
        logger,
    )
    nuisance = run_nuisance_controls(dataset, change, config, logger)
    persistence = dataset["metadata"].contrast_kind == "persistence"
    direction = pd.DataFrame(
        direction_alignment_rows(
            {stage: dataset["projected"][stage][persistence] for stage in STAGES},
            dataset["metadata"].loc[persistence].reset_index(drop=True),
        )
    )
    transition = run_stage_transition(
        direction, change, cross_task, cross_manipulation
    )
    assert set(change.stage) == set(STAGES)
    assert set(cross_task.heldout_task) == {"bandit", "foraging", "solvability"}
    assert set(cross_manipulation.heldout_manipulation) == set(
        COMPONENT_TARGET_BY_MANIPULATION
    )
    assert len(absolute) == 3 * 2 * 3
    assert len(random) == 3 * 2 * 2 * 3
    assert set(nuisance.control) == {"arbitrary_choice", "terminality", "generic_value"}
    assert {"direction_cosine", "task_subspace_overlap"} <= set(direction.kind)
    assert len(transition) == 6
    for name, frame in (
        ("change_decoding.csv", change),
        ("cross_task_change_transfer.csv", cross_task),
        ("cross_manipulation_transfer.csv", cross_manipulation),
        ("absolute_vs_change.csv", absolute),
        ("random_subspace_controls.csv", random),
        ("nuisance_change_controls.csv", nuisance),
        ("direction_alignment.csv", direction),
        ("stage_transition.csv", transition),
    ):
        frame.to_csv(tmp_path / name, index=False)
    generate_figures(tmp_path)
    generate_report(tmp_path)
    required_figures = {
        "change_decoding.png",
        "absolute_vs_change_transfer.png",
        "cross_task_change_transfer.png",
        "cross_manipulation_transfer.png",
        "persistence_vs_nuisance_change.png",
        "direction_alignment.png",
    }
    assert required_figures == {path.name for path in (tmp_path / "figures").glob("*.png")}
    assert (tmp_path / "report.md").read_text().startswith(
        "# Cross-task persistence computation"
    )
