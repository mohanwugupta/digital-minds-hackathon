import random

from analysis.analyze_advantage_probe import exact_match


def test_exact_advantage_matching_recovers_decoding_and_persistence_links():
    rng = random.Random(121)
    rows = []
    for episode in range(40):
        latent_advantage = rng.gauss(0, 1)
        for round_index in (6, 7, 8):
            rows.append(
                {
                    "episode_id": f"episode-{episode}",
                    "round": round_index,
                    "previous_outcome": -2.0,
                    "loss_streak": 1,
                    "prior_score": rng.gauss(0, 1),
                    "continuation_advantage": latent_advantage
                    + rng.gauss(0, 0.05),
                    "ridge_advantage": latent_advantage + rng.gauss(0, 0.05),
                    "persistence_logit": 0.8 * latent_advantage
                    + rng.gauss(0, 0.05),
                }
            )

    result = exact_match(rows)
    probe_target = result["models"]["probe_to_advantage"]["coefficients"][
        "ridge_advantage"
    ]
    probe_persistence = result["models"]["probe_advantage_to_persistence"][
        "coefficients"
    ]["ridge_advantage"]

    assert result["eligible_strata"] == 3
    assert result["states"] == 120
    assert probe_target["standardized_beta"] > 0.9
    assert probe_persistence["standardized_beta"] > 0.9
