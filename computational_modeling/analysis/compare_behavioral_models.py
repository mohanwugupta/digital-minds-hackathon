"""Compare value-learning and heuristic accounts of bandit stopping.

The primary outcome is STOP versus CONTINUE.  Models are evaluated on held-out
episodes and observations are weighted so a 100-round episode does not count
more than an episode that stops after two rounds.  RW learning rates are chosen
inside each training fold; held-out episodes never select their own parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata


MODEL_FEATURES = {
    "time": ["log_round"],
    "heuristic": ["log_round", "previous_outcome", "loss_streak"],
    "rw": ["log_round", "rw_best", "rw_gap"],
    "rw_hybrid": [
        "log_round", "previous_outcome", "loss_streak", "rw_best", "rw_gap"
    ],
    "bayesian": ["log_round", "bayes_best", "bayes_gap"],
    "bayesian_hybrid": [
        "log_round", "previous_outcome", "loss_streak", "bayes_best", "bayes_gap"
    ],
}


def add_behavioral_states(frame: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Add pre-choice RW, Beta-Bernoulli, and recent-history states."""
    required = {"episode_id", "round", "sampled_action", "subsequent_reward"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = frame.sort_values(["episode_id", "round"]).copy().reset_index(drop=True)
    rows: list[dict[str, float]] = []
    for _, episode in out.groupby("episode_id", sort=False):
        q = {"A": 0.5, "B": 0.5}  # expected points under a uniform success prior
        successes = {"A": 1.0, "B": 1.0}
        failures = {"A": 1.0, "B": 1.0}
        previous = 0.0
        streak = 0
        for row in episode.itertuples():
            bayes = {
                arm: 5.0 * successes[arm] / (successes[arm] + failures[arm]) - 2.0
                for arm in ("A", "B")
            }
            rows.append(
                {
                    "rw_A": q["A"], "rw_B": q["B"],
                    "rw_best": max(q.values()), "rw_gap": abs(q["A"] - q["B"]),
                    "bayes_A": bayes["A"], "bayes_B": bayes["B"],
                    "bayes_best": max(bayes.values()),
                    "bayes_gap": abs(bayes["A"] - bayes["B"]),
                    "previous_outcome": previous, "loss_streak": streak,
                    "log_round": np.log1p(float(row.round)),
                }
            )
            action = str(row.sampled_action)
            reward = float(row.subsequent_reward)
            if action in q:
                q[action] += alpha * (reward - q[action])
                if reward > 0:
                    successes[action] += 1
                    streak = 0
                else:
                    failures[action] += 1
                    streak += 1
                previous = reward
    states = pd.DataFrame(rows)
    for column in states.columns:
        out[column] = states[column].to_numpy()
    out["stop"] = (out["sampled_action"] == "C").astype(int)
    return out


def episode_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("episode_id")["episode_id"].transform("size").to_numpy()
    weights = 1.0 / counts
    return weights / weights.mean()


def _fit_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray, l2: float = 1.0):
    x_aug = np.column_stack([np.ones(len(x)), x])

    def objective(beta):
        z = x_aug @ beta
        loss = np.sum(weights * (np.logaddexp(0.0, z) - y * z))
        penalty = 0.5 * l2 * np.sum(beta[1:] ** 2)
        gradient = x_aug.T @ (weights * (_sigmoid(z) - y))
        gradient[1:] += l2 * beta[1:]
        return loss + penalty, gradient

    fit = minimize(objective, np.zeros(x_aug.shape[1]), jac=True, method="L-BFGS-B")
    if not fit.success:
        raise RuntimeError(f"Logistic fit failed: {fit.message}")
    return fit.x


def _sigmoid(z):
    z = np.asarray(z)
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def _auc(y: np.ndarray, probability: np.ndarray) -> float:
    positives = y == 1
    n_pos, n_neg = positives.sum(), (~positives).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(probability)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _predict(train, test, features):
    mean = train[features].mean().to_numpy()
    scale = train[features].std(ddof=0).replace(0, 1).to_numpy()
    x_train = (train[features].to_numpy() - mean) / scale
    x_test = (test[features].to_numpy() - mean) / scale
    beta = _fit_logistic(
        x_train, train["stop"].to_numpy(), episode_balanced_weights(train)
    )
    return _sigmoid(np.column_stack([np.ones(len(test)), x_test]) @ beta), beta


def compare_models(
    frame: pd.DataFrame,
    folds: int = 5,
    alpha_grid: tuple[float, ...] = tuple(np.linspace(0.05, 0.95, 19)),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return held-out predictions and fold-level selected RW alphas."""
    base = add_behavioral_states(frame, alpha=0.5)
    episodes = np.array(sorted(base["episode_id"].unique()))
    fold_ids = np.arange(len(episodes)) % folds
    predictions, selections = [], []
    rng = np.random.default_rng(2026)
    rng.shuffle(fold_ids)
    for fold in range(folds):
        test_ids = set(episodes[fold_ids == fold])
        train_raw = frame[~frame["episode_id"].isin(test_ids)]
        test_raw = frame[frame["episode_id"].isin(test_ids)]
        # Select alpha by episode-balanced in-sample likelihood on training data.
        # Alpha is a latent-state hyperparameter; all policy coefficients are
        # refit below and evaluation remains strictly on held-out episodes.
        alpha_losses = []
        for alpha in alpha_grid:
            candidate = add_behavioral_states(train_raw, alpha)
            probability, _ = _predict(candidate, candidate, MODEL_FEATURES["rw_hybrid"])
            y = candidate["stop"].to_numpy()
            w = episode_balanced_weights(candidate)
            loss = -np.average(
                y * np.log(np.clip(probability, 1e-9, 1))
                + (1 - y) * np.log(np.clip(1 - probability, 1e-9, 1)), weights=w
            )
            alpha_losses.append(loss)
        alpha = float(alpha_grid[int(np.argmin(alpha_losses))])
        train = add_behavioral_states(train_raw, alpha)
        test = add_behavioral_states(test_raw, alpha)
        selections.append({"fold": fold, "alpha": alpha})
        for model, features in MODEL_FEATURES.items():
            probability, beta = _predict(train, test, features)
            part = test[["episode_id", "round", "sampled_action", "stop"]].copy()
            part["fold"] = fold
            part["model"] = model
            part["probability"] = probability
            part["alpha"] = alpha if model.startswith("rw") else np.nan
            predictions.append(part)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(selections)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, part in predictions.groupby("model"):
        y, p = part["stop"].to_numpy(), part["probability"].to_numpy()
        w = episode_balanced_weights(part)
        log_loss = -np.average(
            y * np.log(np.clip(p, 1e-9, 1))
            + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)), weights=w
        )
        rows.append(
            {"model": model, "log_loss": log_loss, "auc": _auc(y, p),
             "brier": np.average((y - p) ** 2, weights=w)}
        )
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def paired_episode_comparisons(
    predictions: pd.DataFrame, reference: str = "heuristic", draws: int = 10000
) -> pd.DataFrame:
    """Bootstrap episode-mean log-loss improvements over a reference model."""
    clipped = predictions.copy()
    clipped["row_loss"] = -(
        clipped["stop"] * np.log(np.clip(clipped["probability"], 1e-9, 1))
        + (1 - clipped["stop"])
        * np.log(np.clip(1 - clipped["probability"], 1e-9, 1))
    )
    episode_loss = clipped.groupby(["episode_id", "model"])["row_loss"].mean().unstack()
    rng = np.random.default_rng(2026)
    rows = []
    for model in episode_loss.columns:
        if model == reference:
            continue
        # Positive delta means the candidate has lower loss than the reference.
        delta = (episode_loss[reference] - episode_loss[model]).dropna().to_numpy()
        boot = np.empty(draws)
        for draw in range(draws):
            boot[draw] = rng.choice(delta, len(delta), replace=True).mean()
        rows.append(
            {
                "model": model,
                "reference": reference,
                "log_loss_improvement": delta.mean(),
                "ci_low": np.quantile(boot, 0.025),
                "ci_high": np.quantile(boot, 0.975),
                "p_two_sided": 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("log_loss_improvement", ascending=False)


def write_report(summary, comparisons, selections, output: Path, n_episodes: int, n_states: int):
    lines = [
        "# Behavioral computational-model comparison", "",
        f"Data: {n_episodes} episodes and {n_states} decision states.", "",
        "Primary outcome: sampled STOP versus CONTINUE. Evaluation uses five held-out "
        "episode folds and equal total weight per episode.", "",
        "| Model | Log loss (lower better) | AUC | Brier |", "|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(f"| {row.model} | {row.log_loss:.4f} | {row.auc:.3f} | {row.brier:.4f} |")
    lines += ["", f"Selected RW learning rates by fold: {selections.alpha.tolist()}.", ""]
    lines += [
        "Episode-bootstrap comparisons use the mean loss within each episode. Positive "
        "differences favor the candidate over the heuristic.", "",
        "| Candidate | Improvement over heuristic | 95% CI | Two-sided p |",
        "|---|---:|---:|---:|",
    ]
    for row in comparisons.itertuples():
        lines.append(
            f"| {row.model} | {row.log_loss_improvement:+.4f} | "
            f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}] | {row.p_two_sided:.4f} |"
        )
    heuristic = float(summary.loc[summary.model == "heuristic", "log_loss"].iloc[0])
    rw = float(summary.loc[summary.model == "rw", "log_loss"].iloc[0])
    hybrid = float(summary.loc[summary.model == "rw_hybrid", "log_loss"].iloc[0])
    lines += [
        "## Interpretation", "",
        f"RW alone changes observation-weighted held-out log loss by {heuristic-rw:+.4f} relative to the "
        "recent-history heuristic; RW plus the heuristic changes it by "
        f"{heuristic-hybrid:+.4f}. Positive values favor the value model.", "",
        "This comparison tests predictive sufficiency, not whether the network literally "
        "implements Rescorla--Wagner learning. A learning model should only be treated as "
        "behaviorally explanatory when it improves held-out prediction beyond the simpler "
        "loss-streak/time model.", "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/bandit_pilot.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("computational_modeling/results")
    )
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    predictions, selections = compare_models(frame)
    summary = summarize_predictions(predictions)
    comparisons = paired_episode_comparisons(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "heldout_predictions.csv", index=False)
    summary.to_csv(args.output_dir / "model_comparison.csv", index=False)
    comparisons.to_csv(args.output_dir / "paired_episode_comparisons.csv", index=False)
    selections.to_csv(args.output_dir / "selected_alphas.csv", index=False)
    payload = {"n_episodes": int(frame.episode_id.nunique()), "n_states": len(frame),
               "models": summary.to_dict(orient="records"),
               "paired_episode_comparisons": comparisons.to_dict(orient="records"),
               "selected_alphas": selections.alpha.tolist()}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    write_report(summary, comparisons, selections, args.output_dir / "report.md",
                 payload["n_episodes"], payload["n_states"])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
