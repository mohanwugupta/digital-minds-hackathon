"""Behavioral-time transition alignment for inferred commitment states."""

from __future__ import annotations


def transition_aligned_summary(
    records: list[dict],
    latent_state: list[float],
    *,
    before: int = 3,
    after: int = 1,
) -> dict:
    """Align w and current choice logit around high-to-low commitment transitions."""

    if len(records) != len(latent_state) or before < 1 or after < 0:
        raise ValueError("invalid transition-alignment inputs")
    ordered_values = sorted(float(value) for value in latent_state)
    low = ordered_values[int(0.25 * (len(ordered_values) - 1))]
    high = ordered_values[int(0.75 * (len(ordered_values) - 1))]
    by_episode = {}
    for index, row in enumerate(records):
        by_episode.setdefault(str(row["episode_id"]), []).append(index)
    aligned = []
    for indices in by_episode.values():
        indices = sorted(indices, key=lambda index: int(records[index]["round"]))
        for position in range(1, len(indices)):
            previous, current = indices[position - 1], indices[position]
            if float(latent_state[previous]) <= high or float(latent_state[current]) >= low:
                continue
            transition_id = f"{records[current]['episode_id']}:{position}"
            for offset in range(-before, after + 1):
                location = position + offset
                if 0 <= location < len(indices):
                    index = indices[location]
                    aligned.append(
                        {
                            "task": str(records[index]["task"]),
                            "episode_id": str(records[index]["episode_id"]),
                            "transition_id": transition_id,
                            "transition_offset": offset,
                            "latent_state": float(latent_state[index]),
                            "persistence_logit": float(records[index]["persistence_logit"]),
                        }
                    )
    cells = {}
    for row in aligned:
        key = (row["task"], row["transition_offset"])
        cells.setdefault(key, []).append(row)
    trajectory = [
        {
            "task": task,
            "transition_offset": offset,
            "mean_latent_state": sum(row["latent_state"] for row in rows) / len(rows),
            "mean_persistence_logit": sum(row["persistence_logit"] for row in rows) / len(rows),
            "states": len(rows),
        }
        for (task, offset), rows in sorted(cells.items())
    ]
    return {
        "high_threshold": high,
        "low_threshold": low,
        "transition_events": len(
            {row["transition_id"] for row in aligned}
        ),
        "trajectory": trajectory,
        "time_axis": "behavioral_time_not_transformer_depth",
    }
