"""Balanced rank-1/rank-2/rank-4 persistence-contrast subspaces."""

from __future__ import annotations

from analysis.persistence_contrasts import equal_task_manipulation_weights


INITIAL_RANKS = (1, 2, 4)


def validate_initial_rank(rank: int) -> int:
    rank = int(rank)
    if rank not in INITIAL_RANKS:
        raise ValueError("initial persistence search rank must be 1, 2, or 4")
    return rank


def _family_centroids(deltas, rows):
    import torch

    weights = torch.tensor(
        equal_task_manipulation_weights(rows), dtype=deltas.dtype, device=deltas.device
    )
    families = {}
    for index, row in enumerate(rows):
        key = (str(row["task"]), str(row["manipulation"]))
        families.setdefault(key, []).append(index)
    centroids = []
    for key in sorted(families):
        indices = torch.tensor(families[key], dtype=torch.long, device=deltas.device)
        local_weights = weights[indices]
        local_weights = local_weights / local_weights.sum()
        centroids.append((deltas[indices] * local_weights[:, None]).sum(dim=0))
    return torch.stack(centroids), weights, sorted(families)


def fit_balanced_subspace(deltas, rows, *, rank: int) -> dict:
    """Fit an uncentered low-rank basis to equal-weight family centroids."""

    import torch

    rank = validate_initial_rank(rank)
    if deltas.ndim != 2 or len(deltas) != len(rows):
        raise ValueError("subspace input must be rows x width with matching metadata")
    if not torch.isfinite(deltas).all():
        raise ValueError("subspace deltas must be finite")
    centroids, observation_weights, families = _family_centroids(deltas.float(), rows)
    maximum_rank = min(int(deltas.shape[0]), int(deltas.shape[1]))
    if rank > maximum_rank:
        raise ValueError(
            f"rank {rank} exceeds available family/feature rank {maximum_rank}"
        )
    mean = centroids.mean(dim=0)
    mean_norm = torch.linalg.vector_norm(mean)
    if float(mean_norm) <= 1e-12:
        # SVD remains valid in the multidimensional case, but rank-1 is defined
        # by the PRD as E[delta h] and must not silently become a PCA direction.
        if rank == 1:
            basis = torch.zeros((int(deltas.shape[1]), 1), dtype=deltas.dtype)
            singular_values = torch.zeros(1, dtype=deltas.dtype)
            return {
                "basis": basis,
                "orientation_vector": basis[:, 0],
                "rank": rank,
                "family_centroids": centroids,
                "families": families,
                "observation_weights": observation_weights,
                "mean_direction_norm": 0.0,
                "singular_values": singular_values,
                "no_signal": True,
            }
    if rank == 1:
        basis = (mean / mean_norm).reshape(-1, 1)
        singular_values = mean_norm.reshape(1).to(dtype=deltas.dtype)
    else:
        weighted = deltas.float() * observation_weights.sqrt()[:, None]
        if int(weighted.numel()) <= 2_000_000:
            _u, singular_values, vh = torch.linalg.svd(
                weighted, full_matrices=False
            )
            basis = vh[:rank].T.contiguous()
        else:
            # Randomized low-rank SVD avoids a full hidden_width x hidden_width
            # decomposition for real all-layer banks. Freeze its RNG locally.
            with torch.random.fork_rng():
                torch.manual_seed(0)
                _u, singular_values, v = torch.pca_lowrank(
                    weighted, q=rank, center=False, niter=4
                )
            basis = v[:, :rank].contiguous()
        # Orient each arbitrary SVD sign toward the aggregate persistence shift.
        for column in range(rank):
            if float(torch.dot(basis[:, column], mean)) < 0:
                basis[:, column].mul_(-1)
    projected_mean = basis @ (basis.T @ mean)
    projected_mean_norm = torch.linalg.vector_norm(projected_mean)
    orientation_vector = (
        projected_mean / projected_mean_norm
        if float(projected_mean_norm) > 1e-12
        else torch.zeros_like(mean)
    )
    return {
        "basis": basis.detach().cpu(),
        "orientation_vector": orientation_vector.detach().cpu(),
        "rank": rank,
        "family_centroids": centroids.detach().cpu(),
        "families": families,
        "observation_weights": observation_weights.detach().cpu(),
        "mean_direction_norm": float(mean_norm),
        "singular_values": singular_values[:rank].detach().cpu(),
        "no_signal": False,
    }


def evaluate_subspace(candidate: dict, deltas) -> dict:
    import torch

    if deltas.ndim != 2:
        raise ValueError("held-out subspace data must be rows x width")
    basis = candidate["basis"].to(dtype=deltas.dtype, device=deltas.device)
    if int(basis.shape[0]) != int(deltas.shape[1]):
        raise ValueError("candidate and held-out activation widths differ")
    total = deltas.square().sum(dim=1)
    projected = (deltas @ basis).square().sum(dim=1)
    valid = total > 1e-12
    capture = torch.where(valid, projected / total.clamp_min(1e-12), torch.zeros_like(total))
    centroid = deltas.mean(dim=0)
    mean_projection = torch.linalg.vector_norm(centroid @ basis)
    orientation = candidate.get("orientation_vector")
    if orientation is None:
        orientation = basis[:, 0]
    else:
        orientation = orientation.to(dtype=deltas.dtype, device=deltas.device)
    signed = deltas @ orientation
    return {
        "captured_energy_fraction": float(capture.mean()),
        "mean_projection_norm": float(mean_projection),
        "mean_signed_projection": float(signed.mean()),
        "positive_projection_fraction": float((signed > 0).float().mean()),
        "observations": len(deltas),
        "rank": int(candidate["rank"]),
    }


def matched_random_subspace_scores(
    *, width: int, rank: int, evaluation_deltas, count: int, seed: int
) -> dict:
    """Evaluate isotropic rank-matched controls without fitting evaluation data."""

    import torch

    validate_initial_rank(rank)
    if count < 1 or width < rank:
        raise ValueError("invalid matched-random subspace dimensions")
    generator = torch.Generator(device=evaluation_deltas.device).manual_seed(int(seed))
    scores = []
    for _ in range(count):
        random_matrix = torch.randn(
            width,
            rank,
            generator=generator,
            dtype=evaluation_deltas.dtype,
            device=evaluation_deltas.device,
        )
        basis, _r = torch.linalg.qr(random_matrix, mode="reduced")
        candidate = {
            "basis": basis,
            "orientation_vector": basis[:, 0],
            "rank": rank,
        }
        scores.append(
            evaluate_subspace(candidate, evaluation_deltas)[
                "captured_energy_fraction"
            ]
        )
    ordered = sorted(scores)
    percentile_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "count": count,
        "captured_energy_fraction_95th": float(ordered[percentile_index]),
        "mean": float(sum(scores) / len(scores)),
        "scores": scores,
    }
