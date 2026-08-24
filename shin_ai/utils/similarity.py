"""Distance conversion and diversity re-ranking shared by every semantic lookup.

Chroma collections in this project are created without an explicit
``hnsw:space``, so they run on Chroma's default: **squared L2**. Every
embedding stored is unit-normalised, and for unit vectors

    ||a - b||^2 = 2 - 2(a . b) = 2 * cosine_distance(a, b)

so a reported distance converts to cosine distance exactly by halving it.
Relevance gates are expressed in cosine distance because that number is
interpretable (0 = identical, 1 = orthogonal) and independent of whichever
space a future collection is built in.
"""

from __future__ import annotations

import asyncio
from typing import Any

MAX_COSINE_DISTANCE = 2.0


def cosine_distance_from_chroma(distance: float) -> float:
    """Convert one Chroma squared-L2 distance to cosine distance."""
    return float(distance) / 2.0


def within_distance(distance: float, max_cosine_distance: float) -> bool:
    """Return whether a Chroma-reported distance passes a cosine-distance gate."""
    return cosine_distance_from_chroma(distance) <= max_cosine_distance


def select_mmr_indices(
    query_embedding: Any,
    candidate_embeddings: Any,
    limit: int,
    *,
    lambda_param: float = 0.65,
) -> list[int]:
    """Maximal Marginal Relevance, returning selected candidate indices.

    Balances relevance to the query against redundancy among the picks. The
    diversity term is computed against the whole candidate set at once rather
    than by re-scanning the selected list per candidate, which keeps the cost
    linear in candidates per selected item.
    """
    import numpy as np

    candidates = np.asarray(candidate_embeddings, dtype=np.float64)
    if candidates.ndim != 2 or 0 in candidates.shape or limit <= 0:
        return []

    query = np.asarray(query_embedding, dtype=np.float64).reshape(-1)

    # Embeddings are unit-normalised, so a dot product is the cosine
    # similarity; normalising defensively keeps this correct if that ever
    # stops holding.
    candidates = candidates / np.clip(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-12, None)
    query = query / max(float(np.linalg.norm(query)), 1e-12)

    relevance = candidates @ query
    similarity = candidates @ candidates.T

    total = candidates.shape[0]
    limit = min(limit, total)
    selected: list[int] = []
    # Running max similarity from each candidate to anything already selected.
    redundancy = np.zeros(total, dtype=np.float64)
    available = np.ones(total, dtype=bool)

    for _ in range(limit):
        scores = lambda_param * relevance - (1.0 - lambda_param) * redundancy
        scores[~available] = -np.inf
        best = int(np.argmax(scores))
        if not np.isfinite(scores[best]):
            break
        selected.append(best)
        available[best] = False
        redundancy = np.maximum(redundancy, similarity[best])

    return selected


async def select_mmr_indices_async(
    query_embedding: Any,
    candidate_embeddings: Any,
    limit: int,
    *,
    lambda_param: float = 0.65,
) -> list[int]:
    """Run MMR off the event loop; candidate pools can reach several hundred."""
    return await asyncio.to_thread(
        select_mmr_indices,
        query_embedding,
        candidate_embeddings,
        limit,
        lambda_param=lambda_param,
    )
