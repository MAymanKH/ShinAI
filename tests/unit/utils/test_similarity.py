import asyncio
import math

import pytest

from shin_ai.utils.similarity import (
    cosine_distance_from_chroma,
    select_mmr_indices,
    select_mmr_indices_async,
    within_distance,
)


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


class TestDistanceConversion:
    """Chroma reports squared L2; for unit vectors that is exactly 2x cosine."""

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ([1.0, 0.0], [1.0, 0.0]),
            ([1.0, 0.0], [0.0, 1.0]),
            ([1.0, 0.0], [-1.0, 0.0]),
            ([1.0, 1.0], [1.0, 0.2]),
            ([0.3, -0.9, 0.4], [0.8, 0.1, -0.2]),
        ],
    )
    def test_matches_true_cosine_distance(self, first, second) -> None:
        a, b = _unit(first), _unit(second)
        squared_l2 = sum((x - y) ** 2 for x, y in zip(a, b, strict=True))
        cosine = 1.0 - sum(x * y for x, y in zip(a, b, strict=True))
        assert cosine_distance_from_chroma(squared_l2) == pytest.approx(cosine, abs=1e-9)

    def test_identical_vectors_are_zero_distance(self) -> None:
        assert cosine_distance_from_chroma(0.0) == 0.0

    def test_opposite_vectors_are_max_distance(self) -> None:
        # squared L2 of antipodal unit vectors is 4 -> cosine distance 2
        assert cosine_distance_from_chroma(4.0) == 2.0

    def test_gate_is_inclusive_at_the_boundary(self) -> None:
        assert within_distance(0.32, 0.16) is True
        assert within_distance(0.33, 0.16) is False

    def test_gate_converts_before_comparing(self) -> None:
        # A 0.16 cosine gate accepts reported squared-L2 up to 0.32. Applying
        # the threshold to the raw reported value instead -- the bug this
        # module exists to prevent -- would accept everything up to 0.16.
        assert within_distance(0.30, 0.16) is True
        assert within_distance(0.43, 0.16) is False
        assert within_distance(0.58, 0.16) is False


class TestMMR:
    def test_picks_the_most_relevant_first(self) -> None:
        query = [1.0, 0.0]
        candidates = [[0.0, 1.0], [1.0, 0.0], [0.7, 0.7]]
        assert select_mmr_indices(query, candidates, 1) == [1]

    def test_prefers_a_diverse_second_pick_over_a_near_duplicate(self) -> None:
        query = [1.0, 0.0, 0.0]
        # 0 is the best match; 1 is a near-duplicate of it and only marginally
        # less relevant; 2 is slightly less relevant but distinct.
        candidates = [[0.8, 0.6, 0.0], [0.79, 0.61, 0.0], [0.75, 0.0, 0.66]]
        assert select_mmr_indices(query, candidates, 2) == [0, 2]

    def test_pure_relevance_when_lambda_is_one(self) -> None:
        query = [1.0, 0.0, 0.0]
        candidates = [[0.8, 0.6, 0.0], [0.79, 0.61, 0.0], [0.75, 0.0, 0.66]]
        assert select_mmr_indices(query, candidates, 2, lambda_param=1.0) == [0, 1]

    def test_returns_every_candidate_when_limit_exceeds_pool(self) -> None:
        query = [1.0, 0.0]
        candidates = [[1.0, 0.0], [0.0, 1.0]]
        assert sorted(select_mmr_indices(query, candidates, 99)) == [0, 1]

    def test_never_repeats_an_index(self) -> None:
        query = [1.0, 0.0, 0.0]
        candidates = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]
        selected = select_mmr_indices(query, candidates, 4)
        assert len(selected) == len(set(selected)) == 4

    @pytest.mark.parametrize("candidates", [[], [[]]])
    def test_empty_candidates_return_empty(self, candidates) -> None:
        assert select_mmr_indices([1.0, 0.0], candidates, 5) == []

    def test_non_positive_limit_returns_empty(self) -> None:
        assert select_mmr_indices([1.0, 0.0], [[1.0, 0.0]], 0) == []

    def test_handles_unnormalised_input(self) -> None:
        query = [3.0, 0.0]
        candidates = [[0.0, 7.0], [9.0, 0.0]]
        assert select_mmr_indices(query, candidates, 1) == [1]

    def test_async_wrapper_matches_sync_result(self) -> None:
        query = [1.0, 0.0, 0.0]
        candidates = [[0.8, 0.6, 0.0], [0.79, 0.61, 0.0], [0.75, 0.0, 0.66]]
        expected = select_mmr_indices(query, candidates, 2)
        assert asyncio.run(select_mmr_indices_async(query, candidates, 2)) == expected
