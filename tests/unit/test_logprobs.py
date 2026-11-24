# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Checks packing contracts through the public logprobs helpers using CPU tensors."""

import math
from typing import NamedTuple

import pytest
import torch

from tests.support.logprobs import ReportedLogprobs
from vllm_tt_plugin.logprobs import build_device_logprobs, build_logprobs_from_topk


class TopkPayload(NamedTuple):
    """Groups candidate scores, candidate IDs, and the sampled ID for each row.

    B counts batch rows and K counts candidates per row.

    Attributes
    ----------
    logprobs : torch.Tensor
        Sorted host log probabilities of shape ``[B, K]``.
    indices : torch.Tensor
        Candidate token IDs of shape ``[B, K]``, aligned with the scores.
    sampled_token_ids : torch.Tensor
        One sampled token ID per row, of shape ``[B]``.
    """

    logprobs: torch.Tensor
    indices: torch.Tensor
    sampled_token_ids: torch.Tensor


class ExpectedLogprobs(NamedTuple):
    """Describes packed output as explicit Python values independent of tensor code.

    ``token_ids`` and ``values`` contain one list per output row; ``ranks``
    contains the corresponding vocabulary rank for each row.
    """

    token_ids: list[list[int]]
    values: list[list[float]]
    ranks: list[int]


class TestBuildLogprobsFromTopk:
    """Checks candidate packing, vocabulary ranks, and buffer ownership."""

    @pytest.fixture
    def topk_payload(self) -> TopkPayload:
        """Supplies two rows sampling opposite ends of a three-candidate set."""
        return TopkPayload(
            logprobs=torch.tensor(
                [[-0.25, -0.5, -1.0], [-1.5, -2.0, -3.0]],
                dtype=torch.float64,
            ),
            indices=torch.tensor([[11, 12, 13], [21, 22, 23]], dtype=torch.int64),
            sampled_token_ids=torch.tensor([11, 23], dtype=torch.int32),
        )

    @pytest.mark.parametrize(
        "num_candidates, expected",
        [
            pytest.param(
                0,
                ExpectedLogprobs(
                    token_ids=[[11], [23]], values=[[-0.25], [-3.0]], ranks=[1, 3]
                ),
                id="sampled-only",
            ),
            pytest.param(
                1,
                ExpectedLogprobs(
                    token_ids=[[11, 11], [23, 21]],
                    values=[[-0.25, -0.25], [-3.0, -1.5]],
                    ranks=[1, 3],
                ),
                id="candidate-prefix",
            ),
            pytest.param(
                3,
                ExpectedLogprobs(
                    token_ids=[[11, 11, 12, 13], [23, 21, 22, 23]],
                    values=[[-0.25, -0.25, -0.5, -1.0], [-3.0, -1.5, -2.0, -3.0]],
                    ranks=[1, 3],
                ),
                id="all-candidates",
            ),
        ],
    )
    def test_packs_sampled_token_then_candidate_prefix(
        self,
        topk_payload: TopkPayload,
        num_candidates: int,
        expected: ExpectedLogprobs,
    ) -> None:
        """Preserves rows and candidate order, including a duplicated sampled ID."""
        result = build_logprobs_from_topk(
            logprobs=topk_payload.logprobs,
            indices=topk_payload.indices,
            sampled_token_ids=topk_payload.sampled_token_ids,
            max_num_logprobs=num_candidates,
        )

        torch.testing.assert_close(
            result.logprob_token_ids,
            torch.tensor(expected.token_ids, dtype=torch.int32),
        )
        torch.testing.assert_close(
            result.logprobs, torch.tensor(expected.values, dtype=torch.float32)
        )
        torch.testing.assert_close(
            result.selected_token_ranks, torch.tensor(expected.ranks, dtype=torch.int32)
        )

    @pytest.mark.parametrize(
        "token, expected_rank",
        [
            pytest.param(11, 2, id="first-top-tie"),
            pytest.param(12, 2, id="second-top-tie"),
            pytest.param(13, 5, id="first-lower-tie"),
            pytest.param(15, 5, id="last-lower-tie"),
            pytest.param(16, 6, id="lowest-candidate"),
        ],
    )
    def test_rank_counts_ties_beyond_returned_prefix(
        self, token: int, expected_rank: int
    ) -> None:
        """Counts every supplied score at or above the sample, even outside N=1."""
        logprobs = torch.tensor([[-0.5, -0.5, -1.0, -1.0, -1.0, -2.0]])
        indices = torch.tensor([[11, 12, 13, 14, 15, 16]], dtype=torch.int32)

        result = build_logprobs_from_topk(logprobs, indices, torch.tensor([token]), 1)

        assert result.selected_token_ranks.tolist() == [expected_rank]

    def test_rank_at_candidate_cutoff_is_a_lower_bound(self) -> None:
        """With three tied maxima and two supplied, the available rank is two."""
        logprobs = torch.full((1, 2), math.log(1 / 3))
        indices = torch.tensor([[11, 12]], dtype=torch.int32)

        result = build_logprobs_from_topk(logprobs, indices, torch.tensor([12]), 1)

        assert result.selected_token_ranks.tolist() == [2]

    def test_candidate_count_has_no_model_or_api_specific_cap(self) -> None:
        """Supports K=N=33 independently of model or API request limits."""
        logprobs = -torch.arange(1, 34, dtype=torch.float32).unsqueeze(0)
        indices = torch.arange(100, 133, dtype=torch.int32).unsqueeze(0)

        result = build_logprobs_from_topk(logprobs, indices, torch.tensor([132]), 33)

        assert result.logprob_token_ids.tolist() == [[132, *range(100, 133)]]
        assert result.logprobs.shape == (1, 34)
        assert result.logprobs[0, [0, 1, -1]].tolist() == [-33.0, -1.0, -33.0]
        assert result.selected_token_ranks.tolist() == [33]

    @pytest.mark.parametrize("column_ids", [False, True], ids=["flat", "column"])
    @pytest.mark.parametrize(
        "candidate_dtype, sampled_dtype",
        [
            pytest.param(torch.int32, torch.uint32, id="unsigned-sample"),
            pytest.param(torch.uint32, torch.int64, id="unsigned-candidates"),
        ],
    )
    def test_normalizes_sample_shape_and_integer_dtypes(
        self,
        column_ids: bool,
        candidate_dtype: torch.dtype,
        sampled_dtype: torch.dtype,
    ) -> None:
        """Matches signed and unsigned IDs for either supported sample shape."""
        sampled = torch.tensor([12], dtype=sampled_dtype)
        if column_ids:
            sampled = sampled.reshape(1, 1)

        result = build_logprobs_from_topk(
            torch.tensor([[-0.25, -0.5]]),
            torch.tensor([[11, 12]], dtype=candidate_dtype),
            sampled,
            0,
        )

        torch.testing.assert_close(
            result.logprob_token_ids, torch.tensor([[12]], dtype=torch.int32)
        )
        torch.testing.assert_close(result.logprobs, torch.tensor([[-0.5]]))
        assert result.selected_token_ranks.tolist() == [2]

    def test_packed_buffers_are_independent_of_inputs(
        self, topk_payload: TopkPayload
    ) -> None:
        """Packing and later output mutation both leave the inputs unchanged."""
        snapshots = tuple(tensor.clone() for tensor in topk_payload)

        result = build_logprobs_from_topk(
            logprobs=topk_payload.logprobs,
            indices=topk_payload.indices,
            sampled_token_ids=topk_payload.sampled_token_ids,
            max_num_logprobs=2,
        )

        for actual, snapshot in zip(topk_payload, snapshots, strict=True):
            torch.testing.assert_close(actual, snapshot)
        result.logprob_token_ids.zero_()
        result.logprobs.zero_()
        result.selected_token_ranks.zero_()
        for actual, snapshot in zip(topk_payload, snapshots, strict=True):
            torch.testing.assert_close(actual, snapshot)


class TestBuildDeviceLogprobs:
    """Checks payload dispatch and alignment of selected rows with sampled IDs."""

    @pytest.mark.parametrize("column_ids", [False, True], ids=["flat", "column"])
    @pytest.mark.parametrize(
        "payload, num_candidates, expected",
        [
            pytest.param(
                (
                    torch.tensor([[-0.25, -0.5], [-1.0, -2.0], [-3.0, -4.0]]),
                    torch.tensor([[10, 11], [20, 21], [30, 31]]),
                ),
                1,
                ExpectedLogprobs(
                    token_ids=[[31, 30], [10, 10]],
                    values=[[-4.0, -3.0], [-0.25, -0.25]],
                    ranks=[2, 1],
                ),
                id="topk",
            ),
            pytest.param(
                ReportedLogprobs(
                    torch.tensor([-0.25, -1.0, -4.0], dtype=torch.float64),
                    torch.tensor([4, 9, 7], dtype=torch.int64),
                ),
                5,
                ExpectedLogprobs(
                    token_ids=[[31], [10]],
                    values=[[-4.0], [-0.25]],
                    ranks=[7, 4],
                ),
                id="reported-rank-tuple",
            ),
            pytest.param(
                torch.tensor([-0.25, -1.0, -4.0], dtype=torch.float64),
                5,
                ExpectedLogprobs(
                    token_ids=[[31], [10]],
                    values=[[-4.0], [-0.25]],
                    ranks=[-1, -1],
                ),
                id="unknown-rank-tensor",
            ),
        ],
    )
    def test_selects_payload_rows_without_reindexing_sampled_ids(
        self,
        column_ids: bool,
        payload: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | ReportedLogprobs,
        num_candidates: int,
        expected: ExpectedLogprobs,
    ) -> None:
        """Reorders payloads, prioritizes reported ranks, and ignores N for scalars.

        Rows [2, 0] select from three payload rows. The two sampled IDs already
        follow that order; indexing those IDs again would be incorrect.
        """
        sampled = torch.tensor([31, 10], dtype=torch.int32)
        if column_ids:
            sampled = sampled.reshape(2, 1)
        rows = torch.tensor([2, 0])
        payload_tensors = (
            (payload,) if isinstance(payload, torch.Tensor) else tuple(payload)
        )
        inputs = (sampled, rows, *payload_tensors)
        snapshots = tuple(tensor.clone() for tensor in inputs)

        result = build_device_logprobs(payload, sampled, rows, num_candidates)

        torch.testing.assert_close(
            result.logprob_token_ids,
            torch.tensor(expected.token_ids, dtype=torch.int32),
        )
        torch.testing.assert_close(
            result.logprobs, torch.tensor(expected.values, dtype=torch.float32)
        )
        torch.testing.assert_close(
            result.selected_token_ranks, torch.tensor(expected.ranks, dtype=torch.int32)
        )
        for actual, snapshot in zip(inputs, snapshots, strict=True):
            torch.testing.assert_close(actual, snapshot)
