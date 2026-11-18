# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Unit tests for device logprob packing.

These tests use synthetic data only (no device required) to verify that
build_logprobs_from_topk and build_device_logprobs pack device logprobs into
LogprobsTensors for the downstream vLLM pipeline, and that a payload returned
as host decode output reaches that packing through the decode controller. A
selected token's rank follows vLLM: the number of vocabulary scores at or above
the selected one, ties included, so the most likely token has rank 1.
"""

import dataclasses
import math
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from tests.test_block_model_runner import _bind_committed_width
from tests.test_decode_reload_contract import _accepted_decode_hooks, _submission_input
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.logprobs import build_device_logprobs, build_logprobs_from_topk
from vllm_tt_plugin.model_runner import TTModelRunner


class TestBuildLogprobsFromTopk:
    """Tests for build_logprobs_from_topk."""

    def _make_sorted_topk(self, sz: int, k: int = 32):
        """Create synthetic sorted top-K data.

        Returns (top_k_logprobs[sz, k], top_k_indices[sz, k])
        where logprobs are sorted descending per row.
        """
        # Generate sorted descending logprobs
        logprobs = torch.sort(torch.randn(sz, k), dim=-1, descending=True).values
        # Generate unique token indices per row
        indices = torch.stack([torch.randperm(10000)[:k] for _ in range(sz)])
        return logprobs, indices.to(torch.int32)

    def test_basic_shape(self):
        """Output shape is [sz, max_num_logprobs + 1]."""
        sz, N = 4, 5
        logprobs, indices = self._make_sorted_topk(sz)
        sampled = indices[:, 0]  # sampled = top-1 token

        result = build_logprobs_from_topk(logprobs, indices, sampled, N)

        assert result.logprob_token_ids.shape == (sz, N + 1)
        assert result.logprobs.shape == (sz, N + 1)
        assert result.selected_token_ranks.shape == (sz,)

    def test_sampled_token_at_col0(self):
        """Column 0 always contains the sampled token ID and logprob."""
        sz = 8
        logprobs, indices = self._make_sorted_topk(sz)
        # Sampled token is at rank 3 in the sorted list
        sampled = indices[:, 3].clone()

        result = build_logprobs_from_topk(logprobs, indices, sampled, 5)

        assert torch.equal(result.logprob_token_ids[:, 0], sampled)
        expected_lps = logprobs[:, 3]
        assert torch.allclose(result.logprobs[:, 0], expected_lps, atol=1e-6)

    def test_top_n_at_cols_1_to_n(self):
        """Columns 1..N contain top-N tokens from sorted list."""
        sz, N = 4, 5
        logprobs, indices = self._make_sorted_topk(sz)
        sampled = indices[:, 0]

        result = build_logprobs_from_topk(logprobs, indices, sampled, N)

        assert torch.equal(
            result.logprob_token_ids[:, 1 : N + 1],
            indices[:, :N].to(torch.int32),
        )
        assert torch.allclose(
            result.logprobs[:, 1 : N + 1],
            logprobs[:, :N].to(torch.float32),
            atol=1e-6,
        )

    def test_selected_token_ranks(self):
        """Without ties, the rank is the sampled token's 1-based position."""
        sz = 4
        logprobs, indices = self._make_sorted_topk(sz)

        for position in [0, 1, 5, 31]:
            sampled = indices[:, position].clone()
            result = build_logprobs_from_topk(logprobs, indices, sampled, 5)
            assert torch.all(result.selected_token_ranks == position + 1)

    def test_sampled_at_rank1(self):
        """Edge case: the sampled token is the unique most likely one (rank 1)."""
        sz = 4
        logprobs, indices = self._make_sorted_topk(sz)
        sampled = indices[:, 0]

        result = build_logprobs_from_topk(logprobs, indices, sampled, 10)

        assert torch.all(result.selected_token_ranks == 1)
        assert torch.equal(result.logprob_token_ids[:, 0], sampled)

    def test_tied_scores_count_toward_the_rank(self):
        """Every score equal to the sampled one counts, whichever tie was drawn."""
        logprobs = torch.tensor([[-0.5, -0.5, -1.0, -1.0, -1.0, -2.0]])
        indices = torch.tensor([[11, 12, 13, 14, 15, 16]], dtype=torch.int32)

        expected = {11: 2, 12: 2, 13: 5, 15: 5, 16: 6}
        for token, rank in expected.items():
            sampled = torch.tensor([token], dtype=torch.int32)
            result = build_logprobs_from_topk(logprobs, indices, sampled, 3)
            assert result.selected_token_ranks.tolist() == [rank], token

    def test_ties_at_the_lowest_returned_score_count_the_returned_ties(self):
        """With 33 tied maxima and 32 returned, the rank counts the 32 returned.

        Tokens outside the returned set can tie the lowest returned score, so a
        sampled token at that score gets the returned count, a lower bound on
        its vocabulary rank; above that score the count is exact.
        """
        k = 32
        logprobs = torch.full((1, k), math.log(1 / 33))
        indices = torch.arange(100, 100 + k, dtype=torch.int32).unsqueeze(0)
        sampled = indices[:, 7].clone()

        result = build_logprobs_from_topk(logprobs, indices, sampled, 5)

        assert result.selected_token_ranks.tolist() == [k]

    def test_max_num_logprobs_zero(self):
        """max_num_logprobs=0 → shape [sz, 1] (sampled token only)."""
        sz = 4
        logprobs, indices = self._make_sorted_topk(sz)
        sampled = indices[:, 2]

        result = build_logprobs_from_topk(logprobs, indices, sampled, 0)

        assert result.logprob_token_ids.shape == (sz, 1)
        assert result.logprobs.shape == (sz, 1)
        assert torch.equal(result.logprob_token_ids[:, 0], sampled)

    def test_dtypes(self):
        """Output dtypes match LogprobsTensors contract."""
        logprobs, indices = self._make_sorted_topk(4)
        sampled = indices[:, 0]

        result = build_logprobs_from_topk(logprobs, indices, sampled, 5)

        assert result.logprob_token_ids.dtype == torch.int32
        assert result.logprobs.dtype == torch.float32
        assert result.selected_token_ranks.dtype == torch.int32

    def test_per_user_different_sampled(self):
        """Each user can have a different sampled token at different ranks."""
        sz = 4
        logprobs, indices = self._make_sorted_topk(sz)
        # User 0 sampled rank 0, user 1 rank 5, user 2 rank 31, user 3 rank 2
        ranks = [0, 5, 31, 2]
        sampled = torch.tensor(
            [indices[i, r].item() for i, r in enumerate(ranks)], dtype=torch.int32
        )

        result = build_logprobs_from_topk(logprobs, indices, sampled, 10)

        for i, r in enumerate(ranks):
            assert result.selected_token_ranks[i].item() == r + 1
            assert result.logprob_token_ids[i, 0].item() == sampled[i].item()


class _ReportedLogprobs(NamedTuple):
    sampled_logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor


class TestBuildDeviceLogprobs:
    """Tests for build_device_logprobs: row selection and the scalar rank."""

    def test_topk_payload_ranks_the_selected_rows(self):
        logprobs = torch.tensor(
            [[-0.1, -2.0, -3.0], [-0.5, -0.5, -4.0], [-1.0, -1.5, -2.5]]
        )
        indices = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.int32)
        rows = torch.tensor([2, 1])
        sampled = torch.tensor([8, 4], dtype=torch.int32)

        result = build_device_logprobs((logprobs, indices), sampled, rows, 1)

        assert result.selected_token_ranks.tolist() == [2, 2]
        assert result.logprobs[:, 0].tolist() == [-1.5, -0.5]

    def test_scalar_payload_takes_the_rank_the_model_reports(self):
        """A single logprob cannot be ranked here; a model that reports the
        rank alongside it gets that rank, row for row."""
        payload = _ReportedLogprobs(
            sampled_logprobs=torch.tensor([-0.25, -1.5, -0.75]),
            selected_token_ranks=torch.tensor([1, 7, 3]),
        )
        rows = torch.tensor([1, 2])
        sampled = torch.tensor([42, 43], dtype=torch.int32)

        result = build_device_logprobs(payload, sampled, rows, 0)

        assert result.logprob_token_ids.tolist() == [[42], [43]]
        assert result.logprobs.tolist() == [[-1.5], [-0.75]]
        assert result.selected_token_ranks.tolist() == [7, 3]
        assert result.selected_token_ranks.dtype == torch.int32

    def test_scalar_payload_without_a_rank_keeps_the_unknown_marker(self):
        """A bare logprob carries no rank, so none is fabricated: -1 stays."""
        rows = torch.tensor([0, 1])
        sampled = torch.tensor([42, 43], dtype=torch.int32)

        result = build_device_logprobs(torch.tensor([-0.25, -1.5]), sampled, rows, 0)

        assert result.logprobs.tolist() == [[-0.25], [-1.5]]
        assert result.selected_token_ranks.tolist() == [-1, -1]


class _HostOutputModel:
    """An adapter whose decode_forward already returns host output, so it
    implements neither read_decode_output nor process_decode_output_host."""

    decode_input_update_contract = 1
    model_capabilities = {"supports_async_decode": True}

    def __init__(self, output):
        self.output = output

    def decode_forward(self, **kwargs):
        return self.output


class TestHostDecodeOutput:
    """Tests that a nested logprob payload in host decode output passes the
    decode controller and reaches the runner's output packing."""

    @pytest.mark.parametrize("async_read", [False, True], ids=["sync", "async"])
    @pytest.mark.parametrize(
        "payload, rank",
        [
            pytest.param(
                (
                    torch.tensor([[-0.5, -1.5, -2.0]]),
                    torch.tensor([[41, 42, 43]], dtype=torch.int32),
                ),
                2,
                id="topk",
            ),
            pytest.param(
                _ReportedLogprobs(torch.tensor([-1.5]), torch.tensor([7])),
                7,
                id="reported-rank",
            ),
        ],
    )
    def test_payload_rank_reaches_packing(self, payload, rank, async_read):
        token_ids = torch.tensor([[42]], dtype=torch.int32)
        controller = TTAsyncDecodeController(
            _accepted_decode_hooks(
                SimpleNamespace(
                    model=_HostOutputModel((token_ids, payload)),
                    trace_mode="decode_only",
                    kv_caches=object(),
                    request_specific_rope=False,
                    supports_stage_outputs=False,
                )
            )
        )
        model_input = _submission_input(device_sampling=True)
        model_input.tt_sampling_params = dataclasses.replace(
            model_input.tt_sampling_params,
            num_logprobs=torch.zeros(1, dtype=torch.int64),
            enable_log_probs=torch.ones(1, dtype=torch.bool),
        )

        submission = controller.submit_decode(
            model_input, read_from_device=not async_read, async_read=async_read
        )
        finalized = controller.finalize_decode(submission)
        runner = _bind_committed_width(
            SimpleNamespace(
                _num_speculative_tokens=0,
                _output_tokens_per_step=1,
                _is_block_output_model=False,
                tt_per_lane_max_num_seqs=1,
            )
        )
        _, logprobs_per_dp = TTModelRunner._get_output_tokens(
            runner,
            finalized.tt_out,
            finalized.tt_log_probs,
            submission.sampling_params,
            SimpleNamespace(
                intermediate_prefill_mask=None,
                max_num_logprobs=[0],
                grammar_bitmask=[None],
            ),
            submission.batch_size_per_dp,
            perform_device_sampling=True,
            is_decode=True,
        )

        (logprobs,) = logprobs_per_dp
        assert logprobs.logprob_token_ids.tolist() == [[42]]
        assert logprobs.logprobs.tolist() == [[-1.5]]
        assert logprobs.selected_token_ranks.tolist() == [rank]
