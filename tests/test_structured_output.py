# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.

import pytest
import torch

from vllm_tt_plugin.structured_output import (
    capture_structured_decode_request_ids,
    reorder_grammar_bitmask_for_tt_batch,
)


def test_capture_structured_decode_request_ids_requires_every_submitted_row():
    with pytest.raises(RuntimeError, match="absent.*req-1"):
        capture_structured_decode_request_ids(
            {"req-0", "req-1"},
            ["req-0", None],
        )


def test_reorder_grammar_bitmask_uses_forward_row_order():
    bitmask = torch.tensor(
        [
            [10, 11],
            [20, 21],
            [30, 31],
            [40, 41],
        ],
        dtype=torch.int32,
    )

    reordered = reorder_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["req-0", "req-1", "req-2", "req-3"],
        row_req_ids=["req-1", "req-3"],
        batch_length=2,
    )

    assert torch.equal(
        reordered,
        torch.tensor(
            [
                [20, 21],
                [40, 41],
            ],
            dtype=torch.int32,
        ),
    )


def test_reorder_grammar_bitmask_ignores_requests_absent_from_the_forward():
    """A structured request the forward did not run must not claim a row."""
    bitmask = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ],
        dtype=torch.int32,
    )

    reordered = reorder_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["req-0", "req-1"],
        row_req_ids=["req-1", "req-9"],
        batch_length=2,
    )

    assert torch.equal(
        reordered,
        torch.tensor(
            [
                [20, 21],
                [-1, -1],
            ],
            dtype=torch.int32,
        ),
    )


def test_reorder_grammar_bitmask_leaves_uncovered_rows_all_allowed():
    """Decode pads to the wire batch size, so rows can outnumber requests."""
    bitmask = torch.tensor([[10, 11]], dtype=torch.int32)

    reordered = reorder_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["req-0"],
        row_req_ids=["req-0"],
        batch_length=3,
    )

    assert torch.equal(
        reordered,
        torch.tensor([[10, 11], [-1, -1], [-1, -1]], dtype=torch.int32),
    )


def test_reorder_grammar_bitmask_handles_forward_narrower_than_batch():
    """A prefill build can drop rows, so a request's persistent batch index can
    exceed the forward's row count."""
    bitmask = torch.tensor([[10, 11]], dtype=torch.int32)

    # Persistent batch req-0..req-3; the forward kept only req-0 and req-2.
    reordered = reorder_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["req-2"],
        row_req_ids=["req-0", "req-2"],
        batch_length=2,
    )

    assert torch.equal(
        reordered,
        torch.tensor([[-1, -1], [10, 11]], dtype=torch.int32),
    )


def test_reorder_grammar_bitmask_rejects_missing_local_structured_row():
    bitmask = torch.tensor([[10, 11]], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="missing TT batch request IDs.*req-1"):
        reorder_grammar_bitmask_for_tt_batch(
            bitmask=bitmask,
            structured_output_request_ids=["req-0"],
            row_req_ids=["req-0", "req-1"],
            batch_length=2,
            expected_structured_output_request_ids={"req-0", "req-1"},
        )


def test_reorder_grammar_bitmask_rejects_duplicate_source_request_id():
    bitmask = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)

    with pytest.raises(ValueError, match="duplicate request IDs.*req-0"):
        reorder_grammar_bitmask_for_tt_batch(
            bitmask=bitmask,
            structured_output_request_ids=["req-0", "req-0"],
            row_req_ids=["req-0"],
            batch_length=1,
            expected_structured_output_request_ids={"req-0"},
        )


def test_reorder_grammar_bitmask_rejects_expected_id_absent_from_snapshot():
    bitmask = torch.tensor([[10, 11]], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="absent from the submitted TT row"):
        reorder_grammar_bitmask_for_tt_batch(
            bitmask=bitmask,
            structured_output_request_ids=["req-0"],
            row_req_ids=["replacement", None],
            batch_length=2,
            expected_structured_output_request_ids={"req-0"},
        )
