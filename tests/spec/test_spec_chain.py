# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``supports_chained_spec_verify``: the declaration check, and the rows a
chained verify sends and takes back once its predecessor commits.

``tests/test_mamba_spec_async.py`` drives chained verifies through the
scheduler, the runner and a device model end to end.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vllm_tt_plugin.spec_admission import validate_chained_spec_verify
from vllm_tt_plugin.spec_chain import chained_rows, fill_chained_input
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

K = 3


@pytest.mark.parametrize(
    ("capabilities", "match"),
    [
        ({"supports_chained_spec_verify": 1}, "is a bool"),
        ({"supports_chained_spec_verify": True}, "without supports_async_spec"),
        (
            {
                "supports_chained_spec_verify": True,
                "supports_async_spec_decode": False,
            },
            "without supports_async_spec",
        ),
    ],
)
def test_a_chained_verify_declaration_needs_the_deferred_readback(capabilities, match):
    with pytest.raises(ValueError, match=match):
        validate_chained_spec_verify(capabilities, "M")


@pytest.mark.parametrize(
    "capabilities",
    [
        None,
        {},
        {"supports_chained_spec_verify": False},
        {"supports_chained_spec_verify": True, "supports_async_spec_decode": True},
    ],
)
def test_declarations_the_plugin_accepts(capabilities):
    validate_chained_spec_verify(capabilities, "M")


@dataclass
class _Input:
    row_req_ids: list[str]
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    draft_token_ids: torch.Tensor
    num_valid_drafts: torch.Tensor
    accepted_counts: torch.Tensor


def test_chained_rows_send_the_positions_of_a_full_acceptance():
    tokens, positions, drafts, num_valid, counts = chained_rows(
        torch.tensor([100, 7], dtype=torch.int32), K, 2
    )
    assert bool((tokens == PLACEHOLDER_TOKEN_ID).all())
    assert positions.tolist() == [[104, 105, 106, 107], [11, 12, 13, 14]]
    assert bool((drafts == PLACEHOLDER_TOKEN_ID).all())
    assert num_valid.tolist() == [2, 2] and counts.tolist() == [1, 1]


def test_the_predecessor_commit_fills_each_row_from_its_own_count():
    """Each row takes its predecessor's last committed token, at the position
    after the commit, and as many of the drafter's drafts as both the sent
    cap and the drafter's offer allow; a padding row stays one."""
    predecessor = _Input(
        ["a", "b", "c"],
        torch.zeros(4, K + 1, dtype=torch.int32),
        torch.tensor(
            [
                [50 + j for j in range(K + 1)],
                [80, 81, -1, -1],
                [9, 10, 11, 12],
                [-1] * 4,
            ]
        ),
        torch.zeros(4, K, dtype=torch.int32),
        torch.tensor([3, 1, 3, 0], dtype=torch.int32),
        torch.ones(4, dtype=torch.int32),
    )
    tokens, positions, drafts, num_valid, counts = chained_rows(
        torch.tensor([50, 80, 9], dtype=torch.int32), K, K
    )
    pad = lambda t, v: torch.cat([t, torch.full((1, *t.shape[1:]), v, dtype=t.dtype)])  # noqa: E731
    successor = _Input(
        ["a", "b", "c"],
        pad(tokens, 0),
        pad(positions, -1),
        pad(drafts, PLACEHOLDER_TOKEN_ID),
        torch.cat([num_valid, torch.zeros(1, dtype=torch.int32)]),
        torch.cat([counts, torch.ones(1, dtype=torch.int32)]),
    )
    committed = torch.tensor(
        [[5, 6, 7, 8], [9, 10, -1, -1], [11, -1, -1, -1], [0, -1, -1, -1]],
        dtype=torch.int32,
    )
    counts = torch.tensor([4, 2, 1, 1], dtype=torch.int32)
    proposal = torch.arange(16, dtype=torch.int32).reshape(4, 4)[:, :K] + 20
    filled = fill_chained_input(
        successor, predecessor, committed, counts, proposal, [K, K, 0, 0], 1000
    )
    assert filled.input_tokens.tolist() == [
        [8, 20, 21, 22],
        [10, 24, 25, 26],
        [11, -1, -1, -1],
        [0, 0, 0, 0],
    ]
    assert filled.input_positions.tolist() == [
        [54, 55, 56, 57],
        [82, 83, 84, 85],
        [10, -1, -1, -1],
        [-1, -1, -1, -1],
    ]
    assert filled.num_valid_drafts.tolist() == [3, 3, 0, 0]
    assert filled.accepted_counts.tolist() == [4, 2, 1, 1]
    assert filled.draft_token_ids[:3].tolist() == [
        [20, 21, 22],
        [24, 25, 26],
        [-1, -1, -1],
    ]
    with pytest.raises(RuntimeError, match="max_model_len"):
        fill_chained_input(
            successor, predecessor, committed, counts, proposal, None, 57
        )
