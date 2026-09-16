# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the greedy accept walk over a verify's ``argmax_ids``.

``accept_greedy_drafts`` is the accept rule for the ``"argmax_ids"`` mode,
where the model returns what the target would have chosen at each of the
``1+K`` candidate positions and no logits cross the boundary. The walk takes
each draft that matches and stops at the first that does not, committing the
target's own choice at the position that rejected.

These tests need no model and no logits: the walk's whole input is one integer
block, one draft block and one count per row.
"""

import pytest
import torch

from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, accept_greedy_drafts

# region Test helpers


def _walk(argmax, drafts, num_valid):
    return accept_greedy_drafts(
        argmax_ids=torch.tensor(argmax, dtype=torch.int32),
        draft_token_ids=torch.tensor(drafts, dtype=torch.int32),
        num_valid_drafts=torch.tensor(num_valid, dtype=torch.int32),
    )


# endregion Test helpers

# region The walk


def test_every_draft_matching_commits_one_more_token():
    """All K drafts accepted commits K+1: the drafts plus the bonus."""
    # Column j is the choice draft j must match; the last column is the bonus.
    committed, counts = _walk([[11, 12, 13, 7]], [[11, 12, 13]], [3])

    assert counts.tolist() == [4]
    assert committed.tolist() == [[11, 12, 13, 7]]


def test_a_mismatch_commits_the_target_s_own_choice_there():
    """Rejecting at position j commits j+1 tokens, the last one corrected."""
    # Draft 0 matches; draft 1 is 12 where the target wants 99, so the row
    # commits the draft it accepted and then 99 in place of the rejected one.
    committed, counts = _walk([[11, 99, 13, 7]], [[11, 12, 13]], [3])

    assert counts.tolist() == [2]
    assert committed.tolist() == [[11, 99, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]]


def test_rejecting_the_first_draft_still_commits_one_token():
    """A count of 0 is never produced: the corrected token always commits."""
    committed, counts = _walk([[99, 98, 7]], [[11, 12]], [2])

    assert counts.tolist() == [1]
    assert committed.tolist() == [[99, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]]


def test_a_row_with_no_drafts_commits_its_bonus():
    """A draftless row is an ordinary decode step inside the wide block."""
    # With no drafts, column 0 is this row's bonus position.
    committed, counts = _walk([[7, 11, 12, 13]], [[11, 12, 13]], [0])

    assert counts.tolist() == [1]
    assert committed.tolist() == [
        [7, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
    ]


def test_a_short_row_bonuses_past_its_own_last_draft():
    """A row carrying fewer than K drafts bonuses at its count, not at K."""
    # One valid draft, accepted, so column 1 is this row's bonus position.
    committed, counts = _walk([[11, 55, 13, 7]], [[11, 12, 13]], [1])

    assert counts.tolist() == [2]
    assert committed.tolist() == [[11, 55, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]]


def test_padding_past_a_row_s_count_can_neither_accept_nor_reject():
    """Columns past the count hold whatever the previous step left there.

    A padding column that happens to match must not extend the row, and one
    that happens to mismatch must not shorten it. Only the row's own count
    decides where the walk stops.
    """
    # Both rows accept their one valid draft. Row 0's padding column 2 agrees
    # with draft 2 and row 1's disagrees, and neither may change the count.
    committed, counts = _walk(
        [[11, 50, 13, 7], [11, 60, 99, 7]],
        [[11, 12, 13], [11, 12, 13]],
        [1, 1],
    )

    assert counts.tolist() == [2, 2]
    assert committed[0].tolist() == [11, 50, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
    assert committed[1].tolist() == [11, 60, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]


def test_one_row_s_rejection_does_not_shorten_another():
    """The walk is per row. A batch-wide stop is the classic bug here."""
    committed, counts = _walk(
        [
            [11, 12, 13, 7],  # accepts all three, then the bonus
            [99, 98, 97, 7],  # rejects draft 0
            [11, 99, 98, 7],  # rejects draft 1
        ],
        [[11, 12, 13]] * 3,
        [3, 3, 3],
    )

    assert counts.tolist() == [4, 1, 2]
    assert committed[0].tolist() == [11, 12, 13, 7]


def test_a_padding_row_commits_its_own_column_zero():
    """A padded decode row carries no drafts, so it commits one token."""
    committed, counts = _walk([[0, 0, 0, 0]], [[-1, -1, -1]], [0])

    assert counts.tolist() == [1]
    assert committed[0, 0] == 0


def test_the_committed_prefix_and_the_count_agree_on_every_row():
    committed, counts = _walk(
        [[11, 12, 13, 7], [11, 40, 0, 7], [7, 11, 12, 13]],
        [[11, 12, 13]] * 3,
        [3, 1, 0],
    )

    for row, count in enumerate(counts.tolist()):
        assert (committed[row, :count] != PLACEHOLDER_TOKEN_ID).all()
        assert (committed[row, count:] == PLACEHOLDER_TOKEN_ID).all()


# endregion The walk

# region Refusals


def test_a_mis_ranked_argmax_block_is_refused():
    with pytest.raises(ValueError, match="must be 2-D"):
        accept_greedy_drafts(
            argmax_ids=torch.zeros(4, dtype=torch.int32),
            draft_token_ids=torch.zeros(1, 3, dtype=torch.int32),
            num_valid_drafts=torch.zeros(1, dtype=torch.int32),
        )


def test_a_draft_width_that_disagrees_with_the_block_is_refused():
    with pytest.raises(ValueError, match=r"draft_token_ids must be \[1, 3\]"):
        accept_greedy_drafts(
            argmax_ids=torch.zeros(1, 4, dtype=torch.int32),
            draft_token_ids=torch.zeros(1, 4, dtype=torch.int32),
            num_valid_drafts=torch.zeros(1, dtype=torch.int32),
        )


def test_a_count_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match=r"num_valid_drafts must be \[2\]"):
        accept_greedy_drafts(
            argmax_ids=torch.zeros(2, 4, dtype=torch.int32),
            draft_token_ids=torch.zeros(2, 3, dtype=torch.int32),
            num_valid_drafts=torch.zeros(1, dtype=torch.int32),
        )


# endregion Refusals


def test_a_placeholder_draft_inside_the_valid_prefix_is_refused():
    """The padding marker is not a token the model can have chosen.

    Nothing downstream can catch it. The model is handed the same column, a
    conformant model returns it unchanged, the walk's comparison matches, and
    ``PLACEHOLDER_TOKEN_ID`` commits as an output token: an id the detokenizer
    rejects, reached only after the whole response has been built. A row whose
    ``num_valid_drafts`` counts more drafts than the caller delivered is a
    caller defect, so it fails here by name.
    """
    argmax_ids = torch.tensor([[PLACEHOLDER_TOKEN_ID, 7, 8, 9]], dtype=torch.int32)
    drafts = torch.tensor(
        [[PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]],
        dtype=torch.int32,
    )

    with pytest.raises(ValueError, match="PLACEHOLDER_TOKEN_ID as a draft"):
        accept_greedy_drafts(argmax_ids, drafts, torch.tensor([3], dtype=torch.int32))


def test_padding_past_a_row_s_count_is_still_allowed():
    """The same marker outside the prefix is the builder's own padding.

    A row carrying fewer drafts than the block is wide pads the rest, and that
    is not an error: the walk never reads those columns.
    """
    argmax_ids = torch.tensor([[11, 7, 8, 9]], dtype=torch.int32)
    drafts = torch.tensor(
        [[11, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]], dtype=torch.int32
    )

    committed, counts = accept_greedy_drafts(
        argmax_ids, drafts, torch.tensor([1], dtype=torch.int32)
    )

    assert int(counts[0]) == 2
    assert committed[0, :2].tolist() == [11, 7]
