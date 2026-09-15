# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Tests for the host accept walk.

``accept_speculated_tokens`` decides how long a prefix of each row of a
``[B, 1+K]`` candidate block commits, and which token ids those are. It is the
step that makes speculation lossless, so the tests come in two kinds.

The structural tests pin the walk's shape: where a rejection stops a row, what
the count is, where the bonus lands, and that one row's rejection never
shortens another's. They are made deterministic by construction rather than by
seeding: a target distribution that puts almost all its mass on the drafted id
accepts whatever uniform is drawn, one that puts almost none rejects it, and a
residual concentrated on a single id recovers that id whatever the exponential
race draws. So these tests read the walk's logic and never its randomness.

The distributional test pins the property the whole module exists for: a
committed token is distributed as the target, not as the drafter. It runs the
trials as rows of one batch, which is both fast and a check that the per-row
randomness really is independent.
"""

import pytest
import torch

from vllm_tt_plugin.spec_accept import AcceptResult, accept_speculated_tokens
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

VOCAB = 8
# A logit gap wide enough that softmax puts more than 1 - 1e-6 on the winner,
# so "almost all the mass" is exact for every assertion below.
DOMINANT = 30.0

# region Test helpers


def _logits(rows: int, width: int, winners: list[list[int]]) -> torch.Tensor:
    """Logits whose column ``j`` of row ``i`` puts ~all mass on ``winners[i][j]``."""
    out = torch.zeros(rows, width, VOCAB)
    for row, columns in enumerate(winners):
        for column, winner in enumerate(columns):
            out[row, column, winner] = DOMINANT
    return out


def _greedy(rows: int) -> torch.Tensor:
    return torch.zeros(rows, dtype=torch.float32)


def _random(rows: int) -> torch.Tensor:
    return torch.ones(rows, dtype=torch.float32)


def _walk(logits, drafts, num_valid, temperature, **kwargs) -> AcceptResult:
    return accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=torch.tensor(drafts, dtype=torch.int32),
        num_valid_drafts=torch.tensor(num_valid, dtype=torch.int32),
        temperature=temperature,
        **kwargs,
    )


# endregion Test helpers

# region The walk's structure


@pytest.mark.parametrize("temperature_of", [_greedy, _random])
def test_every_draft_accepted_commits_one_more_token(temperature_of):
    """A row that accepts all K drafts commits K+1: the drafts plus the bonus."""
    # Columns 0..2 agree with the drafts; column 3 is the bonus position.
    logits = _logits(1, 4, [[1, 2, 3, 7]])
    result = _walk(logits, [[1, 2, 3]], [3], temperature_of(1))

    assert result.accepted_counts.tolist() == [4]
    assert result.committed_token_ids.tolist() == [[1, 2, 3, 7]]


@pytest.mark.parametrize("temperature_of", [_greedy, _random])
def test_a_rejection_commits_the_correction_in_the_draft_s_place(temperature_of):
    """Rejecting at position j commits j+1 tokens, the last one corrected.

    The corrected token is the target's own choice at that position, so the row
    still makes progress. A count of 0 is never produced.
    """
    # Column 1 wants token 5 where the draft says 2, so the row stops there.
    logits = _logits(1, 4, [[1, 5, 3, 7]])
    result = _walk(logits, [[1, 2, 3]], [3], temperature_of(1))

    assert result.accepted_counts.tolist() == [2]
    assert result.committed_token_ids.tolist() == [
        [1, 5, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
    ]


@pytest.mark.parametrize("temperature_of", [_greedy, _random])
def test_rejecting_the_first_draft_still_commits_one_token(temperature_of):
    logits = _logits(1, 3, [[4, 5, 6]])
    result = _walk(logits, [[1, 2]], [2], temperature_of(1))

    assert result.accepted_counts.tolist() == [1]
    assert result.committed_token_ids[0, 0] == 4


@pytest.mark.parametrize("temperature_of", [_greedy, _random])
def test_a_row_with_no_drafts_commits_its_bonus(temperature_of):
    """The uniform 1+K width means a draftless row is an ordinary decode step.

    Its bonus sits at column 0, because that is the column past its last draft,
    and the count is 1.
    """
    logits = _logits(1, 4, [[6, 0, 0, 0]])
    result = _walk(logits, [[1, 2, 3]], [0], temperature_of(1))

    assert result.accepted_counts.tolist() == [1]
    assert result.committed_token_ids[0, 0] == 6


@pytest.mark.parametrize("temperature_of", [_greedy, _random])
def test_a_short_row_puts_its_bonus_past_its_own_last_draft(temperature_of):
    """A row carrying fewer than K drafts bonuses at its own count, not at K."""
    # Two valid drafts out of three. Column 2 is this row's bonus position.
    logits = _logits(1, 4, [[1, 2, 6, 0]])
    result = _walk(logits, [[1, 2, 3]], [2], temperature_of(1))

    assert result.accepted_counts.tolist() == [3]
    assert result.committed_token_ids.tolist() == [[1, 2, 6, PLACEHOLDER_TOKEN_ID]]


@pytest.mark.parametrize("temperature_of", [_greedy, _random])
def test_one_row_s_rejection_does_not_shorten_another(temperature_of):
    """The walk is per row. A batch-wide stop would be the classic bug here."""
    logits = _logits(
        3,
        4,
        [
            [1, 2, 3, 7],  # accepts all three
            [5, 0, 0, 0],  # rejects immediately
            [1, 5, 0, 0],  # rejects at position 1
        ],
    )
    result = _walk(logits, [[1, 2, 3]] * 3, [3, 3, 3], temperature_of(3))

    assert result.accepted_counts.tolist() == [4, 1, 2]


def test_a_mixed_batch_walks_each_row_by_its_own_temperature():
    """Greedy and random rows in one call, each on its own path."""
    logits = _logits(2, 3, [[1, 2, 7], [1, 2, 7]])
    result = accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=torch.tensor([[1, 2], [1, 2]], dtype=torch.int32),
        num_valid_drafts=torch.tensor([2, 2], dtype=torch.int32),
        temperature=torch.tensor([0.0, 1.0]),
    )

    assert result.accepted_counts.tolist() == [3, 3]
    assert result.committed_token_ids.tolist() == [[1, 2, 7], [1, 2, 7]]


@pytest.mark.parametrize("temperature_of", [_greedy, _random])
def test_a_draft_past_a_row_s_count_cannot_commit(temperature_of):
    """Columns past ``num_valid_drafts`` hold no candidate, whatever they hold.

    A row's draft columns beyond its count are padding, and the padding is
    whatever the previous step left there. Reading them as candidates would let
    a stale id commit at a position the scheduler never allocated, so the walk
    has to stop at the row's own count and not at the first mismatch it can
    find anywhere in the block.
    """
    # One valid draft. Column 1's padding happens to match the target and
    # column 2's happens not to, which is the arrangement that would push the
    # walk past the row's count if the count were ignored.
    logits = _logits(1, 4, [[1, 5, 0, 0]])
    result = _walk(logits, [[1, 5, 6]], [1], temperature_of(1))

    assert result.accepted_counts.tolist() == [2]
    assert result.committed_token_ids.tolist() == [
        [1, 5, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
    ]


@pytest.mark.parametrize("temperature_of", [_greedy, _random])
def test_placeholder_padded_draft_columns_are_tolerated(temperature_of):
    """The runner pads draft columns with ``PLACEHOLDER_TOKEN_ID``.

    That is what ``TTModelRunner`` puts past a row's valid draft count, and it
    is not a token id: ``torch.gather`` refuses a negative index rather than
    ignoring it, so a walk that gathers the padding without guarding it fails
    on the first short row a real server produces.
    """
    logits = _logits(1, 4, [[1, 5, 0, 0]])
    result = _walk(
        logits,
        [[1, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]],
        [1],
        temperature_of(1),
    )

    assert result.accepted_counts.tolist() == [2]
    assert result.committed_token_ids.tolist() == [
        [1, 5, PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]
    ]


def test_counts_and_ids_agree_for_every_row_of_a_padded_batch():
    """The committed prefix holds real ids and the tail holds placeholders."""
    logits = _logits(3, 4, [[1, 5, 0, 0], [1, 2, 3, 7], [6, 0, 0, 0]])
    result = _walk(logits, [[1, 2, 3]] * 3, [3, 3, 0], _random(3))

    ids = result.committed_token_ids
    for row, count in enumerate(result.accepted_counts.tolist()):
        assert (ids[row, :count] != PLACEHOLDER_TOKEN_ID).all()
        assert (ids[row, count:] == PLACEHOLDER_TOKEN_ID).all()


# endregion The walk's structure

# region The distribution


def _empirical(committed: torch.Tensor) -> torch.Tensor:
    return (
        torch.bincount(committed.to(torch.int64), minlength=VOCAB) / committed.numel()
    )


@pytest.mark.parametrize("with_draft_probs", [True, False])
def test_the_committed_token_is_distributed_as_the_target(with_draft_probs):
    """The property the module exists for: acceptance is lossless.

    A draft is sampled from the drafter's distribution and then accepted or
    corrected. The committed token must come out distributed as the target,
    never as the drafter, or speculation would silently change what the server
    generates. Trials are rows of one batch.

    Run with the drafter's distribution supplied and withheld. Withheld is the
    n-gram case, where the drafter is a point mass at the id it proposed, and
    the two must agree because a point mass is what the supplied case reduces
    to.
    """
    torch.manual_seed(0)
    trials = 40000
    target = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.06, 0.03, 0.01, 0.00])
    drafter = torch.tensor([0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.00])

    drafts = torch.multinomial(drafter, trials, replacement=True).to(torch.int32)
    logits = target.log().view(1, 1, VOCAB).expand(trials, 2, VOCAB).contiguous()
    draft_probs = (
        drafter.view(1, 1, VOCAB).expand(trials, 1, VOCAB).contiguous()
        if with_draft_probs
        else None
    )

    result = accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=drafts.view(trials, 1),
        num_valid_drafts=torch.ones(trials, dtype=torch.int32),
        temperature=torch.ones(trials),
        draft_probs=draft_probs,
    )

    empirical = _empirical(result.committed_token_ids[:, 0])
    assert torch.allclose(empirical, target, atol=0.01), (
        f"committed distribution {empirical.tolist()} is not the target "
        f"{target.tolist()}"
    )


def test_a_point_mass_drafter_matches_withholding_the_drafter_entirely():
    """Supplying a point-mass ``draft_probs`` equals supplying none.

    Both reduce the accept test to ``p >= u`` and the residual to the target
    with the drafted id removed, so a drafter that reports a degenerate
    distribution and one that reports nothing must behave identically.
    """
    torch.manual_seed(0)
    trials = 20000
    target = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    drafts = torch.randint(0, 4, (trials, 1), dtype=torch.int32)
    logits = target.log().view(1, 1, VOCAB).expand(trials, 2, VOCAB).contiguous()

    point_mass = torch.zeros(trials, 1, VOCAB)
    point_mass.scatter_(2, drafts.to(torch.int64).unsqueeze(2), 1.0)

    common = dict(
        target_logits=logits,
        draft_token_ids=drafts,
        num_valid_drafts=torch.ones(trials, dtype=torch.int32),
        temperature=torch.ones(trials),
    )
    torch.manual_seed(1)
    with_mass = accept_speculated_tokens(**common, draft_probs=point_mass)
    torch.manual_seed(1)
    without = accept_speculated_tokens(**common, draft_probs=None)

    assert torch.equal(with_mass.accepted_counts, without.accepted_counts)
    assert torch.equal(with_mass.committed_token_ids, without.committed_token_ids)


def test_a_short_row_s_bonus_comes_from_the_target_not_the_residual():
    """A row that accepts every draft bonuses from the target distribution.

    The bonus column of a row carrying fewer than K drafts is inside the block
    rather than past it, so it sits where a padding draft also sits. Drawing
    that column from the rejection residual instead of from the target would
    bias it by exactly the mass of whatever padding id happened to be there,
    and nothing downstream would notice.
    """
    torch.manual_seed(0)
    trials = 40000
    # Column 0 always accepts its draft, so every row fully accepts its single
    # real draft and reaches the bonus at column 1.
    accepted_id = 3
    bonus = torch.tensor([0.10, 0.15, 0.20, 0.50, 0.05, 0.00, 0.00, 0.00])
    # The padding id at column 1 carries half the bonus mass, so a residual
    # drawn there instead would be visibly short of it.
    padding_id = int(bonus.argmax())

    logits = torch.zeros(trials, 3, VOCAB)
    logits[:, 0, accepted_id] = DOMINANT
    logits[:, 1] = bonus.log()
    drafts = torch.full((trials, 2), padding_id, dtype=torch.int32)
    drafts[:, 0] = accepted_id

    result = accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=drafts,
        num_valid_drafts=torch.ones(trials, dtype=torch.int32),
        temperature=torch.ones(trials),
    )

    assert (result.accepted_counts == 2).all()
    empirical = _empirical(result.committed_token_ids[:, 1])
    assert torch.allclose(empirical, bonus, atol=0.01), (
        f"bonus distribution {empirical.tolist()} is not the target {bonus.tolist()}"
    )


def test_temperature_reshapes_the_target_before_the_accept_test():
    """The distribution preserved is the one temperature produced.

    Acceptance is lossless with respect to the post-temperature target, so a
    temperature applied the wrong way round would still look self-consistent:
    every count and every id would be valid, and only the shape of the output
    distribution would be wrong.
    """
    torch.manual_seed(0)
    trials = 40000
    temperature = 0.5
    raw = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5])
    expected = (raw / temperature).softmax(dim=-1)

    logits = raw.view(1, 1, VOCAB).expand(trials, 2, VOCAB).contiguous()
    drafts = torch.randint(0, VOCAB, (trials, 1), dtype=torch.int32)

    result = accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=drafts,
        num_valid_drafts=torch.ones(trials, dtype=torch.int32),
        temperature=torch.full((trials,), temperature),
    )

    empirical = _empirical(result.committed_token_ids[:, 0])
    assert torch.allclose(empirical, expected, atol=0.01), (
        f"committed distribution {empirical.tolist()} is not the "
        f"temperature-{temperature} target {expected.tolist()}"
    )


def test_a_seeded_row_is_reproducible():
    """A request with a seed draws from its own generator, so it repeats."""
    logits = torch.randn(2, 3, VOCAB)
    drafts = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    def run() -> AcceptResult:
        return accept_speculated_tokens(
            target_logits=logits,
            draft_token_ids=drafts,
            num_valid_drafts=torch.tensor([2, 2], dtype=torch.int32),
            temperature=torch.ones(2),
            generators={
                0: torch.Generator().manual_seed(7),
                1: torch.Generator().manual_seed(8),
            },
        )

    first, second = run(), run()
    assert torch.equal(first.committed_token_ids, second.committed_token_ids)
    assert torch.equal(first.accepted_counts, second.accepted_counts)


# endregion The distribution

# region Sampling constraints


def test_top_k_of_one_makes_a_random_row_behave_greedily():
    """The constraints apply before the accept test, not after it.

    The distribution acceptance must preserve is the one the operator asked
    for. With top-k of 1 the target is a point mass at its argmax, so a random
    row commits exactly what a greedy row would.
    """
    logits = torch.randn(4, 3, VOCAB)
    drafts = torch.tensor([[1, 2]] * 4, dtype=torch.int32)
    num_valid = torch.tensor([2, 2, 2, 2], dtype=torch.int32)

    greedy = accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=drafts,
        num_valid_drafts=num_valid,
        temperature=torch.zeros(4),
    )
    constrained = accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=drafts,
        num_valid_drafts=num_valid,
        temperature=torch.ones(4),
        top_k=torch.ones(4, dtype=torch.int32),
    )

    assert torch.equal(greedy.accepted_counts, constrained.accepted_counts)
    assert torch.equal(greedy.committed_token_ids, constrained.committed_token_ids)


def test_the_caller_s_logits_are_not_modified():
    """A caller that also wants logprobs needs the raw logits afterwards."""
    logits = torch.randn(2, 3, VOCAB)
    before = logits.clone()

    accept_speculated_tokens(
        target_logits=logits,
        draft_token_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        num_valid_drafts=torch.tensor([2, 2], dtype=torch.int32),
        temperature=torch.full((2,), 0.7),
        top_p=torch.full((2,), 0.9),
    )

    assert torch.equal(logits, before)


# endregion Sampling constraints

# region Refusals


def test_a_mis_ranked_logits_tensor_is_refused():
    with pytest.raises(ValueError, match="must be 3-D"):
        accept_speculated_tokens(
            target_logits=torch.randn(2, VOCAB),
            draft_token_ids=torch.zeros(2, 1, dtype=torch.int32),
            num_valid_drafts=torch.ones(2, dtype=torch.int32),
            temperature=torch.ones(2),
        )


def test_a_draft_width_that_disagrees_with_the_block_is_refused():
    with pytest.raises(ValueError, match=r"draft_token_ids must be \[2, 2\]"):
        accept_speculated_tokens(
            target_logits=torch.randn(2, 3, VOCAB),
            draft_token_ids=torch.zeros(2, 3, dtype=torch.int32),
            num_valid_drafts=torch.ones(2, dtype=torch.int32),
            temperature=torch.ones(2),
        )


@pytest.mark.parametrize("bad", [-1, 3])
def test_a_draft_count_outside_the_block_is_refused(bad: int):
    with pytest.raises(ValueError, match=r"num_valid_drafts entries must lie in"):
        accept_speculated_tokens(
            target_logits=torch.randn(1, 3, VOCAB),
            draft_token_ids=torch.zeros(1, 2, dtype=torch.int32),
            num_valid_drafts=torch.tensor([bad], dtype=torch.int32),
            temperature=torch.ones(1),
        )


def test_a_draft_probs_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match=r"draft_probs must be \[1, 2, 8\]"):
        accept_speculated_tokens(
            target_logits=torch.randn(1, 3, VOCAB),
            draft_token_ids=torch.zeros(1, 2, dtype=torch.int32),
            num_valid_drafts=torch.ones(1, dtype=torch.int32),
            temperature=torch.ones(1),
            draft_probs=torch.rand(1, 3, VOCAB),
        )


def test_a_result_whose_count_overruns_its_ids_is_refused():
    """The pairing is checked where it is produced.

    An off-by-one in a walk corrupts output tokens and raises nowhere on its
    own, so ``AcceptResult`` refuses a count that claims more committed tokens
    than the id block carries.
    """
    ids = torch.tensor([[5, PLACEHOLDER_TOKEN_ID]], dtype=torch.int32)
    with pytest.raises(ValueError, match="PLACEHOLDER_TOKEN_ID inside"):
        AcceptResult(
            committed_token_ids=ids,
            accepted_counts=torch.tensor([2], dtype=torch.int32),
        )


def test_a_result_whose_count_undercounts_its_ids_is_refused():
    ids = torch.tensor([[5, 6]], dtype=torch.int32)
    with pytest.raises(ValueError, match="real token id past"):
        AcceptResult(
            committed_token_ids=ids,
            accepted_counts=torch.tensor([1], dtype=torch.int32),
        )


def test_a_zero_count_is_refused():
    ids = torch.tensor(
        [[PLACEHOLDER_TOKEN_ID, PLACEHOLDER_TOKEN_ID]], dtype=torch.int32
    )
    with pytest.raises(ValueError, match=r"must lie in \[1, 2\]"):
        AcceptResult(
            committed_token_ids=ids,
            accepted_counts=torch.tensor([0], dtype=torch.int32),
        )


# endregion Refusals
