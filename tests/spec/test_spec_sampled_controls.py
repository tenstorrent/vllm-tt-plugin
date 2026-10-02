# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The logits walk applies grammars, token filters and logprobs as vLLM does.

``test_spec_sampled_accept.py`` holds ``accept_sampled_drafts`` to its
references for temperature, top-k, top-p, min_p and the penalties. This file
does the same for the controls a ``logits`` verify applies beside those: the
grammar bitmask, allowed_token_ids, bad_words and min_tokens, which rule tokens
out, and logprobs, which report on what was committed.

The references are the same two kinds.

vLLM's own ``Sampler``, for a row that carries no drafts, fed what the plugin's
ordinary host path feeds it: logits the grammar bitmask has already masked,
and the token filters in its ``SamplingMetadata``. The row must commit the same
token, leave its generator in the same state, and report the same logprobs.

An independent small-vocabulary reference, for rows that carry drafts. It
enumerates the joint distribution of the first committed tokens with each
filter applied in plain Python at every position, and a simulation that drafts,
validates, verifies and commits must reproduce it whatever was drafted. The
grammar case runs twice: once with the drafts truncated at the first one the
grammar rejects, which is what the scheduler does to drafts it validates, and
once with every draft kept and the bitmask rows past the first rejected one
left unconstrained, which is what reaches the walk when the scheduler filled
the rows from drafts it validated but the runner had already verified more.
"""

from __future__ import annotations

import math
from itertools import product

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdateBuilder, LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

import vllm_tt_plugin.spec_accept as spec_accept
from vllm_tt_plugin.spec_accept import (
    SpecPenalties,
    SpecSamplingInputs,
    SpecTokenFilters,
    accept_sampled_drafts,
    apply_speculative_token_filters,
)
from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID

from .test_spec_sampled_accept import (
    PROMPT,
    SMALL_VOCAB,
    _argmax_drafter,
    _causal_table,
    _chi_square,
    _chi_square_critical,
    _reference_distribution,
    _sampling_drafter,
    _wrong_drafter,
)

VOCAB = 32

# region Helpers


def _pack(allowed: torch.Tensor) -> torch.Tensor:
    """xgrammar's packed layout for a ``[..., V]`` bool allowlist."""
    words = (allowed.shape[-1] + 31) // 32
    padded = torch.zeros((*allowed.shape[:-1], words * 32), dtype=torch.bool)
    padded[..., : allowed.shape[-1]] = allowed
    bits = padded.reshape(*allowed.shape[:-1], words, 32).to(torch.int64)
    weights = torch.tensor([1 << b for b in range(32)], dtype=torch.int64)
    packed = (bits * weights).sum(-1)
    # Two's complement, which is how an int32 word holds bit 31.
    return torch.where(packed >= 2**31, packed - 2**32, packed).to(torch.int32)


def _min_tokens_processor(
    params: list[SamplingParams | None], outputs: list[list[int]]
) -> MinTokensLogitsProcessor:
    """vLLM's min_tokens processor, fed the batch through its own update path."""
    processor = MinTokensLogitsProcessor(None, torch.device("cpu"), False)
    builder = BatchUpdateBuilder()
    for row, (param, output) in enumerate(zip(params, outputs)):
        if param is not None:
            builder.added.append((row, param, None, output))
    processor.update_state(builder.get_and_reset(len(params)))
    return processor


def _ordinary(
    logits: torch.Tensor,
    *,
    temperature: torch.Tensor,
    generators: dict[int, torch.Generator],
    history: list[list[int]],
    grammar: torch.Tensor | None = None,
    allowed_mask: torch.Tensor | None = None,
    bad_words: dict[int, list[list[int]]] | None = None,
    min_tokens: MinTokensLogitsProcessor | None = None,
    num_logprobs: int | None = None,
    top_k: torch.Tensor | None = None,
    penalties: dict | None = None,
):
    """One ordinary step through vLLM's ``Sampler``, as the plugin's host path runs it.

    The grammar is applied to the logits first, by masking, which is what
    ``TTModelRunner.apply_grammar_bitmask`` does before it calls the sampler.
    """
    rows, vocab = logits.shape
    logits = logits.clone()
    if grammar is not None:
        logits.masked_fill_(spec_accept._grammar_disallowed(grammar, vocab), -math.inf)
    zeros = torch.zeros(rows)
    penalties = penalties or {}
    return Sampler()(
        logits=logits,
        sampling_metadata=SamplingMetadata(
            temperature=temperature,
            all_greedy=bool((temperature == 0).all()),
            all_random=bool((temperature != 0).all()),
            top_p=torch.ones(rows),
            top_k=(
                torch.full((rows,), vocab, dtype=torch.int32)
                if top_k is None
                else top_k
            ),
            generators=generators,
            max_num_logprobs=num_logprobs,
            no_penalties=not penalties,
            prompt_token_ids=penalties.get("prompt"),
            frequency_penalties=penalties.get("frequency", zeros),
            presence_penalties=penalties.get("presence", zeros),
            repetition_penalties=penalties.get("repetition", torch.ones(rows)),
            output_token_ids=history,
            allowed_token_ids_mask=allowed_mask,
            bad_words_token_ids=bad_words or {},
            logitsprocs=LogitsProcessors([] if min_tokens is None else [min_tokens]),
        ),
    )


def _generators(temperature: torch.Tensor, seed: int) -> dict[int, torch.Generator]:
    """A seeded generator per random row; vLLM gives a greedy request none."""
    return {
        row: torch.Generator().manual_seed(seed * 10 + row)
        for row in range(int(temperature.shape[0]))
        if float(temperature[row]) > 0
    }


# endregion Helpers

# region A row without drafts is an ordinary sampled step


FILTER_CASES = ["grammar", "allowed", "bad-words", "min-tokens", "logprobs", "all"]


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("case", FILTER_CASES)
def test_a_draftless_row_draws_the_ordinary_sampler_s_token(seed, case):
    """Same token, same generator state, same logprobs as one ordinary step.

    Rows mix greedy and seeded random sampling, so the filters are seen by
    both the argmax and the random draw. A filter applied at the wrong column,
    or applied after the draw, changes the committed token on some seed. Token
    0 survives every filter, so no row is left with nothing to sample.
    """
    torch.manual_seed(3000 + seed)
    rows, width = 4, 4
    logits = torch.randn(rows, width, VOCAB) * 2.0
    temperature = torch.tensor([1.0, 0.0, 0.7, 1.3])
    history = [[3, 5, 9], [9], [], [1, 9, 9, 4]]
    top_k = torch.tensor([VOCAB, VOCAB, 6, VOCAB], dtype=torch.int32)

    grammar = allowed_mask = None
    bad_words: dict[int, list[list[int]]] = {}
    min_params: list[SamplingParams | None] = [None] * rows
    num_logprobs = None
    penalties = None
    if case in ("grammar", "all"):
        allowed = torch.rand(rows, VOCAB, generator=torch.Generator().manual_seed(seed))
        allowed = allowed < 0.6
        allowed[:, 0] = True
        grammar = _pack(allowed)
    if case in ("allowed", "all"):
        allowed_mask = torch.ones(rows, VOCAB, dtype=torch.bool)
        allowed_mask[0, [0, 2, 4, 6, 8]] = False
        allowed_mask[2, [0, 1, 2, 3]] = False
        allowed_mask[1] = False
        allowed_mask[3] = False
    if case in ("bad-words", "all"):
        # Single-token words, and words whose prefix the history ends with.
        bad_words = {0: [[7], [9, 11]], 1: [[9, 2], [9, 3]], 3: [[9, 4, 5], [4, 6]]}
    if case in ("min-tokens", "all"):
        min_params = [
            SamplingParams(min_tokens=5, stop_token_ids=[1, 2, 4]),
            SamplingParams(min_tokens=1, stop_token_ids=[3]),
            None,
            SamplingParams(min_tokens=4, stop_token_ids=[5]),
        ]
    if case in ("logprobs", "all"):
        num_logprobs = 3
    if case == "all":
        penalties = {
            "prompt": torch.tensor([[1, 2, VOCAB]] * rows, dtype=torch.int64),
            "presence": torch.tensor([0.5, 0.0, 0.3, 0.0]),
            "frequency": torch.tensor([0.0, 0.4, 0.2, 0.0]),
            "repetition": torch.tensor([1.2, 1.0, 1.0, 1.5]),
        }

    min_tokens = {
        row: (
            params.min_tokens - len(history[row]),
            tuple(sorted(params.all_stop_token_ids)),
        )
        for row, params in enumerate(min_params)
        if params is not None and params.min_tokens > len(history[row])
    }
    filters = SpecTokenFilters(
        grammar_bitmask=(
            None
            if grammar is None
            else grammar.unsqueeze(1).expand(rows, width, -1).contiguous()
        ),
        allowed_token_ids_mask=allowed_mask,
        bad_words_token_ids=bad_words,
        output_token_ids=history,
        min_tokens=min_tokens,
    )
    walked_generators = _generators(temperature, seed)
    walked = accept_sampled_drafts(
        logits,
        torch.zeros(rows, width - 1, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        SpecSamplingInputs(
            vocab_size=VOCAB,
            temperature=temperature,
            top_k=top_k,
            generators=walked_generators,
            penalties=(
                None
                if penalties is None
                else SpecPenalties(
                    prompt_token_ids=penalties["prompt"],
                    output_token_ids=history,
                    presence=penalties["presence"],
                    frequency=penalties["frequency"],
                    repetition=penalties["repetition"],
                )
            ),
            filters=filters,
            num_logprobs=num_logprobs,
        ),
    )

    ordinary_generators = _generators(temperature, seed)
    ordinary = _ordinary(
        logits[:, 0, :],
        temperature=temperature,
        generators=ordinary_generators,
        history=history,
        grammar=grammar,
        allowed_mask=allowed_mask,
        bad_words=bad_words,
        min_tokens=_min_tokens_processor(min_params, history),
        num_logprobs=num_logprobs,
        top_k=top_k,
        penalties=penalties,
    )

    assert walked.accepted_counts.tolist() == [1] * rows
    assert walked.committed_token_ids[:, 0].tolist() == (
        ordinary.sampled_token_ids.view(-1).tolist()
    )
    for row in range(rows):
        if row in walked_generators:
            assert torch.equal(
                walked_generators[row].get_state(),
                ordinary_generators[row].get_state(),
            ), f"row {row} advanced its generator differently from an ordinary step"
    if num_logprobs is None:
        assert walked.logprobs is None
        return
    expected = ordinary.logprobs_tensors
    assert walked.logprobs.cu_num_generated_tokens == list(range(rows + 1))
    assert torch.equal(walked.logprobs.logprob_token_ids, expected.logprob_token_ids)
    assert torch.equal(walked.logprobs.logprobs, expected.logprobs)
    assert torch.equal(
        walked.logprobs.selected_token_ranks, expected.selected_token_ranks
    )


# endregion A row without drafts is an ordinary sampled step

# region Each column reads its own state


def test_a_bad_word_bans_its_last_token_after_the_column_s_own_history():
    """Column ``j``'s history is the committed output plus drafts ``0..j-1``.

    The word ``[4, 6]`` bans 6 wherever the history ends with 4: at column 0,
    whose history is the committed output ``[.., 4]``, and at column 2, after
    the drafts ``[1, 4]``. Columns 1 and 3 follow 1 and 2 and may commit 6.
    A padding column past the row's drafts reads only the valid ones.
    """
    rows, width = 2, 4
    logits = torch.zeros(rows, width, 8)
    drafts = torch.tensor([[1, 4, 2], [4, 9, 9]], dtype=torch.int32)
    num_valid = torch.tensor([3, 1], dtype=torch.int32)
    filtered = apply_speculative_token_filters(
        logits,
        drafts,
        num_valid,
        SpecTokenFilters(
            bad_words_token_ids={0: [[4, 6]], 1: [[4, 6], [5]]},
            output_token_ids=[[3, 4], []],
        ),
    )
    banned = filtered[..., 6] == -math.inf
    assert banned.tolist() == [[True, False, True, False], [False, True, True, True]]
    # A one-token word bans its token everywhere.
    assert bool((filtered[1, :, 5] == -math.inf).all())
    assert bool((filtered[0, :, 5] == 0).all())
    assert bool((logits == 0).all()), "the caller's logits were modified"


def test_min_tokens_masks_the_stop_tokens_only_while_the_output_is_short():
    """A column ``j`` masks the stop tokens while ``j < min_tokens - len(output)``.

    So two columns for a row two tokens short, every column for a row further
    short than the block is wide, and none for a row with nothing remaining.
    """
    rows, width = 3, 4
    filtered = apply_speculative_token_filters(
        torch.zeros(rows, width, 8),
        torch.zeros(rows, width - 1, dtype=torch.int32),
        torch.full((rows,), width - 1, dtype=torch.int32),
        SpecTokenFilters(min_tokens={0: (2, (1, 7)), 1: (9, (3,)), 2: (0, (2,))}),
    )
    masked = filtered == -math.inf
    assert masked[0, :, 1].tolist() == [True, True, False, False]
    assert masked[0, :, 7].tolist() == [True, True, False, False]
    assert masked[1, :, 3].tolist() == [True] * width
    assert not bool(masked[2].any())
    assert int(masked.sum()) == 2 * 2 + width


def test_each_column_reads_its_own_grammar_row():
    """A greedy row walks the grammar column by column.

    The target prefers token 7 everywhere, and each column's bitmask allows
    only the draft offered there, so every draft is the masked argmax and is
    accepted; the bonus column allows only 5. Moving one column's row by one
    rejects at that column and commits what that row allows instead.
    """
    width = 4
    logits = torch.zeros(1, width, 8)
    logits[..., 7] = 4.0
    logits[0, 2, 6] = 1.0
    drafts = torch.tensor([[2, 3, 4]], dtype=torch.int32)

    def walk(rows_allowed):
        allowed = torch.zeros(1, width, 8, dtype=torch.bool)
        for column, tokens in enumerate(rows_allowed):
            allowed[0, column, tokens] = True
        return accept_sampled_drafts(
            logits,
            drafts,
            torch.tensor([3], dtype=torch.int32),
            SpecSamplingInputs(
                vocab_size=8,
                temperature=torch.zeros(1),
                filters=SpecTokenFilters(grammar_bitmask=_pack(allowed)),
            ),
        )

    accepted = walk([[2], [3], [4], [5]])
    assert accepted.committed_token_ids.tolist() == [[2, 3, 4, 5]]

    rejected = walk([[2], [3], [6], [5]])
    assert rejected.accepted_counts.tolist() == [3]
    assert rejected.committed_token_ids[0, :3].tolist() == [2, 3, 6]


def test_a_draft_the_grammar_forbids_is_rejected_even_on_a_zero_draw(monkeypatch):
    """Probability zero is a rejection, not a coin flip that can come up 0.

    The accept test is ``p(d) >= u``. A uniform draw of exactly 0 would accept
    a draft the target gives no probability, and a grammar-forbidden draft
    reaches the walk whenever the scheduler did not validate it.
    """
    monkeypatch.setattr(
        spec_accept,
        "_uniform",
        lambda rows, num_drafts, num_valid, generators: torch.zeros(
            rows, num_drafts, dtype=torch.float64
        ),
    )
    width = 3
    logits = torch.zeros(1, width, 8)
    allowed = torch.ones(1, width, 8, dtype=torch.bool)
    allowed[0, 1, 5] = False
    result = accept_sampled_drafts(
        logits,
        torch.tensor([[2, 5]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        SpecSamplingInputs(
            vocab_size=8,
            temperature=torch.ones(1),
            filters=SpecTokenFilters(grammar_bitmask=_pack(allowed)),
        ),
    )
    assert result.accepted_counts.tolist() == [2]
    assert int(result.committed_token_ids[0, 0]) == 2
    assert int(result.committed_token_ids[0, 1]) != 5


def test_logprobs_are_raw_at_each_committed_column():
    """Each committed token reports the raw logprobs of its own column.

    Raw means after the grammar and before everything else, as the ordinary
    host sampler reports them, so the penalties, temperature and token filters
    here change what is committed but not what is reported for it. The rows
    commit different counts, which ``cu_num_generated_tokens`` locates.
    """
    torch.manual_seed(7)
    rows, width = 3, 4
    logits = torch.randn(rows, width, VOCAB)
    allowed = torch.ones(rows, width, VOCAB, dtype=torch.bool)
    allowed[:, :, 3] = False
    drafts = logits[:, :-1].argmax(-1).to(torch.int32)
    num_valid = torch.tensor([3, 1, 0], dtype=torch.int32)
    result = accept_sampled_drafts(
        logits,
        drafts,
        num_valid,
        SpecSamplingInputs(
            vocab_size=VOCAB,
            temperature=torch.tensor([0.0, 0.8, 1.2]),
            penalties=SpecPenalties(
                prompt_token_ids=torch.full((rows, 1), VOCAB, dtype=torch.int64),
                output_token_ids=[[1, 1], [2], []],
                presence=torch.full((rows,), 0.7),
                frequency=torch.full((rows,), 0.4),
                repetition=torch.full((rows,), 1.3),
            ),
            filters=SpecTokenFilters(
                grammar_bitmask=_pack(allowed),
                bad_words_token_ids={0: [[5]]},
                output_token_ids=[[1, 1], [2], []],
            ),
            num_logprobs=4,
        ),
    )
    counts = result.accepted_counts.tolist()
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    assert result.logprobs.cu_num_generated_tokens == starts

    raw = logits.masked_fill(~allowed, -math.inf).log_softmax(-1)
    flat = 0
    for row, count in enumerate(counts):
        for column in range(count):
            token = int(result.committed_token_ids[row, column])
            column_logprobs = raw[row, column]
            top = column_logprobs.topk(4)
            assert int(result.logprobs.logprob_token_ids[flat, 0]) == token
            assert float(result.logprobs.logprobs[flat, 0]) == pytest.approx(
                float(column_logprobs[token])
            )
            assert result.logprobs.logprob_token_ids[flat, 1:].tolist() == (
                top.indices.tolist()
            )
            assert torch.allclose(result.logprobs.logprobs[flat, 1:], top.values)
            assert int(result.logprobs.selected_token_ranks[flat]) == int(
                (column_logprobs >= column_logprobs[token]).sum()
            )
            flat += 1
    assert flat == int(result.logprobs.logprobs.shape[0])


@pytest.mark.parametrize(
    "broken, match",
    [
        ({"allowed_token_ids_mask": torch.zeros(1, VOCAB, dtype=torch.bool)}, "mask"),
        ({"grammar_bitmask": torch.full((2, 1, 1), -1, dtype=torch.int32)}, "grammar"),
        ({"bad_words_token_ids": {5: [[1]]}, "output_token_ids": [[], []]}, "rows"),
        ({"bad_words_token_ids": {0: [[1, 2]]}}, "output_token_ids"),
        ({"min_tokens": {2: (1, (3,))}}, "rows"),
    ],
    ids=["allowed-rows", "grammar-width", "bad-word-row", "bad-word-history", "min"],
)
def test_a_filter_of_the_wrong_shape_is_refused_by_name(broken, match):
    """A short filter would broadcast or index past the block, silently."""
    with pytest.raises(ValueError, match=match):
        accept_sampled_drafts(
            torch.zeros(2, 3, VOCAB),
            torch.zeros(2, 2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            SpecSamplingInputs(
                vocab_size=VOCAB,
                temperature=torch.ones(2),
                filters=SpecTokenFilters(**broken),
            ),
        )


# endregion Each column reads its own state

# region The committed distribution, against an independent reference


def _grammar_allows(history: list[int], token: int) -> bool:
    """A small grammar whose state is the output's last token and its length.

    Token 0 is always allowed, because ``_causal_table`` never rules it out
    and a context with nothing allowed has no distribution.
    """
    if token == 0:
        return True
    if history and token == history[-1]:
        return False
    return (token + len(history)) % 4 != 1


def _allowed_by_filters(history: list[int], filters: dict) -> list[bool]:
    """Which tokens every filter allows after ``history``, in plain Python."""
    allowed = [True] * SMALL_VOCAB
    for token in range(SMALL_VOCAB):
        if filters.get("grammar") and not _grammar_allows(history, token):
            allowed[token] = False
        if "allowed" in filters and token not in filters["allowed"]:
            allowed[token] = False
        for word in filters.get("bad_words", ()):
            prefix = list(word[:-1])
            if word[-1] == token and (not prefix or history[-len(prefix) :] == prefix):
                allowed[token] = False
        if "min_tokens" in filters:
            count, stops = filters["min_tokens"]
            if len(history) < count and token in stops:
                allowed[token] = False
    return allowed


def _filtered_reference_joint(table, controls, filters, length):
    joint = {}
    for sequence in product(range(SMALL_VOCAB), repeat=length):
        probability = 1.0
        token, position = PROMPT[-1], len(PROMPT) - 1
        history: list[int] = []
        for emitted in sequence:
            raw = table[token, position].tolist()
            allowed = _allowed_by_filters(history, filters)
            raw = [v if ok else -math.inf for v, ok in zip(raw, allowed)]
            probability *= _reference_distribution(raw, history, PROMPT, controls)[
                emitted
            ]
            if probability == 0.0:
                break
            history.append(emitted)
            token, position = emitted, position + 1
        if probability > 0.0:
            joint[sequence] = probability
    return joint


def _grammar_rows(
    output: list[int], drafts: list[int], width: int, validated: int
) -> torch.Tensor:
    """``[width, V]`` allowlists the way the scheduler fills a request's rows.

    Row ``j`` is the grammar after the output and the first ``j`` drafts, up to
    the first draft the scheduler found invalid. Past that row the scheduler
    saw ``-1`` and wrote all ones, except the bonus row, which it fills at the
    state after the valid drafts.
    """
    rows = torch.ones(width, SMALL_VOCAB, dtype=torch.bool)
    for column in range(width):
        if column <= validated:
            state = output + drafts[:column]
        elif column == width - 1:
            state = output + drafts[:validated]
        else:
            continue
        rows[column] = torch.tensor(
            [_grammar_allows(state, token) for token in range(SMALL_VOCAB)]
        )
    return rows


def _filtered_simulate(table, controls, filters, drafter, length, rows, k, seed, mode):
    """Draft, validate, verify and commit, with every filter on every column.

    ``mode`` says what happens to drafts the grammar rejects. ``truncate`` is
    a synchronous scheduler: it keeps the valid prefix only. ``keep`` verifies
    every draft, with the scheduler's rows past the first invalid one left
    unconstrained, so the walk alone must reject.
    """
    torch.manual_seed(seed)
    outputs: list[list[int]] = [[] for _ in range(rows)]
    allowed_mask = None
    if "allowed" in filters:
        allowed_mask = torch.ones(rows, SMALL_VOCAB, dtype=torch.bool)
        allowed_mask[:, list(filters["allowed"])] = False
    bad_words = (
        {row: [list(w) for w in filters["bad_words"]] for row in range(rows)}
        if "bad_words" in filters
        else {}
    )
    step = 0
    while min(len(out) for out in outputs) < length:
        last = torch.tensor(
            [out[-1] if out else PROMPT[-1] for out in outputs], dtype=torch.int64
        )
        start = torch.tensor(
            [len(PROMPT) - 1 + len(out) for out in outputs], dtype=torch.int64
        )
        num_valid = torch.tensor(
            [(row + step) % (k + 1) for row in range(rows)], dtype=torch.int32
        )
        drafts = torch.full((rows, k), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        token = last.clone()
        for column in range(k):
            token = drafter(token, start + column)
            drafts[:, column] = torch.where(
                column < num_valid,
                token.to(torch.int32),
                torch.tensor(PLACEHOLDER_TOKEN_ID),
            )

        bitmask = None
        if filters.get("grammar"):
            masks = []
            for row in range(rows):
                row_drafts = drafts[row, : int(num_valid[row])].tolist()
                valid = 0
                while valid < len(row_drafts) and _grammar_allows(
                    outputs[row] + row_drafts[:valid], row_drafts[valid]
                ):
                    valid += 1
                if mode == "truncate":
                    num_valid[row] = valid
                    row_drafts = row_drafts[:valid]
                masks.append(_grammar_rows(outputs[row], row_drafts, k + 1, valid))
            bitmask = _pack(torch.stack(masks))
        # Padding tails are replaced after truncation, so a truncated draft
        # cannot be read as a candidate.
        drafts = torch.where(
            torch.arange(k).unsqueeze(0) < num_valid.unsqueeze(1),
            drafts,
            torch.tensor(PLACEHOLDER_TOKEN_ID, dtype=torch.int32),
        )

        inputs = torch.cat(
            [last.unsqueeze(1), drafts.to(torch.int64).clamp(min=0)], dim=1
        )
        positions = start.unsqueeze(1) + torch.arange(k + 1).unsqueeze(0)
        logits = table[inputs, positions.clamp(max=table.shape[1] - 1)]

        min_tokens = {}
        if "min_tokens" in filters:
            count, stops = filters["min_tokens"]
            min_tokens = {
                row: (count - len(out), tuple(stops))
                for row, out in enumerate(outputs)
                if count > len(out)
            }
        penalties = None
        if any(name in controls for name in ("presence", "frequency", "repetition")):
            penalties = SpecPenalties(
                prompt_token_ids=torch.tensor([PROMPT] * rows, dtype=torch.int64),
                output_token_ids=[list(out) for out in outputs],
                presence=torch.full((rows,), controls.get("presence", 0.0)),
                frequency=torch.full((rows,), controls.get("frequency", 0.0)),
                repetition=torch.full((rows,), controls.get("repetition", 1.0)),
            )
        result = accept_sampled_drafts(
            logits,
            drafts,
            num_valid,
            SpecSamplingInputs(
                vocab_size=SMALL_VOCAB,
                temperature=torch.full((rows,), controls.get("temperature", 1.0)),
                top_k=torch.full(
                    (rows,), controls.get("top_k", SMALL_VOCAB), dtype=torch.int32
                ),
                penalties=penalties,
                filters=SpecTokenFilters(
                    grammar_bitmask=bitmask,
                    allowed_token_ids_mask=allowed_mask,
                    bad_words_token_ids=bad_words,
                    output_token_ids=[list(out) for out in outputs],
                    min_tokens=min_tokens,
                ),
            ),
        )
        for row in range(rows):
            count = int(result.accepted_counts[row])
            outputs[row].extend(result.committed_token_ids[row, :count].tolist())
        step += 1
    return [tuple(out[:length]) for out in outputs]


@pytest.mark.parametrize(
    "filters, controls, mode",
    [
        ({"grammar": True}, {"temperature": 1.0}, "truncate"),
        ({"grammar": True}, {"temperature": 1.0}, "keep"),
        ({"allowed": (0, 1, 2, 4)}, {"temperature": 0.8}, "truncate"),
        ({"bad_words": ((1, 2), (2, 4), (3,))}, {"temperature": 1.0}, "truncate"),
        ({"min_tokens": (2, (3, 5))}, {"temperature": 1.1}, "truncate"),
        (
            {
                "grammar": True,
                "allowed": (0, 1, 2, 3, 4),
                "bad_words": ((1, 2),),
                "min_tokens": (2, (3,)),
            },
            {"temperature": 0.8, "top_k": 4, "presence": 0.5, "repetition": 1.3},
            "keep",
        ),
    ],
    ids=["grammar", "grammar-unvalidated", "allowed", "bad-words", "min-tokens", "all"],
)
@pytest.mark.parametrize("drafter_name", ["argmax", "wrong", "sampling"])
def test_the_committed_sequences_follow_the_filtered_joint_distribution(
    filters, controls, mode, drafter_name
):
    """Three committed tokens per row, jointly, under every filter.

    The joint covers what follows an accepted draft and what follows a
    correction, so a filter applied to column 0 alone, or a bad word matched
    against the committed output only, fails here even though a draftless
    row would agree with the ordinary sampler.
    """
    table = _causal_table(seed=11)
    drafter = {
        "argmax": lambda: _argmax_drafter(table),
        "wrong": lambda: _wrong_drafter,
        "sampling": lambda: _sampling_drafter(table),
    }[drafter_name]()
    rows = 24000
    sequences = _filtered_simulate(
        table, controls, filters, drafter, length=3, rows=rows, k=3, seed=5, mode=mode
    )
    observed: dict = {}
    for sequence in sequences:
        observed[sequence] = observed.get(sequence, 0) + 1

    expected = _filtered_reference_joint(table, controls, filters, length=3)
    statistic, df = _chi_square(observed, expected, rows)

    assert statistic < _chi_square_critical(df), (
        f"chi-square {statistic:.1f} over {df} degrees of freedom: the committed "
        f"sequences under {filters} and {controls} with the {drafter_name} "
        "drafter are not distributed as the filtered target"
    )


def test_the_filtered_joint_check_catches_a_bad_word_matched_on_the_output_only(
    monkeypatch,
):
    """Its power, against bad words read from the committed output alone.

    That walk agrees with the ordinary sampler on every row without drafts, so
    only a drafted column whose own history completes a word's prefix shows it.
    """
    original = spec_accept.apply_speculative_token_filters

    def output_only(target_logits, draft_token_ids, num_valid_drafts, filters):
        return original(
            target_logits,
            draft_token_ids,
            torch.zeros_like(num_valid_drafts),
            filters,
        )

    monkeypatch.setattr(spec_accept, "apply_speculative_token_filters", output_only)
    table = _causal_table(seed=11)
    filters = {"bad_words": ((1, 2), (2, 4), (3,))}
    controls = {"temperature": 1.0}
    sequences = _filtered_simulate(
        table, controls, filters, _argmax_drafter(table), 3, 24000, 3, 5, "truncate"
    )
    observed: dict = {}
    for sequence in sequences:
        observed[sequence] = observed.get(sequence, 0) + 1
    expected = _filtered_reference_joint(table, controls, filters, 3)
    # A banned continuation appears, which the reference gives probability 0.
    with pytest.raises(AssertionError, match="probability zero"):
        _chi_square(observed, expected, len(sequences))


# endregion The committed distribution, against an independent reference
